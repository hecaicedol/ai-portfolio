"""DAG executor.

Runs a DAG layer-by-layer. Within a layer, nodes run in parallel
(asyncio.gather). On failure the replanner is invoked and its
replacement subgraph is spliced into the DAG in-place — the next
iteration recomputes layers and continues. HITL gating pauses a node
until an out-of-band approval lands.

Param interpolation supports `{{nX.field.path}}` references against the
output of previously-completed nodes — the same syntax as GitHub Actions
expressions.
"""
from __future__ import annotations

import asyncio
import re
from typing import Any, Awaitable, Callable

from planner.dag_parser import DAG, DAGNode, topological_layers
from planner.workflow_memory import WorkflowMemory
from executor.hitl import HITLBroker
from executor.replanner import Replanner


EventCallback = Callable[[dict[str, Any]], Awaitable[None]]


async def _noop_event(_: dict[str, Any]) -> None:
    return None


# {{ node_id.dotted.path }}
_INTERP_RE = re.compile(r"\{\{\s*(\w+)\.([\w\.]+)\s*\}\}")


def _interp_string(s: str, completed: dict[str, dict[str, Any]]) -> Any:
    """Resolve {{nX.field}} references inside a string.

    If the WHOLE string is a single reference and the resolved value is
    not a string, return the value itself (preserve type). Otherwise
    substitute and stitch back as a string.
    """
    match = _INTERP_RE.fullmatch(s.strip())
    if match:
        return _resolve_path(match.group(1), match.group(2), completed, fallback=s)

    def replace(m: re.Match[str]) -> str:
        val = _resolve_path(m.group(1), m.group(2), completed, fallback=m.group(0))
        return str(val) if val is not None else m.group(0)

    return _INTERP_RE.sub(replace, s)


def _resolve_path(node_id: str, dotted: str, completed: dict[str, dict[str, Any]], fallback: Any) -> Any:
    record = completed.get(node_id)
    if not record:
        return fallback
    cur: Any = record.get("output")
    for piece in dotted.split("."):
        if isinstance(cur, dict):
            cur = cur.get(piece)
        else:
            return fallback
        if cur is None:
            return fallback
    return cur


def _interpolate(params: Any, completed: dict[str, dict[str, Any]]) -> Any:
    if isinstance(params, str):
        return _interp_string(params, completed)
    if isinstance(params, dict):
        return {k: _interpolate(v, completed) for k, v in params.items()}
    if isinstance(params, list):
        return [_interpolate(v, completed) for v in params]
    return params


def _splice_replacements(
    dag: DAG,
    failed_node: DAGNode,
    replacements: list[DAGNode],
) -> None:
    """Replace `failed_node` in `dag` with `replacements`, in-place.

    Rules:
    - The first replacement inherits `failed_node.depends_on` if it has
      no depends_on of its own.
    - The last replacement is the new terminal. Any node that previously
      depended on `failed_node` now depends on the terminal replacement.
    """
    if not replacements:
        return
    if not replacements[0].depends_on:
        replacements[0].depends_on = list(failed_node.depends_on)
    terminal_id = replacements[-1].id
    # Insert replacements (filter out any colliding ids, just in case)
    existing_ids = {n.id for n in dag.nodes}
    for r in replacements:
        if r.id in existing_ids:
            raise ValueError(f"Replacement id {r.id!r} collides with existing DAG node")
        dag.nodes.append(r)
        existing_ids.add(r.id)
    # Rewire downstream
    for n in dag.nodes:
        if failed_node.id in n.depends_on:
            new_deps = [d for d in n.depends_on if d != failed_node.id]
            new_deps.append(terminal_id)
            n.depends_on = new_deps
    # Remove the failed node
    dag.nodes[:] = [n for n in dag.nodes if n.id != failed_node.id]


class DAGExecutor:
    """Drives a DAG: parallel within layers, HITL between, replan on
    failure, persist on success."""

    def __init__(
        self,
        *,
        mcp_clients: dict[str, Any],
        replanner: Replanner | None,
        hitl: HITLBroker,
        workflow_memory: WorkflowMemory,
    ) -> None:
        self.mcp_clients = mcp_clients
        self.replanner = replanner
        self.hitl = hitl
        self.workflow_memory = workflow_memory

    async def run(
        self,
        *,
        workflow_id: str,
        dag: DAG,
        on_event: EventCallback | None = None,
    ) -> dict[str, Any]:
        on_event = on_event or _noop_event
        completed: dict[str, dict[str, Any]] = {}
        failed: list[str] = []

        await on_event({"type": "workflow_start", "workflow_id": workflow_id, "goal": dag.goal})

        max_iterations = 50  # guard against runaway replans
        for _ in range(max_iterations):
            layers = topological_layers(dag)
            pending_layer: list[str] | None = None
            for layer in layers:
                pending = [
                    nid for nid in layer
                    if nid not in completed and nid not in failed
                ]
                if pending:
                    pending_layer = pending
                    break
            if pending_layer is None:
                break

            by_id = {n.id: n for n in dag.nodes}
            await on_event({"type": "layer_start", "nodes": list(pending_layer)})
            await asyncio.gather(*(
                self._execute_node(
                    workflow_id=workflow_id,
                    node=by_id[nid],
                    dag=dag,
                    completed=completed,
                    failed=failed,
                    on_event=on_event,
                )
                for nid in pending_layer
            ))
            await on_event({"type": "layer_complete", "nodes": list(pending_layer)})

        metrics = {
            "nodes_completed": len(completed),
            "nodes_failed": len(failed),
            "nodes_in_dag": len(dag.nodes),
        }
        if not failed:
            try:
                await self.workflow_memory.save(goal=dag.goal, dag=dag, metrics=metrics)
            except NotImplementedError:
                # Slice-2 backend not wired; that's fine in tests.
                pass

        await on_event({
            "type": "workflow_complete",
            "workflow_id": workflow_id,
            "completed_nodes": list(completed.keys()),
            "failed_nodes": list(failed),
            "metrics": metrics,
        })
        return {
            "workflow_id": workflow_id,
            "dag": dag.model_dump(),
            "completed": completed,
            "failed": failed,
            "metrics": metrics,
        }

    async def _execute_node(
        self,
        *,
        workflow_id: str,
        node: DAGNode,
        dag: DAG,
        completed: dict[str, dict[str, Any]],
        failed: list[str],
        on_event: EventCallback,
    ) -> None:
        params = _interpolate(node.params, completed)

        if node.requires_approval:
            await self.hitl.request(
                workflow_id=workflow_id,
                node_id=node.id,
                payload={"tool": node.tool, "action": node.action, "params": params},
            )
            await on_event({"type": "awaiting_approval", "node": node.id})
            result = await self.hitl.wait_for(workflow_id=workflow_id, node_id=node.id)
            if not result.get("approved"):
                failed.append(node.id)
                await on_event({"type": "node_rejected", "node": node.id})
                return
            edited = result.get("edited_params")
            if edited:
                params = edited

        client = self.mcp_clients.get(node.tool)
        if client is None:
            failed.append(node.id)
            await on_event({
                "type": "node_failed",
                "node": node.id,
                "error": f"no MCP client wired for tool {node.tool!r}",
            })
            return

        try:
            output = await client.call(node.action, params)
        except Exception as exc:
            await on_event({
                "type": "node_failed",
                "node": node.id,
                "error": str(exc),
            })
            if self.replanner is None:
                failed.append(node.id)
                return
            try:
                replacements = await self.replanner.replan_node(
                    failed_node=node, dag=dag, error=str(exc),
                )
                _splice_replacements(dag, node, replacements)
                await on_event({
                    "type": "replanned",
                    "failed_node": node.id,
                    "replacement_count": len(replacements),
                    "replacement_ids": [r.id for r in replacements],
                })
            except Exception as exc2:
                failed.append(node.id)
                await on_event({
                    "type": "replan_failed",
                    "node": node.id,
                    "error": str(exc2),
                })
            return

        completed[node.id] = {"output": output, "params": params}
        await on_event({
            "type": "node_complete",
            "node": node.id,
            "tool": node.tool,
            "action": node.action,
            "output": output,
        })
