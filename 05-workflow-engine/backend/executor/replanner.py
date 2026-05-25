"""On node failure, asks the model for a replacement subgraph.

The model is injected (same contract as PlannerAgent). The Replanner
itself does not splice — it just returns the new nodes. The DAGExecutor
performs the splice so it can also rewire downstream `depends_on`.
"""
from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from planner.dag_parser import DAG, DAGNode
from planner.planner_agent import _extract_json


REPLANNER_SYSTEM_PROMPT = """You are a Workflow Replanner. A node in a
running DAG failed. Produce a replacement subgraph (one or more nodes)
that achieves the same intent through a different path.

Output format (strict JSON):
{
  "nodes": [
    {"id": "...", "name": "...", "tool": "...", "action": "...",
     "params": {...}, "requires_approval": bool, "depends_on": [...]}
  ]
}

Rules:
- Use unique ids that do NOT collide with the existing DAG's ids.
- The first node's depends_on may be left empty; the executor will
  inject the failed node's depends_on automatically.
- Tools and actions must come from the same catalogue as the planner.
- Reply with ONE JSON object — no prose before or after.
"""


class Replanner:
    MAX_JSON_RETRIES: int = 2

    def __init__(self, *, model: Any, tool_catalogue: dict[str, Any]) -> None:
        self.model = model
        self.tool_catalogue = tool_catalogue

    async def replan_node(
        self,
        *,
        failed_node: DAGNode,
        dag: DAG,
        error: str,
    ) -> list[DAGNode]:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            sys_msg: Any = SystemMessage(content=REPLANNER_SYSTEM_PROMPT)
            user_factory = lambda body: HumanMessage(content=body)
        except ImportError:  # pragma: no cover
            sys_msg = {"role": "system", "content": REPLANNER_SYSTEM_PROMPT}
            user_factory = lambda body: {"role": "user", "content": body}

        body = (
            f"<failed_node>\n{failed_node.model_dump_json(indent=2)}\n</failed_node>\n\n"
            f"<error>\n{error}\n</error>\n\n"
            f"<current_dag>\n{dag.model_dump_json(indent=2)}\n</current_dag>\n\n"
            f"<tool_catalogue>\n{json.dumps(self.tool_catalogue, indent=2, sort_keys=True)}\n</tool_catalogue>"
        )

        last_error: Exception | None = None
        for _ in range(self.MAX_JSON_RETRIES + 1):
            response = await self.model.ainvoke([sys_msg, user_factory(body)])
            try:
                payload = _extract_json(response.content)
                nodes = [DAGNode(**n) for n in payload.get("nodes", [])]
                if not nodes:
                    raise ValueError("Replanner returned an empty node list")
                return nodes
            except (json.JSONDecodeError, ValidationError, KeyError, TypeError, ValueError) as exc:
                last_error = exc
                continue
        raise RuntimeError(
            f"Replanner: invalid JSON / empty replacement after {self.MAX_JSON_RETRIES + 1} attempts "
            f"(last error: {last_error})"
        )
