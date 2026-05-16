from typing import Any, TypedDict

from planner.dag_parser import DAG, topological_layers


class ExecutorState(TypedDict, total=False):
    workflow_id: str
    dag: DAG
    completed: dict[str, dict[str, Any]]
    failed: dict[str, dict[str, Any]]
    awaiting_approval: list[str]
    results: dict[str, Any]


class DAGExecutor:
    """
    Builds a LangGraph dynamically from the DAG and runs it.

    Behavior:
      - Each topological layer runs in parallel via asyncio.gather.
      - For nodes with requires_approval=true: emit 'awaiting_approval' SSE
        event, write a pending entry in Redis, and BLOCK that node until
        executor.hitl.resolve() unblocks it (via Redis pubsub).
      - On node failure: invoke replanner.replan_node(failed_node, dag, error)
        and splice the replacement subgraph into the current DAG.
      - On full completion: workflow_memory.save(goal, dag, metrics).

    Param interpolation: '{{n1.title}}' in a node's params is resolved against
    state.completed[n1].output before the MCP call.
    """

    def __init__(self, *, mcp_clients: dict[str, Any], replanner, hitl, workflow_memory, redis) -> None:
        self.mcp_clients = mcp_clients
        self.replanner = replanner
        self.hitl = hitl
        self.workflow_memory = workflow_memory
        self.redis = redis

    async def run(self, *, workflow_id: str, dag: DAG, on_event) -> dict[str, Any]:
        raise NotImplementedError


def _interpolate(params: dict, completed: dict[str, dict[str, Any]]) -> dict:
    """Resolve {{nX.field}} references against completed node outputs."""
    raise NotImplementedError
