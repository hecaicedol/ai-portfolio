from typing import Any
from planner.dag_parser import DAG, DAGNode


class Replanner:
    """
    On node failure, asks Claude for a replacement subgraph.

    Inputs:
      - The failed node (id, tool, action, params, error message)
      - The current DAG (so the replanner understands downstream dependencies)
      - The same tool catalogue the planner used

    Output: a small DAG (1–N nodes) that replaces the failed node. The Replanner
    splices it in: the new subgraph inherits the failed node's depends_on, and
    any node that depended on the failed node now depends on the new subgraph's
    terminal node(s).
    """

    def __init__(self, *, model: str, api_key: str, tool_catalogue: dict[str, Any]) -> None:
        self.model = model
        self.api_key = api_key
        self.tool_catalogue = tool_catalogue

    async def replan_node(self, *, failed_node: DAGNode, dag: DAG, error: str) -> list[DAGNode]:
        raise NotImplementedError
