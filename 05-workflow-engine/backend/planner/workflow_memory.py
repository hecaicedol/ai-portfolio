from typing import Any
from planner.dag_parser import DAG


class WorkflowMemory:
    """
    pgvector-backed store of past successful workflows.

      save(goal, dag, execution_metrics) → embed(goal); INSERT
      find_similar(goal, k=3) → cosine similarity over embedded goals
      recent(limit) → audit log for the frontend history view
    """

    def __init__(self, dsn: str, *, embed) -> None:
        self.dsn = dsn
        self.embed = embed

    async def connect(self) -> None:
        raise NotImplementedError

    async def save(self, *, goal: str, dag: DAG, metrics: dict[str, Any]) -> int:
        raise NotImplementedError

    async def find_similar(self, goal: str, k: int = 3) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        raise NotImplementedError
