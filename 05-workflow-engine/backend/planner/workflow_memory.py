"""Past-successful-workflow store with cosine similarity over the
embedded goal text. Two backends, one Protocol:

  * `InMemoryWorkflowMemory` — pure-Python, hash-bag embed, used by tests
    and the no-budget demos.
  * `PostgresWorkflowMemory` — pgvector backend (stubbed for Slice 2).
"""
from __future__ import annotations

import math
import time
from typing import Any, Awaitable, Callable, Protocol

from planner.dag_parser import DAG


EmbedFn = Callable[[str], Awaitable[list[float]]]


class WorkflowMemory(Protocol):
    async def connect(self) -> None: ...
    async def save(self, *, goal: str, dag: DAG, metrics: dict[str, Any]) -> int: ...
    async def find_similar(self, goal: str, k: int = 3) -> list[dict[str, Any]]: ...
    async def recent(self, limit: int = 50) -> list[dict[str, Any]]: ...


def _cosine(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


class InMemoryWorkflowMemory:
    def __init__(self, *, embed: EmbedFn) -> None:
        self.embed = embed
        self._records: list[dict[str, Any]] = []

    async def connect(self) -> None:
        return None

    async def save(self, *, goal: str, dag: DAG, metrics: dict[str, Any]) -> int:
        emb = await self.embed(goal)
        rec = {
            "id": len(self._records),
            "goal": goal,
            "dag": dag.model_dump(),
            "metrics": dict(metrics),
            "embedding": emb,
            "saved_at": time.time(),
        }
        self._records.append(rec)
        return rec["id"]

    async def find_similar(self, goal: str, k: int = 3) -> list[dict[str, Any]]:
        if not self._records:
            return []
        q = await self.embed(goal)
        scored = [(_cosine(q, r["embedding"]), r) for r in self._records]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "goal": r["goal"],
                "dag": r["dag"],
                "metrics": r["metrics"],
                "similarity": s,
            }
            for s, r in scored[:k]
        ]

    async def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        return [
            {"id": r["id"], "goal": r["goal"], "saved_at": r["saved_at"], "metrics": r["metrics"]}
            for r in self._records[-limit:][::-1]
        ]


class PostgresWorkflowMemory:
    """pgvector-backed store. Stubbed for Slice 2 — schema sketch only."""

    def __init__(self, dsn: str, *, embed: EmbedFn) -> None:
        self.dsn = dsn
        self.embed = embed
        self._pool = None

    async def connect(self) -> None:
        raise NotImplementedError("PostgresWorkflowMemory is Slice 2")

    async def save(self, *, goal: str, dag: DAG, metrics: dict[str, Any]) -> int:
        raise NotImplementedError

    async def find_similar(self, goal: str, k: int = 3) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        raise NotImplementedError
