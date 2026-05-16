import asyncio
from datetime import datetime
from typing import Any

from graph.neo4j_client import Neo4jClient


class StalenessReport(dict):
    """Latest stale-knowledge report. Stored in-memory; persist to disk if you need durability."""


class StalenessAgent:
    """
    Background task that runs every 24h:
      1. Queries Neo4j for nodes whose `updated_at` is older than `threshold_days`.
      2. Groups stale nodes by domain/topic (uses GDS community detection if available,
         otherwise groups by label).
      3. Calls Claude to summarize: what's outdated and what might have changed,
         producing a markdown report.
      4. Stores the report (in-memory + exposes via API).
    """

    def __init__(self, *, neo4j: Neo4jClient, threshold_days: int = 30, period_seconds: int = 86_400) -> None:
        self.neo4j = neo4j
        self.threshold_days = threshold_days
        self.period_seconds = period_seconds
        self._latest: dict[str, Any] | None = None
        self._task: asyncio.Task | None = None

    def latest_report(self) -> dict[str, Any] | None:
        return self._latest

    async def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            self._task = None

    async def _loop(self) -> None:
        while True:
            try:
                self._latest = await self._build_report()
            except Exception as exc:  # noqa: BLE001
                self._latest = {"error": str(exc), "at": datetime.utcnow().isoformat()}
            await asyncio.sleep(self.period_seconds)

    async def _build_report(self) -> dict[str, Any]:
        raise NotImplementedError
