"""Background task that periodically asks the graph store for nodes
whose `updated_at` is older than `threshold_days` and produces a
markdown report grouped by label. A summarizer model is optional — if
not injected, the report is built deterministically from the node
properties so the agent can run in environments without API keys."""
from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from graph.neo4j_client import GraphStore


Summarizer = Callable[[str, list[dict[str, Any]]], Awaitable[str]]


async def _default_summarize(label: str, nodes: list[dict[str, Any]]) -> str:
    """Deterministic fallback when no LLM summarizer is wired."""
    sample = ", ".join(n["properties"].get("name", n["id"]) for n in nodes[:5])
    extra = "" if len(nodes) <= 5 else f", +{len(nodes) - 5} more"
    return f"{len(nodes)} {label}(s) past freshness threshold: {sample}{extra}."


class StalenessAgent:
    def __init__(
        self,
        *,
        graph: GraphStore,
        threshold_days: int = 30,
        period_seconds: int = 86_400,
        summarizer: Summarizer | None = None,
    ) -> None:
        self.graph = graph
        self.threshold_days = threshold_days
        self.period_seconds = period_seconds
        self.summarize = summarizer or _default_summarize
        self._latest: dict[str, Any] | None = None
        self._task: asyncio.Task[None] | None = None

    def latest_report(self) -> dict[str, Any] | None:
        return self._latest

    async def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, BaseException):
                pass
            self._task = None

    async def _loop(self) -> None:
        while True:
            try:
                self._latest = await self._build_report()
            except Exception as exc:  # noqa: BLE001
                self._latest = {"error": str(exc), "at": datetime.now(timezone.utc).isoformat()}
            await asyncio.sleep(self.period_seconds)

    async def _build_report(self) -> dict[str, Any]:
        stale = await self.graph.stale_nodes(days=self.threshold_days)
        by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for node in stale:
            by_label[node["label"]].append(node)
        sections: list[dict[str, Any]] = []
        for label, nodes in sorted(by_label.items()):
            summary = await self.summarize(label, nodes)
            sections.append({
                "label": label,
                "count": len(nodes),
                "summary": summary,
                "node_ids": [n["id"] for n in nodes],
            })
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "threshold_days": self.threshold_days,
            "total_stale": len(stale),
            "sections": sections,
        }
