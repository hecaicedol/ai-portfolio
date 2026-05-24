"""Metrics persistence layer.

Defines:
  - `MetricsStore` — Protocol describing the surface the evaluator and
    optimizer agents depend on.
  - `InMemoryMetricsStore` — pure-Python implementation used by tests
    and dev mode. Keeps everything in lists; rolling averages are
    computed on the fly. Drop-in replacement for the Postgres backend
    so the eval / optimization loop can be exercised end-to-end without
    a database.
  - `PostgresMetricsStore` — production backend, still stubs.

Both implementations share the same record schema:
  - `QueryRecord` (per-query metrics + latency breakdown + cost)
  - `TuningEvent` (optimizer's before/after parameter snapshot)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Protocol


# ── records ─────────────────────────────────────────────────────────────────

@dataclass
class QueryRecord:
    store: str
    faithfulness: float
    answer_relevancy: float
    context_recall: float
    latency_retrieval_ms: float
    latency_rerank_ms: float
    latency_generation_ms: float
    token_cost_usd: float
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class TuningEvent:
    store: str
    regressed: str
    before: dict[str, float]
    after: dict[str, float]
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ── protocol ────────────────────────────────────────────────────────────────

class MetricsStore(Protocol):
    """Surface the evaluator + optimizer agents depend on."""

    async def record_query(self, record: QueryRecord) -> None: ...
    async def rolling_avg(self, *, store: str, metric: str, window: int = 20) -> float: ...
    async def log_tuning_event(self, event: TuningEvent) -> None: ...
    async def window_for(self, store: str, window: int = 100) -> dict[str, Any]: ...
    async def overview(self) -> dict[str, Any]: ...


# ── in-memory backend ───────────────────────────────────────────────────────

class InMemoryMetricsStore:
    """Concrete `MetricsStore` backed by Python lists. Useful for tests
    and any dev mode that doesn't want to spin up Postgres."""

    name = "in_memory_metrics"

    def __init__(self) -> None:
        self._queries: list[QueryRecord] = []
        self._tuning: list[TuningEvent] = []

    async def record_query(self, record: QueryRecord) -> None:
        self._queries.append(record)

    async def rolling_avg(
        self, *, store: str, metric: str, window: int = 20
    ) -> float:
        recent = _recent_for_store(self._queries, store, window)
        if not recent:
            return 0.0
        values = [getattr(r, metric) for r in recent]
        return sum(values) / len(values)

    async def log_tuning_event(self, event: TuningEvent) -> None:
        self._tuning.append(event)

    async def window_for(self, store: str, window: int = 100) -> dict[str, Any]:
        recent = _recent_for_store(self._queries, store, window)
        return {
            "store": store,
            "n": len(recent),
            "records": [asdict(r) for r in recent],
            "tuning_events": [
                asdict(t) for t in self._tuning if t.store == store
            ],
        }

    async def overview(self) -> dict[str, Any]:
        stores: dict[str, list[QueryRecord]] = {}
        for r in self._queries:
            stores.setdefault(r.store, []).append(r)
        return {
            "stores": {
                store: {
                    "n": len(rows),
                    "avg_faithfulness": _mean(r.faithfulness for r in rows),
                    "avg_answer_relevancy": _mean(r.answer_relevancy for r in rows),
                    "avg_context_recall": _mean(r.context_recall for r in rows),
                    "avg_total_latency_ms": _mean(
                        r.latency_retrieval_ms + r.latency_rerank_ms + r.latency_generation_ms
                        for r in rows
                    ),
                    "total_token_cost_usd": sum(r.token_cost_usd for r in rows),
                    "tuning_events": sum(1 for t in self._tuning if t.store == store),
                }
                for store, rows in stores.items()
            },
            "total_queries": len(self._queries),
            "total_tuning_events": len(self._tuning),
        }


# ── postgres backend (still stub) ───────────────────────────────────────────

class PostgresMetricsStore:
    """PostgreSQL-backed metrics log. Not implemented yet — InMemoryMetricsStore
    covers the dev + test paths. Slice 3 will wire this up against the same
    Postgres instance the rest of the stack uses.

    Tables (planned):
      query_metrics(id, ts, store, faithfulness, answer_relevancy,
                    context_recall, latency_retrieval_ms, latency_rerank_ms,
                    latency_generation_ms, token_cost_usd)
      tuning_events(id, ts, store, regressed, before jsonb, after jsonb)
    """

    name = "postgres_metrics"

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    async def connect(self) -> None:
        raise NotImplementedError

    async def record_query(self, record: QueryRecord) -> None:
        raise NotImplementedError

    async def rolling_avg(self, *, store: str, metric: str, window: int = 20) -> float:
        raise NotImplementedError

    async def log_tuning_event(self, event: TuningEvent) -> None:
        raise NotImplementedError

    async def window_for(self, store: str, window: int = 100) -> dict[str, Any]:
        raise NotImplementedError

    async def overview(self) -> dict[str, Any]:
        raise NotImplementedError


# ── helpers ─────────────────────────────────────────────────────────────────

def _recent_for_store(
    records: list[QueryRecord], store: str, window: int
) -> list[QueryRecord]:
    return [r for r in records if r.store == store][-window:]


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
