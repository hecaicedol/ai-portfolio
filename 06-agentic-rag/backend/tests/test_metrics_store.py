"""Tests for the InMemoryMetricsStore."""
from __future__ import annotations

import pytest

from evaluation.metrics_store import InMemoryMetricsStore, QueryRecord, TuningEvent


def _record(
    store: str = "pgvector",
    faithfulness: float = 0.9,
    answer_relevancy: float = 0.9,
    context_recall: float = 0.9,
) -> QueryRecord:
    return QueryRecord(
        store=store,
        faithfulness=faithfulness,
        answer_relevancy=answer_relevancy,
        context_recall=context_recall,
        latency_retrieval_ms=10.0,
        latency_rerank_ms=5.0,
        latency_generation_ms=400.0,
        token_cost_usd=0.001,
    )


@pytest.mark.asyncio
async def test_record_query_persists_in_order():
    s = InMemoryMetricsStore()
    await s.record_query(_record(faithfulness=0.7))
    await s.record_query(_record(faithfulness=0.8))
    overview = await s.overview()
    assert overview["total_queries"] == 2


@pytest.mark.asyncio
async def test_rolling_avg_over_specified_window():
    s = InMemoryMetricsStore()
    for v in [0.2, 0.3, 0.4, 0.9, 0.95]:
        await s.record_query(_record(faithfulness=v))
    # last 3 values: 0.4 + 0.9 + 0.95 = 2.25 / 3 = 0.75
    avg = await s.rolling_avg(store="pgvector", metric="faithfulness", window=3)
    assert abs(avg - 0.75) < 1e-9


@pytest.mark.asyncio
async def test_rolling_avg_isolates_by_store():
    s = InMemoryMetricsStore()
    await s.record_query(_record(store="pgvector", faithfulness=0.5))
    await s.record_query(_record(store="qdrant", faithfulness=0.9))
    avg_pg = await s.rolling_avg(store="pgvector", metric="faithfulness", window=10)
    avg_qd = await s.rolling_avg(store="qdrant", metric="faithfulness", window=10)
    assert avg_pg == 0.5
    assert avg_qd == 0.9


@pytest.mark.asyncio
async def test_rolling_avg_empty_returns_zero():
    s = InMemoryMetricsStore()
    avg = await s.rolling_avg(store="pgvector", metric="faithfulness", window=10)
    assert avg == 0.0


@pytest.mark.asyncio
async def test_log_tuning_event_persists_per_store():
    s = InMemoryMetricsStore()
    await s.log_tuning_event(
        TuningEvent(
            store="pgvector",
            regressed="faithfulness",
            before={"k": 10},
            after={"k": 8},
        )
    )
    overview = await s.overview()
    assert overview["total_tuning_events"] == 1
    assert overview["stores"] == {}  # no queries yet


@pytest.mark.asyncio
async def test_window_for_returns_per_store_records():
    s = InMemoryMetricsStore()
    await s.record_query(_record(store="pgvector", faithfulness=0.7))
    await s.record_query(_record(store="qdrant", faithfulness=0.9))
    snapshot = await s.window_for("pgvector", window=10)
    assert snapshot["n"] == 1
    assert snapshot["records"][0]["faithfulness"] == 0.7


@pytest.mark.asyncio
async def test_overview_aggregates_per_store():
    s = InMemoryMetricsStore()
    for _ in range(3):
        await s.record_query(_record(store="pgvector", faithfulness=0.9))
    for _ in range(2):
        await s.record_query(_record(store="qdrant", faithfulness=0.6))
    overview = await s.overview()
    assert overview["stores"]["pgvector"]["n"] == 3
    assert overview["stores"]["qdrant"]["n"] == 2
    assert abs(overview["stores"]["pgvector"]["avg_faithfulness"] - 0.9) < 1e-9
    assert abs(overview["stores"]["qdrant"]["avg_faithfulness"] - 0.6) < 1e-9
