"""Tests for the OptimizerAgent: per-metric remediation, clamping, revert."""
from __future__ import annotations

from dataclasses import asdict

import pytest

from evaluation.metrics_store import InMemoryMetricsStore
from evaluation.optimizer_agent import (
    REMEDIATIONS,
    OptimizerAgent,
    RetrievalParams,
    _BOUNDS,
)


@pytest.mark.asyncio
async def test_tune_applies_faithfulness_remediation():
    """Low faithfulness → tighten: smaller k, higher threshold, smaller rerank_top_n."""
    store = InMemoryMetricsStore()
    opt = OptimizerAgent(metrics_store=store, params=RetrievalParams(k=10, rerank_top_n=5, similarity_threshold=0.6))
    new = await opt.tune(store_name="pgvector", regressed_metric="faithfulness")
    assert new.k == 8                              # 10 + (-2)
    assert new.rerank_top_n == 4                   # 5 + (-1)
    assert abs(new.similarity_threshold - 0.65) < 1e-9   # 0.6 + 0.05


@pytest.mark.asyncio
async def test_tune_applies_recall_remediation():
    """Low recall → cast wider net: bigger k, lower threshold."""
    store = InMemoryMetricsStore()
    opt = OptimizerAgent(metrics_store=store, params=RetrievalParams(k=10, similarity_threshold=0.6))
    new = await opt.tune(store_name="qdrant", regressed_metric="context_recall")
    assert new.k == 15                             # 10 + 5
    assert abs(new.similarity_threshold - 0.55) < 1e-9   # 0.6 - 0.05


@pytest.mark.asyncio
async def test_tune_applies_answer_relevancy_remediation():
    """Low answer_relevancy → trust the reranker more, fewer candidates."""
    store = InMemoryMetricsStore()
    opt = OptimizerAgent(metrics_store=store, params=RetrievalParams(rerank_top_n=5, rerank_weight=1.0))
    new = await opt.tune(store_name="pgvector", regressed_metric="answer_relevancy")
    assert new.rerank_top_n == 4                   # 5 + (-1)
    assert abs(new.rerank_weight - 1.2) < 1e-9     # 1.0 + 0.2


@pytest.mark.asyncio
async def test_tune_clamps_to_bounds():
    """k can't go below 1, similarity_threshold can't exceed 1.0."""
    store = InMemoryMetricsStore()
    opt = OptimizerAgent(
        metrics_store=store,
        params=RetrievalParams(k=1, rerank_top_n=1, similarity_threshold=0.99),
    )
    new = await opt.tune(store_name="pgvector", regressed_metric="faithfulness")
    assert new.k == _BOUNDS["k"][0]
    assert new.rerank_top_n == _BOUNDS["rerank_top_n"][0]
    assert new.similarity_threshold <= _BOUNDS["similarity_threshold"][1]


@pytest.mark.asyncio
async def test_tune_logs_event_with_before_and_after():
    store = InMemoryMetricsStore()
    initial = RetrievalParams(k=10, similarity_threshold=0.6)
    opt = OptimizerAgent(metrics_store=store, params=initial)
    await opt.tune(store_name="pgvector", regressed_metric="faithfulness")
    overview = await store.overview()
    assert overview["total_tuning_events"] == 1
    snapshot = await store.window_for("pgvector", window=10)
    event = snapshot["tuning_events"][0]
    assert event["regressed"] == "faithfulness"
    assert event["before"]["k"] == 10
    assert event["after"]["k"] == 8


@pytest.mark.asyncio
async def test_revert_restores_previous_params():
    store = InMemoryMetricsStore()
    original = RetrievalParams(k=10, similarity_threshold=0.6)
    opt = OptimizerAgent(metrics_store=store, params=original)

    await opt.tune(store_name="pgvector", regressed_metric="faithfulness")
    assert opt.params.k == 8
    assert opt.params.similarity_threshold > 0.6

    reverted = await opt.revert(store_name="pgvector", reason="no_improvement")
    assert reverted.k == 10
    assert reverted.similarity_threshold == 0.6

    # Both events (tune + revert) should be on the timeline
    overview = await store.overview()
    assert overview["total_tuning_events"] == 2


@pytest.mark.asyncio
async def test_revert_with_empty_history_is_noop():
    store = InMemoryMetricsStore()
    opt = OptimizerAgent(metrics_store=store, params=RetrievalParams())
    before = opt.params
    after = await opt.revert(store_name="pgvector")
    assert after == before
    overview = await store.overview()
    assert overview["total_tuning_events"] == 0


def test_remediation_map_covers_all_tracked_metrics():
    """Belt and suspenders: every Metric literal has a remediation defined."""
    expected = {"faithfulness", "answer_relevancy", "context_recall"}
    assert set(REMEDIATIONS.keys()) == expected
