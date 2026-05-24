"""Tests for the EvaluatorAgent.

The Ragas scorer is injected as a fake callable so the agent's logic is
what gets verified (recording, rolling avg lookup, optimizer trigger),
not the third-party library.
"""
from __future__ import annotations

import pytest

from evaluation.evaluator_agent import (
    EvaluatorAgent,
    LatencyBreakdown,
    QualityMetrics,
)
from evaluation.metrics_store import InMemoryMetricsStore, QueryRecord
from evaluation.optimizer_agent import OptimizerAgent, RetrievalParams


def make_scorer(metrics: QualityMetrics):
    """Build a fake scorer that returns the same metrics every time."""

    async def scorer(*, question, answer, contexts, ground_truth):  # noqa: ARG001
        return metrics

    return scorer


def make_sequence_scorer(*metrics_list: QualityMetrics):
    """Build a fake scorer that returns successive metrics on each call."""
    queue = list(metrics_list)

    async def scorer(*, question, answer, contexts, ground_truth):  # noqa: ARG001
        if not queue:
            raise RuntimeError("scorer ran out of scripted metrics")
        return queue.pop(0)

    return scorer


def _seed_history(metrics_store: InMemoryMetricsStore, n: int, **fields):
    """Pre-fill the metrics store with N identical records to set up the rolling window."""
    base = {
        "store": "pgvector",
        "faithfulness": 0.9,
        "answer_relevancy": 0.9,
        "context_recall": 0.9,
        "latency_retrieval_ms": 10.0,
        "latency_rerank_ms": 5.0,
        "latency_generation_ms": 400.0,
        "token_cost_usd": 0.001,
        **fields,
    }
    for _ in range(n):
        metrics_store._queries.append(QueryRecord(**base))


@pytest.mark.asyncio
async def test_evaluate_records_metrics_and_returns_them():
    store = InMemoryMetricsStore()
    opt = OptimizerAgent(metrics_store=store, params=RetrievalParams())
    metrics = QualityMetrics(faithfulness=0.9, answer_relevancy=0.9, context_recall=0.9)
    agent = EvaluatorAgent(
        metrics_store=store,
        optimizer=opt,
        scorer=make_scorer(metrics),
        threshold=0.75,
        window=5,
    )

    result = await agent.evaluate(
        store_name="pgvector",
        question="q",
        answer="a",
        contexts=["c"],
        latency=LatencyBreakdown(retrieval_ms=10, rerank_ms=5, generation_ms=400),
        token_cost_usd=0.002,
    )

    assert result == metrics
    overview = await store.overview()
    assert overview["total_queries"] == 1
    # No tuning event because metrics are well above threshold
    assert overview["total_tuning_events"] == 0


@pytest.mark.asyncio
async def test_evaluate_does_not_trigger_optimizer_when_all_metrics_pass():
    store = InMemoryMetricsStore()
    opt = OptimizerAgent(metrics_store=store, params=RetrievalParams(k=10))
    metrics = QualityMetrics(faithfulness=0.9, answer_relevancy=0.85, context_recall=0.85)
    agent = EvaluatorAgent(
        metrics_store=store,
        optimizer=opt,
        scorer=make_scorer(metrics),
        threshold=0.75,
        window=5,
    )
    await agent.evaluate(
        store_name="pgvector",
        question="q",
        answer="a",
        contexts=[],
        latency=LatencyBreakdown(),
        token_cost_usd=0.0,
    )
    assert opt.params.k == 10  # untouched


@pytest.mark.asyncio
async def test_evaluate_triggers_optimizer_on_low_faithfulness():
    store = InMemoryMetricsStore()
    opt = OptimizerAgent(metrics_store=store, params=RetrievalParams(k=10, similarity_threshold=0.6))

    # Pre-seed the rolling window with low faithfulness
    _seed_history(store, n=4, faithfulness=0.5, answer_relevancy=0.9, context_recall=0.9)

    low_quality = QualityMetrics(faithfulness=0.5, answer_relevancy=0.9, context_recall=0.9)
    agent = EvaluatorAgent(
        metrics_store=store,
        optimizer=opt,
        scorer=make_scorer(low_quality),
        threshold=0.75,
        window=5,
    )

    await agent.evaluate(
        store_name="pgvector",
        question="q",
        answer="a",
        contexts=[],
        latency=LatencyBreakdown(),
        token_cost_usd=0.0,
    )

    # Faithfulness remediation: k -2, threshold +0.05
    assert opt.params.k == 8
    assert abs(opt.params.similarity_threshold - 0.65) < 1e-9


@pytest.mark.asyncio
async def test_evaluator_picks_metric_with_largest_drop_when_multiple_below_threshold():
    """If two metrics dip under threshold, the optimizer is triggered for
    the one whose rolling avg is the lowest (largest drop)."""
    store = InMemoryMetricsStore()
    opt = OptimizerAgent(metrics_store=store, params=RetrievalParams(k=10, rerank_top_n=5))

    # context_recall avg = 0.5, answer_relevancy avg = 0.6, faithfulness avg = 0.9
    _seed_history(store, n=4, faithfulness=0.9, answer_relevancy=0.6, context_recall=0.5)

    next_metrics = QualityMetrics(faithfulness=0.9, answer_relevancy=0.6, context_recall=0.5)
    agent = EvaluatorAgent(
        metrics_store=store,
        optimizer=opt,
        scorer=make_scorer(next_metrics),
        threshold=0.75,
        window=5,
    )
    await agent.evaluate(
        store_name="pgvector",
        question="q",
        answer="a",
        contexts=[],
        latency=LatencyBreakdown(),
        token_cost_usd=0.0,
    )

    # context_recall has the lowest rolling avg → its remediation applied: k +5
    assert opt.params.k == 15


@pytest.mark.asyncio
async def test_evaluator_isolates_per_store():
    """A low metric on pgvector should not tune the params we're using on
    qdrant. Here we just confirm the rolling avg lookup is store-scoped."""
    store = InMemoryMetricsStore()
    opt_pg = OptimizerAgent(metrics_store=store, params=RetrievalParams(k=10))

    # Pre-seed pgvector with BAD scores, qdrant with GREAT scores
    _seed_history(store, n=4, faithfulness=0.4)
    _seed_history(store, n=4, store="qdrant", faithfulness=0.95)

    pg_metrics = QualityMetrics(faithfulness=0.4, answer_relevancy=0.9, context_recall=0.9)
    qd_metrics = QualityMetrics(faithfulness=0.95, answer_relevancy=0.9, context_recall=0.9)
    agent = EvaluatorAgent(
        metrics_store=store,
        optimizer=opt_pg,
        scorer=make_sequence_scorer(qd_metrics, pg_metrics),
        threshold=0.75,
        window=5,
    )

    # Querying qdrant first: its rolling avg is great → no tuning
    await agent.evaluate(
        store_name="qdrant",
        question="q",
        answer="a",
        contexts=[],
        latency=LatencyBreakdown(),
        token_cost_usd=0.0,
    )
    assert opt_pg.params.k == 10  # untouched

    # Now querying pgvector: rolling avg of faithfulness is below threshold → tune
    await agent.evaluate(
        store_name="pgvector",
        question="q",
        answer="a",
        contexts=[],
        latency=LatencyBreakdown(),
        token_cost_usd=0.0,
    )
    assert opt_pg.params.k == 8
