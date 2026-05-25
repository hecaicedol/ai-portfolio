"""Tests for evaluation.ragas_scorer.

The real Ragas call needs Claude + the ragas library + a dataset of
test rows; we never exercise that path here. We exercise the contract
that the EvaluatorAgent depends on (Scorer Protocol) using FixedScorer
and the `compute_fn` injection point on RagasScorer.
"""
from __future__ import annotations

import pytest

from evaluation.evaluator_agent import EvaluatorAgent, LatencyBreakdown, QualityMetrics
from evaluation.metrics_store import InMemoryMetricsStore
from evaluation.optimizer_agent import OptimizerAgent, RetrievalParams
from evaluation.ragas_scorer import FixedScorer, RagasScorer


@pytest.mark.asyncio
async def test_fixed_scorer_always_returns_same_metrics():
    metrics = QualityMetrics(faithfulness=0.8, answer_relevancy=0.7, context_recall=0.6)
    s = FixedScorer(metrics)
    out = await s(question='q', answer='a', contexts=[], ground_truth=None)
    assert out == metrics
    assert s.calls == 1
    # Calling again returns the same metrics, increments counter
    await s(question='q2', answer='a2', contexts=['x'], ground_truth='y')
    assert s.calls == 2


@pytest.mark.asyncio
async def test_ragas_scorer_respects_compute_fn_injection():
    """compute_fn lets us avoid pulling in Ragas during tests."""
    async def fake_compute(*, question, answer, contexts, ground_truth):
        # Echo something the test can verify
        return {
            'faithfulness': 0.93,
            'answer_relevancy': 0.71,
            'context_recall': 0.85,
        }

    scorer = RagasScorer(compute_fn=fake_compute)
    out = await scorer(question='q', answer='a', contexts=['c1', 'c2'], ground_truth='gt')
    assert out.faithfulness == pytest.approx(0.93)
    assert out.answer_relevancy == pytest.approx(0.71)
    assert out.context_recall == pytest.approx(0.85)
    assert scorer.calls == 1


@pytest.mark.asyncio
async def test_ragas_scorer_clamps_out_of_range_values():
    """Some Ragas versions can produce >1.0 or negative values on edge
    cases (numerical noise); our wrapper clamps to [0, 1] before handing
    the result back to EvaluatorAgent (which has Pydantic validation)."""
    async def noisy_compute(*, question, answer, contexts, ground_truth):
        return {'faithfulness': 1.05, 'answer_relevancy': -0.02, 'context_recall': 0.42}

    scorer = RagasScorer(compute_fn=noisy_compute)
    out = await scorer(question='q', answer='a', contexts=[], ground_truth=None)
    assert out.faithfulness == 1.0
    assert out.answer_relevancy == 0.0
    assert out.context_recall == 0.42


@pytest.mark.asyncio
async def test_ragas_scorer_plugs_into_evaluator_agent():
    """End-to-end: EvaluatorAgent + InMemoryMetricsStore + OptimizerAgent
    + a RagasScorer with a fake compute_fn. This is the realistic wiring
    for the production benchmark — only the compute_fn differs."""
    store = InMemoryMetricsStore()
    opt = OptimizerAgent(metrics_store=store, params=RetrievalParams(k=10))
    async def passing_compute(*, question, answer, contexts, ground_truth):
        return {'faithfulness': 0.9, 'answer_relevancy': 0.88, 'context_recall': 0.92}

    scorer = RagasScorer(compute_fn=passing_compute)
    agent = EvaluatorAgent(
        metrics_store=store, optimizer=opt, scorer=scorer,
        threshold=0.75, window=5,
    )
    result = await agent.evaluate(
        store_name='pgvector',
        question='What is RRF?', answer='Reciprocal Rank Fusion …',
        contexts=['…'], ground_truth='RRF combines ranked lists',
        latency=LatencyBreakdown(retrieval_ms=12, rerank_ms=4, generation_ms=380),
        token_cost_usd=0.003,
    )
    assert result.faithfulness == 0.9
    overview = await store.overview()
    assert overview['total_queries'] == 1
    # Scores are above threshold, so no tuning was triggered
    assert overview['total_tuning_events'] == 0
    assert opt.params.k == 10  # untouched
