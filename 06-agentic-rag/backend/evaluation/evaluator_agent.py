"""Evaluator agent.

After every query, scores it with Ragas-like metrics, persists everything
to `MetricsStore`, and — if the rolling window for any metric drops below
threshold — triggers `OptimizerAgent` with the metric that dropped most.

The Ragas scorer itself is injected (the production version calls Ragas
+ Claude). Tests inject a callable that returns pre-computed metrics so
the agent's *logic* is what's verified, not the third-party library.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable, Protocol

from pydantic import BaseModel, Field

from evaluation.metrics_store import MetricsStore, QueryRecord
from evaluation.optimizer_agent import Metric, OptimizerAgent


class QualityMetrics(BaseModel):
    faithfulness: float = Field(..., ge=0.0, le=1.0)
    answer_relevancy: float = Field(..., ge=0.0, le=1.0)
    context_recall: float = Field(..., ge=0.0, le=1.0)

    @property
    def overall(self) -> float:
        return (self.faithfulness + self.answer_relevancy + self.context_recall) / 3.0


class Scorer(Protocol):
    """Callable that turns a (question, answer, contexts, ground_truth) tuple
    into QualityMetrics. Production: wraps Ragas + Claude. Tests: a fake."""

    async def __call__(
        self,
        *,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None,
    ) -> QualityMetrics: ...


_TRACKED: tuple[Metric, ...] = ("faithfulness", "answer_relevancy", "context_recall")


@dataclass
class LatencyBreakdown:
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    generation_ms: float = 0.0


class EvaluatorAgent:
    """Runs after every query (background-task style; doesn't block the
    response). Computes Ragas metrics, persists, and decides whether to
    poke the optimizer."""

    def __init__(
        self,
        *,
        metrics_store: MetricsStore,
        optimizer: OptimizerAgent,
        scorer: Scorer,
        threshold: float = 0.75,
        window: int = 20,
    ) -> None:
        self.metrics_store = metrics_store
        self.optimizer = optimizer
        self.scorer = scorer
        self.threshold = threshold
        self.window = window

    async def evaluate(
        self,
        *,
        store_name: str,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
        latency: LatencyBreakdown,
        token_cost_usd: float,
    ) -> QualityMetrics:
        metrics = await self.scorer(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
        )

        await self.metrics_store.record_query(
            QueryRecord(
                store=store_name,
                faithfulness=metrics.faithfulness,
                answer_relevancy=metrics.answer_relevancy,
                context_recall=metrics.context_recall,
                latency_retrieval_ms=latency.retrieval_ms,
                latency_rerank_ms=latency.rerank_ms,
                latency_generation_ms=latency.generation_ms,
                token_cost_usd=token_cost_usd,
            )
        )

        regressed = await self._find_regressed_metric(store_name)
        if regressed is not None:
            await self.optimizer.tune(store_name=store_name, regressed_metric=regressed)

        return metrics

    async def _find_regressed_metric(self, store_name: str) -> Metric | None:
        """Return the metric whose rolling avg is the most below `threshold`,
        or None if every metric is fine."""
        below: dict[Metric, float] = {}
        for m in _TRACKED:
            avg = await self.metrics_store.rolling_avg(
                store=store_name, metric=m, window=self.window
            )
            if avg < self.threshold:
                below[m] = avg
        if not below:
            return None
        return min(below.items(), key=lambda kv: kv[1])[0]
