"""Production Scorer that wraps the Ragas library.

EvaluatorAgent already accepts any callable matching the `Scorer` Protocol
(see `evaluator_agent.py`). This module provides the *real* implementation
that calls Ragas + Claude — and a thin fake (`FixedScorer`) that just
returns canned metrics, useful for benchmarks where we want deterministic
inputs to the optimizer logic.

Ragas is a heavy dependency (pandas, datasets, huggingface-hub). We import
it lazily inside `__call__` so the module itself is cheap to import in
tests and dev tooling.
"""
from __future__ import annotations

from typing import Any

from evaluation.evaluator_agent import QualityMetrics


class FixedScorer:
    """Always returns the same metrics. Useful for tests of EvaluatorAgent /
    OptimizerAgent that need a predictable score regardless of inputs."""

    def __init__(self, metrics: QualityMetrics) -> None:
        self.metrics = metrics
        self.calls = 0

    async def __call__(
        self,
        *,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None,
    ) -> QualityMetrics:
        self.calls += 1
        return self.metrics


class RagasScorer:
    """Production Ragas wrapper.

    Computes the three metrics our EvaluatorAgent tracks
    (faithfulness, answer_relevancy, context_recall) by invoking Ragas
    with an Anthropic-backed LLM + Voyage embeddings.

    The `_compute` hook is split out so unit tests can inject a stub
    rather than actually pulling in Ragas + Claude.

    Parameters
    ----------
    anthropic_api_key
        Key for the Claude model Ragas calls.
    model
        Claude model identifier; defaults to claude-sonnet-4-5.
    compute_fn
        Optional override of the underlying Ragas call — tests pass a
        coroutine returning `dict[str, float]` so the suite never has to
        import Ragas itself.
    """

    def __init__(
        self,
        *,
        anthropic_api_key: str | None = None,
        model: str = "claude-sonnet-4-5",
        compute_fn: Any | None = None,
    ) -> None:
        self.anthropic_api_key = anthropic_api_key
        self.model = model
        self._compute = compute_fn or self._compute_via_ragas
        self.calls = 0

    async def __call__(
        self,
        *,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None,
    ) -> QualityMetrics:
        self.calls += 1
        raw = await self._compute(
            question=question, answer=answer,
            contexts=contexts, ground_truth=ground_truth,
        )
        return QualityMetrics(
            faithfulness=_clamp(raw.get("faithfulness", 0.0)),
            answer_relevancy=_clamp(raw.get("answer_relevancy", 0.0)),
            context_recall=_clamp(raw.get("context_recall", 0.0)),
        )

    async def _compute_via_ragas(
        self,
        *,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None,
    ) -> dict[str, float]:
        """Real Ragas evaluation. Imports are lazy so the module itself
        stays cheap; this method runs only in production paths."""
        # Lazy imports to keep tests + dev light
        from ragas import evaluate  # type: ignore
        from ragas.metrics import (  # type: ignore
            faithfulness, answer_relevancy, context_recall,
        )
        from datasets import Dataset  # type: ignore
        from langchain_anthropic import ChatAnthropic  # type: ignore

        llm = ChatAnthropic(
            model=self.model,
            api_key=self.anthropic_api_key,
            temperature=0,
        )
        # Ragas expects datasets-style rows with these exact column names
        row = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [ground_truth or ""],
        }
        ds = Dataset.from_dict(row)
        metrics = [faithfulness, answer_relevancy]
        if ground_truth:
            metrics.append(context_recall)

        result = evaluate(ds, metrics=metrics, llm=llm)
        # Ragas returns a result object exposing per-metric scores
        return {
            "faithfulness": float(result.get("faithfulness", 0.0)) if "faithfulness" in result else 0.0,
            "answer_relevancy": float(result.get("answer_relevancy", 0.0)) if "answer_relevancy" in result else 0.0,
            "context_recall": float(result.get("context_recall", 0.0)) if "context_recall" in result else 0.0,
        }


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, float(v)))
