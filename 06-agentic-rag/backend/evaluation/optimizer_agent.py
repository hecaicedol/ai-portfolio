"""Auto-optimizer agent.

Watches the rolling Ragas scores from `MetricsStore` and, when one metric
drops below threshold, applies a per-metric remediation to `RetrievalParams`
and logs a `TuningEvent`. If the next evaluation window doesn't improve,
the agent reverts.

This file implements the optimization math (no LLM call). It's testable
end-to-end against `InMemoryMetricsStore`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Literal

from evaluation.metrics_store import MetricsStore, TuningEvent


Metric = Literal["faithfulness", "answer_relevancy", "context_recall"]


@dataclass
class RetrievalParams:
    k: int = 10
    rerank_top_n: int = 5
    similarity_threshold: float = 0.6
    rerank_weight: float = 1.0


REMEDIATIONS: dict[Metric, dict[str, float]] = {
    # Low faithfulness → reduce context noise, raise the bar
    "faithfulness": {"k": -2, "similarity_threshold": +0.05, "rerank_top_n": -1},
    # Low recall → cast a wider net
    "context_recall": {"k": +5, "similarity_threshold": -0.05},
    # Low answer relevancy → trust the reranker more, fewer candidates
    "answer_relevancy": {"rerank_weight": +0.2, "rerank_top_n": -1},
}


# Per-field clamps so the optimizer can't drift into nonsense.
_BOUNDS = {
    "k": (1, 100),
    "rerank_top_n": (1, 50),
    "similarity_threshold": (0.0, 1.0),
    "rerank_weight": (0.0, 5.0),
}


def _clamp(field_name: str, value: float) -> float:
    lo, hi = _BOUNDS[field_name]
    return max(lo, min(hi, value))


def _apply(params: RetrievalParams, deltas: dict[str, float]) -> RetrievalParams:
    new = replace(params)
    for field_name, delta in deltas.items():
        current = getattr(new, field_name)
        bounded = _clamp(field_name, current + delta)
        # k and rerank_top_n must be ints
        if field_name in ("k", "rerank_top_n"):
            bounded = int(round(bounded))
        setattr(new, field_name, bounded)
    return new


class OptimizerAgent:
    def __init__(self, *, metrics_store: MetricsStore, params: RetrievalParams) -> None:
        self.metrics_store = metrics_store
        self.params = params
        self._history: list[RetrievalParams] = []

    async def tune(self, *, store_name: str, regressed_metric: Metric) -> RetrievalParams:
        """Apply the remediation for the regressed metric, log a TuningEvent,
        and update `self.params`. Returns the new params."""
        deltas = REMEDIATIONS[regressed_metric]
        before = self.params
        after = _apply(before, deltas)

        await self.metrics_store.log_tuning_event(
            TuningEvent(
                store=store_name,
                regressed=regressed_metric,
                before=asdict(before),
                after=asdict(after),
            )
        )

        self._history.append(before)
        self.params = after
        return after

    async def revert(self, *, store_name: str, reason: str = "no_improvement") -> RetrievalParams:
        """Undo the last `tune()`. Logs a TuningEvent so the timeline records
        the rollback. No-op if there's nothing to revert."""
        if not self._history:
            return self.params
        before = self.params
        after = self._history.pop()

        await self.metrics_store.log_tuning_event(
            TuningEvent(
                store=store_name,
                regressed=f"revert::{reason}",
                before=asdict(before),
                after=asdict(after),
            )
        )
        self.params = after
        return after
