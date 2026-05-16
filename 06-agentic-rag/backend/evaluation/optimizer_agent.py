from dataclasses import dataclass
from typing import Literal


Metric = Literal["faithfulness", "answer_relevancy", "context_recall"]


@dataclass
class RetrievalParams:
    k: int = 10
    rerank_top_n: int = 5
    similarity_threshold: float = 0.6
    rerank_weight: float = 1.0


REMEDIATIONS: dict[Metric, dict[str, float]] = {
    # Reduce context noise: fewer, higher-quality chunks
    "faithfulness":      {"k": -2, "similarity_threshold": +0.05, "rerank_top_n": -1},
    # Expand retrieval: more chunks, looser threshold
    "context_recall":    {"k": +5, "similarity_threshold": -0.05},
    # Improve relevance ordering: more rerank emphasis
    "answer_relevancy":  {"rerank_weight": +0.2, "rerank_top_n": -1},
}


class OptimizerAgent:
    """
    Triggered by EvaluatorAgent when rolling-window quality drops.

    Workflow:
      1. Identify the metric with the largest drop.
      2. Apply per-metric remediation deltas (REMEDIATIONS).
      3. Persist the new params + log a "tuning event" with before/after.
      4. Evaluate next 10 queries with new params; if improvement < 5%,
         revert and try the next remediation.
    """

    def __init__(self, *, metrics_store, params: RetrievalParams) -> None:
        self.metrics_store = metrics_store
        self.params = params

    async def tune(self, *, store_name: str, regressed_metric: Metric) -> RetrievalParams:
        deltas = REMEDIATIONS[regressed_metric]
        new_params = RetrievalParams(**vars(self.params))
        for field, delta in deltas.items():
            setattr(new_params, field, max(1, getattr(new_params, field) + delta))
        await self.metrics_store.log_tuning_event(
            store=store_name, regressed=regressed_metric, before=self.params, after=new_params
        )
        self.params = new_params
        return new_params
