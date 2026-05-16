from typing import Any
from pydantic import BaseModel, Field


class QualityMetrics(BaseModel):
    faithfulness: float = Field(..., ge=0.0, le=1.0)
    answer_relevancy: float = Field(..., ge=0.0, le=1.0)
    context_recall: float = Field(..., ge=0.0, le=1.0)
    overall: float = Field(..., ge=0.0, le=1.0)


class EvaluatorAgent:
    """
    Runs after every query (in background, doesn't block the response):
      1. Computes Ragas metrics: faithfulness, answer_relevancy, context_recall
      2. Records latency breakdown (retrieval / reranking / generation) and token cost
      3. Persists everything to metrics_store
      4. If rolling avg over last `window` queries < `threshold`,
         triggers OptimizerAgent with the metric that dropped most.
    """

    def __init__(
        self,
        *,
        metrics_store,
        optimizer,
        threshold: float = 0.75,
        window: int = 20,
    ) -> None:
        self.metrics_store = metrics_store
        self.optimizer = optimizer
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
        latency_breakdown: dict[str, float],
        token_cost: float,
    ) -> QualityMetrics:
        raise NotImplementedError
