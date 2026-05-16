from typing import Any


class MetricsStore:
    """
    PostgreSQL-backed metrics log.

    Tables:
      query_metrics(id, ts, store, question_hash, faithfulness, answer_relevancy,
                    context_recall, latency_retrieval_ms, latency_rerank_ms,
                    latency_generation_ms, token_cost_usd)

      tuning_events(id, ts, store, regressed_metric, before_params jsonb,
                    after_params jsonb)

    Used by:
      - EvaluatorAgent to record each query's metrics.
      - OptimizerAgent to log tuning events.
      - Frontend benchmark dashboard to render time-series charts.
    """

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    async def connect(self) -> None:
        raise NotImplementedError

    async def record_query(self, **fields: Any) -> None:
        raise NotImplementedError

    async def rolling_avg(self, *, store: str, metric: str, window: int = 20) -> float:
        raise NotImplementedError

    async def window_for(self, store: str, window: int = 100) -> dict[str, Any]:
        raise NotImplementedError

    async def overview(self) -> dict[str, Any]:
        """Aggregates per-store time-series for the live benchmark dashboard."""
        raise NotImplementedError

    async def log_tuning_event(self, **fields: Any) -> None:
        raise NotImplementedError
