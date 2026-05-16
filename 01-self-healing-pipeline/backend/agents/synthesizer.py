from typing import Any

from api.schemas import CriticReport


class SynthesizerAgent:
    """
    Final assembly step. Combines validated extraction with audit metadata
    (critic scores, retry trace) into a single, auditable output object.
    No LLM call — this is pure orchestration logic.
    """

    def run(
        self,
        *,
        extracted: dict[str, Any],
        critic_report: CriticReport,
        iterations: int,
        errors_history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "data": extracted,
            "audit": {
                "score": critic_report.overall_score,
                "principles": [p.model_dump() for p in critic_report.principles],
                "iterations": iterations,
                "self_healed": iterations > 1,
                "errors_history": errors_history,
            },
        }
