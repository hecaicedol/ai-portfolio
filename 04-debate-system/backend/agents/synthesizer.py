from typing import Any
from pydantic import BaseModel

from agents.base_agent import Statement


class ExecutiveMemo(BaseModel):
    recommended_decision: str
    confidence: float
    consensus_level: float
    key_supporting_arguments: list[str]
    key_risks_to_monitor: list[str]
    dissenting_views: list[str]
    next_steps: list[str]


class SynthesizerAgent:
    """
    Reads all three rounds and emits a structured executive memo.
    Also computes the consensus score from final-round stances.
    """

    def __init__(self, model) -> None:
        self.model = model

    async def synthesize(self, *, problem: str, all_statements: list[Statement]) -> ExecutiveMemo:
        raise NotImplementedError

    @staticmethod
    def consensus(final_statements: list[Statement]) -> float:
        """
        Map stances to a numeric axis and compute 1 - normalized variance.
        strong_no = -2, no = -1, neutral = 0, yes = 1, strong_yes = 2
        Returns 0.0 (max disagreement) to 1.0 (full consensus).
        """
        axis = {"strong_no": -2, "no": -1, "neutral": 0, "yes": 1, "strong_yes": 2}
        values = [axis[s.stance] for s in final_statements]
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return max(0.0, 1.0 - variance / 4.0)
