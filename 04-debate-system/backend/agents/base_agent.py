from typing import Literal
from pydantic import BaseModel, Field

Stance = Literal["strong_yes", "yes", "neutral", "no", "strong_no"]
RoundType = Literal["opening", "rebuttal", "final"]


class Statement(BaseModel):
    role: str
    round: int
    content: str
    key_points: list[str]
    confidence: float = Field(..., ge=0.0, le=1.0)
    stance: Stance


class BaseDebateAgent:
    """
    Base class for all five debate agents.
    Subclasses set `role_name`, `perspective`, and `system_prompt`.
    """

    role_name: str = "agent"
    perspective: str = ""
    system_prompt: str = ""

    def __init__(self, model) -> None:
        self.model = model

    async def generate_statement(
        self,
        *,
        problem: str,
        debate_history: list[Statement],
        round_type: RoundType,
        round_number: int,
    ) -> Statement:
        """
        Compose user message:
          <problem>...</problem>
          <previous_statements>... full history with role labels ...</previous_statements>
          <round_type>opening|rebuttal|final</round_type>
          <instruction>{role-specific instruction for this round}</instruction>

        For rebuttals: instruct the agent to target the 2 strongest opposing points.
        For finals: allow stance change with explicit justification.
        """
        raise NotImplementedError
