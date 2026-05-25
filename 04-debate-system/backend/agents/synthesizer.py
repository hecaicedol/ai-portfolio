"""Reads every statement of the debate and emits a structured executive
memo. Consensus is computed from the final-round stances — separately
from the LLM call, so it is deterministic and auditable."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ValidationError

from agents.base_agent import Statement, _extract_json


class ExecutiveMemo(BaseModel):
    recommended_decision: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    consensus_level: float = Field(..., ge=0.0, le=1.0)
    key_supporting_arguments: list[str]
    key_risks_to_monitor: list[str]
    dissenting_views: list[str]
    next_steps: list[str]


SYNTH_SYSTEM_PROMPT = """You are the Synthesizer for a multi-agent debate.

Read every statement across the three rounds and produce a single
executive memo. You must reflect the actual debate — do not invent
arguments that were never made. If the debate has unresolved dissent,
record it under dissenting_views rather than smoothing it over.

Reply with a single JSON object — nothing before or after:

{
  "recommended_decision": "<one-paragraph recommendation>",
  "confidence": <float in [0, 1]>,
  "key_supporting_arguments": ["<arg>", "<arg>", ...],
  "key_risks_to_monitor": ["<risk>", "<risk>", ...],
  "dissenting_views": ["<view>", ...],
  "next_steps": ["<step>", "<step>", ...]
}
"""


class SynthesizerAgent:
    """Reads all three rounds and emits a structured executive memo.

    Also computes the consensus score from final-round stances —
    deterministically, outside the LLM.
    """

    MAX_JSON_RETRIES: int = 2

    def __init__(self, model: Any) -> None:
        self.model = model

    async def synthesize(
        self,
        *,
        problem: str,
        all_statements: list[Statement],
    ) -> ExecutiveMemo:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            sys_msg: Any = SystemMessage(content=SYNTH_SYSTEM_PROMPT)
            user_msg_factory = lambda body: HumanMessage(content=body)
        except ImportError:  # pragma: no cover
            sys_msg = {"role": "system", "content": SYNTH_SYSTEM_PROMPT}
            user_msg_factory = lambda body: {"role": "user", "content": body}

        final_stances = [s for s in all_statements if s.round == 3]
        consensus_level = self.consensus(final_stances)
        history = "\n\n".join(
            f"[round {s.round} · {s.role} · stance={s.stance} · conf={s.confidence:.2f}]\n{s.content}"
            for s in all_statements
        )
        body = (
            f"<problem>\n{problem}\n</problem>\n\n"
            f"<debate_history>\n{history}\n</debate_history>\n\n"
            f"<consensus_level>{consensus_level:.3f}</consensus_level>\n"
        )

        last_error: Exception | None = None
        for _ in range(self.MAX_JSON_RETRIES + 1):
            response = await self.model.ainvoke([sys_msg, user_msg_factory(body)])
            try:
                payload = _extract_json(response.content)
                payload["consensus_level"] = consensus_level
                return ExecutiveMemo(**payload)
            except (ValidationError, KeyError, ValueError, TypeError) as exc:
                last_error = exc
                continue
        raise RuntimeError(
            f"Synthesizer: invalid JSON after {self.MAX_JSON_RETRIES + 1} attempts "
            f"(last error: {last_error})"
        )

    @staticmethod
    def consensus(final_statements: list[Statement]) -> float:
        """Map stances to a numeric axis and compute 1 - normalized variance.

        strong_no = -2, no = -1, neutral = 0, yes = 1, strong_yes = 2
        Variance is divided by 4 (the max possible variance on a {-2, 2}
        axis with two-way disagreement) so the score lives in [0, 1].
        """
        axis = {"strong_no": -2, "no": -1, "neutral": 0, "yes": 1, "strong_yes": 2}
        values = [axis[s.stance] for s in final_statements]
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return max(0.0, 1.0 - variance / 4.0)
