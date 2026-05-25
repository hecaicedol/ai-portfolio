"""Base class for the five debate agents.

Each subclass sets `role_name`, `perspective`, and `system_prompt`. The
shared `generate_statement()` builds the prompt, invokes the injected
model (production wires `ChatAnthropic`, tests wire `ScriptedDebateLLM`),
and parses a strict JSON envelope into a `Statement`. Two retries on
JSON failure — matches the contract used in P1/P3 so failures are loud,
not silent.
"""
from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError


Stance = Literal["strong_yes", "yes", "neutral", "no", "strong_no"]
RoundType = Literal["opening", "rebuttal", "final"]


class Statement(BaseModel):
    role: str
    round: int
    content: str
    key_points: list[str]
    confidence: float = Field(..., ge=0.0, le=1.0)
    stance: Stance


_STANCES: tuple[Stance, ...] = (
    "strong_no", "no", "neutral", "yes", "strong_yes",
)


ROUND_INSTRUCTIONS: dict[RoundType, str] = {
    "opening": (
        "OPENING ROUND. Lay out your strongest case from your perspective. "
        "Enumerate 3 concrete key points. State an honest initial stance."
    ),
    "rebuttal": (
        "REBUTTAL ROUND. Read the openings above. Identify the 2 strongest "
        "opposing points and explain why they are weaker than they look. "
        "If they survive scrutiny, say so — you may shift stance."
    ),
    "final": (
        "FINAL ROUND. State your final position. If any argument shifted "
        "you, name it explicitly. Your stance must be one of "
        "strong_yes / yes / neutral / no / strong_no."
    ),
}


JSON_ENVELOPE_HINT = """\
Reply with a single JSON object — nothing before or after — of the form:

{
  "content": "<2-4 sentence argument>",
  "key_points": ["<bullet>", "<bullet>", "<bullet>"],
  "confidence": <float in [0, 1]>,
  "stance": "<strong_no|no|neutral|yes|strong_yes>"
}
"""


def _format_history(history: list[Statement]) -> str:
    if not history:
        return "(no previous statements)"
    lines = []
    for s in history:
        lines.append(
            f"[round {s.round} · {s.role} · stance={s.stance} · conf={s.confidence:.2f}]"
            f"\n{s.content}"
        )
    return "\n\n".join(lines)


def _extract_json(raw: str) -> dict[str, Any]:
    """Pull the first JSON object out of a raw LLM reply. Tolerates code
    fences and surrounding chatter — same recovery pattern as P1's critic."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```\s*$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


class BaseDebateAgent:
    """Base class for all five debate agents.

    Subclasses set `role_name`, `perspective`, and `system_prompt`.
    """

    role_name: str = "agent"
    perspective: str = ""
    system_prompt: str = ""

    MAX_JSON_RETRIES: int = 2

    def __init__(self, model: Any) -> None:
        self.model = model

    async def generate_statement(
        self,
        *,
        problem: str,
        debate_history: list[Statement],
        round_type: RoundType,
        round_number: int,
    ) -> Statement:
        # Lazy import so the package is usable in environments without
        # langchain installed (the tests use a duck-typed ScriptedDebateLLM
        # that accepts plain dicts).
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            sys_msg: Any = SystemMessage(content=self.system_prompt)
            user_msg_factory = lambda body: HumanMessage(content=body)
        except ImportError:  # pragma: no cover — covered indirectly by tests
            sys_msg = {"role": "system", "content": self.system_prompt}
            user_msg_factory = lambda body: {"role": "user", "content": body}

        instruction = ROUND_INSTRUCTIONS[round_type]
        history_block = _format_history(debate_history)
        body = (
            f"<problem>\n{problem}\n</problem>\n\n"
            f"<previous_statements>\n{history_block}\n</previous_statements>\n\n"
            f"<round_type>{round_type}</round_type>\n"
            f"<instruction>\n{instruction}\n\n{JSON_ENVELOPE_HINT}</instruction>"
        )

        last_error: Exception | None = None
        for _ in range(self.MAX_JSON_RETRIES + 1):
            response = await self.model.ainvoke([sys_msg, user_msg_factory(body)])
            try:
                payload = _extract_json(response.content)
                stance = payload.get("stance", "neutral")
                if stance not in _STANCES:
                    stance = "neutral"
                return Statement(
                    role=self.role_name,
                    round=round_number,
                    content=str(payload.get("content", "")),
                    key_points=[str(p) for p in payload.get("key_points", [])],
                    confidence=float(payload.get("confidence", 0.5)),
                    stance=stance,
                )
            except (json.JSONDecodeError, ValidationError, KeyError, TypeError) as exc:
                last_error = exc
                continue
        raise RuntimeError(
            f"{self.role_name}: model returned invalid JSON for {self.MAX_JSON_RETRIES + 1} "
            f"consecutive attempts (last error: {last_error})"
        )
