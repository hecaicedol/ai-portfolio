import json
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from api.schemas import CriticReport, PrincipleScore
from memory.episodic import EpisodicMemory

CRITIC_SYSTEM_PROMPT = """You are a Critic agent grounded in Constitutional AI.
You evaluate an Extractor's output against four principles and return JSON only.

Principles (each scored 0.0 – 1.0):
1. completeness   — All required fields are present and non-empty.
2. accuracy       — Every extracted value is verifiable from the source document.
3. consistency    — Fields do not contradict each other (e.g. line items must sum to total).
4. format         — Output matches expected types (numbers numeric, dates ISO-8601).

Output schema (strict):
{
  "overall_score": float,
  "principles": [
    {"principle": "completeness", "score": float, "feedback": "..."},
    {"principle": "accuracy",     "score": float, "feedback": "..."},
    {"principle": "consistency",  "score": float, "feedback": "..."},
    {"principle": "format",       "score": float, "feedback": "..."}
  ]
}

- Be specific in feedback. "Missing tax_id field" is good; "incomplete" is not.
- overall_score must equal the unweighted mean of the four principle scores.
- If past similar errors are provided, weight your judgment with them — do not be lenient
  about a class of mistake the system has made before.
"""


class CriticAgent:
    def __init__(self, model: ChatAnthropic, memory: EpisodicMemory, pass_threshold: float) -> None:
        self.model = model
        self.memory = memory
        self.pass_threshold = pass_threshold

    async def run(
        self,
        *,
        document_type: str,
        source: str,
        extracted: dict[str, Any],
        structural: dict[str, Any],
    ) -> CriticReport:
        query = f"document_type={document_type} fields={list(extracted.keys())}"
        past = await self.memory.retrieve_similar_errors(query, k=3)
        past_block = json.dumps(past, default=str) if past else "[]"

        user_message = (
            f"<document_type>{document_type}</document_type>\n"
            f"<source>\n{source}\n</source>\n"
            f"<extracted>\n{json.dumps(extracted, indent=2, default=str)}\n</extracted>\n"
            f"<structural_check>\n{json.dumps(structural)}\n</structural_check>\n"
            f"<past_similar_errors>\n{past_block}\n</past_similar_errors>"
        )

        response = await self.model.ainvoke(
            [SystemMessage(content=CRITIC_SYSTEM_PROMPT), HumanMessage(content=user_message)]
        )
        raw = _safe_json(response.content)
        principles = [PrincipleScore(**p) for p in raw["principles"]]
        overall = float(raw.get("overall_score", sum(p.score for p in principles) / len(principles)))
        return CriticReport(
            overall_score=overall,
            principles=principles,
            passes=overall >= self.pass_threshold,
            similar_past_errors=[e.get("error_type", "") for e in past],
        )


def _safe_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise
