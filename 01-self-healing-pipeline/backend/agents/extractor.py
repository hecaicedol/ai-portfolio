import json
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

EXTRACTOR_SYSTEM_PROMPT = """You are a strict structured-data extraction agent.
You receive a business document (invoice, contract, receipt, etc.) and must return
a JSON object with all relevant fields you can identify.

Rules:
- Output ONLY a valid JSON object, no prose, no markdown fences.
- Use null for unknown fields, never invent values.
- Numbers must be numeric (not strings).
- Dates must be ISO 8601 (YYYY-MM-DD).
- If a "critic_feedback" section is present in the user message, address each point.
"""


class ExtractorAgent:
    def __init__(self, model: ChatAnthropic) -> None:
        self.model = model

    async def run(
        self,
        *,
        document_type: str,
        content: str,
        critic_feedback: list[str] | None = None,
    ) -> dict[str, Any]:
        user_parts = [
            f"<document_type>{document_type}</document_type>",
            f"<document>\n{content}\n</document>",
        ]
        if critic_feedback:
            joined = "\n".join(f"- {fb}" for fb in critic_feedback)
            user_parts.append(f"<critic_feedback>\n{joined}\n</critic_feedback>")
        user_message = "\n\n".join(user_parts)

        response = await self.model.ainvoke(
            [
                SystemMessage(content=EXTRACTOR_SYSTEM_PROMPT),
                HumanMessage(content=user_message),
            ]
        )
        return _safe_json(response.content)


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
