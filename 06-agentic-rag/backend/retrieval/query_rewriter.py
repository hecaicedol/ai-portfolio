"""LLM-based query rewriter.

Claude generates `n` alternative phrasings of the query so the hybrid search
catches chunks whose vocabulary differs from the user's wording.

The constructor takes the model as a parameter (an object with an `.ainvoke`
method that takes a list of LangChain messages and returns an object with a
`.content` attribute). This is the same pattern used by P1's agents — it
keeps tests cheap (inject `ScriptedLLM`) and production simple (inject
`ChatAnthropic`).
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

SYSTEM_PROMPT = """You are a query-rewriting agent. Given a user question,
return exactly N alternative phrasings as a JSON array of strings.

Make the rewrites lexically diverse: vary domain vocabulary, level of formality,
and granularity. Do NOT answer the question — only rephrase it.

Output ONLY a JSON array of N strings, no prose, no markdown fences.
Example output: ["how does X work?", "explain mechanism of X", "X internals"]"""

MAX_PARSE_ATTEMPTS = 3


class QueryRewriterParseError(RuntimeError):
    """Raised when the rewriter cannot produce a valid JSON array of rewrites
    after MAX_PARSE_ATTEMPTS attempts."""


class QueryRewriter:
    def __init__(self, model: Any) -> None:
        self.model = model

    async def rewrite(self, query: str, n: int = 3) -> list[str]:
        user_message = (
            f"Question:\n{query}\n\n"
            f"Return exactly {n} alternative phrasings as a JSON array."
        )

        last_exc: Exception | None = None
        for attempt in range(MAX_PARSE_ATTEMPTS):
            extra = ""
            if attempt > 0:
                extra = (
                    "\n\nIMPORTANT: your previous response was not a valid JSON "
                    "array of strings. Return ONLY the array now."
                )
            response = await self.model.ainvoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT + extra),
                    HumanMessage(content=user_message),
                ]
            )
            try:
                parsed = _safe_json_list(response.content)
                rewrites = [str(item).strip() for item in parsed if isinstance(item, str) and str(item).strip()]
                if rewrites:
                    return rewrites[:n]
                raise ValueError("rewriter returned an empty list")
            except (json.JSONDecodeError, ValueError) as exc:
                last_exc = exc

        raise QueryRewriterParseError(
            f"Query rewriter could not produce a valid JSON array after {MAX_PARSE_ATTEMPTS} attempts"
        ) from last_exc


def _safe_json_list(text: str) -> list[Any]:
    """Tolerant JSON-array parser. Strips markdown fences; if the response
    contains prose around an array, slices to the outermost `[ ... ]`."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()

    try:
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise ValueError("response parsed but was not a JSON array")
        return parsed
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
        if not isinstance(parsed, list):
            raise ValueError("response parsed but was not a JSON array")
        return parsed
