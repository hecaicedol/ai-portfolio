"""Calls an LLM with a strict system prompt to extract entities and
relationships from a document chunk. The model is injected (production:
ChatAnthropic; tests: ScriptedLLM) so the same code path runs without
API keys."""
from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, ValidationError


class EntityType(str, Enum):
    person = "person"
    organization = "organization"
    project = "project"
    concept = "concept"
    event = "event"
    document = "document"
    date = "date"
    amount = "amount"


class Entity(BaseModel):
    id: str = Field(..., description="Stable id — slugified canonical name")
    name: str
    type: EntityType
    properties: dict[str, Any] = Field(default_factory=dict)
    aliases: list[str] = Field(default_factory=list)


class Relationship(BaseModel):
    source_id: str
    target_id: str
    type: str = Field(..., description="UPPER_SNAKE_CASE relation, e.g. WORKS_ON, OWNS, DEPENDS_ON")
    properties: dict[str, Any] = Field(default_factory=dict)


class ExtractedKnowledge(BaseModel):
    entities: list[Entity]
    relationships: list[Relationship]


EXTRACTOR_SYSTEM_PROMPT = """You extract structured knowledge from business documents.
Return ONLY a JSON object matching this schema:
{
  "entities": [
    {"id": "alice-johnson", "name": "Alice Johnson", "type": "person",
     "properties": {"role": "CTO"}, "aliases": ["A. Johnson"]}
  ],
  "relationships": [
    {"source_id": "alice-johnson", "target_id": "project-atlas",
     "type": "OWNS", "properties": {"since": "2024-03-01"}}
  ]
}

Rules:
- entity.id = lowercase, dash-separated, deterministic across documents.
- Use existing ids if the entity is the same person/org/project — do not duplicate.
- relationship.type = UPPER_SNAKE_CASE verb phrase.
- Skip entities you cannot ground in the text.
- Reply with ONE JSON object — no prose before or after.
"""


def _extract_json(raw: str) -> dict[str, Any]:
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


class EntityExtractor:
    MAX_JSON_RETRIES: int = 2

    def __init__(self, *, model: Any) -> None:
        self.model = model

    async def extract(self, *, document_text: str, document_id: str) -> ExtractedKnowledge:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            sys_msg: Any = SystemMessage(content=EXTRACTOR_SYSTEM_PROMPT)
            user_factory = lambda body: HumanMessage(content=body)
        except ImportError:  # pragma: no cover
            sys_msg = {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT}
            user_factory = lambda body: {"role": "user", "content": body}

        body = (
            f"<document_id>{document_id}</document_id>\n\n"
            f"<text>\n{document_text}\n</text>"
        )

        last_error: Exception | None = None
        for _ in range(self.MAX_JSON_RETRIES + 1):
            response = await self.model.ainvoke([sys_msg, user_factory(body)])
            try:
                payload = _extract_json(response.content)
                return ExtractedKnowledge(**payload)
            except (json.JSONDecodeError, ValidationError, KeyError, TypeError) as exc:
                last_error = exc
                continue
        raise RuntimeError(
            f"EntityExtractor: invalid JSON after {self.MAX_JSON_RETRIES + 1} attempts "
            f"(last error: {last_error})"
        )
