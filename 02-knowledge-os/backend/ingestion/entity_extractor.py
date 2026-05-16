from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


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


class EntityExtractor:
    """
    Calls Claude with a strict system prompt to extract entities and relationships
    from a document chunk. Returns ExtractedKnowledge (Pydantic-validated).
    """

    SYSTEM_PROMPT = """You extract structured knowledge from business documents.
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
"""

    def __init__(self, *, model: str, api_key: str) -> None:
        self.model = model
        self.api_key = api_key

    async def extract(self, *, document_text: str, document_id: str) -> ExtractedKnowledge:
        raise NotImplementedError("Call Claude with SYSTEM_PROMPT + document; parse JSON; validate as ExtractedKnowledge")
