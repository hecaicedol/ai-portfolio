"""Tests for EntityExtractor with ScriptedLLM."""
from __future__ import annotations

import pytest

from ingestion.entity_extractor import EntityExtractor
from tests.conftest import ScriptedLLM


@pytest.mark.asyncio
async def test_extract_parses_clean_json():
    llm = ScriptedLLM(extractor_responses=[{
        "entities": [
            {"id": "alice", "name": "Alice", "type": "person", "properties": {"role": "CTO"}},
            {"id": "atlas", "name": "Project Atlas", "type": "project", "properties": {}},
        ],
        "relationships": [
            {"source_id": "alice", "target_id": "atlas", "type": "OWNS"},
        ],
    }])
    ex = EntityExtractor(model=llm)
    out = await ex.extract(document_text="Alice owns Atlas.", document_id="doc-1")
    assert [e.id for e in out.entities] == ["alice", "atlas"]
    assert out.relationships[0].type == "OWNS"


@pytest.mark.asyncio
async def test_extract_tolerates_code_fence_and_retries():
    llm = ScriptedLLM(extractor_responses=[
        "garbage that is not JSON",
        "```json\n" + '{"entities": [], "relationships": []}' + "\n```",
    ])
    ex = EntityExtractor(model=llm)
    out = await ex.extract(document_text="empty doc", document_id="doc-2")
    assert out.entities == []


@pytest.mark.asyncio
async def test_extract_raises_after_max_retries():
    llm = ScriptedLLM(extractor_responses=["nope", "still nope", "yep nope"])
    ex = EntityExtractor(model=llm)
    with pytest.raises(RuntimeError, match="invalid JSON"):
        await ex.extract(document_text="x", document_id="doc-3")
