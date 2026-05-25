"""Tests for memory.semantic_memory.InMemorySemantic."""
from __future__ import annotations

import pytest

from memory.semantic_memory import InMemorySemantic, SIMILARITY_DEDUPE_THRESHOLD
from tests.conftest import fake_embed


@pytest.mark.asyncio
async def test_upsert_inserts_and_assigns_id():
    sm = InMemorySemantic(embed=fake_embed)
    await sm.connect()
    fid = await sm.upsert_fact(fact="LangGraph uses StateGraph", source="docs", confidence=0.9)
    assert fid == 1
    fid2 = await sm.upsert_fact(fact="MemGPT has three memory tiers", source="paper", confidence=0.8)
    assert fid2 == 2
    out = await sm.retrieve_relevant_knowledge("LangGraph", k=5)
    assert len(out) >= 1


@pytest.mark.asyncio
async def test_upsert_dedupes_when_cosine_above_threshold():
    sm = InMemorySemantic(embed=fake_embed)
    await sm.connect()
    fid1 = await sm.upsert_fact(fact="RRF combines ranked lists", source="A", confidence=0.7)
    # Identical fact should dedupe (cosine 1.0 > 0.92)
    fid2 = await sm.upsert_fact(fact="RRF combines ranked lists", source="B", confidence=0.8)
    assert fid2 == fid1
    out = await sm.retrieve_relevant_knowledge("RRF", k=5)
    target = next(f for f in out if f["id"] == fid1)
    assert "A" in target["sources"] and "B" in target["sources"]
    assert target["confidence"] == 0.8  # max of the two


@pytest.mark.asyncio
async def test_upsert_inserts_distinct_facts_when_below_threshold():
    sm = InMemorySemantic(embed=fake_embed)
    await sm.connect()
    a = await sm.upsert_fact(
        fact="pgvector uses ivfflat for ANN", source="docs", confidence=0.9,
    )
    b = await sm.upsert_fact(
        fact="Qdrant uses HNSW for ANN", source="docs", confidence=0.9,
    )
    assert a != b
    assert len(sm.facts) == 2


@pytest.mark.asyncio
async def test_supersede_marks_old_fact_and_hides_from_retrieval():
    sm = InMemorySemantic(embed=fake_embed)
    await sm.connect()
    old_id = await sm.upsert_fact(fact="Claude 3 is the latest", source="A", confidence=0.6)
    # Distinct enough to not dedupe — use very different vocabulary
    new_id = await sm.upsert_fact(fact="Sonnet 4.5 superseded earlier generations", source="A", confidence=0.9)
    await sm.supersede(old_fact_id=old_id, new_fact_id=new_id)
    out = await sm.retrieve_relevant_knowledge("Claude", k=5)
    assert all(f["id"] != old_id for f in out)


@pytest.mark.asyncio
async def test_retrieve_knowledge_sorts_by_similarity():
    sm = InMemorySemantic(embed=fake_embed)
    await sm.connect()
    await sm.upsert_fact(fact="LangGraph orchestration patterns", source="x", confidence=0.9)
    await sm.upsert_fact(fact="cooking sourdough at high altitude", source="y", confidence=0.9)
    out = await sm.retrieve_relevant_knowledge("LangGraph", k=2)
    assert out[0]["fact"].startswith("LangGraph")
    assert out[0]["similarity"] >= out[-1]["similarity"]


def test_dedupe_threshold_constant_is_exposed():
    assert 0.0 < SIMILARITY_DEDUPE_THRESHOLD < 1.0
