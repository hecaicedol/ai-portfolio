"""Tests for memory.memgpt_controller.MemGPTController."""
from __future__ import annotations

import pytest

from memory.episodic_memory import InMemoryEpisodic
from memory.memgpt_controller import MemGPTController
from memory.semantic_memory import InMemorySemantic
from memory.working_memory import WorkingMemory
from tests.conftest import fake_embed


@pytest.fixture
async def controller():
    em = InMemoryEpisodic(embed=fake_embed)
    await em.connect()
    sm = InMemorySemantic(embed=fake_embed)
    await sm.connect()
    wm = WorkingMemory(max_tokens=100)  # tight, to force eviction quickly
    return MemGPTController(working=wm, episodic=em, semantic=sm, session_id="test-sess")


@pytest.mark.asyncio
async def test_remember_below_budget_does_not_archive(controller):
    await controller.remember(key="goal", content="study rrf", kind="goal")
    # Nothing archived to episodic yet (budget not exceeded)
    archive = await controller.episodic.retrieve_archive("test-sess")
    assert archive == []


@pytest.mark.asyncio
async def test_remember_over_budget_auto_archives_evictions(controller):
    # Stuff working memory until it overflows, then verify eviction was archived
    for i in range(8):
        await controller.remember(
            key=f"snip-{i}",
            content=("filler text " * 20) + f"# {i}",
            kind="snippet",
        )
    archive = await controller.episodic.retrieve_archive("test-sess")
    assert len(archive) > 0
    # First evictions correspond to the earliest snippets (FIFO)
    contents = [a["content"] for a in archive]
    assert any("# 0" in c for c in contents)


@pytest.mark.asyncio
async def test_get_full_context_assembles_working_episodic_semantic(controller):
    # Seed episodic and semantic memory with known content
    await controller.episodic.save_session(
        session_id="prev",
        summary="previous study of rrf and hybrid retrieval",
        key_findings=[],
    )
    await controller.semantic.upsert_fact(
        fact="RRF with k=60 beats vector-only retrieval",
        source="cormack-2009",
        confidence=0.9,
    )
    await controller.remember(key="goal", content="what is rrf?", kind="goal")

    ctx = await controller.get_full_context("rrf")
    assert "Working memory" in ctx
    assert "Relevant past sessions" in ctx
    assert "Relevant durable knowledge" in ctx
    assert "previous study" in ctx
    assert "RRF with k=60" in ctx


@pytest.mark.asyncio
async def test_consolidate_writes_session_and_facts(controller):
    findings = [
        {"fact": "BM25 captures exact keyword matches", "source": "robertson", "confidence": 0.95},
        {"fact": "Vector retrieval captures semantic paraphrases", "source": "voyage", "confidence": 0.9},
    ]
    await controller.consolidate(
        summary="learned hybrid retrieval fundamentals",
        key_findings=findings,
    )
    sessions = await controller.episodic.retrieve_relevant_sessions("hybrid retrieval", k=5)
    assert any(s["session_id"] == "test-sess" for s in sessions)
    knowledge = await controller.semantic.retrieve_relevant_knowledge("BM25", k=5)
    assert any("BM25" in f["fact"] for f in knowledge)


@pytest.mark.asyncio
async def test_consolidate_dedupes_repeated_facts(controller):
    findings = [
        {"fact": "RRF with k=60 is standard", "source": "A", "confidence": 0.7},
        {"fact": "RRF with k=60 is standard", "source": "B", "confidence": 0.9},
    ]
    await controller.consolidate(summary="x", key_findings=findings)
    # Only one fact lives in semantic memory; sources are unioned
    knowledge = await controller.semantic.retrieve_relevant_knowledge("RRF k=60", k=5)
    target = next(f for f in knowledge if "RRF" in f["fact"])
    assert {"A", "B"}.issubset(set(target["sources"]))
    assert target["confidence"] == 0.9
