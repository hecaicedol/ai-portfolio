"""Tests for memory.episodic_memory.InMemoryEpisodic."""
from __future__ import annotations

import pytest

from memory.episodic_memory import InMemoryEpisodic
from tests.conftest import fake_embed


@pytest.mark.asyncio
async def test_save_and_retrieve_relevant_session():
    em = InMemoryEpisodic(embed=fake_embed)
    await em.connect()
    await em.save_session(
        session_id="s1",
        summary="study reciprocal rank fusion and BM25 hybrid retrieval",
        key_findings=[{"fact": "RRF with k=60 beats vector-only"}],
    )
    await em.save_session(
        session_id="s2",
        summary="cooking recipes for sourdough bread",
        key_findings=[],
    )
    hits = await em.retrieve_relevant_sessions("RRF retrieval benchmarks", k=2)
    assert hits[0]["session_id"] == "s1"
    assert hits[0]["similarity"] > 0


@pytest.mark.asyncio
async def test_archive_round_trip_per_session():
    em = InMemoryEpisodic(embed=fake_embed)
    await em.connect()
    await em.archive(session_id="s1", kind="snippet", content="alpha")
    await em.archive(session_id="s1", kind="tool_output", content="beta")
    await em.archive(session_id="s2", kind="snippet", content="gamma")
    archive_s1 = await em.retrieve_archive("s1")
    assert {a["content"] for a in archive_s1} == {"alpha", "beta"}
    assert all(a["session_id"] == "s1" for a in archive_s1)


@pytest.mark.asyncio
async def test_save_session_overwrites_same_id():
    em = InMemoryEpisodic(embed=fake_embed)
    await em.connect()
    await em.save_session(session_id="x", summary="v1", key_findings=[])
    await em.save_session(session_id="x", summary="v2 (revised)", key_findings=[])
    hits = await em.retrieve_relevant_sessions("anything", k=5)
    # Only one row for session x
    assert sum(1 for h in hits if h["session_id"] == "x") == 1
    assert hits[0]["summary"] == "v2 (revised)"


@pytest.mark.asyncio
async def test_retrieve_relevant_sessions_empty_returns_empty():
    em = InMemoryEpisodic(embed=fake_embed)
    await em.connect()
    out = await em.retrieve_relevant_sessions("anything", k=3)
    assert out == []
