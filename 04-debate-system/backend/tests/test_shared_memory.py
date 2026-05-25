"""Tests for InMemoryDebateSharedMemory."""
from __future__ import annotations

import asyncio

import pytest

from debate.shared_memory import InMemoryDebateSharedMemory


@pytest.mark.asyncio
async def test_append_preserves_order():
    mem = InMemoryDebateSharedMemory()
    for i in range(5):
        await mem.append("s1", {"i": i})
    history = await mem.history("s1")
    assert [h["i"] for h in history] == [0, 1, 2, 3, 4]


@pytest.mark.asyncio
async def test_sessions_are_isolated():
    mem = InMemoryDebateSharedMemory()
    await mem.append("s1", {"who": "alpha"})
    await mem.append("s2", {"who": "beta"})
    assert (await mem.history("s1"))[0]["who"] == "alpha"
    assert (await mem.history("s2"))[0]["who"] == "beta"


@pytest.mark.asyncio
async def test_metadata_round_trips_and_records_started_at():
    mem = InMemoryDebateSharedMemory()
    await mem.set_metadata("s1", "Should we ship feature X?")
    meta = await mem.get_metadata("s1")
    assert meta["problem"] == "Should we ship feature X?"
    assert isinstance(meta["started_at"], float)


@pytest.mark.asyncio
async def test_expired_session_is_evicted():
    mem = InMemoryDebateSharedMemory(ttl_seconds=0)
    await mem.append("s1", {"x": 1})
    await asyncio.sleep(0.001)
    assert await mem.history("s1") == []
