"""Tests for memory.working_memory.WorkingMemory."""
from __future__ import annotations

from memory.working_memory import WorkingMemory


def test_add_returns_no_evictions_when_under_budget():
    wm = WorkingMemory(max_tokens=10_000)
    evicted = wm.add(key="goal", content="study reciprocal rank fusion", kind="goal")
    assert evicted == []
    assert wm.used() > 0
    snap = wm.snapshot()
    assert len(snap) == 1
    assert snap[0]["key"] == "goal"


def test_add_evicts_oldest_when_over_budget():
    # Tiny budget forces eviction. Each add returns ONLY the entries it
    # evicted on that call, so collect evictions across all three.
    wm = WorkingMemory(max_tokens=20)
    all_evicted = []
    all_evicted += wm.add(key="a", content="alpha " * 10, kind="snippet")
    all_evicted += wm.add(key="b", content="beta "  * 10, kind="snippet")
    all_evicted += wm.add(key="c", content="gamma " * 10, kind="snippet")
    evicted_keys = {e.key for e in all_evicted}
    assert "a" in evicted_keys, f"FIFO: oldest should evict first (got {evicted_keys})"
    assert "a" not in {x["key"] for x in wm.snapshot()}


def test_get_context_includes_every_live_entry():
    wm = WorkingMemory(max_tokens=10_000)
    wm.add(key="goal", content="What is RRF?", kind="goal")
    wm.add(key="plan", content="step 1: arxiv search", kind="plan")
    ctx = wm.get_context()
    assert "[GOAL] goal" in ctx
    assert "[PLAN] plan" in ctx
    assert "RRF" in ctx
    assert "arxiv" in ctx


def test_snapshot_truncates_long_previews():
    wm = WorkingMemory(max_tokens=10_000)
    long_text = "word " * 500
    wm.add(key="big", content=long_text, kind="snippet")
    snap = wm.snapshot()
    assert len(snap[0]["preview"]) <= 200
