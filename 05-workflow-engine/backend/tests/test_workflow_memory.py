"""Tests for InMemoryWorkflowMemory."""
from __future__ import annotations

import pytest

from planner.dag_parser import DAG, DAGNode
from planner.workflow_memory import InMemoryWorkflowMemory
from tests.conftest import fake_embed


def _dag(goal: str) -> DAG:
    return DAG(goal=goal, nodes=[DAGNode(id="n1", name="n1", tool="github", action="open_pr")])


@pytest.mark.asyncio
async def test_save_and_recent_round_trip():
    mem = InMemoryWorkflowMemory(embed=fake_embed)
    await mem.save(goal="ship feature X", dag=_dag("ship feature X"), metrics={"nodes": 1})
    await mem.save(goal="ship feature Y", dag=_dag("ship feature Y"), metrics={"nodes": 2})
    recent = await mem.recent(limit=10)
    # Most-recent-first
    assert [r["goal"] for r in recent] == ["ship feature Y", "ship feature X"]


@pytest.mark.asyncio
async def test_find_similar_ranks_by_cosine():
    mem = InMemoryWorkflowMemory(embed=fake_embed)
    await mem.save(goal="open a pull request and tag the reviewer", dag=_dag("a"), metrics={})
    await mem.save(goal="schedule a calendar invite for tomorrow", dag=_dag("b"), metrics={})
    await mem.save(goal="open a pull request and assign the author", dag=_dag("c"), metrics={})
    hits = await mem.find_similar("open a pull request and request review", k=2)
    assert len(hits) == 2
    # The two PR-related goals should beat the calendar one
    top_goals = {h["goal"] for h in hits}
    assert "schedule a calendar invite for tomorrow" not in top_goals
    assert hits[0]["similarity"] >= hits[1]["similarity"]


@pytest.mark.asyncio
async def test_find_similar_on_empty_memory_returns_empty():
    mem = InMemoryWorkflowMemory(embed=fake_embed)
    assert await mem.find_similar("anything") == []
