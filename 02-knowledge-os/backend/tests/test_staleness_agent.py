"""Tests for StalenessAgent."""
from __future__ import annotations

import time
from typing import Any

import pytest

from graph.neo4j_client import InMemoryGraphStore
from graph.staleness_agent import StalenessAgent


@pytest.mark.asyncio
async def test_build_report_groups_by_label():
    store = InMemoryGraphStore()
    for nid in ("alice", "bob"):
        await store.upsert_node(label="person", properties={"id": nid, "name": nid})
    await store.upsert_node(label="project", properties={"id": "atlas", "name": "Atlas"})
    # Mark everything stale
    cutoff = time.time() - 60 * 86_400
    for n in store._nodes.values():
        n["updated_at"] = cutoff

    agent = StalenessAgent(graph=store, threshold_days=30)
    report = await agent._build_report()
    assert report["total_stale"] == 3
    labels = {s["label"] for s in report["sections"]}
    assert labels == {"person", "project"}
    person_section = next(s for s in report["sections"] if s["label"] == "person")
    assert person_section["count"] == 2


@pytest.mark.asyncio
async def test_summarizer_is_called_for_each_label():
    calls: list[tuple[str, int]] = []

    async def fake_summarize(label: str, nodes: list[dict[str, Any]]) -> str:
        calls.append((label, len(nodes)))
        return f"LLM said: {label} has {len(nodes)} stale"

    store = InMemoryGraphStore()
    await store.upsert_node(label="person", properties={"id": "alice", "name": "Alice"})
    store._nodes["alice"]["updated_at"] = time.time() - 60 * 86_400

    agent = StalenessAgent(graph=store, threshold_days=30, summarizer=fake_summarize)
    report = await agent._build_report()
    assert calls == [("person", 1)]
    assert report["sections"][0]["summary"] == "LLM said: person has 1 stale"


@pytest.mark.asyncio
async def test_empty_graph_produces_empty_report():
    agent = StalenessAgent(graph=InMemoryGraphStore(), threshold_days=30)
    report = await agent._build_report()
    assert report["total_stale"] == 0
    assert report["sections"] == []
