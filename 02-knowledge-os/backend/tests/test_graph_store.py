"""Tests for InMemoryGraphStore — the pure-Python implementation behind
the GraphStore Protocol."""
from __future__ import annotations

import time

import pytest

from graph.neo4j_client import InMemoryGraphStore
from tests.conftest import fake_embed


@pytest.mark.asyncio
async def test_upsert_node_creates_then_updates():
    store = InMemoryGraphStore()
    nid = await store.upsert_node(
        label="person",
        properties={"id": "alice", "name": "Alice Johnson", "role": "CTO"},
    )
    assert nid == "alice"
    # Update — same id, different role
    await store.upsert_node(
        label="person",
        properties={"id": "alice", "role": "CEO"},
    )
    snap = await store.snapshot()
    alice = next(n for n in snap["nodes"] if n["id"] == "alice")
    assert alice["properties"]["role"] == "CEO"
    assert alice["properties"]["name"] == "Alice Johnson"  # preserved across update


@pytest.mark.asyncio
async def test_upsert_node_requires_id():
    store = InMemoryGraphStore()
    with pytest.raises(ValueError, match="must include an 'id'"):
        await store.upsert_node(label="person", properties={"name": "no id"})


@pytest.mark.asyncio
async def test_upsert_relationship_deduplicates():
    store = InMemoryGraphStore()
    await store.upsert_node(label="person", properties={"id": "alice"})
    await store.upsert_node(label="project", properties={"id": "atlas"})
    await store.upsert_relationship(from_id="alice", to_id="atlas", rel_type="OWNS", properties={"since": "2024"})
    await store.upsert_relationship(from_id="alice", to_id="atlas", rel_type="OWNS", properties={"since": "2025"})
    snap = await store.snapshot()
    owns_edges = [e for e in snap["edges"] if e["type"] == "OWNS"]
    assert len(owns_edges) == 1


@pytest.mark.asyncio
async def test_vector_search_ranks_by_cosine():
    store = InMemoryGraphStore()
    for name in ("alice johnson cto", "bob marketing", "carol engineering"):
        await store.upsert_node(
            label="person",
            properties={"id": name.split()[0], "name": name},
            embedding=await fake_embed(name),
        )
    hits = await store.vector_search(
        label="person",
        embedding=await fake_embed("alice johnson"),
        k=2,
    )
    assert hits[0]["id"] == "alice"
    assert hits[0]["score"] >= hits[1]["score"]


@pytest.mark.asyncio
async def test_traverse_respects_max_hops():
    store = InMemoryGraphStore()
    for nid in ("a", "b", "c", "d"):
        await store.upsert_node(label="concept", properties={"id": nid})
    # a -> b -> c -> d
    await store.upsert_relationship(from_id="a", to_id="b", rel_type="LINKS")
    await store.upsert_relationship(from_id="b", to_id="c", rel_type="LINKS")
    await store.upsert_relationship(from_id="c", to_id="d", rel_type="LINKS")
    result = (await store.traverse(start_node_ids=["a"], max_hops=2))[0]
    visited = {n["id"] for n in result["nodes"]}
    assert visited == {"a", "b", "c"}, f"got {visited}"


@pytest.mark.asyncio
async def test_stale_nodes_returns_old_records():
    store = InMemoryGraphStore()
    await store.upsert_node(label="doc", properties={"id": "fresh"})
    await store.upsert_node(label="doc", properties={"id": "stale"})
    # Hack the timestamp to 60 days ago
    store._nodes["stale"]["updated_at"] = time.time() - 60 * 86_400
    stale = await store.stale_nodes(days=30)
    assert [n["id"] for n in stale] == ["stale"]
