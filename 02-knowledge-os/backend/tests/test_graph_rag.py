"""End-to-end tests for GraphRAGEngine.

Wires an InMemoryGraphStore + EntityExtractor + synthesizer LLM and
checks that:
- A query about an entity in the graph returns an answer that cites the
  matched node.
- The RRF fusion + graph traversal surfaces neighbors that the LLM can
  cite as supporting evidence.
- When the extractor returns no entities, the engine falls back to
  embedding the whole question.
"""
from __future__ import annotations

import pytest

from graph.graph_rag import GraphRAGEngine
from graph.neo4j_client import InMemoryGraphStore
from ingestion.entity_extractor import EntityExtractor
from tests.conftest import ScriptedLLM, fake_embed


async def _seed_alice_atlas(store: InMemoryGraphStore) -> None:
    await store.upsert_node(
        label="person",
        properties={"id": "alice", "name": "Alice Johnson", "role": "CTO"},
        embedding=await fake_embed("Alice Johnson CTO"),
    )
    await store.upsert_node(
        label="project",
        properties={"id": "atlas", "name": "Project Atlas", "stage": "GA"},
        embedding=await fake_embed("Project Atlas payments platform"),
    )
    await store.upsert_node(
        label="organization",
        properties={"id": "acme", "name": "Acme Corp"},
        embedding=await fake_embed("Acme Corp parent company"),
    )
    await store.upsert_relationship(from_id="alice", to_id="atlas", rel_type="OWNS")
    await store.upsert_relationship(from_id="atlas", to_id="acme", rel_type="BELONGS_TO")


@pytest.mark.asyncio
async def test_query_cites_matched_node():
    store = InMemoryGraphStore()
    await _seed_alice_atlas(store)
    llm = ScriptedLLM(
        extractor_responses=[{
            "entities": [{"id": "alice-query", "name": "Alice Johnson", "type": "person",
                          "properties": {}}],
            "relationships": [],
        }],
        synth_responses=[{
            "answer": "Alice Johnson is the CTO and owns Project Atlas.",
            "cited_node_ids": ["alice", "atlas"],
            "confidence": 0.9,
        }],
    )
    engine = GraphRAGEngine(
        graph=store,
        extractor=EntityExtractor(model=llm),
        synthesizer_model=llm,
        embed=fake_embed,
    )
    answer = await engine.query("Who owns Project Atlas?", max_hops=1)
    assert answer.confidence == 0.9
    cited_ids = {c["id"] for c in answer.cited_nodes}
    assert "alice" in cited_ids
    assert "atlas" in cited_ids


@pytest.mark.asyncio
async def test_query_traverses_to_one_hop_neighbor():
    store = InMemoryGraphStore()
    await _seed_alice_atlas(store)
    # Synthesizer cites a node reachable only via traversal (acme, 1 hop from atlas)
    llm = ScriptedLLM(
        extractor_responses=[{
            "entities": [{"id": "atlas-query", "name": "Project Atlas", "type": "project",
                          "properties": {}}],
            "relationships": [],
        }],
        synth_responses=[{
            "answer": "Project Atlas belongs to Acme Corp.",
            "cited_node_ids": ["atlas", "acme"],
            "confidence": 0.85,
        }],
    )
    engine = GraphRAGEngine(
        graph=store,
        extractor=EntityExtractor(model=llm),
        synthesizer_model=llm,
        embed=fake_embed,
    )
    answer = await engine.query("What organization owns Project Atlas?", max_hops=2)
    cited_ids = {c["id"] for c in answer.cited_nodes}
    assert "acme" in cited_ids, "traversal should surface 1-hop neighbor"


@pytest.mark.asyncio
async def test_query_falls_back_to_question_embedding_when_extractor_finds_none():
    store = InMemoryGraphStore()
    await _seed_alice_atlas(store)
    llm = ScriptedLLM(
        extractor_responses=[{"entities": [], "relationships": []}],
        synth_responses=[{
            "answer": "Acme Corp owns Project Atlas.",
            "cited_node_ids": ["atlas"],
            "confidence": 0.6,
        }],
    )
    engine = GraphRAGEngine(
        graph=store,
        extractor=EntityExtractor(model=llm),
        synthesizer_model=llm,
        embed=fake_embed,
    )
    answer = await engine.query("Project Atlas overview")
    # The engine still produced an answer (fallback path used the question embedding)
    assert answer.cited_nodes
    assert answer.cited_nodes[0]["id"] == "atlas"


@pytest.mark.asyncio
async def test_query_returns_empty_cites_when_synth_lists_unknown_ids():
    store = InMemoryGraphStore()
    await _seed_alice_atlas(store)
    llm = ScriptedLLM(
        extractor_responses=[{
            "entities": [{"id": "alice-q", "name": "Alice Johnson", "type": "person",
                          "properties": {}}],
            "relationships": [],
        }],
        synth_responses=[{
            "answer": "Insufficient information.",
            "cited_node_ids": ["nonexistent-node"],
            "confidence": 0.2,
        }],
    )
    engine = GraphRAGEngine(
        graph=store,
        extractor=EntityExtractor(model=llm),
        synthesizer_model=llm,
        embed=fake_embed,
    )
    answer = await engine.query("Who is Alice?")
    # Unknown cited ids are silently dropped — answer field still preserved
    assert answer.cited_nodes == []
    assert answer.confidence == 0.2
