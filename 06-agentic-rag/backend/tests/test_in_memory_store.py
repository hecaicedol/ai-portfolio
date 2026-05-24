"""Tests for `stores.in_memory_store.InMemoryVectorStore`."""
from __future__ import annotations

import pytest

from stores.base_store import EnrichedChunk
from stores.in_memory_store import InMemoryVectorStore
from tests.conftest import fake_embed


@pytest.mark.asyncio
async def test_index_documents_stores_chunks(sample_chunks):
    store = InMemoryVectorStore()
    res = await store.index_documents(sample_chunks)
    assert res.indexed == len(sample_chunks)
    assert res.failed == 0
    assert res.store == "in_memory"
    stats = await store.get_stats()
    assert stats.doc_count == len(sample_chunks)


@pytest.mark.asyncio
async def test_index_documents_skips_chunks_without_embedding():
    store = InMemoryVectorStore()
    docs = [
        EnrichedChunk(
            id="ok",
            content="indexable",
            enriched_content="indexable",
            embedding=fake_embed("indexable"),
        ),
        EnrichedChunk(
            id="bad",
            content="missing embedding",
            enriched_content="missing embedding",
            embedding=None,
        ),
    ]
    res = await store.index_documents(docs)
    assert res.indexed == 1
    assert res.failed == 1


@pytest.mark.asyncio
async def test_similarity_search_returns_most_similar_first(sample_chunks):
    store = InMemoryVectorStore()
    await store.index_documents(sample_chunks)

    # Query roughly aligned with ml-2 (RRF) — embedding overlaps on "rank fusion"
    q_embedding = fake_embed("rank fusion of retrieval lists")
    results = await store.similarity_search(query_embedding=q_embedding, k=3)

    assert len(results) == 3
    assert results[0].chunk_id == "ml-2"
    assert results[0].score >= results[1].score >= results[2].score


@pytest.mark.asyncio
async def test_similarity_search_empty_store_returns_empty():
    store = InMemoryVectorStore()
    results = await store.similarity_search(query_embedding=fake_embed("anything"), k=5)
    assert results == []


@pytest.mark.asyncio
async def test_similarity_search_zero_query_returns_empty(sample_chunks):
    store = InMemoryVectorStore()
    await store.index_documents(sample_chunks)
    results = await store.similarity_search(query_embedding=[0.0] * 64, k=5)
    assert results == []


@pytest.mark.asyncio
async def test_similarity_search_respects_metadata_filter(sample_chunks):
    store = InMemoryVectorStore()
    await store.index_documents(sample_chunks)

    q = fake_embed("python decorators event loop")
    results = await store.similarity_search(
        query_embedding=q, k=5, filters={"topic": "py"}
    )
    assert len(results) == 2  # py-1 and py-2 only
    assert all(r.chunk_id.startswith("py-") for r in results)


@pytest.mark.asyncio
async def test_keyword_search_returns_bm25_relevant(sample_chunks):
    store = InMemoryVectorStore()
    await store.index_documents(sample_chunks)

    results = await store.keyword_search(query="pgvector cosine", k=3)
    assert len(results) >= 1
    assert results[0].chunk_id == "infra-1"


@pytest.mark.asyncio
async def test_keyword_search_empty_store():
    store = InMemoryVectorStore()
    results = await store.keyword_search(query="anything", k=5)
    assert results == []


@pytest.mark.asyncio
async def test_get_stats_reports_corpus_size(sample_chunks):
    store = InMemoryVectorStore()
    await store.index_documents(sample_chunks)
    stats = await store.get_stats()
    assert stats.doc_count == len(sample_chunks)
    assert stats.index_size_mb > 0
