"""Tests for QdrantStore — no real Qdrant instance required.

Fake client mocks the four async methods QdrantStore actually calls:
upsert, search, scroll, count.
"""
from __future__ import annotations

import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from stores.base_store import EnrichedChunk
from stores.qdrant_store import QdrantStore


def _fake_client(*, search_hits=None, scroll_records=None, count_value=0):
    cli = SimpleNamespace()
    cli.upsert = AsyncMock(return_value=None)
    cli.search = AsyncMock(return_value=search_hits or [])
    cli.scroll = AsyncMock(return_value=(scroll_records or [], None))
    cli.count = AsyncMock(return_value=SimpleNamespace(count=count_value))
    cli.get_collection = AsyncMock(return_value=SimpleNamespace())
    cli.create_collection = AsyncMock(return_value=None)
    return cli


@pytest.mark.asyncio
async def test_index_documents_upserts_with_enriched_content():
    cli = _fake_client()
    store = QdrantStore(url='http://qdrant', client=cli)
    docs = [
        EnrichedChunk(id='c1', content='raw', enriched_content='ctx · raw',
                      embedding=[0.1] * 1024, metadata={'topic': 'x'}),
        EnrichedChunk(id='c2', content='r2', enriched_content='ctx · r2',
                      embedding=[0.2] * 1024, metadata={}),
    ]
    res = await store.index_documents(docs)
    assert res.indexed == 2
    assert res.failed == 0
    cli.upsert.assert_awaited_once()
    sent = cli.upsert.call_args.kwargs['points']
    assert len(sent) == 2
    # Enriched content goes into payload (matches the Anthropic technique)
    assert sent[0]['payload']['content'] == 'ctx · raw'
    assert sent[0]['payload']['topic'] == 'x'
    # The chunk_id is preserved in payload so we can map back from Qdrant's UUID-style ids
    assert sent[0]['payload']['chunk_id'] == 'c1'


@pytest.mark.asyncio
async def test_index_documents_skips_chunks_without_embedding():
    cli = _fake_client()
    store = QdrantStore(url='http://qdrant', client=cli)
    docs = [
        EnrichedChunk(id='ok', content='c', enriched_content='c', embedding=[0.1] * 1024),
        EnrichedChunk(id='no', content='c', enriched_content='c', embedding=None),
    ]
    res = await store.index_documents(docs)
    assert res.indexed == 1
    assert res.failed == 1


@pytest.mark.asyncio
async def test_similarity_search_passes_query_vector_and_returns_hits():
    hits = [
        SimpleNamespace(id='uuid-1', score=0.93,
                        payload={'chunk_id': 'c1', 'content': 'about cats', 'topic': 'animal'}),
        SimpleNamespace(id='uuid-2', score=0.81,
                        payload={'chunk_id': 'c2', 'content': 'about dogs', 'topic': 'animal'}),
    ]
    cli = _fake_client(search_hits=hits)
    store = QdrantStore(url='http://qdrant', client=cli)
    results = await store.similarity_search(query_embedding=[0.1] * 1024, k=2)
    assert len(results) == 2
    assert results[0].chunk_id == 'c1'
    assert results[0].score == pytest.approx(0.93)
    assert results[0].metadata == {'topic': 'animal'}
    cli.search.assert_awaited_once()
    call_kwargs = cli.search.call_args.kwargs
    assert call_kwargs['limit'] == 2
    assert call_kwargs['query_vector'] == [0.1] * 1024


@pytest.mark.asyncio
async def test_keyword_search_uses_match_text_filter():
    records = [
        SimpleNamespace(id='uuid-x', payload={'chunk_id': 'x', 'content': 'pgvector setup'}),
    ]
    cli = _fake_client(scroll_records=records)
    store = QdrantStore(url='http://qdrant', client=cli)
    out = await store.keyword_search(query='pgvector', k=3)
    assert len(out) == 1
    assert out[0].chunk_id == 'x'
    cli.scroll.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_stats_returns_count():
    cli = _fake_client(count_value=4242)
    store = QdrantStore(url='http://qdrant', client=cli)
    stats = await store.get_stats()
    assert stats.doc_count == 4242
