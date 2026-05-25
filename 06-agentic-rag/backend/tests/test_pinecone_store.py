"""Tests for PineconeStore — no real Pinecone account required."""
from __future__ import annotations

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from stores.base_store import EnrichedChunk
from stores.pinecone_store import PineconeStore


def _fake_client(*, query_matches=None, describe_count=0):
    cli = MagicMock()
    cli.upsert = MagicMock(return_value=None)
    cli.query = MagicMock(return_value={'matches': query_matches or []})
    cli.describe_index_stats = MagicMock(return_value={'total_vector_count': describe_count})
    return cli


@pytest.mark.asyncio
async def test_index_documents_upserts_vectors_with_enriched_content_in_metadata():
    cli = _fake_client()
    store = PineconeStore(api_key='k', index_name='bench', client=cli)
    docs = [
        EnrichedChunk(id='c1', content='raw', enriched_content='ctx · raw',
                      embedding=[0.1] * 1024, metadata={'topic': 'a'}),
    ]
    res = await store.index_documents(docs)
    assert res.indexed == 1
    cli.upsert.assert_called_once()
    sent = cli.upsert.call_args.kwargs['vectors']
    assert sent[0]['id'] == 'c1'                    # IDs are passed through as-is
    assert sent[0]['metadata']['content'] == 'ctx · raw'
    assert sent[0]['metadata']['topic'] == 'a'


@pytest.mark.asyncio
async def test_similarity_search_returns_matches_with_score_and_metadata():
    matches = [
        {'id': 'c1', 'score': 0.92, 'metadata': {'content': 'about cats', 'topic': 'animal'}},
        {'id': 'c2', 'score': 0.81, 'metadata': {'content': 'about dogs', 'topic': 'animal'}},
    ]
    cli = _fake_client(query_matches=matches)
    store = PineconeStore(api_key='k', index_name='bench', client=cli)
    results = await store.similarity_search(query_embedding=[0.1] * 1024, k=2)
    assert len(results) == 2
    assert results[0].chunk_id == 'c1'
    assert results[0].content == 'about cats'
    assert results[0].metadata == {'topic': 'animal'}
    assert results[0].score == pytest.approx(0.92)
    cli.query.assert_called_once()
    kw = cli.query.call_args.kwargs
    assert kw['top_k'] == 2
    assert kw['include_metadata'] is True


@pytest.mark.asyncio
async def test_similarity_search_passes_metadata_filter_through():
    cli = _fake_client()
    store = PineconeStore(api_key='k', index_name='bench', client=cli)
    await store.similarity_search(
        query_embedding=[0.1] * 1024, k=3,
        filters={'topic': 'animal', 'lang': 'en'},
    )
    kw = cli.query.call_args.kwargs
    assert kw['filter'] == {'topic': 'animal', 'lang': 'en'}


@pytest.mark.asyncio
async def test_keyword_search_returns_empty_until_sidecar_is_wired():
    """The contract: until the BM25 sidecar lands, return [] rather than
    raising — the hybrid searcher then falls back to vector-only for
    this store. This is the explicit limitation called out in the
    benchmark report."""
    store = PineconeStore(api_key='k', index_name='bench', client=_fake_client())
    out = await store.keyword_search(query='anything', k=5)
    assert out == []


@pytest.mark.asyncio
async def test_get_stats_reports_total_vector_count():
    cli = _fake_client(describe_count=8_888)
    store = PineconeStore(api_key='k', index_name='bench', client=cli)
    stats = await store.get_stats()
    assert stats.doc_count == 8888
