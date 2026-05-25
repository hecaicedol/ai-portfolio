"""Tests for retrieval.reranker.CohereReranker.

Uses a FakeHttpClient so the tests never touch the network. The fake
records every request and returns whatever response object the test
hands it — same surface as httpx.AsyncClient.post().
"""
from __future__ import annotations

import pytest
from types import SimpleNamespace

from retrieval.reranker import CohereReranker, COHERE_RERANK_URL
from stores.base_store import SearchResult


def _result(chunk_id: str, content: str, score: float = 0.5) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id, content=content,
        score=score, metadata={}, latency_ms=0.0,
    )


class FakeHttp:
    """Records the last call and returns a canned JSON-like payload."""

    def __init__(self, payload):
        self.payload = payload
        self.last_url = None
        self.last_headers = None
        self.last_body = None

    async def post(self, url, *, headers, json):
        self.last_url = url
        self.last_headers = headers
        self.last_body = json
        return SimpleNamespace(json=lambda: self.payload)


@pytest.mark.asyncio
async def test_rerank_reorders_by_relevance_score():
    candidates = [
        _result('a', 'about cats'),
        _result('b', 'about dogs'),
        _result('c', 'unrelated noise'),
    ]
    # Cohere returns c last (most relevant), b first
    http = FakeHttp({'results': [
        {'index': 1, 'relevance_score': 0.95},
        {'index': 0, 'relevance_score': 0.55},
        {'index': 2, 'relevance_score': 0.10},
    ]})
    rr = CohereReranker(api_key='fake-key', top_n=3, http=http)
    out = await rr.rerank(query='dogs', candidates=candidates)

    assert [r.chunk_id for r in out] == ['b', 'a', 'c']
    assert out[0].score == pytest.approx(0.95)


@pytest.mark.asyncio
async def test_rerank_respects_top_n_clamp():
    candidates = [_result(f'c{i}', f'doc {i}') for i in range(8)]
    http = FakeHttp({'results': [
        {'index': i, 'relevance_score': 1.0 - i * 0.1} for i in range(8)
    ]})
    rr = CohereReranker(api_key='fake-key', top_n=3, http=http)
    out = await rr.rerank(query='q', candidates=candidates)
    assert len(out) == 3
    # top_n is also sent to Cohere so the API doesn't waste tokens on the tail
    assert http.last_body['top_n'] == 3


@pytest.mark.asyncio
async def test_rerank_empty_candidates_short_circuits():
    http = FakeHttp({'results': []})
    rr = CohereReranker(api_key='fake-key', http=http)
    out = await rr.rerank(query='q', candidates=[])
    assert out == []
    # And the HTTP client must NOT have been called
    assert http.last_url is None


@pytest.mark.asyncio
async def test_rerank_sends_correct_endpoint_and_auth():
    http = FakeHttp({'results': [{'index': 0, 'relevance_score': 0.5}]})
    rr = CohereReranker(api_key='SECRET', model='rerank-multilingual-v3.0', http=http)
    await rr.rerank(query='hi', candidates=[_result('a', 'hi there')])

    assert http.last_url == COHERE_RERANK_URL
    assert http.last_headers['Authorization'] == 'Bearer SECRET'
    assert http.last_headers['Content-Type'] == 'application/json'
    assert http.last_body['model'] == 'rerank-multilingual-v3.0'
    assert http.last_body['query'] == 'hi'
    assert http.last_body['documents'] == ['hi there']


@pytest.mark.asyncio
async def test_rerank_silently_drops_out_of_range_indices():
    """If Cohere ever returns an index past the candidate list, skip it."""
    http = FakeHttp({'results': [
        {'index': 0, 'relevance_score': 0.9},
        {'index': 99, 'relevance_score': 0.8},  # bogus, must be skipped
        {'index': 1, 'relevance_score': 0.7},
    ]})
    candidates = [_result('a', 'A'), _result('b', 'B')]
    rr = CohereReranker(api_key='k', http=http)
    out = await rr.rerank(query='q', candidates=candidates)
    assert [r.chunk_id for r in out] == ['a', 'b']
