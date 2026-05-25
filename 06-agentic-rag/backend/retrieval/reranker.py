"""Cross-encoder reranker via Cohere Rerank v3.

Cohere's rerank endpoint takes a query + a list of documents and returns
each document with a relevance score (0–1). It runs a cross-encoder model
under the hood, which consistently beats embedding similarity on top-K
precision.

Design notes for testability
- HTTP client is injected. Production gives it an `httpx.AsyncClient`
  configured with the Cohere base URL + bearer token; tests inject a fake
  whose `.post()` returns a stubbed response object.
- API contract (Cohere docs):
    POST https://api.cohere.com/v2/rerank
    {"model": "rerank-english-v3.0", "query": "...", "documents": [...]}
  Response: {"results": [{"index": int, "relevance_score": float}, ...]}
- The reranker preserves the candidate dict shape (SearchResult) — only
  the order and the `score` field change.
"""
from __future__ import annotations

from typing import Any, Protocol

from stores.base_store import SearchResult


COHERE_RERANK_URL = "https://api.cohere.com/v2/rerank"


class _HttpClient(Protocol):
    async def post(self, url: str, *, headers: dict, json: dict) -> Any: ...


class CohereReranker:
    """Reorder candidate hits with Cohere Rerank.

    Parameters
    ----------
    api_key
        Bearer token. Sent as Authorization header.
    model
        Cohere model id (default rerank-english-v3.0).
    top_n
        Trim the reranked list to this many results.
    http
        Async HTTP client; defaults to a lazy httpx.AsyncClient at
        first use so tests can avoid pulling httpx as a hard dep.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "rerank-english-v3.0",
        top_n: int = 5,
        http: _HttpClient | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.top_n = top_n
        self._http = http

    def _client(self) -> _HttpClient:
        if self._http is None:
            import httpx  # lazy import
            self._http = httpx.AsyncClient(timeout=15.0)
        return self._http

    async def rerank(
        self,
        *,
        query: str,
        candidates: list[SearchResult],
    ) -> list[SearchResult]:
        if not candidates:
            return []

        # Cohere v2 takes plain document strings; we attach the index to
        # remap back to our SearchResult objects.
        body = {
            "model": self.model,
            "query": query,
            "documents": [c.content for c in candidates],
            "top_n": min(self.top_n, len(candidates)),
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = await self._client().post(
            COHERE_RERANK_URL, headers=headers, json=body
        )
        # Response shape: {"results": [{"index": 0, "relevance_score": 0.97}, ...]}
        data = response.json() if callable(getattr(response, 'json', None)) else response
        results = data.get("results", [])

        reranked: list[SearchResult] = []
        for r in results:
            idx = r.get("index")
            score = r.get("relevance_score", 0.0)
            if idx is None or idx >= len(candidates):
                continue
            orig = candidates[idx]
            reranked.append(SearchResult(
                chunk_id=orig.chunk_id,
                content=orig.content,
                score=float(score),
                metadata=orig.metadata,
                latency_ms=orig.latency_ms,
            ))
        return reranked[: self.top_n]
