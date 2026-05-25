"""Qdrant backend for the vector store interface.

Why bother with Qdrant alongside pgvector + Pinecone in the benchmark:
  • Native sparse-vector support — could replace the BM25 sidecar if we
    ever wanted a single-engine hybrid.
  • Payload filtering is first-class (faster than pgvector's metadata
    GIN scan above a few million rows).
  • HNSW index has more knobs (M, ef_construct, ef_search) than
    pgvector's ivfflat.

Design notes for testability
- The Qdrant client is injected. Production wires
  `qdrant_client.AsyncQdrantClient`; tests pass a fake that exposes
  the same `.upsert / .search / .scroll / .count` async methods.
- Distance metric: Cosine. Qdrant's `score` is already cosine
  similarity ∈ [-1, 1] (we float-pass it through to SearchResult.score).
- Keyword search uses a payload-side text-match filter (`MatchText`);
  Qdrant doesn't natively expose BM25, and the benchmark report calls
  that limitation out explicitly.
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from stores.base_store import (
    BaseVectorStore,
    EnrichedChunk,
    IndexResult,
    SearchResult,
    StoreStats,
)


COLLECTION_NAME = "chunks"
EMBED_DIM = 1024


class QdrantStore(BaseVectorStore):
    name = "qdrant"

    def __init__(
        self,
        *,
        url: str,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.url = url
        self.api_key = api_key
        self._client = client

    def _ensure_client(self):
        if self._client is None:
            from qdrant_client import AsyncQdrantClient  # type: ignore
            self._client = AsyncQdrantClient(url=self.url, api_key=self.api_key)
        return self._client

    async def ensure_collection(self) -> None:
        from qdrant_client.http.models import Distance, VectorParams  # type: ignore
        cli = self._ensure_client()
        exists = False
        try:
            await cli.get_collection(COLLECTION_NAME)
            exists = True
        except Exception:
            exists = False
        if not exists:
            await cli.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )

    async def index_documents(self, docs: list[EnrichedChunk]) -> IndexResult:
        cli = self._ensure_client()
        start = time.perf_counter()
        indexed = 0
        failed = 0
        points = []
        for d in docs:
            if d.embedding is None:
                failed += 1
                continue
            payload = dict(d.metadata or {})
            payload["content"] = d.enriched_content or d.content
            payload["chunk_id"] = d.id
            points.append({
                "id": _to_point_id(d.id),
                "vector": d.embedding,
                "payload": payload,
            })
            indexed += 1
        if points:
            await cli.upsert(collection_name=COLLECTION_NAME, points=points)
        return IndexResult(
            store=self.name,
            indexed=indexed,
            failed=failed,
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    async def similarity_search(
        self,
        *,
        query_embedding: list[float],
        k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        cli = self._ensure_client()
        start = time.perf_counter()
        hits = await cli.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=k,
            query_filter=_build_filter(filters) if filters else None,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        return [_hit_to_result(h, latency_ms) for h in hits]

    async def keyword_search(self, *, query: str, k: int = 10) -> list[SearchResult]:
        cli = self._ensure_client()
        start = time.perf_counter()
        # Plain dict matches Qdrant's HTTP API model — the real qdrant-client
        # accepts dicts and auto-converts. Keeps tests free of an SDK install.
        qf = {"must": [{"key": "content", "match": {"text": query}}]}
        records, _ = await cli.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=qf,
            limit=k,
            with_payload=True,
            with_vectors=False,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        return [
            SearchResult(
                chunk_id=r.payload.get('chunk_id', str(r.id)),
                content=r.payload.get('content', ''),
                score=1.0,  # scroll has no relevance score
                metadata={k_: v for k_, v in r.payload.items() if k_ not in ('content', 'chunk_id')},
                latency_ms=latency_ms,
            )
            for r in records
        ]

    async def get_stats(self) -> StoreStats:
        cli = self._ensure_client()
        info = await cli.count(collection_name=COLLECTION_NAME, exact=True)
        return StoreStats(
            doc_count=int(getattr(info, 'count', 0)),
            index_size_mb=0.0,
            avg_query_latency_ms=0.0,
        )


def _to_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_OID, chunk_id))


def _build_filter(filters: dict[str, Any]):
    # Plain dict; the qdrant-client SDK accepts dicts and auto-converts to
    # the typed models, so the production path works the same.
    return {"must": [{"key": k, "match": {"value": v}} for k, v in filters.items()]}


def _hit_to_result(hit, latency_ms: float) -> SearchResult:
    payload = getattr(hit, 'payload', {}) or {}
    return SearchResult(
        chunk_id=payload.get('chunk_id', str(getattr(hit, 'id', ''))),
        content=payload.get('content', ''),
        score=float(getattr(hit, 'score', 0.0)),
        metadata={k: v for k, v in payload.items() if k not in ('content', 'chunk_id')},
        latency_ms=latency_ms,
    )
