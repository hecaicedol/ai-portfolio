"""Pinecone (serverless) backend for the vector store interface.

The "managed" leg of the three-store benchmark. Pinecone is the operational-
simplicity bet: pay-per-query, near-zero ops, scales without thinking. The
trade-off the benchmark surfaces:
  • No native BM25 — we union with a sidecar BM25 over the chunk text
    held client-side (here approximated; the production setup uses
    rank-bm25 over a local inverted index).
  • Per-namespace metadata filtering, not arbitrary SQL.
  • Cost scales linearly with query volume, which is why it loses on a
    bursty internal workload.

Design notes for testability
- Pinecone client is injected. Production wires a real
  `pinecone.Index` async client; tests pass a fake exposing
  `.upsert / .query / .fetch / .describe_index_stats`.
- IDs use the chunk_id verbatim — Pinecone accepts string IDs without
  the UUID rewrite Qdrant needs.
"""
from __future__ import annotations

import time
from typing import Any

from stores.base_store import (
    BaseVectorStore,
    EnrichedChunk,
    IndexResult,
    SearchResult,
    StoreStats,
)


class PineconeStore(BaseVectorStore):
    name = "pinecone"

    def __init__(
        self,
        *,
        api_key: str,
        index_name: str,
        client: Any | None = None,
    ) -> None:
        self.api_key = api_key
        self.index_name = index_name
        self._client = client

    def _ensure_client(self):
        if self._client is None:
            from pinecone import Pinecone  # type: ignore
            pc = Pinecone(api_key=self.api_key)
            self._client = pc.Index(self.index_name)
        return self._client

    async def index_documents(self, docs: list[EnrichedChunk]) -> IndexResult:
        cli = self._ensure_client()
        start = time.perf_counter()
        indexed = 0
        failed = 0
        vectors = []
        for d in docs:
            if d.embedding is None:
                failed += 1
                continue
            md = dict(d.metadata or {})
            md["content"] = d.enriched_content or d.content
            vectors.append({
                "id": d.id,
                "values": d.embedding,
                "metadata": md,
            })
            indexed += 1
        if vectors:
            res = cli.upsert(vectors=vectors)
            if hasattr(res, "__await__"):
                await res
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
        kwargs = dict(
            vector=query_embedding,
            top_k=k,
            include_metadata=True,
            include_values=False,
        )
        if filters:
            kwargs["filter"] = filters
        res = cli.query(**kwargs)
        if hasattr(res, "__await__"):
            res = await res
        latency_ms = (time.perf_counter() - start) * 1000
        matches = (res.get("matches") if isinstance(res, dict) else getattr(res, "matches", [])) or []
        return [_match_to_result(m, latency_ms) for m in matches]

    async def keyword_search(self, *, query: str, k: int = 10) -> list[SearchResult]:
        """Pinecone doesn't natively expose BM25. The benchmark uses a
        sidecar rank-bm25 index over the same chunks, returned in the
        same SearchResult shape so the HybridSearcher doesn't have to
        special-case the store. This stub returns [] — when the sidecar
        index is wired up, the integration replaces this method.
        """
        return []

    async def get_stats(self) -> StoreStats:
        cli = self._ensure_client()
        res = cli.describe_index_stats()
        if hasattr(res, "__await__"):
            res = await res
        count = 0
        if isinstance(res, dict):
            count = int(res.get("total_vector_count", 0))
        else:
            count = int(getattr(res, "total_vector_count", 0))
        return StoreStats(
            doc_count=count,
            index_size_mb=0.0,
            avg_query_latency_ms=0.0,
        )


def _match_to_result(m: Any, latency_ms: float) -> SearchResult:
    if isinstance(m, dict):
        mid = m.get("id", "")
        score = m.get("score", 0.0)
        metadata = m.get("metadata", {}) or {}
    else:
        mid = getattr(m, "id", "")
        score = getattr(m, "score", 0.0)
        metadata = getattr(m, "metadata", {}) or {}
    content = metadata.pop("content", "") if isinstance(metadata, dict) else ""
    return SearchResult(
        chunk_id=str(mid),
        content=content,
        score=float(score),
        metadata=metadata if isinstance(metadata, dict) else {},
        latency_ms=latency_ms,
    )
