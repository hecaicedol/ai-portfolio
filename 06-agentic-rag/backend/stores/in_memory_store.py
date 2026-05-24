"""Pure-Python implementation of `BaseVectorStore`.

Used as the development / test backend when no external vector DB is
available. Same interface as `PgVectorStore`, `QdrantStore`, `PineconeStore`,
so the rest of the retrieval pipeline (HybridSearcher, RRF, reranker) works
against this without modification.

Cosine similarity is implemented with numpy.
Keyword search uses `rank_bm25.BM25Okapi`.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from stores.base_store import (
    BaseVectorStore,
    EnrichedChunk,
    IndexResult,
    SearchResult,
    StoreStats,
)


class InMemoryVectorStore(BaseVectorStore):
    """All chunks live in a dict; BM25 index is rebuilt on every `index_documents`
    call. Fine for tests and dev where the corpus is small."""

    name = "in_memory"

    def __init__(self) -> None:
        self._chunks: dict[str, EnrichedChunk] = {}
        self._bm25: BM25Okapi | None = None
        self._bm25_keys: list[str] = []

    # ── ingestion ────────────────────────────────────────────────────────────

    async def index_documents(self, docs: list[EnrichedChunk]) -> IndexResult:
        start = time.perf_counter()
        failed = 0
        for d in docs:
            if d.embedding is None:
                failed += 1
                continue
            self._chunks[d.id] = d
        self._rebuild_bm25()
        return IndexResult(
            store=self.name,
            indexed=len(docs) - failed,
            failed=failed,
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    def _rebuild_bm25(self) -> None:
        if not self._chunks:
            self._bm25 = None
            self._bm25_keys = []
            return
        self._bm25_keys = list(self._chunks.keys())
        tokenized_corpus = [
            _tokenize(self._chunks[k].content) for k in self._bm25_keys
        ]
        self._bm25 = BM25Okapi(tokenized_corpus)

    # ── search ───────────────────────────────────────────────────────────────

    async def similarity_search(
        self,
        *,
        query_embedding: list[float],
        k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        start = time.perf_counter()
        if not self._chunks:
            return []

        q = np.asarray(query_embedding, dtype=np.float32)
        q_norm = float(np.linalg.norm(q))
        if q_norm == 0.0:
            return []

        scored: list[tuple[str, EnrichedChunk, float]] = []
        for cid, chunk in self._chunks.items():
            if filters and not _matches_filters(chunk.metadata, filters):
                continue
            if chunk.embedding is None:
                continue
            e = np.asarray(chunk.embedding, dtype=np.float32)
            e_norm = float(np.linalg.norm(e))
            if e_norm == 0.0:
                continue
            sim = float(np.dot(q, e) / (q_norm * e_norm))
            scored.append((cid, chunk, sim))

        scored.sort(key=lambda t: t[2], reverse=True)
        top = scored[:k]
        latency_ms = (time.perf_counter() - start) * 1000

        return [
            SearchResult(
                chunk_id=cid,
                content=chunk.content,
                score=score,
                metadata=chunk.metadata,
                latency_ms=latency_ms,
            )
            for cid, chunk, score in top
        ]

    async def keyword_search(self, *, query: str, k: int = 10) -> list[SearchResult]:
        start = time.perf_counter()
        if self._bm25 is None or not self._chunks:
            return []

        scores = self._bm25.get_scores(_tokenize(query))
        if len(scores) == 0:
            return []

        order = np.argsort(scores)[::-1][:k]
        latency_ms = (time.perf_counter() - start) * 1000

        results: list[SearchResult] = []
        for idx in order:
            if scores[idx] <= 0.0:
                continue
            cid = self._bm25_keys[int(idx)]
            chunk = self._chunks[cid]
            results.append(
                SearchResult(
                    chunk_id=cid,
                    content=chunk.content,
                    score=float(scores[idx]),
                    metadata=chunk.metadata,
                    latency_ms=latency_ms,
                )
            )
        return results

    # ── stats ────────────────────────────────────────────────────────────────

    async def get_stats(self) -> StoreStats:
        size_bytes = 0
        for c in self._chunks.values():
            size_bytes += len(c.content.encode("utf-8"))
            if c.embedding is not None:
                size_bytes += len(c.embedding) * 4  # float32 estimate
        return StoreStats(
            doc_count=len(self._chunks),
            index_size_mb=size_bytes / 1024 / 1024,
            avg_query_latency_ms=0.5,  # in-memory; constant
        )


# ── helpers ──────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase + whitespace tokenizer. Good enough for BM25 on small corpora;
    the production stores (pgvector / Qdrant) will use their native tokenizers."""
    return [t for t in text.lower().split() if t]


def _matches_filters(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    return all(metadata.get(k) == v for k, v in filters.items())
