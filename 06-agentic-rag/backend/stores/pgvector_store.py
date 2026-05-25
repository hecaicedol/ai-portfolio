"""PostgreSQL + pgvector backend for the vector store interface.

Schema (auto-created via ensure_schema())
    CREATE TABLE chunks (
        id            TEXT PRIMARY KEY,
        content       TEXT NOT NULL,
        metadata      JSONB DEFAULT '{}'::jsonb,
        embedding     vector(1024) NOT NULL,
        tsv           tsvector GENERATED ALWAYS AS (
                        setweight(to_tsvector('english', content), 'A')
                      ) STORED
    );
    CREATE INDEX chunks_embedding_idx ON chunks
        USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
    CREATE INDEX chunks_tsv_idx ON chunks USING GIN (tsv);

Design notes for testability
- `pool_factory` is injectable. Production passes
  `psycopg_pool.AsyncConnectionPool`; tests pass a fake whose
  `.connection()` returns an async-context-manager wrapping a mock
  connection. No real Postgres needed for unit tests.
- All operations record `latency_ms` on every SearchResult — that's
  what the benchmark dashboard plots in P6's recharts UI.
- Cosine distance via `<=>`, so the SELECT uses `1 - (embedding <=> q)`
  as similarity (∈ [0, 1], higher = closer).
- BM25-ish keyword search uses the GIN'd tsvector with
  `ts_rank_cd(tsv, plainto_tsquery(...))`.
"""
from __future__ import annotations

import json
import time
from typing import Any, Callable

from stores.base_store import (
    BaseVectorStore,
    EnrichedChunk,
    IndexResult,
    SearchResult,
    StoreStats,
)


CREATE_SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id        TEXT PRIMARY KEY,
    content   TEXT NOT NULL,
    metadata  JSONB DEFAULT '{}'::jsonb,
    embedding vector(1024) NOT NULL,
    tsv       tsvector GENERATED ALWAYS AS (
                setweight(to_tsvector('english', content), 'A')
              ) STORED
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS chunks_tsv_idx
    ON chunks USING GIN (tsv);
"""

INSERT_SQL = """
INSERT INTO chunks (id, content, metadata, embedding)
VALUES (%s, %s, %s::jsonb, %s::vector)
ON CONFLICT (id) DO UPDATE
    SET content = EXCLUDED.content,
        metadata = EXCLUDED.metadata,
        embedding = EXCLUDED.embedding;
"""

SIMILARITY_SQL = """
SELECT id, content, metadata,
       1 - (embedding <=> %s::vector) AS similarity
FROM chunks
{where_clause}
ORDER BY embedding <=> %s::vector
LIMIT %s;
"""

KEYWORD_SQL = """
SELECT id, content, metadata,
       ts_rank_cd(tsv, plainto_tsquery('english', %s)) AS rank
FROM chunks
WHERE tsv @@ plainto_tsquery('english', %s)
ORDER BY rank DESC
LIMIT %s;
"""

STATS_SQL = """
SELECT count(*),
       coalesce(pg_total_relation_size('chunks') / 1024.0 / 1024.0, 0)
FROM chunks;
"""


class PgVectorStore(BaseVectorStore):
    name = "pgvector"

    def __init__(
        self,
        *,
        dsn: str,
        pool_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.dsn = dsn.replace("postgresql+psycopg://", "postgresql://")
        self._pool_factory = pool_factory
        self._pool: Any | None = None

    # ── Lifecycle ──────────────────────────────────────────────────────

    async def connect(self) -> None:
        if self._pool is not None:
            return
        if self._pool_factory is None:
            # Lazy import so unit tests don't need psycopg installed
            from psycopg_pool import AsyncConnectionPool  # type: ignore
            self._pool = AsyncConnectionPool(self.dsn, min_size=1, max_size=5, open=False)
            await self._pool.open()
        else:
            self._pool = self._pool_factory(self.dsn)
            opener = getattr(self._pool, "open", None)
            if opener is not None:
                res = opener()
                if hasattr(res, "__await__"):
                    await res

    async def close(self) -> None:
        if self._pool is not None:
            closer = getattr(self._pool, "close", None)
            if closer is not None:
                res = closer()
                if hasattr(res, "__await__"):
                    await res
            self._pool = None

    async def ensure_schema(self) -> None:
        await self.connect()
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(CREATE_SCHEMA_SQL)

    # ── Ingestion ──────────────────────────────────────────────────────

    async def index_documents(self, docs: list[EnrichedChunk]) -> IndexResult:
        await self.connect()
        start = time.perf_counter()
        indexed = 0
        failed = 0
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                for d in docs:
                    if d.embedding is None:
                        failed += 1
                        continue
                    await cur.execute(
                        INSERT_SQL,
                        (
                            d.id,
                            d.enriched_content or d.content,
                            json.dumps(d.metadata or {}),
                            d.embedding,
                        ),
                    )
                    indexed += 1
        return IndexResult(
            store=self.name,
            indexed=indexed,
            failed=failed,
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    # ── Search ─────────────────────────────────────────────────────────

    async def similarity_search(
        self,
        *,
        query_embedding: list[float],
        k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        await self.connect()
        start = time.perf_counter()
        where_clause, where_params = _build_metadata_filter(filters)
        sql = SIMILARITY_SQL.format(where_clause=where_clause)
        params = (query_embedding, *where_params, query_embedding, k)
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(sql, params)
                rows = await cur.fetchall()
        latency_ms = (time.perf_counter() - start) * 1000
        return [_row_to_result(r, latency_ms) for r in rows]

    async def keyword_search(self, *, query: str, k: int = 10) -> list[SearchResult]:
        await self.connect()
        start = time.perf_counter()
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(KEYWORD_SQL, (query, query, k))
                rows = await cur.fetchall()
        latency_ms = (time.perf_counter() - start) * 1000
        return [_row_to_result(r, latency_ms) for r in rows]

    async def get_stats(self) -> StoreStats:
        await self.connect()
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(STATS_SQL)
                row = await cur.fetchone()
        return StoreStats(
            doc_count=int(row[0]),
            index_size_mb=float(row[1]),
            avg_query_latency_ms=0.0,
        )


# ── helpers ────────────────────────────────────────────────────────────

def _build_metadata_filter(filters: dict[str, Any] | None) -> tuple[str, tuple]:
    if not filters:
        return "", ()
    clauses = []
    params = []
    for k, v in filters.items():
        clauses.append("metadata->>%s = %s")
        params.extend([k, str(v)])
    return "WHERE " + " AND ".join(clauses), tuple(params)


def _row_to_result(row, latency_ms: float) -> SearchResult:
    chunk_id, content, metadata, score = row
    return SearchResult(
        chunk_id=str(chunk_id),
        content=content,
        score=float(score),
        metadata=metadata or {},
        latency_ms=latency_ms,
    )
