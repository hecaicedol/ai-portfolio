import json
from typing import Any

import psycopg
import voyageai
from psycopg_pool import AsyncConnectionPool

from config import get_settings


class EpisodicMemory:
    """
    Stores past pipeline errors in PostgreSQL with pgvector embeddings.
    The critic agent queries this before evaluating new output, so it
    learns from past failures across runs.
    """

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn.replace("postgresql+psycopg://", "postgresql://")
        self._pool: AsyncConnectionPool | None = None
        self._voyage = voyageai.AsyncClient(api_key=get_settings().voyage_api_key) if get_settings().voyage_api_key else None

    async def connect(self) -> None:
        self._pool = AsyncConnectionPool(self._dsn, min_size=1, max_size=5, open=False)
        await self._pool.open()

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()

    async def _embed(self, text: str) -> list[float]:
        if self._voyage is None:
            # Fallback to deterministic stub for local dev without a Voyage key.
            return _hash_embedding(text)
        result = await self._voyage.embed([text], model=get_settings().embedding_model, input_type="document")
        return result.embeddings[0]

    async def save_error(
        self,
        *,
        document_type: str,
        error_type: str,
        principle: str,
        context: dict[str, Any],
        resolution: str | None = None,
    ) -> int:
        assert self._pool is not None
        embedding_text = f"[{document_type}][{principle}] {error_type} :: {json.dumps(context, default=str)}"
        embedding = await self._embed(embedding_text)
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO episodic_errors
                      (document_type, error_type, principle, context, resolution, embedding)
                    VALUES (%s, %s, %s, %s::jsonb, %s, %s::vector)
                    RETURNING id
                    """,
                    (document_type, error_type, principle, json.dumps(context), resolution, embedding),
                )
                row = await cur.fetchone()
                return int(row[0])

    async def retrieve_similar_errors(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        assert self._pool is not None
        embedding = await self._embed(query)
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT id, created_at, document_type, error_type, principle, context, resolution,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM episodic_errors
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (embedding, embedding, k),
                )
                rows = await cur.fetchall()
                return [
                    {
                        "id": r[0],
                        "created_at": r[1].isoformat(),
                        "document_type": r[2],
                        "error_type": r[3],
                        "principle": r[4],
                        "context": r[5],
                        "resolution": r[6],
                        "similarity": float(r[7]),
                    }
                    for r in rows
                ]

    async def recent_errors(self, limit: int = 20) -> list[dict[str, Any]]:
        assert self._pool is not None
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT id, created_at, document_type, error_type, principle, context, resolution
                    FROM episodic_errors
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = await cur.fetchall()
                return [
                    {
                        "id": r[0],
                        "created_at": r[1].isoformat(),
                        "document_type": r[2],
                        "error_type": r[3],
                        "principle": r[4],
                        "context": r[5],
                        "resolution": r[6],
                    }
                    for r in rows
                ]

    async def record_run(
        self,
        *,
        document_type: str,
        document_hash: str,
        final_score: float,
        retry_count: int,
        success: bool,
        final_output: dict[str, Any],
        errors_history: list[dict[str, Any]],
    ) -> None:
        assert self._pool is not None
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO pipeline_runs
                      (document_type, document_hash, final_score, retry_count, success, final_output, errors_history)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
                    """,
                    (
                        document_type,
                        document_hash,
                        final_score,
                        retry_count,
                        success,
                        json.dumps(final_output, default=str),
                        json.dumps(errors_history, default=str),
                    ),
                )


def _hash_embedding(text: str, dims: int = 1024) -> list[float]:
    """Deterministic fallback embedding for local development without a Voyage key."""
    import hashlib
    digest = hashlib.sha256(text.encode()).digest()
    raw = (digest * ((dims // len(digest)) + 1))[: dims]
    return [(b - 128) / 128.0 for b in raw]
