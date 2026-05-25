"""Semantic memory tier — durable, deduplicated facts across all sessions.

Two backends with the same Protocol surface:
  • InMemorySemantic — list of facts; cosine dedupe via numpy. Testable.
  • PostgresSemantic — production stub (Slice 2).

The deduplication rule: if a new fact's embedding has cosine similarity
> SIMILARITY_DEDUPE_THRESHOLD (0.92) against any existing fact, treat
it as the same fact. Update confidence (max of the two), append the
new source. Return the existing id.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Protocol

from memory.episodic_memory import _cosine, _default_embed, Embed


SIMILARITY_DEDUPE_THRESHOLD = 0.92


@dataclass
class FactRecord:
    id: int
    fact: str
    sources: list[str] = field(default_factory=list)
    confidence: float = 0.5
    embedding: list[float] = field(default_factory=list)
    superseded_by: int | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SemanticMemory(Protocol):
    async def connect(self) -> None: ...
    async def upsert_fact(
        self, *, fact: str, source: str, confidence: float
    ) -> int: ...
    async def retrieve_relevant_knowledge(
        self, query: str, k: int = 5
    ) -> list[dict[str, Any]]: ...
    async def supersede(self, *, old_fact_id: int, new_fact_id: int) -> None: ...


class InMemorySemantic:
    SIMILARITY_DEDUPE_THRESHOLD = SIMILARITY_DEDUPE_THRESHOLD

    def __init__(self, *, embed: Embed | None = None) -> None:
        self.embed = embed or _default_embed
        self.facts: list[FactRecord] = []
        self._next_id = 1
        self._connected = False

    async def connect(self) -> None:
        self._connected = True

    async def upsert_fact(
        self,
        *,
        fact: str,
        source: str,
        confidence: float,
    ) -> int:
        embedding = await self.embed(fact)

        # Dedupe: find any existing live fact with cosine > threshold
        for existing in self.facts:
            if existing.superseded_by is not None:
                continue
            sim = _cosine(embedding, existing.embedding)
            if sim > self.SIMILARITY_DEDUPE_THRESHOLD:
                if source and source not in existing.sources:
                    existing.sources.append(source)
                existing.confidence = max(existing.confidence, float(confidence))
                return existing.id

        # Insert new
        fid = self._next_id
        self._next_id += 1
        self.facts.append(
            FactRecord(
                id=fid,
                fact=fact,
                sources=[source] if source else [],
                confidence=float(confidence),
                embedding=embedding,
            )
        )
        return fid

    async def retrieve_relevant_knowledge(
        self,
        query: str,
        k: int = 5,
    ) -> list[dict[str, Any]]:
        live = [f for f in self.facts if f.superseded_by is None]
        if not live:
            return []
        q = await self.embed(query)
        scored = [(_cosine(q, f.embedding), f) for f in live]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "id": f.id,
                "fact": f.fact,
                "source": ", ".join(f.sources) if f.sources else "",
                "sources": list(f.sources),
                "confidence": f.confidence,
                "similarity": sim,
            }
            for sim, f in scored[:k]
        ]

    async def supersede(
        self,
        *,
        old_fact_id: int,
        new_fact_id: int,
    ) -> None:
        for f in self.facts:
            if f.id == old_fact_id:
                f.superseded_by = new_fact_id
                return


class PostgresSemantic:
    """Production backend over Postgres + pgvector. Stub until Slice 2."""

    SIMILARITY_DEDUPE_THRESHOLD = SIMILARITY_DEDUPE_THRESHOLD

    def __init__(self, dsn: str, *, embed: Embed | None = None) -> None:
        self.dsn = dsn
        self.embed = embed or _default_embed
        self._pool = None

    async def connect(self) -> None:
        raise NotImplementedError("PostgresSemantic — wire in Slice 2")

    async def upsert_fact(self, **kwargs) -> int:
        raise NotImplementedError

    async def retrieve_relevant_knowledge(self, query, k=5):
        raise NotImplementedError

    async def supersede(self, **kwargs) -> None:
        raise NotImplementedError
