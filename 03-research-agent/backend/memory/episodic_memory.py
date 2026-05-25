"""Episodic memory tier — per-session records.

Two backends:
  • InMemoryEpisodic — Python list/dict storage, testable without Postgres.
  • PostgresEpisodic — production stub (Slice 2 will wire it up).

Both implement the EpisodicMemory Protocol so MemGPTController can swap
them transparently. Similarity-based retrieval uses an injected embed()
function (defaults to a deterministic hash-bag, same trick used in P1's
fallback) so tests don't need Voyage.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Protocol

import numpy as np


Embed = Callable[[str], Awaitable[list[float]]]


@dataclass
class SessionRecord:
    session_id: str
    summary: str
    key_findings: list[dict[str, Any]]
    embedding: list[float] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ArchiveEntry:
    session_id: str
    kind: str
    content: str
    embedding: list[float] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class EpisodicMemory(Protocol):
    async def connect(self) -> None: ...
    async def save_session(
        self, *, session_id: str, summary: str, key_findings: list[dict[str, Any]]
    ) -> None: ...
    async def archive(self, *, session_id: str, kind: str, content: str) -> None: ...
    async def retrieve_relevant_sessions(
        self, query: str, k: int = 3
    ) -> list[dict[str, Any]]: ...
    async def retrieve_archive(self, session_id: str) -> list[dict[str, Any]]: ...


# ── Default embed: deterministic hash-bag, identical to P1's dev fallback.
async def _default_embed(text: str, dim: int = 256) -> list[float]:
    vec = [0.0] * dim
    for token in text.lower().split():
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    an = float(np.linalg.norm(av))
    bn = float(np.linalg.norm(bv))
    if an == 0.0 or bn == 0.0:
        return 0.0
    return float(np.dot(av, bv) / (an * bn))


class InMemoryEpisodic:
    """In-memory backend. Used in tests and dev mode where Postgres isn't
    available. Identical surface to PostgresEpisodic (below)."""

    def __init__(self, *, embed: Embed | None = None) -> None:
        self.embed = embed or _default_embed
        self.sessions: list[SessionRecord] = []
        self.archives: list[ArchiveEntry] = []
        self._connected = False

    async def connect(self) -> None:
        self._connected = True

    async def save_session(
        self,
        *,
        session_id: str,
        summary: str,
        key_findings: list[dict[str, Any]],
    ) -> None:
        embedding = await self.embed(summary)
        # Sessions are unique per ID; later saves replace earlier ones
        self.sessions = [s for s in self.sessions if s.session_id != session_id]
        self.sessions.append(
            SessionRecord(
                session_id=session_id,
                summary=summary,
                key_findings=list(key_findings),
                embedding=embedding,
            )
        )

    async def archive(
        self,
        *,
        session_id: str,
        kind: str,
        content: str,
    ) -> None:
        embedding = await self.embed(content)
        self.archives.append(
            ArchiveEntry(
                session_id=session_id,
                kind=kind,
                content=content,
                embedding=embedding,
            )
        )

    async def retrieve_relevant_sessions(
        self,
        query: str,
        k: int = 3,
    ) -> list[dict[str, Any]]:
        if not self.sessions:
            return []
        q = await self.embed(query)
        scored = [
            (_cosine(q, s.embedding), s) for s in self.sessions
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "session_id": s.session_id,
                "summary": s.summary,
                "key_findings": s.key_findings,
                "similarity": sim,
                "created_at": s.created_at.isoformat(),
            }
            for sim, s in scored[:k]
        ]

    async def retrieve_archive(self, session_id: str) -> list[dict[str, Any]]:
        return [
            {
                "session_id": a.session_id,
                "kind": a.kind,
                "content": a.content,
                "created_at": a.created_at.isoformat(),
            }
            for a in self.archives
            if a.session_id == session_id
        ]


class PostgresEpisodic:
    """Production backend. Stub until Slice 2 wires the real pgvector schema
    (sessions + archive tables, ivfflat index on embeddings)."""

    def __init__(self, dsn: str, *, embed: Embed | None = None) -> None:
        self.dsn = dsn
        self.embed = embed or _default_embed
        self._pool = None

    async def connect(self) -> None:
        raise NotImplementedError("PostgresEpisodic — wire in Slice 2")

    async def save_session(self, **kwargs) -> None:
        raise NotImplementedError

    async def archive(self, **kwargs) -> None:
        raise NotImplementedError

    async def retrieve_relevant_sessions(self, query, k=3):
        raise NotImplementedError

    async def retrieve_archive(self, session_id):
        raise NotImplementedError
