"""Debate shared-memory backends.

Two implementations, one Protocol:

* `InMemoryDebateSharedMemory` — pure-Python dict store. Used by tests and
  by single-process deployments where Redis is overkill.
* `RedisDebateSharedMemory` — production backend. Lazy redis import so
  the test suite runs without the redis package installed.

The schema (Redis only):
    debate:{session_id}:statements  → LIST of JSON-encoded Statement dicts
    debate:{session_id}:metadata    → HASH {problem, started_at, ...}

Both stores apply TTL on every write (Redis natively; in-memory tracks an
expiry timestamp and evicts on read).
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from typing import Any, Protocol


class DebateSharedMemory(Protocol):
    async def append(self, session_id: str, statement: dict[str, Any]) -> None: ...
    async def history(self, session_id: str) -> list[dict[str, Any]]: ...
    async def set_metadata(self, session_id: str, problem: str) -> None: ...
    async def get_metadata(self, session_id: str) -> dict[str, Any]: ...


class InMemoryDebateSharedMemory:
    """Process-local store. Survives across requests in the same uvicorn
    worker but not across restarts. Perfect for tests and single-node
    demos."""

    def __init__(self, ttl_seconds: int = 604_800) -> None:
        self.ttl_seconds = ttl_seconds
        self._statements: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._metadata: dict[str, dict[str, Any]] = {}
        self._expiry: dict[str, float] = {}

    def _refresh_expiry(self, session_id: str) -> None:
        self._expiry[session_id] = time.time() + self.ttl_seconds

    def _evict_if_expired(self, session_id: str) -> None:
        exp = self._expiry.get(session_id)
        if exp is not None and exp < time.time():
            self._statements.pop(session_id, None)
            self._metadata.pop(session_id, None)
            self._expiry.pop(session_id, None)

    async def append(self, session_id: str, statement: dict[str, Any]) -> None:
        self._evict_if_expired(session_id)
        self._statements[session_id].append(statement)
        self._refresh_expiry(session_id)

    async def history(self, session_id: str) -> list[dict[str, Any]]:
        self._evict_if_expired(session_id)
        return list(self._statements.get(session_id, []))

    async def set_metadata(self, session_id: str, problem: str) -> None:
        self._evict_if_expired(session_id)
        self._metadata[session_id] = {
            "problem": problem,
            "started_at": time.time(),
        }
        self._refresh_expiry(session_id)

    async def get_metadata(self, session_id: str) -> dict[str, Any]:
        self._evict_if_expired(session_id)
        return dict(self._metadata.get(session_id, {}))


class RedisDebateSharedMemory:
    """Redis-backed implementation. Lazy import so the dev test suite
    doesn't need redis installed."""

    def __init__(self, url: str, ttl_seconds: int = 604_800) -> None:
        import redis.asyncio as aioredis  # lazy
        self._client = aioredis.from_url(url, decode_responses=True)
        self.ttl_seconds = ttl_seconds

    async def append(self, session_id: str, statement: dict[str, Any]) -> None:
        key = f"debate:{session_id}:statements"
        await self._client.rpush(key, json.dumps(statement))
        await self._client.expire(key, self.ttl_seconds)

    async def history(self, session_id: str) -> list[dict[str, Any]]:
        rows = await self._client.lrange(f"debate:{session_id}:statements", 0, -1)
        return [json.loads(r) for r in rows]

    async def set_metadata(self, session_id: str, problem: str) -> None:
        key = f"debate:{session_id}:metadata"
        await self._client.hset(key, mapping={"problem": problem, "started_at": str(time.time())})
        await self._client.expire(key, self.ttl_seconds)

    async def get_metadata(self, session_id: str) -> dict[str, Any]:
        return await self._client.hgetall(f"debate:{session_id}:metadata") or {}
