import json
from typing import Any

import redis.asyncio as aioredis


class DebateSharedMemory:
    """
    Persists debate history in Redis. Schema:

      key:  debate:{session_id}:statements   → LIST of JSON-encoded Statement dicts
      key:  debate:{session_id}:metadata     → HASH {problem, started_at, ttl}

    All entries get TTL applied on write.
    """

    def __init__(self, url: str, ttl_seconds: int = 604_800) -> None:
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
        await self._client.hset(key, mapping={"problem": problem})
        await self._client.expire(key, self.ttl_seconds)
