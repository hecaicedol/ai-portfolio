from typing import Any


class EpisodicMemory:
    """
    Per-session memory in PostgreSQL.

      - save_session(session_id, summary, key_findings) → at session end
      - archive(session_id, kind, content) → for entries evicted from working memory
      - retrieve_relevant_sessions(query, k) → similarity search on session.embedding
      - retrieve_archive(session_id) → reads back evictions from a session
    """

    def __init__(self, dsn: str, *, embed) -> None:
        self.dsn = dsn
        self.embed = embed
        self._pool = None

    async def connect(self) -> None:
        raise NotImplementedError

    async def save_session(self, *, session_id: str, summary: str, key_findings: list[dict[str, Any]]) -> None:
        raise NotImplementedError

    async def archive(self, *, session_id: str, kind: str, content: str) -> None:
        raise NotImplementedError

    async def retrieve_relevant_sessions(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def retrieve_archive(self, session_id: str) -> list[dict[str, Any]]:
        raise NotImplementedError
