from typing import Any


class SemanticMemory:
    """
    Consolidated, durable knowledge in pgvector.

      - upsert_fact(fact, source, confidence) → with embedding-based dedupe
      - retrieve_relevant_knowledge(query, k)
      - supersede(fact_id, new_fact_id) → mark old fact as outdated

    Dedupe rule: if cosine similarity to an existing fact > 0.92, treat as the
    same fact and update confidence/source rather than inserting a duplicate.
    """

    SIMILARITY_DEDUPE_THRESHOLD = 0.92

    def __init__(self, dsn: str, *, embed) -> None:
        self.dsn = dsn
        self.embed = embed
        self._pool = None

    async def connect(self) -> None:
        raise NotImplementedError

    async def upsert_fact(self, *, fact: str, source: str, confidence: float) -> int:
        raise NotImplementedError

    async def retrieve_relevant_knowledge(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def supersede(self, *, old_fact_id: int, new_fact_id: int) -> None:
        raise NotImplementedError
