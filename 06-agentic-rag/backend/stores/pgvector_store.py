from typing import Any
from stores.base_store import BaseVectorStore, EnrichedChunk, IndexResult, SearchResult, StoreStats


class PgVectorStore(BaseVectorStore):
    """
    pgvector implementation.
    Table: chunks(id TEXT PRIMARY KEY, content TEXT, metadata JSONB, embedding vector(1024), tsv TSVECTOR)
    Index: ivfflat on embedding for cosine; GIN on tsv for BM25-ish keyword search.
    """
    name = "pgvector"

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    async def index_documents(self, docs: list[EnrichedChunk]) -> IndexResult:
        raise NotImplementedError

    async def similarity_search(
        self, *, query_embedding: list[float], k: int = 10, filters: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        raise NotImplementedError

    async def keyword_search(self, *, query: str, k: int = 10) -> list[SearchResult]:
        raise NotImplementedError

    async def get_stats(self) -> StoreStats:
        raise NotImplementedError
