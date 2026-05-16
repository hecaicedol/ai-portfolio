from typing import Any
from stores.base_store import BaseVectorStore, EnrichedChunk, IndexResult, SearchResult, StoreStats


class PineconeStore(BaseVectorStore):
    """
    Pinecone (serverless) implementation.
    Index: dimension=1024, metric=cosine, cloud=aws, region=us-east-1.
    """
    name = "pinecone"

    def __init__(self, api_key: str, index_name: str) -> None:
        self.api_key = api_key
        self.index_name = index_name

    async def index_documents(self, docs: list[EnrichedChunk]) -> IndexResult:
        raise NotImplementedError

    async def similarity_search(
        self, *, query_embedding: list[float], k: int = 10, filters: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        raise NotImplementedError

    async def keyword_search(self, *, query: str, k: int = 10) -> list[SearchResult]:
        """Pinecone hybrid (alpha-weighted sparse+dense) — sparse component via BM25 client-side."""
        raise NotImplementedError

    async def get_stats(self) -> StoreStats:
        raise NotImplementedError
