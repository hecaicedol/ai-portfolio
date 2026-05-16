from typing import Any
from stores.base_store import BaseVectorStore, EnrichedChunk, IndexResult, SearchResult, StoreStats


class QdrantStore(BaseVectorStore):
    """
    Qdrant implementation via qdrant-client async.
    Collection config: vectors size=1024, distance=Cosine, on_disk=True for fair comparison.
    """
    name = "qdrant"

    def __init__(self, url: str, collection: str = "rag-benchmark") -> None:
        self.url = url
        self.collection = collection

    async def index_documents(self, docs: list[EnrichedChunk]) -> IndexResult:
        raise NotImplementedError

    async def similarity_search(
        self, *, query_embedding: list[float], k: int = 10, filters: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        raise NotImplementedError

    async def keyword_search(self, *, query: str, k: int = 10) -> list[SearchResult]:
        """Qdrant supports text-match conditions for keyword search."""
        raise NotImplementedError

    async def get_stats(self) -> StoreStats:
        raise NotImplementedError
