from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel, Field


class EnrichedChunk(BaseModel):
    id: str
    content: str
    enriched_content: str  # context-prefixed for embedding
    embedding: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    chunk_id: str
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float


class StoreStats(BaseModel):
    doc_count: int
    index_size_mb: float
    avg_query_latency_ms: float


class IndexResult(BaseModel):
    store: str
    indexed: int
    failed: int
    latency_ms: float


class BaseVectorStore(ABC):
    """
    Common interface for pgvector, Qdrant, Pinecone implementations.

    All methods must:
      - Record `latency_ms` on every search.
      - Support metadata filtering (e.g. tenant_id).
      - Use the same embedding dimension (1024 — Voyage `voyage-3`).
    """

    name: str

    @abstractmethod
    async def index_documents(self, docs: list[EnrichedChunk]) -> IndexResult: ...

    @abstractmethod
    async def similarity_search(
        self, *, query_embedding: list[float], k: int = 10, filters: dict[str, Any] | None = None
    ) -> list[SearchResult]: ...

    @abstractmethod
    async def keyword_search(self, *, query: str, k: int = 10) -> list[SearchResult]: ...

    @abstractmethod
    async def get_stats(self) -> StoreStats: ...
