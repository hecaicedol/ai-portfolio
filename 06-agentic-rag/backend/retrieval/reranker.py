from stores.base_store import SearchResult


class CohereReranker:
    """
    Cross-encoder reranking via Cohere Rerank v3.

    Takes the fused top-K from HybridSearcher, sends to Cohere with the original
    query, and returns the reordered top-N (N < K).
    """

    def __init__(self, *, api_key: str, model: str = "rerank-english-v3.0", top_n: int = 5) -> None:
        self.api_key = api_key
        self.model = model
        self.top_n = top_n

    async def rerank(self, *, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        raise NotImplementedError
