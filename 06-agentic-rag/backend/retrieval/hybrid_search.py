from collections import defaultdict
from stores.base_store import BaseVectorStore, SearchResult


RRF_K = 60  # standard reciprocal rank fusion parameter


class HybridSearcher:
    """
    Per store:
      1. Use Claude to generate 3 alternative phrasings of the query.
      2. For each phrasing:
         - keyword_search (BM25 / FTS / sparse, depending on store)
         - similarity_search (vector)
      3. Merge all returned lists via Reciprocal Rank Fusion (RRF):
           score(d) = sum( 1 / (k + rank_i(d)) ) over each result list
      4. Return top-K fused results.

    The pipeline runs once per store; results are returned per store for the
    benchmark dashboard side-by-side view.
    """

    def __init__(self, *, query_rewriter, embed, k: int = 10) -> None:
        self.query_rewriter = query_rewriter
        self.embed = embed
        self.k = k

    async def search(self, *, store: BaseVectorStore, query: str) -> list[SearchResult]:
        rewrites = await self.query_rewriter.rewrite(query, n=3)
        result_lists: list[list[SearchResult]] = []
        for q in [query, *rewrites]:
            embedding = await self.embed(q)
            vector_hits = await store.similarity_search(query_embedding=embedding, k=self.k)
            keyword_hits = await store.keyword_search(query=q, k=self.k)
            result_lists.append(vector_hits)
            result_lists.append(keyword_hits)
        return self._rrf(result_lists, top_k=self.k)

    @staticmethod
    def _rrf(result_lists: list[list[SearchResult]], top_k: int) -> list[SearchResult]:
        scores: dict[str, float] = defaultdict(float)
        by_id: dict[str, SearchResult] = {}
        for results in result_lists:
            for rank, r in enumerate(results, start=1):
                scores[r.chunk_id] += 1.0 / (RRF_K + rank)
                by_id.setdefault(r.chunk_id, r)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [by_id[chunk_id] for chunk_id, _ in ranked]
