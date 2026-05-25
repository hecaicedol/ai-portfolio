"""Reciprocal Rank Fusion — the same math P6 uses to fuse BM25 + vector,
applied here to fuse graph-traversal hits with vector-search hits.

  rrf(d) = Σ_lists 1 / (k + rank(d, list))

`k=60` is the value from Cormack et al. 2009; the function takes it
as a parameter so the demo can show how the ranking shifts as k changes.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable


def reciprocal_rank_fusion(
    ranked_lists: Iterable[list[str]],
    *,
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked id lists into one ordered by RRF score.

    Each element in `ranked_lists` is a list of document ids ordered
    from best to worst by some retriever. Returns a list of
    `(id, score)` pairs sorted descending by score.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    scores: dict[str, float] = defaultdict(float)
    for lst in ranked_lists:
        for rank, doc_id in enumerate(lst, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
