"""Tests for the RRF helper used to fuse vector + graph rankings."""
from __future__ import annotations

import pytest

from graph.rrf import reciprocal_rank_fusion


def test_rrf_combines_disjoint_lists():
    out = reciprocal_rank_fusion([["a", "b"], ["c", "d"]], k=60)
    ids = [doc_id for doc_id, _ in out]
    # First-place items from each list outrank second-place items
    assert ids.index("a") < ids.index("b")
    assert ids.index("c") < ids.index("d")


def test_rrf_boosts_documents_in_multiple_lists():
    # 'x' is rank-2 in both lists; 'a' is rank-1 in only one list.
    # Their RRF scores:
    #   a: 1/(60+1) ≈ 0.01639
    #   x: 1/(60+2) + 1/(60+2) ≈ 0.03226
    out = reciprocal_rank_fusion([["a", "x"], ["b", "x"]], k=60)
    top_id = out[0][0]
    assert top_id == "x", f"shared doc should win (got {out})"


def test_rrf_rejects_zero_k():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=0)
