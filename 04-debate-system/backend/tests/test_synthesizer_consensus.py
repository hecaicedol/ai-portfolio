"""Consensus is the deterministic, auditable half of the synthesizer.
These tests pin its behavior down to the decimal."""
from __future__ import annotations

import math

from agents.base_agent import Statement
from agents.synthesizer import SynthesizerAgent


def _final(role: str, stance: str, conf: float = 0.7) -> Statement:
    return Statement(
        role=role, round=3, content="…", key_points=["x"],
        confidence=conf, stance=stance,
    )


def test_consensus_empty_is_zero():
    assert SynthesizerAgent.consensus([]) == 0.0


def test_consensus_full_agreement_is_one():
    stmts = [_final(f"a{i}", "yes") for i in range(5)]
    assert math.isclose(SynthesizerAgent.consensus(stmts), 1.0)


def test_consensus_split_yes_no_is_low():
    stmts = [
        _final("a", "strong_yes"),
        _final("b", "strong_no"),
        _final("c", "strong_yes"),
        _final("d", "strong_no"),
        _final("e", "neutral"),
    ]
    score = SynthesizerAgent.consensus(stmts)
    assert 0.0 <= score <= 0.2  # heavy disagreement


def test_consensus_clusters_near_one_stance_is_high():
    # Four agree on 'yes', one neutral
    stmts = [
        _final("a", "yes"), _final("b", "yes"),
        _final("c", "yes"), _final("d", "yes"),
        _final("e", "neutral"),
    ]
    score = SynthesizerAgent.consensus(stmts)
    assert score > 0.9, f"tight cluster should score >0.9 (got {score:.3f})"
