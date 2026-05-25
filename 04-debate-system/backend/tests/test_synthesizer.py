"""Tests for SynthesizerAgent.synthesize."""
from __future__ import annotations

import pytest

from agents.base_agent import Statement
from agents.synthesizer import SynthesizerAgent
from tests.conftest import ScriptedDebateLLM


def _stmt(role: str, round_: int, stance: str) -> Statement:
    return Statement(
        role=role, round=round_, content=f"{role} round {round_}",
        key_points=[f"kp-{role}-{round_}"], confidence=0.7, stance=stance,
    )


@pytest.mark.asyncio
async def test_synthesize_builds_executive_memo_and_uses_computed_consensus():
    # Heavy split: strong_yes vs strong_no on the extremes ⇒ moderate consensus
    all_stmts = [
        _stmt("optimist", 3, "strong_yes"),
        _stmt("skeptic", 3, "strong_no"),
        _stmt("financial", 3, "yes"),
        _stmt("risk", 3, "neutral"),
        _stmt("devils_advocate", 3, "no"),
    ]
    llm = ScriptedDebateLLM(synth_response={
        "recommended_decision": "Proceed conditionally.",
        "confidence": 0.65,
        # The synthesizer ignores any consensus_level the LLM tries to send;
        # we always overwrite with the computed value.
        "consensus_level": 0.99,
        "key_supporting_arguments": ["Optimist on TAM", "Financial on payback"],
        "key_risks_to_monitor": ["Integration risk"],
        "dissenting_views": ["Devil's Advocate: distribution moat is overstated"],
        "next_steps": ["Run a 30-day pilot"],
    })
    synth = SynthesizerAgent(model=llm)
    memo = await synth.synthesize(problem="acquire?", all_statements=all_stmts)
    assert memo.recommended_decision == "Proceed conditionally."
    # Heavily split stances ⇒ moderate consensus, NOT the 0.99 the LLM tried to inject
    assert 0.3 <= memo.consensus_level <= 0.7, f"got {memo.consensus_level:.3f}"
    assert memo.consensus_level != 0.99


@pytest.mark.asyncio
async def test_synthesize_retries_on_invalid_json():
    llm = ScriptedDebateLLM(synth_response=None)
    llm.synth_response = {
        "recommended_decision": "x", "confidence": 0.5,
        "key_supporting_arguments": [], "key_risks_to_monitor": [],
        "dissenting_views": [], "next_steps": [],
    }
    memo = await SynthesizerAgent(model=llm).synthesize(
        problem="p", all_statements=[_stmt("optimist", 3, "yes")],
    )
    assert memo.confidence == 0.5
