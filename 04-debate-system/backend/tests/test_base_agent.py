"""Tests for BaseDebateAgent.generate_statement with ScriptedDebateLLM."""
from __future__ import annotations

import pytest

from agents.optimist import OptimistAgent
from agents.skeptic import SkepticAgent
from tests.conftest import ScriptedDebateLLM


@pytest.mark.asyncio
async def test_generate_statement_parses_clean_json():
    llm = ScriptedDebateLLM(responses_by_role={
        "optimist": [{
            "content": "Three upside vectors.",
            "key_points": ["TAM expansion", "Talent acquisition", "Distribution"],
            "confidence": 0.82,
            "stance": "yes",
        }],
    })
    agent = OptimistAgent(model=llm)
    s = await agent.generate_statement(
        problem="acquire competitor?",
        debate_history=[],
        round_type="opening",
        round_number=1,
    )
    assert s.role == "optimist"
    assert s.round == 1
    assert s.stance == "yes"
    assert len(s.key_points) == 3
    assert 0.0 <= s.confidence <= 1.0


@pytest.mark.asyncio
async def test_generate_statement_tolerates_code_fence():
    llm = ScriptedDebateLLM(responses_by_role={
        "skeptic": [
            "```json\n"
            '{"content":"prior M&A failed","key_points":["case A","case B"],'
            '"confidence":0.6,"stance":"no"}\n'
            "```"
        ],
    })
    agent = SkepticAgent(model=llm)
    s = await agent.generate_statement(
        problem="acquire competitor?",
        debate_history=[],
        round_type="opening",
        round_number=1,
    )
    assert s.stance == "no"
    assert s.key_points == ["case A", "case B"]


@pytest.mark.asyncio
async def test_generate_statement_retries_then_recovers():
    # First reply is broken; second is fine. Agent should still succeed.
    llm = ScriptedDebateLLM(responses_by_role={
        "optimist": [
            "not json at all",
            {"content": "ok", "key_points": ["x"], "confidence": 0.5, "stance": "neutral"},
        ],
    })
    agent = OptimistAgent(model=llm)
    s = await agent.generate_statement(
        problem="p", debate_history=[], round_type="opening", round_number=1,
    )
    assert s.stance == "neutral"


@pytest.mark.asyncio
async def test_generate_statement_unknown_stance_falls_back_to_neutral():
    llm = ScriptedDebateLLM(responses_by_role={
        "optimist": [{
            "content": "ok", "key_points": [],
            "confidence": 0.5, "stance": "MAYBE",
        }],
    })
    agent = OptimistAgent(model=llm)
    s = await agent.generate_statement(
        problem="p", debate_history=[], round_type="opening", round_number=1,
    )
    assert s.stance == "neutral"


@pytest.mark.asyncio
async def test_generate_statement_raises_after_max_retries():
    llm = ScriptedDebateLLM(responses_by_role={
        "optimist": ["bad", "still bad", "yep bad"],
    })
    agent = OptimistAgent(model=llm)
    with pytest.raises(RuntimeError, match="invalid JSON"):
        await agent.generate_statement(
            problem="p", debate_history=[], round_type="opening", round_number=1,
        )
