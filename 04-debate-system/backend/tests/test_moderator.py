"""End-to-end test of the debate graph using ScriptedDebateLLM.

Wires all five agents + the synthesizer + the in-memory shared store
and verifies the full debate produces:
  · 5 + 5 + 5 = 15 statements
  · an ExecutiveMemo with the consensus we compute from final stances
  · events for every round transition through the on_event callback
"""
from __future__ import annotations

import uuid

import pytest

from agents.devils_advocate import DevilsAdvocateAgent
from agents.financial import FinancialAgent
from agents.optimist import OptimistAgent
from agents.risk import RiskAgent
from agents.skeptic import SkepticAgent
from agents.synthesizer import SynthesizerAgent
from debate.moderator import build_debate_graph, run
from debate.shared_memory import InMemoryDebateSharedMemory
from tests.conftest import ScriptedDebateLLM


def _three_round_responses(stances: tuple[str, str, str], conf: float = 0.7):
    """Return three JSON payloads for opening / rebuttal / final."""
    return [
        {"content": f"opening · {stances[0]}", "key_points": ["kp1", "kp2"],
         "confidence": conf, "stance": stances[0]},
        {"content": f"rebuttal · {stances[1]}", "key_points": ["kp"],
         "confidence": conf, "stance": stances[1]},
        {"content": f"final · {stances[2]}", "key_points": ["kp"],
         "confidence": conf, "stance": stances[2]},
    ]


@pytest.mark.asyncio
async def test_full_debate_produces_15_statements_and_an_exec_memo():
    llm = ScriptedDebateLLM(
        responses_by_role={
            "optimist":        _three_round_responses(("yes", "yes", "yes")),
            "skeptic":         _three_round_responses(("no", "no", "no")),
            "financial":       _three_round_responses(("yes", "yes", "yes")),
            "risk":            _three_round_responses(("neutral", "neutral", "neutral")),
            "devils_advocate": _three_round_responses(("no", "no", "no")),
        },
        synth_response={
            "recommended_decision": "Proceed with safeguards.",
            "confidence": 0.7,
            "key_supporting_arguments": ["A", "B"],
            "key_risks_to_monitor": ["R1"],
            "dissenting_views": ["devils: distribution thesis weak"],
            "next_steps": ["pilot 30d"],
        },
    )
    agents = [
        OptimistAgent(model=llm),
        SkepticAgent(model=llm),
        FinancialAgent(model=llm),
        RiskAgent(model=llm),
        DevilsAdvocateAgent(model=llm),
    ]
    synth = SynthesizerAgent(model=llm)
    mem = InMemoryDebateSharedMemory()

    events: list[dict] = []
    async def on_event(e):
        events.append(e)

    graph = build_debate_graph(
        agents=agents, synthesizer=synth, shared_memory=mem, on_event=on_event,
    )
    session_id = str(uuid.uuid4())
    out = await run(
        graph=graph, problem="acquire vendor X?", session_id=session_id,
        shared_memory=mem, on_event=on_event,
    )

    assert out["debate_complete"] is True
    assert len(out["statements"]) == 15  # 5 agents × 3 rounds
    assert out["executive_memo"]["recommended_decision"] == "Proceed with safeguards."

    # All persisted to shared memory
    history = await mem.history(session_id)
    assert len(history) == 15

    # Event stream covers all the round transitions
    event_types = [e["type"] for e in events]
    assert event_types.count("round_start") == 3
    assert event_types.count("round_complete") == 3
    assert "synthesis_start" in event_types
    assert event_types[-1] == "debate_complete"


@pytest.mark.asyncio
async def test_consensus_reflects_final_round_stances():
    # All five end at strong_yes ⇒ consensus = 1.0
    llm = ScriptedDebateLLM(
        responses_by_role={
            role: _three_round_responses(("yes", "yes", "strong_yes"))
            for role in ("optimist", "skeptic", "financial", "risk", "devils_advocate")
        },
        synth_response={
            "recommended_decision": "Unanimous yes — proceed.",
            "confidence": 0.95,
            "key_supporting_arguments": ["all five aligned"],
            "key_risks_to_monitor": [],
            "dissenting_views": [],
            "next_steps": ["execute"],
        },
    )
    agents = [
        OptimistAgent(model=llm), SkepticAgent(model=llm),
        FinancialAgent(model=llm), RiskAgent(model=llm),
        DevilsAdvocateAgent(model=llm),
    ]
    mem = InMemoryDebateSharedMemory()
    graph = build_debate_graph(
        agents=agents, synthesizer=SynthesizerAgent(model=llm), shared_memory=mem,
    )
    out = await run(graph=graph, problem="p", session_id="s", shared_memory=mem)
    assert out["consensus_level"] == 1.0


@pytest.mark.asyncio
async def test_rebuttal_round_sees_all_openings_before_each_agent_speaks():
    """Sequential reads: by the time the LAST rebuttal-round agent speaks,
    history should already contain every opening AND every preceding
    rebuttal from this round."""
    captured: list[int] = []

    class CaptureLLM(ScriptedDebateLLM):
        async def ainvoke(self, messages):
            # Only count when the user prompt mentions rebuttal
            sys = messages[0].content if hasattr(messages[0], "content") else messages[0].get("content", "")
            user = messages[-1].content if hasattr(messages[-1], "content") else messages[-1].get("content", "")
            # Match the exact <round_type> marker — the scripted statement
            # bodies also contain the word "rebuttal" so a naive substring
            # match would also count the final-round calls.
            if "<round_type>rebuttal</round_type>" in user and "Synthesizer" not in sys:
                captured.append(user.count("[round"))
            return await super().ainvoke(messages)

    llm = CaptureLLM(
        responses_by_role={
            role: _three_round_responses(("yes", "yes", "yes"))
            for role in ("optimist", "skeptic", "financial", "risk", "devils_advocate")
        },
        synth_response={
            "recommended_decision": "x", "confidence": 0.5,
            "key_supporting_arguments": [], "key_risks_to_monitor": [],
            "dissenting_views": [], "next_steps": [],
        },
    )
    agents = [
        OptimistAgent(model=llm), SkepticAgent(model=llm),
        FinancialAgent(model=llm), RiskAgent(model=llm),
        DevilsAdvocateAgent(model=llm),
    ]
    mem = InMemoryDebateSharedMemory()
    graph = build_debate_graph(
        agents=agents, synthesizer=SynthesizerAgent(model=llm), shared_memory=mem,
    )
    await run(graph=graph, problem="p", session_id="s", shared_memory=mem)

    # Rebuttal #1 sees 5 openings; rebuttal #5 sees 5 openings + 4 prior rebuttals = 9
    assert captured == [5, 6, 7, 8, 9], f"sequential reads broken: {captured}"
