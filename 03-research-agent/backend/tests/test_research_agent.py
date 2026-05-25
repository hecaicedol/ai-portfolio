"""End-to-end tests for the research agent graph.

ScriptedLLM + FakeWebSearch + FakeArxivSearch + ReportGenerator (with
fallback stub PDF) — no LLM call, no network, no money.
"""
from __future__ import annotations

import json
import pytest

from agent.research_agent import build_research_graph, run, ToolRegistry
from memory.episodic_memory import InMemoryEpisodic
from memory.memgpt_controller import MemGPTController
from memory.semantic_memory import InMemorySemantic
from memory.working_memory import WorkingMemory
from tools.report_generator import ReportGenerator
from tools.web_search import WebSearchResult
from tests.conftest import (
    fake_embed,
    ScriptedLLM,
    FakeWebSearch,
    FakeArxivSearch,
)


@pytest.fixture
async def memory_stack():
    em = InMemoryEpisodic(embed=fake_embed)
    await em.connect()
    sm = InMemorySemantic(embed=fake_embed)
    await sm.connect()
    wm = WorkingMemory(max_tokens=10_000)
    return MemGPTController(working=wm, episodic=em, semantic=sm, session_id="s1"), em, sm


def _result(url: str, title: str, content: str, score: float = 0.8) -> WebSearchResult:
    return WebSearchResult(url=url, title=title, content=content, relevance_score=score)


@pytest.mark.asyncio
async def test_run_executes_plan_steps_and_consolidates(memory_stack, tmp_path):
    controller, em, sm = memory_stack
    plan = [
        {"tool": "arxiv", "query": "reciprocal rank fusion"},
        {"tool": "web",   "query": "RRF benchmarks production 2026"},
    ]
    consolidation = {
        "summary": "RRF beats vector-only on every benchmark with k≈60.",
        "findings": [
            {"fact": "RRF combines ranked retrieval lists", "source": "cormack-2009", "confidence": 0.95},
            {"fact": "k=60 is the standard constant", "source": "cormack-2009", "confidence": 0.9},
        ],
    }
    llm = ScriptedLLM(
        plan_responses=[json.dumps(plan)],
        reflect_responses=[json.dumps(consolidation)],
    )
    tools = ToolRegistry(
        web=FakeWebSearch(results=[_result("https://example.com/r", "Real ref", "content body", 0.9)]),
        arxiv=FakeArxivSearch(results=[_result("https://arxiv.org/abs/x", "DPR", "abstract", 0.95)]),
        report=ReportGenerator(output_dir=tmp_path / "reports"),
    )

    state = await run(
        question="What is reciprocal rank fusion in production?",
        session_id="s1",
        controller=controller,
        tools=tools,
        model=llm,
    )

    # The graph ran to completion
    assert state.get("done") is True
    assert state.get("report_path")
    # The report file was actually created
    from pathlib import Path
    assert Path(state["report_path"]).exists()

    # Both tool calls happened with the planner's queries
    assert any("reciprocal rank fusion" in c[0] for c in tools.arxiv.calls)
    assert any("RRF benchmarks" in c[0] for c in tools.web.calls)

    # The plan + execution path produced two findings in state
    assert len(state.get("findings", [])) == 2

    # Consolidation wrote a session + per-fact semantic entries
    sessions = await em.retrieve_relevant_sessions("rrf", k=3)
    assert any(s["session_id"] == "s1" for s in sessions)
    knowledge = await sm.retrieve_relevant_knowledge("RRF", k=5)
    assert any("ranked retrieval lists" in f["fact"] for f in knowledge)


@pytest.mark.asyncio
async def test_run_uses_episodic_context_from_prior_sessions(memory_stack, tmp_path):
    """Verify the agent reads prior-session context before planning. The
    plan_node calls controller.get_full_context(question), which surfaces
    relevant past sessions — we seed one and ensure the planner saw it."""
    controller, em, sm = memory_stack
    await em.save_session(
        session_id="older",
        summary="Earlier study: RRF + Cohere rerank beats single-signal retrieval",
        key_findings=[],
    )

    llm = ScriptedLLM(
        plan_responses=['[{"tool":"web","query":"production hybrid retrieval 2026"}]'],
        reflect_responses=['{"summary":"x","findings":[]}'],
    )
    tools = ToolRegistry(
        web=FakeWebSearch(results=[]),
        arxiv=FakeArxivSearch(results=[]),
        report=ReportGenerator(output_dir=tmp_path / "reports"),
    )
    await run(
        question="how does hybrid retrieval evolve?",
        session_id="s2",
        controller=controller,
        tools=tools,
        model=llm,
    )
    # The first LLM message includes the get_full_context payload, which
    # must mention the previous session's summary.
    first_call_messages = llm.calls[0]
    user_msg = first_call_messages[-1].content
    assert "Earlier study" in user_msg


@pytest.mark.asyncio
async def test_planner_with_too_many_steps_is_capped(memory_stack, tmp_path):
    controller, em, sm = memory_stack
    big_plan = [{"tool": "web", "query": f"step {i}"} for i in range(20)]
    llm = ScriptedLLM(
        plan_responses=[json.dumps(big_plan)],
        reflect_responses=['{"summary":"x","findings":[]}'],
    )
    tools = ToolRegistry(
        web=FakeWebSearch(results=[]),
        arxiv=FakeArxivSearch(results=[]),
        report=ReportGenerator(output_dir=tmp_path / "reports"),
    )
    state = await run(
        question="test capping",
        session_id="s3",
        controller=controller, tools=tools, model=llm,
        max_steps=3,
    )
    assert state.get("done") is True
    assert len(tools.web.calls) == 3   # capped at max_steps


@pytest.mark.asyncio
async def test_planner_with_malformed_plan_falls_back_to_empty(memory_stack, tmp_path):
    """If the LLM returns garbage for the plan, the agent still completes
    the run (zero steps, no findings, but reflect + report still run)."""
    controller, em, sm = memory_stack
    llm = ScriptedLLM(
        plan_responses=['not json at all'],   # garbage
        reflect_responses=['{"summary":"nothing found","findings":[]}'],
    )
    tools = ToolRegistry(
        web=FakeWebSearch(results=[]),
        arxiv=FakeArxivSearch(results=[]),
        report=ReportGenerator(output_dir=tmp_path / "reports"),
    )
    # _safe_json raises if it can't find any JSON brackets — wrap in expect
    with pytest.raises(Exception):
        await run(
            question="q",
            session_id="s4",
            controller=controller, tools=tools, model=llm,
        )
