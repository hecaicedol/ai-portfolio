"""End-to-end orchestrator tests using ScriptedLLM + FakeMemory.

These exercise the four canonical pipeline paths:
  1. Pass on first attempt (no reflection).
  2. Self-heal on second attempt (one reflection cycle).
  3. Hit max iterations without ever passing (fail state).
  4. record_run is wired correctly and includes the right retry_count.
"""
import json

import pytest

from agents.orchestrator import build_graph, run_pipeline, stream_pipeline
from tests.conftest import FakeMemory, ScriptedLLM


def _good_invoice_extract() -> str:
    return json.dumps(
        {
            "invoice_number": "INV-001",
            "vendor": "Acme Corp",
            "total": 1000.00,
            "issue_date": "2026-01-15",
        }
    )


def _bad_invoice_extract() -> str:
    return json.dumps(
        {
            "invoice_number": "INV-001",
            "vendor": None,
            "total": None,
            "issue_date": None,
        }
    )


def _empty_extract() -> str:
    return json.dumps(
        {
            "invoice_number": None,
            "vendor": None,
            "total": None,
            "issue_date": None,
        }
    )


def _critic_pass(score: float = 0.95) -> str:
    return json.dumps(
        {
            "overall_score": score,
            "principles": [
                {"principle": "completeness", "score": score, "feedback": "All fields present"},
                {"principle": "accuracy", "score": score, "feedback": "Values match source"},
                {"principle": "consistency", "score": score, "feedback": "No contradictions"},
                {"principle": "format", "score": score, "feedback": "Format correct"},
            ],
        }
    )


def _critic_fail(score: float = 0.4) -> str:
    return json.dumps(
        {
            "overall_score": score,
            "principles": [
                {
                    "principle": "completeness",
                    "score": 0.2,
                    "feedback": "vendor, total, issue_date are null",
                },
                {
                    "principle": "accuracy",
                    "score": 0.5,
                    "feedback": "cannot verify null values against source",
                },
                {
                    "principle": "consistency",
                    "score": 0.6,
                    "feedback": "missing fields can't be cross-checked",
                },
                {
                    "principle": "format",
                    "score": 0.3,
                    "feedback": "expected populated fields",
                },
            ],
        }
    )


@pytest.mark.asyncio
async def test_pipeline_passes_on_first_attempt(settings):
    memory = FakeMemory()
    llm = ScriptedLLM(
        extractor_responses=[_good_invoice_extract()],
        critic_responses=[_critic_pass()],
    )
    graph = build_graph(memory=memory, settings=settings, model=llm)
    result = await run_pipeline(
        graph=graph,
        memory=memory,
        document_type="invoice",
        content="Invoice #INV-001 Acme Corp $1000 2026-01-15",
        metadata={},
    )

    assert result.success is True
    assert result.iterations == 1
    assert result.final_score == 0.95
    assert result.extracted_data["vendor"] == "Acme Corp"
    assert result.extracted_data["total"] == 1000.0
    assert result.errors_history == []

    # record_run should have been called once, with retry_count=0
    assert len(memory.runs) == 1
    run = memory.runs[0]
    assert run["success"] is True
    assert run["retry_count"] == 0
    assert run["final_score"] == 0.95
    assert run["document_type"] == "invoice"
    assert len(run["document_hash"]) == 16  # _doc_hash truncates to 16 chars

    # No errors saved because critic passed on first try
    assert memory.errors == []


@pytest.mark.asyncio
async def test_pipeline_self_heals_on_second_attempt(settings):
    memory = FakeMemory()
    llm = ScriptedLLM(
        extractor_responses=[_bad_invoice_extract(), _good_invoice_extract()],
        critic_responses=[_critic_fail(), _critic_pass()],
    )
    graph = build_graph(memory=memory, settings=settings, model=llm)
    result = await run_pipeline(
        graph=graph,
        memory=memory,
        document_type="invoice",
        content="Invoice #INV-001 Acme Corp $1000 2026-01-15",
        metadata={},
    )

    assert result.success is True
    assert result.iterations == 2
    assert result.final_score == 0.95
    assert result.extracted_data["vendor"] == "Acme Corp"
    assert len(result.errors_history) == 1
    assert result.errors_history[0]["iteration"] == 1
    assert result.errors_history[0]["score"] == 0.4

    # record_run called once, retry_count=1 (one reflection)
    assert len(memory.runs) == 1
    assert memory.runs[0]["retry_count"] == 1
    assert memory.runs[0]["success"] is True

    # All 4 principles scored < 0.85 in the first critic pass → 4 errors saved
    assert len(memory.errors) == 4
    saved_principles = {e["principle"] for e in memory.errors}
    assert saved_principles == {"completeness", "accuracy", "consistency", "format"}


@pytest.mark.asyncio
async def test_pipeline_fails_after_max_iterations(settings):
    memory = FakeMemory()
    llm = ScriptedLLM(
        extractor_responses=[_empty_extract(), _empty_extract(), _empty_extract()],
        critic_responses=[_critic_fail(0.3), _critic_fail(0.3), _critic_fail(0.3)],
    )
    graph = build_graph(memory=memory, settings=settings, model=llm)
    result = await run_pipeline(
        graph=graph,
        memory=memory,
        document_type="invoice",
        content="garbage content",
        metadata={},
    )

    assert result.success is False
    assert result.iterations == settings.max_reflection_iterations  # 3
    assert result.final_score == 0.3
    # errors_history has one entry per failed critic pass (iterations 1 and 2;
    # the third never enters reflect because route_after_critic short-circuits to synthesize)
    assert len(result.errors_history) == 2

    # record_run called once with success=False, retry_count=2
    assert len(memory.runs) == 1
    assert memory.runs[0]["success"] is False
    assert memory.runs[0]["retry_count"] == 2

    # 2 reflections × 4 failing principles = 8 errors persisted
    assert len(memory.errors) == 8


@pytest.mark.asyncio
async def test_stream_pipeline_emits_run_started_and_completed(settings):
    memory = FakeMemory()
    llm = ScriptedLLM(
        extractor_responses=[_good_invoice_extract()],
        critic_responses=[_critic_pass()],
    )
    graph = build_graph(memory=memory, settings=settings, model=llm)

    events = []
    async for ev in stream_pipeline(
        graph=graph,
        memory=memory,
        document_type="invoice",
        content="Invoice #INV-001 Acme Corp $1000 2026-01-15",
        metadata={},
    ):
        events.append(ev)

    types = [e.type for e in events]
    assert types[0] == "run_started"
    assert types[-1] == "run_completed"
    # Between them: agent_finished events for extract, validate, critique, synthesize
    agent_names = [e.agent for e in events if e.type == "agent_finished"]
    assert set(agent_names) >= {"extract", "validate", "critique", "synthesize"}

    # Streaming path also calls record_run
    assert len(memory.runs) == 1
    assert memory.runs[0]["success"] is True
