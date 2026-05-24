"""Unit tests for ExtractorAgent and CriticAgent — focused on the JSON-retry fix."""
import json

import pytest

from agents.critic import CriticAgent, CriticParseError
from agents.extractor import ExtractorAgent, ExtractorParseError
from tests.conftest import FakeMemory, ScriptedLLM


@pytest.mark.asyncio
async def test_extractor_returns_parsed_json_on_first_try():
    llm = ScriptedLLM(extractor_responses=[json.dumps({"vendor": "Acme", "total": 99.0})])
    agent = ExtractorAgent(model=llm)
    out = await agent.run(document_type="invoice", content="bla")
    assert out == {"vendor": "Acme", "total": 99.0}
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_extractor_strips_markdown_fences():
    fenced = "```json\n" + json.dumps({"vendor": "Acme"}) + "\n```"
    llm = ScriptedLLM(extractor_responses=[fenced])
    agent = ExtractorAgent(model=llm)
    out = await agent.run(document_type="invoice", content="bla")
    assert out == {"vendor": "Acme"}


@pytest.mark.asyncio
async def test_extractor_retries_on_invalid_json_then_succeeds():
    llm = ScriptedLLM(
        extractor_responses=[
            "I'm sorry, I cannot extract this document.",
            "still no JSON here",
            json.dumps({"vendor": "Eventual"}),
        ]
    )
    agent = ExtractorAgent(model=llm)
    out = await agent.run(document_type="invoice", content="bla")
    assert out == {"vendor": "Eventual"}
    assert len(llm.calls) == 3


@pytest.mark.asyncio
async def test_extractor_raises_after_max_attempts():
    llm = ScriptedLLM(extractor_responses=["nope", "still nope", "third nope"])
    agent = ExtractorAgent(model=llm)
    with pytest.raises(ExtractorParseError):
        await agent.run(document_type="invoice", content="bla")
    assert len(llm.calls) == 3


@pytest.mark.asyncio
async def test_critic_includes_past_errors_in_prompt(settings):
    memory = FakeMemory()
    await memory.save_error(
        document_type="invoice",
        error_type="completeness_below_threshold",
        principle="completeness",
        context={"missing": ["tax_id"]},
    )
    good_critic_response = json.dumps(
        {
            "overall_score": 0.9,
            "principles": [
                {"principle": "completeness", "score": 0.9, "feedback": "ok"},
                {"principle": "accuracy", "score": 0.9, "feedback": "ok"},
                {"principle": "consistency", "score": 0.9, "feedback": "ok"},
                {"principle": "format", "score": 0.9, "feedback": "ok"},
            ],
        }
    )
    llm = ScriptedLLM(critic_responses=[good_critic_response])
    critic = CriticAgent(model=llm, memory=memory, pass_threshold=settings.critic_pass_threshold)
    report = await critic.run(
        document_type="invoice",
        source="raw text",
        extracted={"vendor": "Acme"},
        structural={"structural_pass": True, "missing_fields": []},
    )
    assert report.passes is True
    assert report.overall_score == 0.9
    # The past error's error_type should surface in similar_past_errors
    assert "completeness_below_threshold" in report.similar_past_errors
    # And the LLM call should have included past_similar_errors in the user message
    user_msg = llm.calls[0][1].content
    assert "past_similar_errors" in user_msg
    assert "tax_id" in user_msg


@pytest.mark.asyncio
async def test_critic_retries_on_invalid_json(settings):
    memory = FakeMemory()
    good = json.dumps(
        {
            "overall_score": 0.9,
            "principles": [
                {"principle": "completeness", "score": 0.9, "feedback": "ok"},
                {"principle": "accuracy", "score": 0.9, "feedback": "ok"},
                {"principle": "consistency", "score": 0.9, "feedback": "ok"},
                {"principle": "format", "score": 0.9, "feedback": "ok"},
            ],
        }
    )
    llm = ScriptedLLM(critic_responses=["not json", good])
    critic = CriticAgent(model=llm, memory=memory, pass_threshold=settings.critic_pass_threshold)
    report = await critic.run(
        document_type="invoice",
        source="raw text",
        extracted={"vendor": "Acme"},
        structural={"structural_pass": True, "missing_fields": []},
    )
    assert report.passes is True
    assert len(llm.calls) == 2


@pytest.mark.asyncio
async def test_critic_raises_after_max_attempts(settings):
    memory = FakeMemory()
    llm = ScriptedLLM(critic_responses=["x", "y", "z"])
    critic = CriticAgent(model=llm, memory=memory, pass_threshold=settings.critic_pass_threshold)
    with pytest.raises(CriticParseError):
        await critic.run(
            document_type="invoice",
            source="raw text",
            extracted={"vendor": "Acme"},
            structural={"structural_pass": True, "missing_fields": []},
        )
