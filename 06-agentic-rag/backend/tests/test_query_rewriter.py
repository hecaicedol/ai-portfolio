"""Tests for `retrieval.query_rewriter.QueryRewriter`."""
from __future__ import annotations

import json

import pytest

from retrieval.query_rewriter import QueryRewriter, QueryRewriterParseError
from tests.conftest import ScriptedLLM


@pytest.mark.asyncio
async def test_rewrite_returns_parsed_list():
    llm = ScriptedLLM(responses=[json.dumps(["alt 1", "alt 2", "alt 3"])])
    rewriter = QueryRewriter(model=llm)
    out = await rewriter.rewrite("original question", n=3)
    assert out == ["alt 1", "alt 2", "alt 3"]


@pytest.mark.asyncio
async def test_rewrite_truncates_to_n_when_model_overshoots():
    llm = ScriptedLLM(responses=[json.dumps(["a", "b", "c", "d", "e"])])
    rewriter = QueryRewriter(model=llm)
    out = await rewriter.rewrite("question", n=2)
    assert out == ["a", "b"]


@pytest.mark.asyncio
async def test_rewrite_strips_markdown_fences():
    fenced = "```json\n" + json.dumps(["x", "y"]) + "\n```"
    llm = ScriptedLLM(responses=[fenced])
    rewriter = QueryRewriter(model=llm)
    out = await rewriter.rewrite("q", n=2)
    assert out == ["x", "y"]


@pytest.mark.asyncio
async def test_rewrite_recovers_from_prose_around_array():
    llm = ScriptedLLM(responses=[
        'Sure, here are some rewrites: ["one", "two"] hope this helps!'
    ])
    rewriter = QueryRewriter(model=llm)
    out = await rewriter.rewrite("q", n=2)
    assert out == ["one", "two"]


@pytest.mark.asyncio
async def test_rewrite_retries_on_invalid_json():
    llm = ScriptedLLM(responses=[
        "I cannot rewrite that, sorry.",
        "Still not JSON.",
        json.dumps(["finally", "valid"]),
    ])
    rewriter = QueryRewriter(model=llm)
    out = await rewriter.rewrite("q", n=2)
    assert out == ["finally", "valid"]
    assert len(llm.calls) == 3


@pytest.mark.asyncio
async def test_rewrite_raises_after_max_attempts():
    llm = ScriptedLLM(responses=["no", "still no", "really nope"])
    rewriter = QueryRewriter(model=llm)
    with pytest.raises(QueryRewriterParseError):
        await rewriter.rewrite("q", n=3)
    assert len(llm.calls) == 3


@pytest.mark.asyncio
async def test_rewrite_filters_non_string_items_in_response():
    # Model returns a mixed-type array
    llm = ScriptedLLM(responses=[json.dumps(["ok", 42, None, "also ok"])])
    rewriter = QueryRewriter(model=llm)
    out = await rewriter.rewrite("q", n=3)
    assert out == ["ok", "also ok"]
