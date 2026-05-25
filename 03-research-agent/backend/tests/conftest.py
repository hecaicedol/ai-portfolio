"""Test fixtures for P3: deterministic embed, ScriptedLLM, fake tools.

Mirrors the patterns we used in P6 — no real services, no money spent.
"""
from __future__ import annotations

import hashlib
import math
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import SystemMessage


# ── deterministic embed (same hash-bag trick as P6 / P1) ───────────────

async def fake_embed(text: str, dim: int = 256) -> list[float]:
    vec = [0.0] * dim
    for token in text.lower().split():
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


# ── scripted LLM ───────────────────────────────────────────────────────

class ScriptedLLM:
    """Returns prewritten responses in order, optionally branching by
    which system prompt arrived (so tests can give different replies to
    the planner vs the reflector)."""

    def __init__(
        self,
        *,
        plan_responses: list[str] | None = None,
        reflect_responses: list[str] | None = None,
        responses: list[str] | None = None,
    ) -> None:
        self.plan_responses = list(plan_responses or [])
        self.reflect_responses = list(reflect_responses or [])
        self.responses = list(responses or [])
        self.calls: list[Any] = []

    async def ainvoke(self, messages: list[Any]) -> SimpleNamespace:
        self.calls.append(messages)
        sys = next((m for m in messages if isinstance(m, SystemMessage)), None)
        sys_content = sys.content if sys else ""
        if "research-planning agent" in sys_content and self.plan_responses:
            return SimpleNamespace(content=self.plan_responses.pop(0))
        if "research-consolidation agent" in sys_content and self.reflect_responses:
            return SimpleNamespace(content=self.reflect_responses.pop(0))
        if not self.responses:
            raise RuntimeError("ScriptedLLM: ran out of responses")
        return SimpleNamespace(content=self.responses.pop(0))


# ── fake tools ─────────────────────────────────────────────────────────

class FakeWebSearch:
    """In-memory web search. Each call returns the same canned list,
    regardless of query."""

    def __init__(self, results: list[Any] | None = None) -> None:
        self.results = results or []
        self.calls: list[tuple[str, int]] = []

    async def search(self, query: str, *, k: int = 5):
        self.calls.append((query, k))
        return self.results[:k]


class FakeArxivSearch:
    def __init__(self, results: list[Any] | None = None) -> None:
        self.results = results or []
        self.calls: list[tuple[str, int]] = []

    async def search(self, query: str, *, max_results: int = 10):
        self.calls.append((query, max_results))
        return self.results[:max_results]


@pytest.fixture
def tmp_reports(tmp_path):
    """Pytest tmp_path subdirectory for generated reports."""
    return tmp_path / "reports"
