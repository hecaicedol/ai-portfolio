"""Test fixtures and fakes for the P6 (Agentic RAG) backend.

Provides:
  - `fake_embed` / `async_fake_embed` — deterministic hash-based embeddings
    so similar text produces similar vectors without needing Voyage AI.
  - `ScriptedLLM` — same pattern as P1: returns pre-recorded responses
    based on the system prompt content.
  - `sample_chunks` — a small synthetic corpus.
"""
from __future__ import annotations

import hashlib
import math
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import SystemMessage

from stores.base_store import EnrichedChunk


# ── deterministic embeddings ────────────────────────────────────────────────

EMBED_DIM = 64


def fake_embed(text: str, dim: int = EMBED_DIM) -> list[float]:
    """Hash-based bag-of-words embedding. Same text → same vector. Texts with
    overlapping vocabulary → high cosine similarity. Useful for tests that
    care about retrieval *behavior* without paying for real embeddings."""
    vec = [0.0] * dim
    for token in text.lower().split():
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


async def async_fake_embed(text: str) -> list[float]:
    """Async wrapper used by HybridSearcher's `embed` injection point."""
    return fake_embed(text)


# ── scripted LLM ────────────────────────────────────────────────────────────

class ScriptedLLM:
    """LLM stand-in: returns prewritten responses in order. Used by tests of
    QueryRewriter."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[Any] = []

    async def ainvoke(self, messages: list[Any]) -> SimpleNamespace:
        self.calls.append(messages)
        if not self.responses:
            raise RuntimeError("ScriptedLLM: ran out of scripted responses")
        return SimpleNamespace(content=self.responses.pop(0))


# ── sample corpus ───────────────────────────────────────────────────────────

@pytest.fixture
def sample_chunks() -> list[EnrichedChunk]:
    """5 short docs covering distinct topics — enough to test that retrieval
    pulls the relevant one for a given query, not all of them."""
    raw = [
        ("ml-1", "The transformer architecture introduced self-attention for sequence modeling tasks."),
        ("ml-2", "Reciprocal rank fusion combines multiple ranked retrieval lists into a single ordering."),
        ("py-1", "Python decorators wrap a function to modify its behavior without changing its source."),
        ("py-2", "Asyncio uses an event loop to schedule coroutines cooperatively in a single thread."),
        ("infra-1", "PostgreSQL with the pgvector extension stores embeddings and supports cosine and L2 distance."),
    ]
    return [
        EnrichedChunk(
            id=cid,
            content=content,
            enriched_content=content,
            embedding=fake_embed(content),
            metadata={"topic": cid.split("-")[0]},
        )
        for cid, content in raw
    ]
