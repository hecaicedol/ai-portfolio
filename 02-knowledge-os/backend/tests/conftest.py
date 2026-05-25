"""Test fixtures for P2.

Same fakes-only contract as the rest of the portfolio: deterministic
hash-bag embed + ScriptedLLM that branches by which system prompt
arrived. No Neo4j, no Anthropic, no network.
"""
from __future__ import annotations

import hashlib
import json
import math
from types import SimpleNamespace
from typing import Any


async def fake_embed(text: str, dim: int = 256) -> list[float]:
    vec = [0.0] * dim
    for token in text.lower().split():
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


class ScriptedLLM:
    """Branches by which system prompt arrived so a single instance can
    drive the EntityExtractor AND the GraphRAG synthesizer in the same
    end-to-end test."""

    def __init__(
        self,
        *,
        extractor_responses: list[str | dict] | None = None,
        synth_responses: list[str | dict] | None = None,
        responses: list[str | dict] | None = None,
    ) -> None:
        self.extractor_responses = list(extractor_responses or [])
        self.synth_responses = list(synth_responses or [])
        self.responses = list(responses or [])
        self.calls: list[tuple[str, str]] = []

    async def ainvoke(self, messages: list[Any]) -> SimpleNamespace:
        # Collapse whitespace so substrings split across newlines still match.
        sys = " ".join(_content(messages[0]).split()) if messages else ""
        user = _content(messages[-1]) if messages else ""
        if "extract structured knowledge" in sys and self.extractor_responses:
            self.calls.append(("extractor", user))
            return SimpleNamespace(content=_serialize(self.extractor_responses.pop(0)))
        if "knowledge-graph context" in sys and self.synth_responses:
            self.calls.append(("synth", user))
            return SimpleNamespace(content=_serialize(self.synth_responses.pop(0)))
        if not self.responses:
            raise RuntimeError("ScriptedLLM: no response queued")
        self.calls.append(("generic", user))
        return SimpleNamespace(content=_serialize(self.responses.pop(0)))


def _serialize(payload: str | dict) -> str:
    if isinstance(payload, str):
        return payload
    return json.dumps(payload)


def _content(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("content", ""))
    return str(getattr(message, "content", ""))
