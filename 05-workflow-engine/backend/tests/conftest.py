"""Test fixtures for P5.

Same fakes-only contract as P1/P3/P4: deterministic hash-bag embed,
ScriptedLLM that pops responses in order (or branches by which prompt
arrived), and a FakeMCPClient with per-action canned outputs.
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
    """Pops planner/replanner responses in order. Optionally branches by
    which system prompt arrived so a single instance can drive both."""

    def __init__(
        self,
        *,
        planner_responses: list[str | dict] | None = None,
        replanner_responses: list[str | dict] | None = None,
        responses: list[str | dict] | None = None,
    ) -> None:
        self.planner_responses = list(planner_responses or [])
        self.replanner_responses = list(replanner_responses or [])
        self.responses = list(responses or [])
        self.calls: list[tuple[str, str]] = []

    async def ainvoke(self, messages: list[Any]) -> SimpleNamespace:
        sys = _content(messages[0]) if messages else ""
        user = _content(messages[-1]) if messages else ""
        if "Workflow Planner" in sys and self.planner_responses:
            self.calls.append(("planner", user))
            return SimpleNamespace(content=_serialize(self.planner_responses.pop(0)))
        if "Workflow Replanner" in sys and self.replanner_responses:
            self.calls.append(("replanner", user))
            return SimpleNamespace(content=_serialize(self.replanner_responses.pop(0)))
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


class FakeMCPClient:
    """In-memory MCP client. Maps action names to canned outputs (or to
    callables that compute output from params). Raises whatever you put
    in `errors[action]` instead — useful for triggering replans.
    """

    def __init__(
        self,
        *,
        tool: str,
        outputs: dict[str, Any] | None = None,
        errors: dict[str, Exception] | None = None,
    ) -> None:
        self.tool = tool
        self.outputs = dict(outputs or {})
        self.errors = dict(errors or {})
        self.calls: list[tuple[str, dict]] = []

    async def call(self, action: str, params: dict) -> Any:
        self.calls.append((action, dict(params)))
        if action in self.errors:
            raise self.errors[action]
        if action not in self.outputs:
            raise KeyError(f"FakeMCPClient[{self.tool}]: no output canned for action {action!r}")
        out = self.outputs[action]
        if callable(out):
            return out(params)
        return out
