"""Test fixtures for P4: ScriptedDebateLLM that branches by role.

No Redis, no Anthropic, no network — same fakes-only contract used in
P1, P3, and P6.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any


def _system_content(messages: list[Any]) -> str:
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "system":
            return str(m.get("content", ""))
        if hasattr(m, "type") and getattr(m, "type", "") == "system":
            return str(getattr(m, "content", ""))
        if hasattr(m, "content") and m.__class__.__name__ == "SystemMessage":
            return str(m.content)
    return ""


def _user_content(messages: list[Any]) -> str:
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "user":
            return str(m.get("content", ""))
        if hasattr(m, "type") and getattr(m, "type", "") == "human":
            return str(getattr(m, "content", ""))
        if hasattr(m, "content") and m.__class__.__name__ == "HumanMessage":
            return str(m.content)
    return ""


class ScriptedDebateLLM:
    """A single LLM that fans out scripted replies by which agent's system
    prompt arrives. Each `*_responses` list is consumed in order — one
    entry per round per agent.

    `synth_response` is consumed once when the synthesizer asks.
    """

    def __init__(
        self,
        *,
        responses_by_role: dict[str, list[str | dict[str, Any]]] | None = None,
        synth_response: str | dict[str, Any] | None = None,
    ) -> None:
        self.responses_by_role = {
            role: list(replies) for role, replies in (responses_by_role or {}).items()
        }
        self.synth_response = synth_response
        self.calls: list[tuple[str, str]] = []

    async def ainvoke(self, messages: list[Any]) -> SimpleNamespace:
        sys = _system_content(messages)
        user = _user_content(messages)
        if "Synthesizer" in sys:
            if self.synth_response is None:
                raise RuntimeError("ScriptedDebateLLM: no synthesizer response queued")
            self.calls.append(("synthesizer", user))
            return SimpleNamespace(content=_serialize(self.synth_response))

        role = _role_of(sys)
        queue = self.responses_by_role.get(role, [])
        if not queue:
            raise RuntimeError(f"ScriptedDebateLLM: no response queued for role={role!r}")
        self.calls.append((role, user))
        return SimpleNamespace(content=_serialize(queue.pop(0)))


def _serialize(payload: str | dict[str, Any]) -> str:
    if isinstance(payload, str):
        return payload
    return json.dumps(payload)


def _role_of(system_prompt: str) -> str:
    """Map a system prompt back to the role_name. Match the unique
    'You are the X' opener each role uses. Order matters because Risk's
    prompt contains the literal 'non-financial' substring."""
    sp = system_prompt.lower()
    if "you are the optimist" in sp:
        return "optimist"
    if "you are the skeptic" in sp:
        return "skeptic"
    if "you are the financial agent" in sp:
        return "financial"
    if "you are the risk agent" in sp:
        return "risk"
    if "you are the devil" in sp:
        return "devils_advocate"
    return "unknown"
