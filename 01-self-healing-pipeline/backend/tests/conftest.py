"""Test fixtures and fakes for the self-healing pipeline.

This conftest sets minimal env vars BEFORE importing Settings, so tests can run
without a real .env file or any real API keys / database. The fakes below stand
in for the production LLM and pgvector memory.
"""
import os

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-real")
os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://test:test@localhost/test")

from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import SystemMessage

from config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


class FakeMemory:
    """In-memory stand-in for EpisodicMemory.

    Records every save_error / record_run call so tests can assert on them.
    retrieve_similar_errors just returns the most-recent k errors — no embedding.
    """

    def __init__(self) -> None:
        self.errors: list[dict[str, Any]] = []
        self.runs: list[dict[str, Any]] = []

    async def save_error(
        self,
        *,
        document_type: str,
        error_type: str,
        principle: str,
        context: dict[str, Any],
        resolution: str | None = None,
    ) -> int:
        self.errors.append(
            {
                "document_type": document_type,
                "error_type": error_type,
                "principle": principle,
                "context": context,
                "resolution": resolution,
            }
        )
        return len(self.errors)

    async def retrieve_similar_errors(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        return self.errors[-k:]

    async def recent_errors(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.errors[-limit:]

    async def record_run(
        self,
        *,
        document_type: str,
        document_hash: str,
        final_score: float,
        retry_count: int,
        success: bool,
        final_output: dict[str, Any],
        errors_history: list[dict[str, Any]],
    ) -> None:
        self.runs.append(
            {
                "document_type": document_type,
                "document_hash": document_hash,
                "final_score": final_score,
                "retry_count": retry_count,
                "success": success,
                "final_output": final_output,
                "errors_history": errors_history,
            }
        )


class ScriptedLLM:
    """Returns pre-written responses based on which agent's prompt arrived.

    Distinguishes extractor from critic by inspecting the SystemMessage:
      - Critic prompt contains 'Critic agent'
      - Extractor prompt contains 'structured-data extraction'
    """

    def __init__(
        self,
        *,
        extractor_responses: list[str] | None = None,
        critic_responses: list[str] | None = None,
    ) -> None:
        self.extractor_responses = list(extractor_responses or [])
        self.critic_responses = list(critic_responses or [])
        self.calls: list[Any] = []

    async def ainvoke(self, messages: list[Any]) -> SimpleNamespace:
        self.calls.append(messages)
        sys_msg = next((m for m in messages if isinstance(m, SystemMessage)), None)
        sys_content = sys_msg.content if sys_msg else ""
        if "Critic agent" in sys_content:
            if not self.critic_responses:
                raise RuntimeError("ScriptedLLM: ran out of critic responses")
            return SimpleNamespace(content=self.critic_responses.pop(0))
        if not self.extractor_responses:
            raise RuntimeError("ScriptedLLM: ran out of extractor responses")
        return SimpleNamespace(content=self.extractor_responses.pop(0))
