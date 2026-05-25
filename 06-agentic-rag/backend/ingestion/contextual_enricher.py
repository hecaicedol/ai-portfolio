"""Anthropic's Contextual Retrieval — chunk-context prefixing.

Paper: https://www.anthropic.com/news/contextual-retrieval

The idea: before embedding a chunk, prepend 2–3 sentences of context
explaining where that chunk fits in the original document. Anthropic's
internal evaluation shows ~49% reduction in retrieval failure rate vs
embedding raw chunks. The cost is one LLM call per chunk *at ingest
time* — amortized over the chunk's lifetime in the vector store.

Design notes for testability
- The LLM is injected (any object with `async ainvoke(messages)` →
  object with `.content`). Production wires `ChatAnthropic`; tests use
  the same `ScriptedLLM` pattern P1's agents use.
- JSON-retry is unnecessary here: the LLM is asked for plain text, not
  structured output. We just trim and strip.
- Batches of `batch_size` (default 10) use `asyncio.gather` so a 1000-
  chunk document parallelizes across the rate limit instead of serializing.
- `total_tokens` is accumulated only if the model client exposes usage
  metadata (langchain-anthropic does via `response.usage_metadata`); we
  fall back to a length heuristic so the counter is never zero.
"""
from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel


class ChunkInput(BaseModel):
    id: str
    content: str


class EnrichedOutput(BaseModel):
    id: str
    original: str
    context_prefix: str
    enriched: str
    tokens_used: int


SYSTEM_PROMPT = """You write short context blurbs (2–3 sentences) that explain
where a chunk fits inside a larger document. The blurb is prepended to the
chunk before embedding, so the embedding captures both the local content and
the chunk's document-level role.

Rules:
- Be concise, specific, factual. No filler.
- Do NOT summarize the chunk itself — describe its *place* in the document.
- Plain text only, no markdown, no headings, no JSON.
"""

USER_TEMPLATE = """<document>
{full_doc}
</document>

<chunk>
{chunk}
</chunk>

Write a 2–3 sentence context blurb describing where this chunk sits in the
document and what makes it significant."""


def _estimate_tokens(text: str) -> int:
    """Cheap fallback: ~4 chars per token. Used only when the model client
    doesn't expose real usage metadata."""
    return max(1, len(text) // 4)


class ContextualEnricher:
    """Build context prefixes for every chunk in parallel batches.

    Parameters
    ----------
    model
        Anything with `async ainvoke(messages) -> obj_with_.content`.
    batch_size
        How many chunks to enrich concurrently per gather() call.
    """

    def __init__(self, *, model: Any, batch_size: int = 10) -> None:
        self.model = model
        self.batch_size = batch_size
        self.total_tokens = 0

    async def enrich(
        self,
        *,
        full_doc: str,
        chunks: list[ChunkInput],
    ) -> list[EnrichedOutput]:
        results: list[EnrichedOutput] = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            batch_results = await asyncio.gather(
                *[self._enrich_one(full_doc, c) for c in batch]
            )
            results.extend(batch_results)
        return results

    async def _enrich_one(
        self,
        full_doc: str,
        chunk: ChunkInput,
    ) -> EnrichedOutput:
        user_message = USER_TEMPLATE.format(full_doc=full_doc, chunk=chunk.content)
        response = await self.model.ainvoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_message),
            ]
        )
        context_prefix = (response.content or "").strip()
        enriched = f"{context_prefix}\n\n{chunk.content}".strip()

        # Try real usage metadata first; fall back to a length heuristic.
        usage = getattr(response, "usage_metadata", None) or {}
        in_tok = usage.get("input_tokens") or 0
        out_tok = usage.get("output_tokens") or 0
        tokens = (in_tok + out_tok) if (in_tok or out_tok) else (
            _estimate_tokens(user_message) + _estimate_tokens(context_prefix)
        )
        self.total_tokens += tokens

        return EnrichedOutput(
            id=chunk.id,
            original=chunk.content,
            context_prefix=context_prefix,
            enriched=enriched,
            tokens_used=tokens,
        )
