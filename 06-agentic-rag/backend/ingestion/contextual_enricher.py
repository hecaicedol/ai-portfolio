import asyncio
from typing import Any
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


CONTEXTUAL_PROMPT = """Here is the document:
<document>
{full_doc}
</document>

Here is one chunk of that document:
<chunk>
{chunk}
</chunk>

Provide a 2–3 sentence context describing where this chunk fits in the document
and what makes it significant. Be concise, specific, and factual. Do NOT
summarize the chunk itself — describe its place in the document."""


class ContextualEnricher:
    """
    Implements Anthropic's contextual retrieval enrichment.

    For each chunk, asks Claude for a short context blurb derived from the full
    document. The blurb is prepended to the chunk *before* embedding, so the
    embedding captures both the chunk's content and its document-level role.

    Processed in batches of 10 with asyncio.gather. Tracks total tokens for
    cost reporting per document.
    """

    def __init__(self, *, model: str, api_key: str, batch_size: int = 10) -> None:
        self.model = model
        self.api_key = api_key
        self.batch_size = batch_size
        self.total_tokens = 0

    async def enrich(self, *, full_doc: str, chunks: list[ChunkInput]) -> list[EnrichedOutput]:
        results: list[EnrichedOutput] = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            batch_results = await asyncio.gather(*[self._enrich_one(full_doc, c) for c in batch])
            results.extend(batch_results)
        return results

    async def _enrich_one(self, full_doc: str, chunk: ChunkInput) -> EnrichedOutput:
        raise NotImplementedError
