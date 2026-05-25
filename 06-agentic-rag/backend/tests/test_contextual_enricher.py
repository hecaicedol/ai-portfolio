"""Tests for ingestion.contextual_enricher.

Uses the same ScriptedLLM pattern as the other tests — no real Anthropic
API call, no network, no money spent.
"""
from __future__ import annotations

import pytest
from types import SimpleNamespace

from ingestion.contextual_enricher import (
    ChunkInput,
    ContextualEnricher,
    EnrichedOutput,
)
from tests.conftest import ScriptedLLM


@pytest.mark.asyncio
async def test_enrich_one_prefixes_context_to_chunk():
    llm = ScriptedLLM(responses=[
        'This chunk introduces the RRF formula in section 2 of the paper.',
    ])
    enricher = ContextualEnricher(model=llm, batch_size=1)
    out = await enricher._enrich_one(
        full_doc='Full paper text…',
        chunk=ChunkInput(id='c1', content='score(d) = Σ 1/(k+rank)'),
    )
    assert isinstance(out, EnrichedOutput)
    assert out.id == 'c1'
    assert out.context_prefix.startswith('This chunk introduces')
    assert 'score(d) = Σ 1/(k+rank)' in out.enriched
    assert out.enriched.index('This chunk') < out.enriched.index('score(d)')
    assert out.tokens_used > 0
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_enrich_strips_whitespace_from_model_output():
    llm = ScriptedLLM(responses=['\n\n   The chunk explains BM25 weighting.   \n\n'])
    enricher = ContextualEnricher(model=llm)
    out = await enricher._enrich_one(
        full_doc='doc',
        chunk=ChunkInput(id='c2', content='BM25 = ...'),
    )
    assert out.context_prefix == 'The chunk explains BM25 weighting.'


@pytest.mark.asyncio
async def test_enrich_batches_via_asyncio_gather():
    """If batch_size is 3 and we hand it 7 chunks, the LLM should be called
    seven times in three batches (3 + 3 + 1)."""
    responses = [f'context {i}' for i in range(7)]
    llm = ScriptedLLM(responses=responses)
    enricher = ContextualEnricher(model=llm, batch_size=3)
    chunks = [ChunkInput(id=f'c{i}', content=f'chunk {i}') for i in range(7)]
    out = await enricher.enrich(full_doc='full', chunks=chunks)
    assert len(out) == 7
    assert all(out[i].id == f'c{i}' for i in range(7))
    assert len(llm.calls) == 7


@pytest.mark.asyncio
async def test_total_tokens_accumulates_across_calls():
    llm = ScriptedLLM(responses=['ctx a', 'ctx b'])
    enricher = ContextualEnricher(model=llm)
    await enricher.enrich(
        full_doc='doc',
        chunks=[
            ChunkInput(id='1', content='one'),
            ChunkInput(id='2', content='two'),
        ],
    )
    assert enricher.total_tokens > 0


@pytest.mark.asyncio
async def test_uses_real_usage_metadata_when_available():
    """If the model client returns response.usage_metadata, we should prefer
    it over the length-based estimate."""

    class MetadataLLM:
        async def ainvoke(self, messages):
            return SimpleNamespace(
                content='specific context',
                usage_metadata={'input_tokens': 100, 'output_tokens': 25},
            )

    enricher = ContextualEnricher(model=MetadataLLM())
    out = await enricher._enrich_one(
        full_doc='doc',
        chunk=ChunkInput(id='x', content='chunk'),
    )
    assert out.tokens_used == 125
