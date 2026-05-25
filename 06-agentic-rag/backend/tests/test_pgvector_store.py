"""Tests for PgVectorStore — no real Postgres required.

We inject a fake pool whose `.connection()` returns an async-context-manager
wrapping a fake connection + cursor. Every SQL call is recorded and the
cursor returns scripted rows.

This verifies the SQL contract (correct parameterization, ORDER BY, etc.)
without standing up a database. A separate integration-test module under
pytest.mark.integration would exercise the real Postgres path.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from stores.base_store import EnrichedChunk
from stores.pgvector_store import PgVectorStore


class FakeCursor:
    def __init__(self, scripted_rows=None):
        self.scripted = list(scripted_rows or [])
        self.executed: list[tuple] = []

    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False

    async def execute(self, sql, params=None):
        self.executed.append((sql, params))

    async def fetchall(self):
        return self.scripted.pop(0) if self.scripted else []

    async def fetchone(self):
        return self.scripted.pop(0) if self.scripted else None


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor

    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False

    def cursor(self): return self._cursor


class FakePool:
    def __init__(self, cursor):
        self._cursor = cursor
        self.opened = False

    def open(self):
        self.opened = True

    def connection(self):
        return FakeConnection(self._cursor)


def _make_store(scripted_rows=None):
    cursor = FakeCursor(scripted_rows=scripted_rows or [])
    pool = FakePool(cursor)
    store = PgVectorStore(dsn='postgresql://test', pool_factory=lambda dsn: pool)
    return store, pool, cursor


# ── Tests ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_index_documents_inserts_each_and_records_latency():
    store, pool, cur = _make_store()
    docs = [
        EnrichedChunk(id='a', content='hello world', enriched_content='ctx · hello world',
                      embedding=[0.1] * 1024, metadata={'topic': 'greeting'}),
        EnrichedChunk(id='b', content='goodbye',     enriched_content='ctx · goodbye',
                      embedding=[0.2] * 1024, metadata={}),
    ]
    res = await store.index_documents(docs)
    assert res.indexed == 2
    assert res.failed == 0
    assert res.latency_ms >= 0
    # Two INSERT calls, plus none for the missing embedding case
    insert_calls = [c for c in cur.executed if 'INSERT INTO chunks' in c[0]]
    assert len(insert_calls) == 2
    # The enriched_content (not the raw chunk) is what gets embedded — this
    # matches the Anthropic contextual-retrieval pattern.
    assert insert_calls[0][1][1] == 'ctx · hello world'


@pytest.mark.asyncio
async def test_index_documents_skips_chunks_without_embedding():
    store, pool, cur = _make_store()
    docs = [
        EnrichedChunk(id='ok', content='c', enriched_content='c', embedding=[0.1] * 1024),
        EnrichedChunk(id='no', content='c', enriched_content='c', embedding=None),
    ]
    res = await store.index_documents(docs)
    assert res.indexed == 1
    assert res.failed == 1


@pytest.mark.asyncio
async def test_similarity_search_runs_cosine_order_query():
    rows = [
        ('a', 'about cats', {'topic': 'animal'}, 0.91),
        ('b', 'about dogs', {'topic': 'animal'}, 0.84),
    ]
    store, pool, cur = _make_store(scripted_rows=[rows])
    results = await store.similarity_search(query_embedding=[0.1] * 1024, k=2)
    assert len(results) == 2
    assert results[0].chunk_id == 'a'
    assert results[0].score == pytest.approx(0.91)
    # SQL order: vector parameter appears twice (SELECT and ORDER BY) + limit
    sql, params = cur.executed[-1]
    assert 'embedding <=> %s::vector' in sql
    assert params[-1] == 2  # limit


@pytest.mark.asyncio
async def test_similarity_search_with_metadata_filter_builds_where_clause():
    store, pool, cur = _make_store(scripted_rows=[[]])
    await store.similarity_search(
        query_embedding=[0.1] * 1024, k=5,
        filters={'topic': 'animal'},
    )
    sql, params = cur.executed[-1]
    assert 'WHERE' in sql
    assert "metadata->>%s = %s" in sql
    # params: (q_vec, 'topic', 'animal', q_vec, k)
    assert params[1] == 'topic'
    assert params[2] == 'animal'
    assert params[-1] == 5


@pytest.mark.asyncio
async def test_keyword_search_uses_tsvector_with_plainto_tsquery():
    rows = [('x', 'pgvector ivfflat setup', {}, 0.55)]
    store, pool, cur = _make_store(scripted_rows=[rows])
    out = await store.keyword_search(query='pgvector setup', k=3)
    assert len(out) == 1
    assert out[0].chunk_id == 'x'
    sql, params = cur.executed[-1]
    assert 'plainto_tsquery' in sql
    assert 'ts_rank_cd' in sql
    # Both placeholders fill with the same query string
    assert params[0] == 'pgvector setup'
    assert params[1] == 'pgvector setup'
    assert params[2] == 3


@pytest.mark.asyncio
async def test_get_stats_reads_row_count_and_index_size():
    store, pool, cur = _make_store(scripted_rows=[(1042, 18.7)])
    # fetchone is used here, not fetchall — but our FakeCursor falls through
    # to scripted[0] either way. Force the path explicitly:
    cur.scripted = [(1042, 18.7)]
    # Need fetchone to return the single row
    async def fetchone_impl(): return (1042, 18.7)
    cur.fetchone = fetchone_impl
    stats = await store.get_stats()
    assert stats.doc_count == 1042
    assert stats.index_size_mb == pytest.approx(18.7)


@pytest.mark.asyncio
async def test_ensure_schema_runs_create_extension_and_indexes():
    store, pool, cur = _make_store()
    await store.ensure_schema()
    schema_call = cur.executed[-1][0]
    assert 'CREATE EXTENSION IF NOT EXISTS vector' in schema_call
    assert 'CREATE TABLE IF NOT EXISTS chunks' in schema_call
    assert 'ivfflat' in schema_call
    assert 'tsv' in schema_call and 'tsvector' in schema_call
