CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS episodic_sessions (
    session_id    TEXT PRIMARY KEY,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    goal          TEXT NOT NULL,
    summary       TEXT,
    key_findings  JSONB,
    embedding     vector(1024)
);
CREATE INDEX IF NOT EXISTS episodic_sessions_embedding_idx
    ON episodic_sessions USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE TABLE IF NOT EXISTS episodic_archive (
    id            BIGSERIAL PRIMARY KEY,
    session_id    TEXT NOT NULL REFERENCES episodic_sessions(session_id) ON DELETE CASCADE,
    archived_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    kind          TEXT NOT NULL,
    content       TEXT NOT NULL,
    embedding     vector(1024)
);
CREATE INDEX IF NOT EXISTS episodic_archive_embedding_idx
    ON episodic_archive USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE TABLE IF NOT EXISTS semantic_facts (
    id            BIGSERIAL PRIMARY KEY,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    fact          TEXT NOT NULL,
    source        TEXT,
    confidence    NUMERIC(4,3) NOT NULL DEFAULT 0.5,
    embedding     vector(1024) NOT NULL,
    superseded_by BIGINT
);
CREATE INDEX IF NOT EXISTS semantic_facts_embedding_idx
    ON semantic_facts USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
