CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS episodic_errors (
    id              BIGSERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    document_type   TEXT NOT NULL,
    error_type      TEXT NOT NULL,
    principle       TEXT NOT NULL,
    context         JSONB NOT NULL,
    resolution      TEXT,
    embedding       vector(1024) NOT NULL
);

CREATE INDEX IF NOT EXISTS episodic_errors_doctype_idx
    ON episodic_errors (document_type);

CREATE INDEX IF NOT EXISTS episodic_errors_embedding_idx
    ON episodic_errors USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id               BIGSERIAL PRIMARY KEY,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    document_type    TEXT NOT NULL,
    document_hash    TEXT NOT NULL,
    final_score      NUMERIC(4,3),
    retry_count      INT NOT NULL DEFAULT 0,
    success          BOOLEAN NOT NULL,
    final_output     JSONB,
    errors_history   JSONB
);

CREATE INDEX IF NOT EXISTS pipeline_runs_doctype_idx
    ON pipeline_runs (document_type, created_at DESC);
