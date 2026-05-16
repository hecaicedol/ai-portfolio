# Agentic RAG with Vector DB Benchmark and Auto-Optimization

> Production-grade RAG that indexes the same corpus into pgvector, Qdrant, and Pinecone in parallel — runs hybrid retrieval with contextual enrichment and reranking — and includes an evaluator agent that auto-tunes parameters when quality drops.

[![Status](https://img.shields.io/badge/status-scaffolded-blueviolet)]()
[![Python](https://img.shields.io/badge/python-3.12-blue)]()
[![Pattern](https://img.shields.io/badge/pattern-agentic--rag-7c5cff)]()
[![Eval](https://img.shields.io/badge/eval-Ragas-4dd0c2)]()

---

## The problem

Most RAG portfolio projects answer "can you call a vector DB?" — yes, of course you can, anyone can. The questions production teams actually ask are:

1. *Which* vector DB should we use for our workload?
2. How do we measure that our retrieval is good, with numbers we can defend in a review?
3. What do we do when quality silently degrades after a model swap or a corpus refresh?

This project answers all three with a single system: identical corpus indexed into three stores, identical retrieval pipeline against each, Ragas evaluation on every query, and an optimizer agent that auto-tunes parameters when scores drift.

## The thesis

> *Choosing infrastructure with data, not vibes, is the engineer behavior senior AI teams are screening for. The auto-optimizer demonstrates production thinking.*

---

## Architecture

```
                       ┌──────────────────────────────┐
                       │       Document Ingestion     │
                       │  · semantic chunking         │
                       │  · contextual enrichment     │  (Anthropic technique)
                       └──────────────┬───────────────┘
                                      │
                ┌─────────────────────┼─────────────────────┐
                ▼                     ▼                     ▼
          ┌─────────┐           ┌─────────┐            ┌──────────┐
          │ pgvec   │           │ Qdrant  │            │ Pinecone │
          └────┬────┘           └────┬────┘            └────┬─────┘
               │                     │                      │
               └─────────────────────┴──────────────────────┘
                                     │
                       ┌─────────────▼─────────────┐
                       │     Query pipeline        │
                       │ 1. Query rewriting        │
                       │ 2. BM25 + vector search   │
                       │ 3. RRF fusion             │
                       │ 4. Cohere reranking       │
                       │ 5. Claude synthesis       │
                       └─────────────┬─────────────┘
                                     │
                       ┌─────────────▼─────────────┐
                       │    Evaluator Agent (BG)   │
                       │ Ragas: faithfulness /     │
                       │   answer_relevancy /      │
                       │   context_recall          │
                       └─────────────┬─────────────┘
                                     │ if score drops
                                     ▼
                       ┌───────────────────────────┐
                       │   Optimizer Agent         │
                       │ adjusts k, threshold,     │
                       │ reranking weight          │
                       └───────────────────────────┘
```

### Why this design

| Decision | Alternative | Why we picked this |
|---|---|---|
| **Three vector DBs in parallel** | Pick one upfront | Production teams DON'T know which is best for their workload. Benchmarking with the same corpus + same pipeline isolates the store as the variable. |
| **Anthropic contextual retrieval** | Plain chunks | The Anthropic paper reports a 35–49% reduction in retrieval failures. The cost is one Claude call per chunk *at ingest time*, which is amortized across thousands of queries. |
| **RRF fusion of BM25 + vector** | Vector-only | Reciprocal Rank Fusion is the documented industry default for hybrid retrieval. BM25 catches exact-keyword matches that vector embeddings miss (acronyms, product codes). |
| **Cohere Rerank** | Embedding similarity only | Reranking with a cross-encoder consistently beats embedding similarity on top-K precision. Cheap (~$0.001/query) and high ROI. |
| **Auto-optimizer with per-metric remediation** | Single learning-rate optimizer | Different metric drops have different causes: low faithfulness wants *less* context, low recall wants *more*. Per-metric remediation encodes this domain knowledge. |

---

## Tech stack

| Layer | Choice |
|---|---|
| Vector DBs | `pgvector` 0.7+, Qdrant 1.12, Pinecone (serverless) |
| RAG framework | LlamaIndex |
| Embeddings | Voyage AI `voyage-3` |
| Reranking | Cohere Rerank v3 |
| Evaluation | Ragas |
| LLM | Claude `claude-sonnet-4-5` |
| API | FastAPI |
| Frontend | Next.js + recharts (live benchmark dashboard) |

---

## Repository layout

```
06-agentic-rag/
├── docker-compose.yml
├── .env.example
├── README.md
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── api/main.py
│   ├── ingestion/
│   │   ├── chunker.py
│   │   ├── contextual_enricher.py
│   │   └── indexer.py
│   ├── stores/
│   │   ├── base_store.py
│   │   ├── pgvector_store.py
│   │   ├── qdrant_store.py
│   │   └── pinecone_store.py
│   ├── retrieval/
│   │   ├── query_rewriter.py
│   │   ├── hybrid_search.py
│   │   ├── reranker.py
│   │   └── pipeline.py
│   └── evaluation/
│       ├── evaluator_agent.py
│       ├── optimizer_agent.py
│       └── metrics_store.py
└── frontend/
```

---

## Quick start

```bash
cp .env.example .env
# Set ANTHROPIC_API_KEY, VOYAGE_API_KEY, COHERE_API_KEY
# Optional: PINECONE_API_KEY (skip if you only want to benchmark pgvector vs Qdrant)

docker compose up --build
```

### Ingest a corpus

```bash
curl -X POST http://localhost:8000/api/ingest \
  -F "file=@./samples/whitepaper.pdf"
```

This runs chunking + contextual enrichment, then indexes into all three stores in parallel.

### Query

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the recommended hybrid fusion strategy?"}'
```

Response includes side-by-side results from each store, with per-store Ragas scores.

---

## Build order

1. `backend/stores/base_store.py` + `pgvector_store.py` (start with one)
2. `backend/ingestion/contextual_enricher.py` (the differentiator)
3. `backend/retrieval/hybrid_search.py` + `reranker.py`
4. `backend/stores/qdrant_store.py` + `pinecone_store.py`
5. `backend/evaluation/{evaluator,optimizer}_agent.py`
6. `frontend/` — recharts dashboard

---

## Metrics to track

| Metric | pgvector | Qdrant | Pinecone |
|---|---|---|---|
| Index latency (per 10K chunks) | _TODO_ | _TODO_ | _TODO_ |
| Query latency p50 / p95 | _TODO_ | _TODO_ | _TODO_ |
| Faithfulness (Ragas) | _TODO_ | _TODO_ | _TODO_ |
| Context recall (Ragas) | _TODO_ | _TODO_ | _TODO_ |
| Estimated $ / 1000 queries | _TODO_ | _TODO_ | _TODO_ |
| Auto-optimizer interventions (30d) | _TODO_ | _TODO_ | _TODO_ |

---

## License

MIT.
