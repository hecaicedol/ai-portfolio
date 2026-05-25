# Autonomous Research Agent — MemGPT 3-Tier Memory

> An agent that researches complex topics with **persistent memory across sessions**, implementing the MemGPT pattern of explicit context management.

[![Status](https://img.shields.io/badge/status-core%20memory%20verified-2ec27e)]()
[![Tests](https://img.shields.io/badge/tests-31%20passing-2ec27e)]()
[![Python](https://img.shields.io/badge/python-3.12-blue)]()
[![Memory](https://img.shields.io/badge/memory-MemGPT-ff7a59)]()

---

## The problem

LLMs forget. When a research task spans days or weeks, "stuff it all in the context window" stops working — even at 200K tokens, important details get crowded out by the latest reading. Production agents need a memory architecture as deliberate as a database schema.

This project implements **MemGPT / Letta-style explicit memory management** — three tiers, each with a different lifecycle, and a controller that decides what lives where.

## The thesis

> *Memory architecture is one of the hardest, most-asked-about properties in AI engineering interviews. Implementing it from scratch — not just calling `RetrievalMemory()` — demonstrates depth.*

---

## Architecture — three memory tiers

```
┌──────────────────────────────────────────────────────────────────┐
│                       WORKING MEMORY                             │
│ Held in the LLM context window — bounded by max_tokens budget.   │
│  · Current session goal                                          │
│  · Active research plan                                          │
│  · Last N retrieved snippets                                     │
│  · System prompt + tools                                         │
└────────────────────────────────────────────────────────┬─────────┘
                                                         │ archive
                                                         ▼  oldest
┌──────────────────────────────────────────────────────────────────┐
│                      EPISODIC MEMORY                             │
│ PostgreSQL — one row per session, plus archived working entries. │
│  · Session id, timestamp, goal, summary, key findings            │
│  · Retrieved on new-session start (similarity to new goal)       │
└────────────────────────────────────────────────────────┬─────────┘
                                                         │ consolidate
                                                         ▼  on session end
┌──────────────────────────────────────────────────────────────────┐
│                      SEMANTIC MEMORY                             │
│ pgvector — durable facts extracted across all sessions.          │
│  · Fact, source, confidence, embedding                           │
│  · Retrieved any time, deduped against existing facts            │
└──────────────────────────────────────────────────────────────────┘

                ┌─────────────────────────────────┐
                │      MemGPT CONTROLLER          │
                │  Decides what enters working    │
                │  memory each turn. Moves info   │
                │  between tiers automatically.   │
                └─────────────────────────────────┘
```

### Why this design

| Decision | Alternative | Why we picked this |
|---|---|---|
| **Three explicit tiers** | Single vector store + RAG | Vector RAG is a *retrieval* mechanism, not a memory model. A session's plan ≠ a permanent fact ≠ "what happened last week." Treating them the same loses information about *when* something should be recalled. |
| **Token-budget working memory** | "Just fit it all in 200K" | 200K isn't free — cost and latency scale linearly. A bounded budget forces the controller to make decisions a real production agent would have to make. |
| **Consolidation step** | Save every turn to semantic memory | Naive saves create duplicated, low-quality facts. Consolidation at session end extracts only the durable claims, with provenance. |

---

## Tech stack

| Layer | Choice |
|---|---|
| Agent | LangGraph |
| LLM | Claude `claude-sonnet-4-5` |
| Working memory | Python dataclass with token budget enforcement |
| Episodic memory | PostgreSQL |
| Semantic memory | pgvector |
| Tools | Tavily (web), arXiv API, ReportLab (PDF reports) |
| API | FastAPI + SSE |
| Frontend | Next.js (research view + memory viewer + reports library) |

---

## Repository layout

```
03-research-agent/
├── docker-compose.yml
├── .env.example
├── README.md
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── api/main.py
│   ├── agent/
│   │   ├── research_agent.py     # LangGraph StateGraph
│   │   ├── planner.py
│   │   ├── executor.py
│   │   └── reflector.py
│   ├── memory/
│   │   ├── working_memory.py
│   │   ├── episodic_memory.py
│   │   ├── semantic_memory.py
│   │   └── memgpt_controller.py
│   ├── tools/
│   │   ├── web_search.py
│   │   ├── arxiv_search.py
│   │   └── report_generator.py
│   └── db/init.sql
└── frontend/
```

---

## Quick start

```bash
cp .env.example .env
docker compose up --build
```

### Start a research session

```bash
curl -X POST http://localhost:8000/api/research \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the production trade-offs between MemGPT and naive context stuffing as of 2026?"}'
```

The agent will plan, search, analyze, reflect, and generate a PDF report. Subsequent sessions on related topics retrieve from episodic + semantic memory automatically.

---

## Status — Slice 1 (core memory + graph verified, no API keys spent)

**Built and tested without external services:**

| Component | Status | Tests |
|---|---|---|
| `WorkingMemory` — token-budget FIFO | ✅ | 4 |
| `InMemoryEpisodic` — session save + cosine-similarity recall + archive | ✅ | 4 |
| `InMemorySemantic` — fact upsert with cosine dedupe + source union | ✅ | 6 |
| `MemGPTController` — auto-archive on eviction, full-context assembly, consolidation | ✅ | 5 |
| `WebSearchTool` + `ArxivSearchTool` — httpx injection (no network) | ✅ | 8 |
| `ReportGenerator` — Pydantic validation + lazy ReportLab + text fallback | ✅ | 2 |
| `ResearchAgent` (LangGraph: planner → executor → reflector → reporter) — `ScriptedLLM` drives the state machine end-to-end | ✅ | 4 |
| `PostgresEpisodic` / `PostgresSemantic` — pgvector backends (stubs) | ⏳ | — |
| Frontend (Next.js memory viewer + research view + reports library) | ⏳ | — |

**31 tests passing in 1.02s.** Run them with `cd backend && .venv/Scripts/python -m pytest tests -q`.

### Design notes uncovered while building

- **LangGraph nodes must NOT collide with state keys.** Naive naming (`plan`, `report`) crashes — nodes are renamed to verbs (`planner`, `reporter`). Same pattern surfaced in P1 (`critic` → `critique`).
- **MemGPT controller is the keystone**, not the memory tiers. Tiers are pure data structures; the controller decides what enters working memory each turn and runs consolidation at session end. Testing the controller is what proves the *architecture* works, not just the storage.
- **Cosine-similarity dedupe at insertion time** (threshold 0.92) keeps semantic memory clean. New facts that paraphrase existing ones union their sources and take the max confidence, rather than creating duplicate rows.

### Pending (Slice 2)

1. Wire `PostgresEpisodic` / `PostgresSemantic` to real Postgres + pgvector (docker-compose already declares the service).
2. Frontend views (memory viewer is the recruiter-facing differentiator).
3. Real research session against live Tavily + arXiv + Claude (needs API keys — out of scope for the no-budget run).

---

## Metrics to track

| Metric | Stateless baseline | With 3-tier memory |
|---|---|---|
| Question quality on multi-session topics | _TODO_ | _TODO_ |
| Tokens per session (avg) | _TODO_ | _TODO_ |
| % of new questions answered from semantic memory alone | n/a | _TODO_ |
| Time to produce a 5-page report | _TODO_ | _TODO_ |

---

## License

MIT.
