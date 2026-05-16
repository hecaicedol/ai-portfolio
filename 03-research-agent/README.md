# Autonomous Research Agent — MemGPT 3-Tier Memory

> An agent that researches complex topics with **persistent memory across sessions**, implementing the MemGPT pattern of explicit context management.

[![Status](https://img.shields.io/badge/status-scaffolded-blueviolet)]()
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

## Build order

1. `backend/memory/memgpt_controller.py` — **the core of the project**. Implement first.
2. `backend/memory/{working,episodic,semantic}_memory.py`
3. `backend/agent/research_agent.py` — LangGraph nodes
4. `backend/tools/` — web/arxiv/PDF
5. `frontend/` — three views (research / memory viewer / reports)

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
