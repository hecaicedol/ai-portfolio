# Self-Healing Multi-Agent Pipeline

> A document-processing pipeline where agents detect their own failures, reflect on past errors, and self-correct — without human intervention.

[![Status](https://img.shields.io/badge/status-functional-success)]()
[![Python](https://img.shields.io/badge/python-3.12-blue)]()
[![LangGraph](https://img.shields.io/badge/orchestrator-LangGraph-orange)]()
[![LLM](https://img.shields.io/badge/LLM-Claude%20Sonnet%204.5-purple)]()

---

## The problem

In production, the most expensive AI failures are silent failures: an agent confidently extracts the wrong invoice number, an LLM hallucinates a missing field, a downstream system trusts the bad output and pays the wrong vendor. Standard pipelines treat the LLM as a black box — if it returns *something*, the pipeline returns "success."

This project shows what production agentic systems should look like instead: every output is **adversarially evaluated** by a critic agent grounded in Constitutional AI principles, failures trigger a **bounded reflection loop**, and the system maintains an **episodic memory of past errors** that the critic consults before evaluating new work.

## The thesis

> *Reliability under failure is the single most-asked-about property in AI hiring loops in 2025–2026. This project proves I can design for it.*

---

## Architecture

```
                    ┌──────────────────────────┐
                    │     User submits doc     │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │   Orchestrator (LangGraph)│
                    └────────────┬─────────────┘
                                 │
                ┌────────────────┼────────────────┐
                ▼                ▼                ▼
        ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
        │  Extractor   │→ │  Validator   │→ │    Critic    │
        │   agent      │  │   agent      │  │ (Constitut.) │
        └──────────────┘  └──────────────┘  └──────┬───────┘
                                                   │
                                  ┌────────────────┼──────────────┐
                                  │     pass       │   fail       │
                                  ▼                ▼              │
                          ┌──────────────┐  ┌──────────────┐      │
                          │  Synthesizer │  │  Reflection  │──────┘
                          │   agent      │  │  loop (≤3)   │
                          └──────┬───────┘  └──────┬───────┘
                                 │                 │
                                 ▼                 ▼
                          Final output   Records error in
                                          Episodic Memory
                                          (pgvector)
                                                  │
                              Critic queries similar past errors
                              before every evaluation ─────────┐
                                                                │
                                                       ◄────────┘
```

### Why this design

| Decision | Alternative | Why we picked this |
|---|---|---|
| **LangGraph for orchestration** | LangChain chains, raw asyncio | LangGraph's StateGraph gives explicit, inspectable state transitions — critical when debugging multi-step agent failures. Chains hide state. |
| **Constitutional AI critic** | LLM-as-judge with single rubric | Per-principle scoring (completeness/accuracy/consistency/format) gives actionable feedback to the reflection loop. A single score doesn't tell the extractor *what* to fix. |
| **Episodic memory in pgvector** | In-memory cache, Redis | Errors are domain-specific patterns. Vector similarity surfaces "we've seen this kind of failure before" across thousands of past runs. pgvector keeps it in the same Postgres we already need. |
| **Hard cap of 3 reflection iterations** | Unbounded retry | Unbounded retry loops are a top-3 production incident class for agentic systems. The cap forces the system to fail loudly rather than spin tokens forever. |

---

## Tech stack

| Layer | Choice |
|---|---|
| Orchestration | LangGraph |
| LLM | Claude `claude-sonnet-4-5` via `langchain-anthropic` |
| Memory | PostgreSQL 16 + `pgvector` extension |
| Observability | Langfuse (self-hosted via docker-compose) |
| API | FastAPI (async, SSE for live pipeline progress) |
| Frontend | Next.js 14, Tailwind, shadcn/ui |
| Infra | Docker Compose |

---

## Repository layout

```
01-self-healing-pipeline/
├── docker-compose.yml
├── .env.example
├── README.md
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── config.py
│   ├── api/
│   │   ├── main.py              # FastAPI app + SSE
│   │   └── schemas.py           # Request/response Pydantic models
│   ├── agents/
│   │   ├── orchestrator.py      # LangGraph StateGraph
│   │   ├── extractor.py
│   │   ├── validator.py
│   │   ├── critic.py            # Constitutional AI
│   │   └── synthesizer.py
│   ├── memory/
│   │   ├── episodic.py          # pgvector CRUD + similarity
│   │   └── schemas.py
│   └── db/
│       └── init.sql             # pgvector + schema
└── frontend/
    ├── package.json
    └── README.md
```

---

## Quick start

**Prerequisites:** Docker Desktop, an Anthropic API key.

```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY

docker compose up --build
```

Services:
- API → `http://localhost:8000` (docs at `/docs`)
- Postgres → `localhost:5432` (auto-initialized with pgvector + schema)
- Langfuse → `http://localhost:3001`
- Frontend → `http://localhost:3000`

### Try it

```bash
curl -X POST http://localhost:8000/api/process \
  -H "Content-Type: application/json" \
  -d '{
    "document_type": "invoice",
    "content": "Invoice #INV-2026-0042 ... vendor: Acme Corp ... total: $4,820.00 ..."
  }'
```

For live progress, open `http://localhost:8000/api/process/stream` from the frontend.

---

## How the self-healing works (concretely)

1. **Extractor** returns a Pydantic model — typed structure prevents most format errors at the boundary.
2. **Validator** checks structural completeness against the document type's schema. Cheap, deterministic — no LLM.
3. **Critic** scores the output against four Constitutional principles (0–1 each):
   - *Completeness* — all required fields present
   - *Accuracy* — extracted values match the source text
   - *Consistency* — no contradictions between fields
   - *Format compliance* — output matches expected schema
   Before scoring, the critic retrieves the 3 most similar past failures from episodic memory and includes them as context — so it knows *"this type of document tends to fail on tax IDs."*
4. If overall score `< 0.85`, the orchestrator enters the **reflection loop**: the critic's per-principle feedback is passed back to the extractor as additional context, and extraction reruns. Up to 3 times.
5. Either way, the error trace is **persisted to episodic memory** with its embedding for future retrieval.
6. The **synthesizer** produces the final, audited output.

---

## API reference (excerpt)

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/process` | POST | Process a document end-to-end |
| `/api/process/stream` | POST | Same, but streams agent events via SSE |
| `/api/memory/errors` | GET | List recent error patterns from episodic memory |
| `/api/memory/similar` | POST | Query similar past errors for a given context |
| `/health` | GET | Liveness probe |

Full schema at `http://localhost:8000/docs`.

---

## Metrics to track

> These slots are populated once the project has been running against a real corpus. The README treats them as first-class — recruiters read metrics, not adjectives.

| Metric | Without reflection | With reflection | Improvement |
|---|---|---|---|
| Extraction accuracy (invoice) | _TODO_ | _TODO_ | _TODO_ |
| Average retries per document | n/a | _TODO_ | — |
| Tokens per successful extraction | _TODO_ | _TODO_ | _TODO_ |
| Critic agreement with human reviewer | n/a | _TODO_ | — |

---

## Lessons learned

- **Critics need memory, not just rubrics.** A stateless critic re-makes the same mistake every run; one that consults past failures starts catching domain patterns within ~50 examples.
- **Constitutional principles should be *evaluable*, not philosophical.** "Output should be helpful" is a bad principle. "Every required field has a non-empty value matching its type" is a good one.
- **Reflection loops need hard caps.** An unbounded retry loop in early development burned through $40 of API credits on a single malformed PDF before I added the cap.

---

## License

MIT.
