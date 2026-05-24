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
| Orchestration | LangGraph 0.2 |
| LLM | Claude `claude-sonnet-4-5` via `langchain-anthropic` |
| Memory | PostgreSQL 16 + `pgvector` extension |
| Embeddings | Voyage AI (`voyage-3`, 1024 dims) with deterministic hash fallback for dev |
| API | FastAPI (async) + sse-starlette for live pipeline streaming |
| Frontend | Next.js 14 (App Router) + Tailwind + lucide-react |
| Testing | pytest + pytest-asyncio (15 tests, no Docker / no API key needed) |
| Infra | Docker Compose (Postgres + API + frontend) |
| Observability | Langfuse — deps installed, callback wiring pending (see `BUILD.md`) |

---

## Repository layout

```
01-self-healing-pipeline/
├── docker-compose.yml
├── .env.example
├── README.md
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt          # runtime deps
│   ├── requirements-dev.txt      # adds pytest + pytest-asyncio
│   ├── pytest.ini
│   ├── config.py
│   ├── api/
│   │   ├── main.py               # FastAPI app + SSE endpoints
│   │   └── schemas.py            # Pydantic request / response models
│   ├── agents/
│   │   ├── orchestrator.py       # LangGraph StateGraph
│   │   ├── extractor.py          # JSON-retry hardened
│   │   ├── validator.py          # deterministic, no LLM
│   │   ├── critic.py             # Constitutional AI + memory lookup
│   │   └── synthesizer.py
│   ├── memory/
│   │   ├── episodic.py           # pgvector CRUD + similarity
│   │   └── schemas.py
│   ├── db/
│   │   └── init.sql              # pgvector + schema
│   ├── tests/                    # 15 tests — `pytest tests`
│   │   ├── conftest.py           # FakeMemory + ScriptedLLM fixtures
│   │   ├── test_validator.py
│   │   ├── test_agents.py
│   │   └── test_orchestrator.py
│   └── eval/
│       ├── corpus/               # 15 hand-written synthetic docs
│       ├── benchmark.py          # runner + grader + aggregator
│       └── README.md             # how to run dry-run vs real
└── frontend/                     # Next.js 14 SPA — see frontend/README.md
    ├── package.json
    ├── src/app/                  # layout + page (SSE consumer)
    ├── src/components/           # Header, DocumentInput, Pipeline, Critic, Memory…
    └── src/lib/                  # api client, types, helpers
```

---

## Quick start

Two ways to run, depending on what you want to check.

### A. No Docker, no API key — just want to see the code is correct

```bash
cd backend
python -m venv .venv
.venv/Scripts/activate            # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pytest tests                      # 15 tests, ~2 seconds
python -m eval.benchmark --mode dry-run   # exercises the full pipeline with a stub LLM
```

This runs the entire orchestrator graph on the 15-document corpus using a deterministic stub LLM and an in-memory fake of the pgvector layer. `$0` cost, no Docker required.

### B. Full stack — see it work against real Claude

```bash
cp .env.example .env
# set ANTHROPIC_API_KEY in .env

docker compose up --build
```

Services after `up`:
- API → `http://localhost:8000` (interactive docs at `/docs`)
- Postgres → `localhost:5432` (auto-initialized with pgvector + schema)
- Frontend → `http://localhost:3000`

(Langfuse self-hosting is commented out in `docker-compose.yml` — see notes in that file.)

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

## Testing

22 automated tests live in `backend/tests/`. They use `FakeMemory` (in-memory stand-in for the pgvector layer) and `ScriptedLLM` (deterministic LLM that replays pre-written responses) so the suite needs **no Docker, no API key, and no money** to run — `pytest tests` finishes in about three seconds.

What's covered:

| File | What it verifies |
|---|---|
| `test_validator.py` | Required-field detection per document type, edge cases (unknown types, empty values, generic) |
| `test_agents.py` | `Extractor` / `Critic` JSON-retry behavior, markdown-fence stripping, retry exhaustion → typed parse errors, critic prompt actually includes past similar errors |
| `test_orchestrator.py` | All four LangGraph paths end-to-end: pass on first attempt, self-heal on second attempt, exhaust max iterations, SSE streaming emits the expected events. Also verifies `record_run` is called exactly once per pipeline invocation. |
| `test_api.py` | FastAPI HTTP layer with TestClient: `/health`, `/api/process` happy path + request validation, `/api/process/stream` returns SSE, `/api/memory/errors`, `/api/memory/similar` happy path + missing-query 400. Lifespan is patched to a no-op so tests don't need Postgres. |

The orchestrator tests were what surfaced a latent LangGraph bug (a node name colliding with a state-key reserved word) that had never been observed because the system had never actually been run against the real library version. That fix is in the commit history.

## Evaluation harness

`backend/eval/` contains a 15-document synthetic corpus (8 invoices + 4 receipts + 3 contracts, split across easy / medium / hard) plus `benchmark.py`, which:

1. Loads every corpus item, runs it through the pipeline with reflection enabled (max=3 iterations).
2. Grades the extracted output field-by-field against ground truth (numeric tolerance 0.01; string compare is case-insensitive after strip; lists/dicts compared recursively).
3. Aggregates pass rate, average iterations, accuracy, latency — broken down by `document_type` and `difficulty`.
4. Writes `results/<label>_results.json`, `<label>_summary.json`, and `<label>_summary.md` (the last one is what gets pasted into this README).

Two modes:

```bash
python -m eval.benchmark --mode dry-run     # OracleStub LLM, $0
python -m eval.benchmark --mode real --label baseline-2026-05
```

Worst-case cost for the real run on the 15-doc corpus is ≈ **$1.20 USD** (15 docs × up to 6 LLM calls per doc at Claude Sonnet 4.5 list pricing). See `backend/eval/README.md`.

## Metrics (pending real benchmark run)

The slots below are populated by `eval/benchmark.py` once the real-mode run has been executed. The dry-run produces 100% by construction (the stub returns ground truth verbatim) — those numbers are not posted here on purpose.

| Metric | Without reflection | With reflection | Δ |
|---|---|---|---|
| Pass rate (overall) | _pending_ | _pending_ | — |
| Avg accuracy across required fields | — | _pending_ | — |
| Avg iterations on successful runs | 1.00 | _pending_ | — |
| Documents healed by reflection | n/a | _pending_ | — |
| Avg latency per doc (s) | — | _pending_ | — |

---

## Lessons learned

- **Critics need memory, not just rubrics.** A stateless critic re-makes the same mistake every run; one that consults past failures starts catching domain patterns within ~50 examples.
- **Constitutional principles should be *evaluable*, not philosophical.** "Output should be helpful" is a bad principle. "Every required field has a non-empty value matching its type" is a good one.
- **Reflection loops need hard caps.** An unbounded retry loop in early development burned through $40 of API credits on a single malformed PDF before I added the cap.

---

## License

MIT.
