# Full-Stack AI Engineer Portfolio

> Six production-grade AI engineering projects demonstrating multi-agent systems, advanced RAG, persistent memory, and the Model Context Protocol (MCP).

**Author:** Heims Andrés Caicedo Lopera
**Contact:** hecaicedol@unal.edu.co
**Live portfolio site:** [`./index.html`](./index.html) — deployable to GitHub Pages

---

## Why this portfolio

95% of AI portfolios show a chatbot wrapper around an API. These six projects show what production AI systems actually need:

- **Self-correction**: agents that detect their own failures and recover from them
- **Persistent memory**: state that survives across sessions, not just chat history
- **Hybrid retrieval**: knowledge graphs + vector search + reranking, not just `vector.search()`
- **Multi-agent collaboration**: distinct roles producing emergent insights
- **Dynamic orchestration**: workflows generated at runtime, not hardcoded
- **Production discipline**: real benchmarks, real evaluation loops, real cost tracking

Each project answers a question recruiters in EU/USA actually ask: *"Can this person build something that survives a Monday morning in production?"*

---

## The six projects

| # | Project | Core Technique | Status |
|---|---------|---------------|--------|
| 1 | [Self-Healing Multi-Agent Pipeline](./01-self-healing-pipeline) | Constitutional AI critic + reflection loops + episodic memory | **Functional · tested · evaluated** |
| 2 | [Enterprise Knowledge OS (GraphRAG)](./02-knowledge-os) | Knowledge graph + vector hybrid retrieval (Neo4j + LlamaIndex) | Scaffolded |
| 3 | [Autonomous Research Agent (MemGPT)](./03-research-agent) | Three-tier memory architecture (working/episodic/semantic) | **Core memory verified** · 31 tests |
| 4 | [Multi-Agent Debate System](./04-debate-system) | Society of Mind pattern, 5 specialized agents, consensus measurement | Scaffolded |
| 5 | [Adaptive Workflow Engine (MCP)](./05-workflow-engine) | Meta-agent generates DAGs at runtime, executes via MCP servers | Scaffolded |
| 6 | [Agentic RAG with Vector DB Benchmark](./06-agentic-rag) | Contextual retrieval + hybrid search + auto-optimizer agent | **Core retrieval verified** · 23 tests |

> **Why one flagship + five scaffolds?** A recruiter scanning the portfolio in 90 seconds needs (a) proof of end-to-end execution and (b) breadth of architectural thinking. Project 1 proves execution; the other five prove I can design six different non-trivial systems without copy-pasting the same template.

### Current state of P1 (what "Functional · tested · evaluated" means)

**Built and verified:**
- 5-node LangGraph orchestrating `extract → validate → critique → reflect → synthesize`, with a hard-capped reflection loop (3 iterations max).
- 4 agents: Extractor + Critic (Claude Sonnet 4.5, with 3-attempt JSON-recovery retry), Validator + Synthesizer (deterministic, no LLM).
- Episodic memory layer over pgvector — the critic retrieves the 3 most-similar past errors before scoring each new doc.
- FastAPI backend with REST + SSE streaming endpoints.
- **22 automated tests** passing (`pytest`), covering validator paths, agent JSON-retry behavior, all four orchestrator paths (pass-on-first, self-heal, max-iter fail, streaming), and the FastAPI HTTP endpoints via `TestClient`.
- **15-document synthetic evaluation corpus** + `eval/benchmark.py` harness that runs the pipeline against the corpus, grades each output field-by-field against ground truth, and emits aggregate metrics by document type and difficulty.
- Next.js 14 + Tailwind frontend with live SSE pipeline visualization, per-principle critic scoring, and episodic memory viewer.

**Pending (and honest about it):**
- Real benchmark run against Claude (script is ready; needs an API key + ~$1 budget).
- Public deployment (Railway / Fly.io for API, Vercel for frontend).
- Langfuse observability integration is wired in `build_graph`: the callback handler is registered with `ChatAnthropic` when `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` are present, and is a no-op otherwise. The Langfuse server itself is intentionally not bundled in `docker-compose.yml` because the v2 self-host requires extra services; the slot is open for a managed Langfuse Cloud account.

---

## Shared design principles across all projects

1. **Every project ships with `docker-compose.yml`** — clone, `docker compose up`, it works.
2. **Every project uses Pydantic** for I/O boundaries — no untyped dicts crossing module lines.
3. **Every agent is observable** — Langfuse traces on every LLM call.
4. **Every README answers four questions**: what problem, what architecture, why this approach over alternatives, what metrics prove it works.
5. **Every project has a `frontend/`** — recruiters open the browser, not the terminal.

---

## Repository layout

```
ai-portfolio/
├── index.html                       # Single-page portfolio site (GitHub Pages)
├── README.md                        # This file
├── 01-self-healing-pipeline/        # P1 — functional flagship
├── 02-knowledge-os/                 # P2 — GraphRAG
├── 03-research-agent/               # P3 — MemGPT memory
├── 04-debate-system/                # P4 — Society of Mind
├── 05-workflow-engine/              # P5 — MCP + dynamic DAGs
└── 06-agentic-rag/                  # P6 — Vector DB benchmark
```

---

## How to use this portfolio

**As a recruiter / hiring manager:**
1. Open `index.html` (live site link below) — 60-second overview
2. Pick the project that matches the role's stack
3. Read its README — every README is self-contained
4. Watch the demo or run `docker compose up`

**As a candidate using this as a template:**
1. Fork it
2. Replace `Heims Andrés Caicedo Lopera` everywhere
3. Pick one project per month — implement the scaffolded ones using the prompts in `BUILD.md` inside each folder
4. Deploy each one separately to Railway / Render / Fly.io
5. Track metrics in the README — recruiters love numbers

---

## Tech stack used across projects

| Layer | Tools |
|-------|-------|
| LLM | Claude (Anthropic) — `claude-sonnet-4-5` / `claude-opus-4-7` |
| Orchestration | LangGraph |
| Embeddings | Voyage AI / OpenAI `text-embedding-3-large` |
| Vector stores | pgvector, Qdrant, Pinecone |
| Graph DB | Neo4j (APOC plugin) |
| Reranking | Cohere Rerank v3 |
| Evaluation | Ragas |
| Observability | Langfuse |
| Backend | FastAPI (async), Pydantic v2 |
| Frontend | Next.js 14, Tailwind, shadcn/ui, react-flow, react-force-graph, recharts |
| Infra | Docker Compose, Postgres 16 |
| Protocol | Model Context Protocol (MCP) |

---

## Roadmap (personal)

| Quarter | Goal | Status |
|--|--|--|
| Q1 | P1 fully shipped + deployed + metrics on README | In progress — code + tests + frontend + eval harness done; deployment + real metrics pending |
| Q2 | P6 (Agentic RAG) — second most important for AI roles | Scaffolded |
| Q3 | P3 (Research Agent) — memory architecture is a hiring filter | Slice 1 verified — 31 tests covering working/episodic/semantic + MemGPT controller + research graph |
| Q4 | P2, P4, P5 — finish breadth | Scaffolded |

---

## License

MIT. Use freely. Attribution appreciated but not required.
