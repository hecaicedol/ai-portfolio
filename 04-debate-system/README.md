# Multi-Agent Debate System

> Five agents with distinct perspectives debate a business problem across three rounds and produce a grounded executive recommendation. Society of Mind, applied.

[![Status](https://img.shields.io/badge/status-scaffolded-blueviolet)]()
[![Python](https://img.shields.io/badge/python-3.12-blue)]()
[![Pattern](https://img.shields.io/badge/pattern-Society%20of%20Mind-4dd0c2)]()

---

## The problem

Single-LLM "deliberation" is mostly the same model agreeing with itself in five different fonts. Real strategic decisions need *adversarial* deliberation — and a single chat with five role labels in the system prompt does not actually produce that. Each role needs its own context, its own instructions, its own write/read separation from the others.

This project implements **Society-of-Mind** style debate: five agents with strictly disjoint perspectives, three structured rounds, an emergent consensus score, and an executive memo at the end.

## The thesis

> *Multi-agent isn't "many LLMs in a loop." It's deliberately designed information asymmetry. This project demonstrates the difference.*

---

## Architecture

```
                       ┌─────────────────────────┐
                       │   Problem statement     │
                       └────────────┬────────────┘
                                    │
                       ┌────────────▼────────────┐
                       │   Debate Moderator      │
                       │     (LangGraph)         │
                       └────────────┬────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
  ROUND 1: opening            ROUND 2: rebuttals          ROUND 3: final
   (parallel, 5 agents)       (sequential, each reads     (parallel, agents
                              all openings)               may update stance)
                                    │
                       ┌────────────▼────────────┐
                       │      Synthesizer        │
                       │   produces exec memo    │
                       └─────────────────────────┘

Shared Memory (Redis)
└── All agents read the full debate history before generating
```

The five agents — each with a distinct system prompt:

| Agent | Lens |
|---|---|
| **Optimist** | Opportunities, upside, best-case scenarios |
| **Skeptic** | Weaknesses, past failures, worst-case |
| **Financial** | ROI, cashflow, unit economics |
| **Risk** | Operational, legal, reputational risk |
| **Devil's Advocate** | Always argues against the current consensus |

### Why this design

| Decision | Alternative | Why we picked this |
|---|---|---|
| **One model, five disjoint system prompts** | Five different model families | Cleaner experiment: differences are role-induced, not capability-induced. Cheaper, faster. |
| **Round structure (open / rebut / final)** | Free-form chat | Rounds force agents to *read* others' arguments before responding. Free chat collapses to whoever speaks loudest. |
| **Devil's Advocate that re-targets every round** | Static dissent role | Real dissent isn't a fixed position — it's whatever the majority is currently wrong about. |
| **Redis for shared memory** | In-process state | WebSocket clients reconnect; debate state needs to survive that. Redis with 7-day TTL fits. |
| **Consensus measurement** | Just print the memo | A numeric consensus score (0–1) tells the decision-maker *how settled* the debate was. A 0.95 consensus means "act"; a 0.4 means "this needs more data." |

---

## Tech stack

| Layer | Choice |
|---|---|
| Orchestration | LangGraph (with parallel + sequential rounds) |
| LLM | Claude `claude-sonnet-4-5` |
| Shared memory | Redis (debate history, 7-day TTL) |
| API | FastAPI + WebSockets |
| Frontend | Next.js — animated debate room |
| Infra | Docker Compose |

---

## Repository layout

```
04-debate-system/
├── docker-compose.yml
├── .env.example
├── README.md
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── api/main.py
│   ├── agents/
│   │   ├── base_agent.py
│   │   ├── optimist.py
│   │   ├── skeptic.py
│   │   ├── financial.py
│   │   ├── risk.py
│   │   ├── devils_advocate.py
│   │   └── synthesizer.py
│   └── debate/
│       ├── moderator.py
│       ├── shared_memory.py
│       └── rounds.py
└── frontend/
```

---

## Quick start

```bash
cp .env.example .env
docker compose up --build
```

### Start a debate

Open `http://localhost:3000` and submit a problem. The frontend connects to `ws://localhost:8000/ws/debate/{session_id}` and streams each statement as agents produce them.

Programmatic:
```bash
wscat -c ws://localhost:8000/ws/debate/$(uuidgen)
> {"action":"start","problem":"Should we acquire Vendor X for $4M to in-source our payments stack?"}
```

Stream events:
```json
{"type":"round_start","round":1}
{"type":"agent_statement","agent":"optimist","round":1,"content":"..."}
...
{"type":"debate_complete","consensus":0.78,"executive_memo":{...}}
```

---

## Metrics to track

| Metric | Value |
|---|---|
| Avg time per full debate (3 rounds, 5 agents) | _TODO_ |
| Tokens per debate | _TODO_ |
| Consensus distribution across 50 trial problems | _TODO_ |
| Human-vs-system agreement on "best decision" | _TODO_ |

---

## License

MIT.
