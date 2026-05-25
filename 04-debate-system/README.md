# Multi-Agent Debate System

> Five agents with distinct perspectives debate a business problem across three rounds and produce a grounded executive recommendation. Society of Mind, applied.

[![Status](https://img.shields.io/badge/status-debate%20graph%20verified-2ec27e)]()
[![Tests](https://img.shields.io/badge/tests-18%20passing-2ec27e)]()
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

## Status — Slice 1 (debate graph verified, no API keys spent)

**Built and tested without external services:**

| Component | Status | Tests |
|---|---|---|
| `BaseDebateAgent.generate_statement` — strict JSON envelope + 2-retry recovery | ✅ | 5 |
| `SynthesizerAgent.consensus` — deterministic stance-variance math | ✅ | 4 |
| `SynthesizerAgent.synthesize` — ExecutiveMemo + LLM-injected consensus is overridden by computed value | ✅ | 2 |
| `InMemoryDebateSharedMemory` — TTL-bounded session-isolated store | ✅ | 4 |
| LangGraph debate moderator (opening → rebuttal → final → synthesis) wired end-to-end with 5 ScriptedLLM agents + sequential-read verification | ✅ | 3 |
| `RedisDebateSharedMemory` — lazy redis import, production backend | ⏳ | — |
| Frontend (Next.js animated debate room + WebSocket stream) | ⏳ | — |

**18 tests passing in 2.4 s.** Run them with `cd backend && .venv/Scripts/python -m pytest tests -q`.

### Design notes uncovered while building

- **The synthesizer's consensus score is NOT trusted to the LLM.** It's computed deterministically from the final-round stances using `1 − var/4` on the {strong_no, no, neutral, yes, strong_yes} = {−2, −1, 0, 1, 2} axis. The LLM's `synthesize()` call returns a recommendation, key arguments, risks, dissents, and next steps — but the consensus number is overwritten by the computed value, so the score is auditable.
- **Sequential reads in the rebuttal round matter.** The 1st rebuttal agent sees 5 openings; the 5th rebuttal agent sees 5 openings + 4 prior rebuttals (= 9 statements). A test pins this exact sequence so it can't regress to parallel-execution semantics by accident.
- **Each agent's system prompt is its primary differentiator.** When the test scaffolding needs to know which agent is being called, it inspects the system prompt — the unique `"You are the X"` opener is the contract. Risk's prompt contains the substring `"non-financial"`, which broke my naive role detector until I switched to exact phrase matching.

### Pending (Slice 2)

1. Wire `RedisDebateSharedMemory` against a real Redis instance (docker-compose already declares it).
2. WebSocket `/ws/debate/{session_id}` endpoint that streams every `on_event` to the browser.
3. Next.js animated debate room frontend.
4. Real debate against Claude (needs API keys — out of scope for the no-budget run).

---

## License

MIT.
