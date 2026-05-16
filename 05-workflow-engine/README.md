# Adaptive Workflow Engine — MCP + Dynamic DAGs

> A meta-agent that builds workflow DAGs from natural-language goals at runtime, executes them through MCP servers connected to real enterprise tools, and learns from past successful workflows.

[![Status](https://img.shields.io/badge/status-scaffolded-blueviolet)]()
[![Python](https://img.shields.io/badge/python-3.12-blue)]()
[![Protocol](https://img.shields.io/badge/protocol-MCP-7c5cff)]()
[![Pattern](https://img.shields.io/badge/pattern-meta--agent-orange)]()

---

## The problem

Most agent frameworks let you wire up a fixed graph and call it a workflow. That works until the user's goal doesn't match the graph you wired up. Production teams need agents that build the workflow *for the goal*, not the other way around.

This project: a **meta-agent** that reads a goal in natural language, plans a DAG of tasks, validates it (no cycles, valid tool references), and hands it to an executor that runs nodes through **MCP servers** — the Anthropic protocol that's becoming the integration layer for enterprise AI in 2025–2026.

## The thesis

> *MCP is the most important protocol to understand in 2026. A meta-agent that generates DAGs dynamically is the most advanced pattern in agentic architecture. Combining them in one project is exactly the right level of ambition.*

---

## Architecture

```
            ┌──────────────────────────────────────┐
            │  User goal (natural language)        │
            └────────────────┬─────────────────────┘
                             │
            ┌────────────────▼─────────────────────┐
            │           Planner (meta-agent)       │
            │  · Queries workflow_memory (similar) │
            │  · Generates DAG JSON                │
            │  · Validates (no cycles, valid tools)│
            │  · Presents plan for HITL approval   │
            └────────────────┬─────────────────────┘
                             │
            ┌────────────────▼─────────────────────┐
            │           Executor (LangGraph)       │
            │  · Topological order                 │
            │  · Parallel where no deps            │
            │  · HITL approval on critical nodes   │
            │  · Per-node MCP tool call            │
            │  · On failure → Replanner             │
            └────────────────┬─────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
   ┌─────────┐         ┌─────────┐          ┌──────────┐
   │ GitHub  │         │  Jira   │          │  Slack   │   …more MCP servers
   │   MCP   │         │   MCP   │          │   MCP    │
   └─────────┘         └─────────┘          └──────────┘

  Workflow memory (pgvector)
  └── On success → store DAG with embedding(goal). Planner consults next time.
```

### Why this design

| Decision | Alternative | Why we picked this |
|---|---|---|
| **MCP everywhere** | Direct API calls per integration | MCP is the protocol Anthropic is investing in for tool integrations. Building real MCP servers (not just clients) demonstrates familiarity at the right level. |
| **DAG validation before execution** | Just try and catch | Cycle detection at plan time is cheap and avoids burning tokens on a malformed plan that won't run. |
| **HITL flag per node** | All-or-nothing approval | Real workflows have some nodes you want to review (Slack messages to customers) and many you don't (read-only queries). Per-node granularity is the production-correct shape. |
| **Replanner instead of retry** | Just retry the node | Failures often mean *the plan was wrong*, not that the tool was flaky. Letting the planner observe the failure and re-plan is more powerful than blind retry. |
| **Workflow memory by goal-similarity** | Hardcoded workflow library | The Planner learns the team's actual patterns. After 50 successful workflows, plan latency drops because the planner adapts past wins. |

---

## Tech stack

| Layer | Choice |
|---|---|
| Orchestration | LangGraph (dynamic graph construction) |
| Protocol | MCP (Anthropic's `mcp` Python SDK) |
| LLM | Claude `claude-sonnet-4-5` |
| Memory | PostgreSQL + pgvector |
| Cache | Redis (in-flight DAG state, approvals) |
| API | FastAPI + SSE |
| Frontend | Next.js + `react-flow` for DAG visualization |
| Tools | GitHub, Jira, Slack, Google Drive (as MCP servers) |

---

## Repository layout

```
05-workflow-engine/
├── docker-compose.yml
├── .env.example
├── README.md
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── api/main.py
│   ├── planner/
│   │   ├── planner_agent.py
│   │   ├── dag_parser.py
│   │   └── workflow_memory.py
│   ├── executor/
│   │   ├── engine.py
│   │   ├── replanner.py
│   │   └── hitl.py
│   └── mcp/
│       ├── client.py
│       ├── github_server.py
│       ├── jira_server.py
│       ├── slack_server.py
│       └── gdrive_server.py
└── frontend/
```

---

## Quick start

```bash
cp .env.example .env
docker compose up --build
```

### Submit a goal

```bash
curl -X POST http://localhost:8000/api/workflows \
  -H "Content-Type: application/json" \
  -d '{"goal": "When a new GitHub issue is labeled bug, create a Jira ticket in the BACKEND project and notify #engineering on Slack."}'
```

The response is the proposed DAG (JSON). Approve it via:

```bash
curl -X POST http://localhost:8000/api/workflows/{workflow_id}/approve
```

Watch execution live: `GET /api/workflows/{workflow_id}/stream` (SSE).

---

## DAG schema

```jsonc
{
  "goal": "When a new GitHub issue is labeled bug...",
  "nodes": [
    {
      "id": "n1",
      "name": "Get issue details",
      "tool": "github",
      "action": "get_issue",
      "params": {"repo": "acme/api", "number": 42},
      "requires_approval": false,
      "depends_on": []
    },
    {
      "id": "n2",
      "name": "Create Jira ticket",
      "tool": "jira",
      "action": "create_ticket",
      "params": {"project": "BACKEND", "summary": "{{n1.title}}", "priority": "P2"},
      "requires_approval": true,
      "depends_on": ["n1"]
    }
  ],
  "estimated_duration_minutes": 2
}
```

`{{nX.field}}` syntax pulls outputs from upstream nodes. Validated by `dag_parser.py`.

---

## Build order

1. `backend/mcp/client.py` and one MCP server (start with `github_server.py`)
2. `backend/planner/planner_agent.py` + `dag_parser.py`
3. `backend/executor/engine.py` — LangGraph dynamic build
4. `backend/executor/hitl.py` + `replanner.py`
5. `backend/planner/workflow_memory.py`
6. `frontend/` — `react-flow` DAG view

---

## License

MIT.
