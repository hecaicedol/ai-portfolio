from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Adaptive Workflow Engine", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class GoalRequest(BaseModel):
    goal: str
    auto_approve_non_critical: bool = False


class ApprovalRequest(BaseModel):
    node_id: str
    approved: bool
    edited_params: dict | None = None


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/workflows")
async def submit_goal(req: GoalRequest) -> dict:
    """Planner returns a proposed DAG for review."""
    raise NotImplementedError("planner.planner_agent.PlannerAgent.plan(goal)")


@app.post("/api/workflows/{workflow_id}/approve")
async def approve_workflow(workflow_id: str) -> dict:
    raise NotImplementedError("Kick off executor.engine.run(workflow_id)")


@app.post("/api/workflows/{workflow_id}/nodes/{node_id}/approve")
async def approve_node(workflow_id: str, node_id: str, req: ApprovalRequest) -> dict:
    raise NotImplementedError("executor.hitl.resolve(workflow_id, node_id, approved, edited_params)")


@app.get("/api/workflows/{workflow_id}/stream")
async def stream_workflow(workflow_id: str):
    raise NotImplementedError("SSE stream of node state transitions")


@app.get("/api/workflows/history")
async def history(limit: int = 50) -> dict:
    raise NotImplementedError("planner.workflow_memory.recent(limit)")
