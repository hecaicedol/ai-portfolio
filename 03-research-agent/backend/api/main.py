from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Autonomous Research Agent", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ResearchRequest(BaseModel):
    question: str
    max_steps: int = 12


class MemoryQuery(BaseModel):
    query: str
    layer: str = "semantic"  # working | episodic | semantic
    k: int = 5


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/research")
async def research(req: ResearchRequest) -> dict:
    raise NotImplementedError("Drive agent.research_agent.run(question)")


@app.post("/api/research/stream")
async def research_stream(req: ResearchRequest):
    raise NotImplementedError("SSE stream from agent.research_agent.stream(question)")


@app.post("/api/memory/query")
async def memory_query(q: MemoryQuery) -> dict:
    raise NotImplementedError("Delegate to memory.memgpt_controller.retrieve(layer, query, k)")


@app.get("/api/sessions")
async def list_sessions(limit: int = 50) -> dict:
    raise NotImplementedError("Read from episodic_sessions table")


@app.get("/api/reports/{session_id}")
async def get_report(session_id: str):
    raise NotImplementedError("Stream PDF from tools.report_generator")
