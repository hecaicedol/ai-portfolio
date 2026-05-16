import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from agents.orchestrator import build_graph, run_pipeline, stream_pipeline
from api.schemas import ProcessRequest, ProcessResponse
from config import get_settings
from memory.episodic import EpisodicMemory


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    memory = EpisodicMemory(settings.database_url)
    await memory.connect()
    app.state.memory = memory
    app.state.graph = build_graph(memory=memory, settings=settings)
    yield
    await memory.close()


app = FastAPI(
    title="Self-Healing Multi-Agent Pipeline",
    version="1.0.0",
    description="A document-processing pipeline with Constitutional AI critic, reflection loops, and episodic memory.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/process", response_model=ProcessResponse)
async def process_document(req: ProcessRequest) -> ProcessResponse:
    try:
        result = await run_pipeline(
            graph=app.state.graph,
            document_type=req.document_type,
            content=req.content,
            metadata=req.metadata,
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/process/stream")
async def process_document_stream(req: ProcessRequest):
    async def event_generator():
        async for event in stream_pipeline(
            graph=app.state.graph,
            document_type=req.document_type,
            content=req.content,
            metadata=req.metadata,
        ):
            yield {"event": event.type, "data": json.dumps(event.model_dump(mode="json"))}

    return EventSourceResponse(event_generator())


@app.get("/api/memory/errors")
async def list_errors(limit: int = 20) -> dict[str, list[dict]]:
    rows = await app.state.memory.recent_errors(limit=limit)
    return {"errors": rows}


@app.post("/api/memory/similar")
async def similar_errors(payload: dict) -> dict[str, list[dict]]:
    query = payload.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="'query' is required")
    rows = await app.state.memory.retrieve_similar_errors(query, k=int(payload.get("k", 3)))
    return {"errors": rows}
