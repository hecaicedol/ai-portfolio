from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Agentic RAG · Vector DB Benchmark", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class QueryRequest(BaseModel):
    question: str
    stores: list[str] = ["pgvector", "qdrant", "pinecone"]


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/ingest")
async def ingest(file: UploadFile = File(...)) -> dict:
    """Chunk → contextually enrich → index into all enabled stores in parallel."""
    raise NotImplementedError("ingestion.indexer.run(file)")


@app.post("/api/query")
async def query(req: QueryRequest) -> dict:
    """Run the full retrieval+synthesis pipeline against each requested store."""
    raise NotImplementedError("retrieval.pipeline.run(req.question, stores=req.stores)")


@app.get("/api/metrics/store/{name}")
async def store_metrics(name: str, window: int = 100) -> dict:
    raise NotImplementedError("evaluation.metrics_store.window_for(name, window)")


@app.get("/api/metrics/overview")
async def metrics_overview() -> dict:
    """Live benchmark — latency p50/p95, Ragas scores, costs, optimizer interventions."""
    raise NotImplementedError("evaluation.metrics_store.overview()")
