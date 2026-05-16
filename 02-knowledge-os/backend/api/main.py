from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="Enterprise Knowledge OS",
    version="0.1.0",
    description="GraphRAG over Neo4j with entity extraction and staleness detection.",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class QueryRequest(BaseModel):
    question: str
    max_hops: int = 2


class QueryResponse(BaseModel):
    answer: str
    reasoning_path: list[str]
    cited_nodes: list[dict]
    confidence: float


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/ingest")
async def ingest(file: UploadFile = File(...), source: str = Form("manual")) -> dict:
    """
    Ingest a document: parse → extract entities & relations → upsert into Neo4j.
    Implementation lives in backend/ingestion/pipeline.py.
    """
    raise NotImplementedError("Implement ingestion.pipeline.run(file, source)")


@app.post("/api/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    """
    GraphRAG query: extract query entities → vector-match → traverse → synthesize.
    Implementation lives in backend/graph/graph_rag.py.
    """
    raise NotImplementedError("Implement graph.graph_rag.GraphRAGEngine.query")


@app.get("/api/staleness-report")
async def staleness_report() -> dict:
    """
    Latest report from the background StalenessAgent — nodes not updated > 30 days.
    """
    raise NotImplementedError("Implement graph.staleness_agent.latest_report")


@app.get("/api/graph")
async def graph_snapshot(limit: int = 200) -> dict:
    """
    Snapshot of nodes + edges for the frontend force-graph visualization.
    """
    raise NotImplementedError("Implement graph.neo4j_client.snapshot")
