"""HTTP-layer tests using FastAPI's TestClient.

The production lifespan opens a real Postgres connection, which isn't
available in tests. We replace the app's lifespan with a no-op and set
`app.state.memory` / `app.state.graph` manually per-test so each test
gets a fresh FakeMemory + ScriptedLLM.
"""
import json
from contextlib import asynccontextmanager
from typing import Any

from fastapi.testclient import TestClient

from agents.orchestrator import build_graph
from api.main import app
from config import Settings
from tests.conftest import FakeMemory, ScriptedLLM


@asynccontextmanager
async def _noop_lifespan(_app):
    yield


# Disable the Postgres-touching lifespan once for the whole module
app.router.lifespan_context = _noop_lifespan


def _good_invoice() -> str:
    return json.dumps(
        {
            "invoice_number": "INV-001",
            "vendor": "Acme",
            "total": 100.0,
            "issue_date": "2026-01-15",
        }
    )


def _critic_pass() -> str:
    return json.dumps(
        {
            "overall_score": 0.95,
            "principles": [
                {"principle": p, "score": 0.95, "feedback": "ok"}
                for p in ("completeness", "accuracy", "consistency", "format")
            ],
        }
    )


def _wire_app(
    *,
    extractor_responses: list[str] | None = None,
    critic_responses: list[str] | None = None,
    pre_errors: list[dict[str, Any]] | None = None,
) -> FakeMemory:
    """Replace app.state.memory / app.state.graph with test-controlled
    instances. Returns the FakeMemory so tests can assert on it."""
    memory = FakeMemory()
    for e in pre_errors or []:
        memory.errors.append(e)
    llm = ScriptedLLM(
        extractor_responses=extractor_responses or [],
        critic_responses=critic_responses or [],
    )
    settings = Settings()  # type: ignore[call-arg]
    graph = build_graph(memory=memory, settings=settings, model=llm)
    app.state.memory = memory
    app.state.graph = graph
    return memory


def test_health_endpoint():
    client = TestClient(app)
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_process_endpoint_returns_pass_payload():
    _wire_app(
        extractor_responses=[_good_invoice()],
        critic_responses=[_critic_pass()],
    )
    client = TestClient(app)
    res = client.post(
        "/api/process",
        json={
            "document_type": "invoice",
            "content": "Invoice #INV-001 Acme $100 2026-01-15",
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["success"] is True
    assert body["iterations"] == 1
    assert body["final_score"] == 0.95
    assert body["extracted_data"]["vendor"] == "Acme"
    assert body["critic_report"]["passes"] is True
    assert len(body["critic_report"]["principles"]) == 4


def test_process_endpoint_validates_request():
    _wire_app()
    client = TestClient(app)
    # Missing required `content` field
    res = client.post("/api/process", json={"document_type": "invoice"})
    assert res.status_code == 422


def test_process_stream_endpoint_returns_sse_response():
    """Verify the HTTP wiring: 200 status, SSE content-type, and at least the
    first frame arrives. (Full event-by-event behavior is exercised by
    `test_orchestrator.py::test_stream_pipeline_emits_run_started_and_completed`
    directly against `stream_pipeline`; replicating that here is brittle
    under TestClient because Starlette's sync TestClient and sse_starlette's
    BackgroundTask machinery don't co-operate cleanly without a real loop.)
    """
    _wire_app(
        extractor_responses=[_good_invoice()],
        critic_responses=[_critic_pass()],
    )
    client = TestClient(app)

    with client.stream(
        "POST",
        "/api/process/stream",
        json={
            "document_type": "invoice",
            "content": "Invoice #INV-001 Acme $100 2026-01-15",
        },
    ) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")
        body = b"".join(response.iter_bytes()).decode("utf-8")

    # The first event must arrive — the orchestrator-level test covers the rest.
    assert "event: run_started" in body
    assert "data:" in body


def test_memory_errors_endpoint_returns_recent():
    pre_errors = [
        {
            "document_type": "invoice",
            "error_type": "accuracy_below_threshold",
            "principle": "accuracy",
            "context": {"feedback": "test"},
            "resolution": None,
        },
        {
            "document_type": "receipt",
            "error_type": "completeness_below_threshold",
            "principle": "completeness",
            "context": {"missing": ["date"]},
            "resolution": None,
        },
    ]
    _wire_app(pre_errors=pre_errors)
    client = TestClient(app)

    res = client.get("/api/memory/errors?limit=10")
    assert res.status_code == 200
    body = res.json()
    assert "errors" in body
    assert len(body["errors"]) == 2
    error_types = {e["error_type"] for e in body["errors"]}
    assert error_types == {
        "accuracy_below_threshold",
        "completeness_below_threshold",
    }


def test_memory_similar_endpoint_rejects_missing_query():
    _wire_app()
    client = TestClient(app)
    res = client.post("/api/memory/similar", json={})
    assert res.status_code == 400


def test_memory_similar_endpoint_returns_results():
    pre_errors = [
        {
            "document_type": "invoice",
            "error_type": "consistency_below_threshold",
            "principle": "consistency",
            "context": {"detail": "x"},
            "resolution": None,
        },
    ]
    _wire_app(pre_errors=pre_errors)
    client = TestClient(app)
    res = client.post(
        "/api/memory/similar",
        json={"query": "invoice consistency", "k": 2},
    )
    assert res.status_code == 200
    body = res.json()
    assert "errors" in body
    # FakeMemory.retrieve_similar_errors returns the last k errors (no embedding)
    assert len(body["errors"]) == 1
    assert body["errors"][0]["error_type"] == "consistency_below_threshold"
