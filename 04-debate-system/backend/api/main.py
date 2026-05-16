import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Multi-Agent Debate System", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.websocket("/ws/debate/{session_id}")
async def debate_socket(ws: WebSocket, session_id: str) -> None:
    """
    Client sends:  {"action": "start", "problem": "..."}
    Server streams:
      {"type": "round_start", "round": N}
      {"type": "agent_statement", "agent": "<role>", "round": N, "content": "..."}
      {"type": "round_complete", "round": N, "consensus_level": float}
      {"type": "debate_complete", "executive_memo": {...}}
    """
    await ws.accept()
    try:
        msg = json.loads(await ws.receive_text())
        if msg.get("action") != "start":
            await ws.close(code=1003)
            return
        raise NotImplementedError("Drive debate.moderator.run(problem, on_event=ws.send_json)")
    except WebSocketDisconnect:
        return
