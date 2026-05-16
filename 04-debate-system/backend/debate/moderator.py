from typing import Any, Awaitable, Callable, TypedDict


class DebateState(TypedDict, total=False):
    problem: str
    session_id: str
    statements: list[dict[str, Any]]
    consensus_level: float
    executive_memo: dict[str, Any]
    debate_complete: bool


def build_debate_graph(*, agents: list, synthesizer, shared_memory):
    """
    LangGraph StateGraph:

      round_1_opening   — asyncio.gather(all agents.generate_statement(round_type='opening'))
      round_2_rebuttal  — sequential; each agent reads all round-1 statements
      round_3_final     — asyncio.gather(all final statements)
      synthesize        — produces ExecutiveMemo + consensus score

    Each node:
      - reads current state.statements (debate history)
      - persists new statements to shared_memory (Redis) for crash safety
      - emits events via the on_event callback for WebSocket streaming
    """
    raise NotImplementedError


async def run(
    *,
    graph,
    problem: str,
    session_id: str,
    on_event: Callable[[dict], Awaitable[None]],
) -> dict[str, Any]:
    """Drive the graph; call `on_event` for every state transition so the
    WebSocket client receives live updates."""
    raise NotImplementedError
