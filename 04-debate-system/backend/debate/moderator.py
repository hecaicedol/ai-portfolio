"""LangGraph debate orchestrator.

Three rounds, then synthesis:

    opening_round   → parallel  (asyncio.gather)  · 5 openings
    rebuttal_round  → sequential                  · each agent reads all openings
    final_round     → parallel  (asyncio.gather)  · 5 final positions
    synthesis       → SynthesizerAgent emits ExecutiveMemo + consensus

LangGraph forbids node names that collide with state keys, so the node
names use the `_round` / `synthesis` suffix even though the state keys
(`statements`, `consensus_level`, `executive_memo`, …) wouldn't actually
collide. Same lesson we hit in P1 (critic → critique) and P3 (plan →
planner). Cheap insurance.
"""
from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, TypedDict

from agents.base_agent import BaseDebateAgent, Statement
from agents.synthesizer import ExecutiveMemo, SynthesizerAgent
from debate.shared_memory import DebateSharedMemory


EventCallback = Callable[[dict[str, Any]], Awaitable[None]]


async def _noop_event(_: dict[str, Any]) -> None:
    return None


class DebateState(TypedDict, total=False):
    problem: str
    session_id: str
    statements: list[dict[str, Any]]
    consensus_level: float
    executive_memo: dict[str, Any]
    debate_complete: bool


def _statements_to_models(state: DebateState) -> list[Statement]:
    return [Statement(**s) for s in state.get("statements", [])]


async def _persist_and_emit(
    *,
    statement: Statement,
    state: DebateState,
    shared_memory: DebateSharedMemory,
    on_event: EventCallback,
) -> dict[str, Any]:
    record = statement.model_dump()
    await shared_memory.append(state["session_id"], record)
    await on_event({
        "type": "agent_statement",
        "agent": statement.role,
        "round": statement.round,
        "stance": statement.stance,
        "confidence": statement.confidence,
        "content": statement.content,
        "key_points": statement.key_points,
    })
    return record


def build_debate_graph(
    *,
    agents: list[BaseDebateAgent],
    synthesizer: SynthesizerAgent,
    shared_memory: DebateSharedMemory,
    on_event: EventCallback | None = None,
):
    """Compile the four-node debate graph.

    Returns a compiled LangGraph app whose `ainvoke({"problem": ...,
    "session_id": ...})` drives the full debate.
    """
    from langgraph.graph import END, StateGraph  # lazy

    on_event = on_event or _noop_event

    async def opening_round(state: DebateState) -> dict[str, Any]:
        await on_event({"type": "round_start", "round": 1})
        history = _statements_to_models(state)
        results = await asyncio.gather(*[
            agent.generate_statement(
                problem=state["problem"],
                debate_history=history,
                round_type="opening",
                round_number=1,
            )
            for agent in agents
        ])
        records = []
        for s in results:
            records.append(await _persist_and_emit(
                statement=s, state=state, shared_memory=shared_memory, on_event=on_event,
            ))
        await on_event({"type": "round_complete", "round": 1, "count": len(records)})
        return {"statements": list(state.get("statements", [])) + records}

    async def rebuttal_round(state: DebateState) -> dict[str, Any]:
        await on_event({"type": "round_start", "round": 2})
        running = list(state.get("statements", []))
        # Sequential: each agent reads everything written so far, including
        # the in-progress rebuttals from prior agents.
        for agent in agents:
            history = [Statement(**s) for s in running]
            statement = await agent.generate_statement(
                problem=state["problem"],
                debate_history=history,
                round_type="rebuttal",
                round_number=2,
            )
            running.append(await _persist_and_emit(
                statement=statement, state=state,
                shared_memory=shared_memory, on_event=on_event,
            ))
        await on_event({"type": "round_complete", "round": 2, "count": len(agents)})
        return {"statements": running}

    async def final_round(state: DebateState) -> dict[str, Any]:
        await on_event({"type": "round_start", "round": 3})
        history = _statements_to_models(state)
        results = await asyncio.gather(*[
            agent.generate_statement(
                problem=state["problem"],
                debate_history=history,
                round_type="final",
                round_number=3,
            )
            for agent in agents
        ])
        running = list(state.get("statements", []))
        for s in results:
            running.append(await _persist_and_emit(
                statement=s, state=state, shared_memory=shared_memory, on_event=on_event,
            ))
        await on_event({"type": "round_complete", "round": 3, "count": len(results)})
        return {"statements": running}

    async def synthesis(state: DebateState) -> dict[str, Any]:
        await on_event({"type": "synthesis_start"})
        all_statements = _statements_to_models(state)
        memo: ExecutiveMemo = await synthesizer.synthesize(
            problem=state["problem"],
            all_statements=all_statements,
        )
        await on_event({
            "type": "debate_complete",
            "consensus": memo.consensus_level,
            "executive_memo": memo.model_dump(),
        })
        return {
            "consensus_level": memo.consensus_level,
            "executive_memo": memo.model_dump(),
            "debate_complete": True,
        }

    graph = StateGraph(DebateState)
    graph.add_node("opening_round", opening_round)
    graph.add_node("rebuttal_round", rebuttal_round)
    graph.add_node("final_round", final_round)
    graph.add_node("synthesis", synthesis)
    graph.set_entry_point("opening_round")
    graph.add_edge("opening_round", "rebuttal_round")
    graph.add_edge("rebuttal_round", "final_round")
    graph.add_edge("final_round", "synthesis")
    graph.add_edge("synthesis", END)
    return graph.compile()


async def run(
    *,
    graph: Any,
    problem: str,
    session_id: str,
    shared_memory: DebateSharedMemory,
    on_event: EventCallback | None = None,
) -> dict[str, Any]:
    """Drive the graph. The `shared_memory` argument is also persisted
    as the debate's metadata so the API layer can render history later."""
    on_event = on_event or _noop_event
    await shared_memory.set_metadata(session_id, problem)
    await on_event({"type": "debate_start", "session_id": session_id, "problem": problem})
    state: DebateState = {
        "problem": problem,
        "session_id": session_id,
        "statements": [],
    }
    return await graph.ainvoke(state)
