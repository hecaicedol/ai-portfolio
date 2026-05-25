"""LangGraph research-agent state machine.

Flow
    plan_node       — model reads controller.get_full_context(question)
                      (working + relevant episodic + relevant semantic),
                      emits a JSON plan of {step, tool, query} dicts.
    execute_node    — runs the current step's tool (web / arxiv); stores
                      finding via controller.remember().
    route_node      — conditional: more steps remain → execute again,
                      else → reflect.
    reflect_node    — model summarizes findings into durable facts; the
                      controller consolidates → episodic.save_session +
                      semantic.upsert_fact for each finding.
    report_node     — generates a structured PDF via ReportGenerator
                      and stores the path in state.

The model and tools are injected so the whole graph can be exercised
end-to-end in tests with a ScriptedLLM + a fake tool registry.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from memory.memgpt_controller import MemGPTController
from tools.report_generator import ReportGenerator, ReportInput, ReportSection, ReportSource


class ResearchState(TypedDict, total=False):
    session_id: str
    question: str
    plan: list[dict[str, Any]]
    current_step: int
    findings: list[dict[str, Any]]
    consolidated: list[dict[str, Any]]
    report_path: str
    done: bool


PLAN_SYSTEM = """You are a research-planning agent.

Given a research question, return a JSON array of 2–5 steps. Each step
is an object with EXACTLY these keys:
  - "tool"   : one of "web", "arxiv"
  - "query"  : the search query string

Output ONLY the JSON array, no prose, no markdown fences.
Example:
[{"tool":"arxiv","query":"reciprocal rank fusion"},
 {"tool":"web","query":"production RAG retrieval benchmarks 2026"}]
"""


REFLECT_SYSTEM = """You are a research-consolidation agent.

You will receive the original question, the working-memory context, and
the findings collected this session. Return a JSON object with:
  - "summary": one-paragraph executive summary
  - "findings": [{"fact": "...", "source": "...", "confidence": 0.0-1.0}, ...]
                Each fact is a durable claim worth storing in semantic memory.

Output ONLY the JSON, no prose, no markdown fences.
"""


@dataclass
class ToolRegistry:
    """Bundle of the tools the agent can call. Each entry is an object with
    an async .search(query, ...) that returns a list of objects with
    `.model_dump()` (Pydantic) or dict-shaped attributes."""
    web: Any
    arxiv: Any
    report: ReportGenerator


def _safe_json(text: str) -> Any:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for open_c, close_c in (("[", "]"), ("{", "}")):
            start = text.find(open_c)
            end = text.rfind(close_c)
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start : end + 1])
        raise


def _result_to_dict(r: Any) -> dict[str, Any]:
    """Normalize a tool result (Pydantic or dict) into a plain dict."""
    if hasattr(r, "model_dump"):
        return r.model_dump()
    if isinstance(r, dict):
        return r
    return {"value": str(r)}


def build_research_graph(
    *,
    controller: MemGPTController,
    tools: ToolRegistry,
    model: Any,
    max_steps: int = 5,
):
    """Build the StateGraph. `model` must expose `async .ainvoke(messages)`."""

    async def plan_node(state: ResearchState) -> dict[str, Any]:
        context = await controller.get_full_context(state["question"])
        await controller.remember(
            key="research_goal", content=state["question"], kind="goal",
        )
        response = await model.ainvoke([
            SystemMessage(content=PLAN_SYSTEM),
            HumanMessage(
                content=f"<context>\n{context}\n</context>\n\n"
                        f"Question: {state['question']}"
            ),
        ])
        plan = _safe_json(response.content)
        if not isinstance(plan, list):
            plan = []
        plan = plan[:max_steps]
        await controller.remember(
            key="research_plan",
            content=json.dumps(plan, indent=2),
            kind="plan",
        )
        return {"plan": plan, "current_step": 0, "findings": []}

    async def execute_node(state: ResearchState) -> dict[str, Any]:
        idx = state.get("current_step", 0)
        plan = state.get("plan", [])
        if idx >= len(plan):
            return {"current_step": idx}

        step = plan[idx]
        tool_name = step.get("tool", "web")
        query = step.get("query", state["question"])

        if tool_name == "arxiv":
            raw_results = await tools.arxiv.search(query, max_results=5)
        else:
            raw_results = await tools.web.search(query, k=5)

        results = [_result_to_dict(r) for r in raw_results]
        finding = {
            "step": idx,
            "tool": tool_name,
            "query": query,
            "results": results,
        }
        await controller.remember(
            key=f"finding_{idx}",
            content=json.dumps(finding, default=str)[:4000],
            kind="tool_output",
        )

        existing = state.get("findings", [])
        return {
            "findings": existing + [finding],
            "current_step": idx + 1,
        }

    def route_after_execute(state: ResearchState) -> str:
        idx = state.get("current_step", 0)
        plan = state.get("plan", [])
        return "execute" if idx < len(plan) else "reflect"

    async def reflect_node(state: ResearchState) -> dict[str, Any]:
        context = controller.working.get_context()
        findings_text = json.dumps(state.get("findings", []), default=str)[:6000]
        response = await model.ainvoke([
            SystemMessage(content=REFLECT_SYSTEM),
            HumanMessage(
                content=f"<question>{state['question']}</question>\n"
                        f"<working_memory>\n{context}\n</working_memory>\n"
                        f"<findings>\n{findings_text}\n</findings>"
            ),
        ])
        consolidation = _safe_json(response.content)
        if not isinstance(consolidation, dict):
            consolidation = {"summary": "", "findings": []}

        summary = consolidation.get("summary", "") or "(no summary)"
        durable = consolidation.get("findings", []) or []

        await controller.consolidate(summary=summary, key_findings=durable)
        return {"consolidated": durable}

    async def report_node(state: ResearchState) -> dict[str, Any]:
        sections: list[ReportSection] = []
        sources: list[ReportSource] = []
        for finding in state.get("findings", []):
            heading = f"Step {finding['step'] + 1} — {finding['tool']}: {finding['query']}"
            body_parts = []
            for r in finding["results"]:
                title = r.get("title") or r.get("url") or "(no title)"
                snippet = r.get("content") or r.get("abstract") or ""
                body_parts.append(f"- {title}: {snippet[:240]}")
                if r.get("url"):
                    sources.append(ReportSource(
                        title=str(title)[:120],
                        url=str(r.get("url"))[:300],
                        credibility=float(r.get("relevance_score", 0.5)),
                    ))
            sections.append(ReportSection(heading=heading, body="\n".join(body_parts)))

        durable_block = "\n".join(
            f"- ({f.get('confidence', 0.5):.2f}) {f.get('fact', '')}"
            for f in state.get("consolidated", [])
        ) or "(none)"

        payload = ReportInput(
            title=f"Research report — {state['question'][:80]}",
            executive_summary=durable_block,
            methodology=(
                f"{len(state.get('plan', []))}-step plan executed across "
                f"{sum(1 for f in state.get('findings', []))} tool calls."
            ),
            sections=sections,
            sources=sources,
            confidence_assessment="Stored as durable facts; see semantic memory.",
        )
        path = tools.report.generate(payload, session_id=state["session_id"])
        return {"report_path": str(path), "done": True}

    # LangGraph forbids node names that collide with state keys, and the
    # state has `plan`, `findings`, `consolidated`, `report_path` already.
    # Node names use verbs to avoid collisions.
    graph = StateGraph(ResearchState)
    graph.add_node("planner", plan_node)
    graph.add_node("executor", execute_node)
    graph.add_node("reflector", reflect_node)
    graph.add_node("reporter", report_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "executor")
    graph.add_conditional_edges(
        "executor",
        route_after_execute,
        {"execute": "executor", "reflect": "reflector"},
    )
    graph.add_edge("reflector", "reporter")
    graph.add_edge("reporter", END)

    return graph.compile()


async def run(
    *,
    question: str,
    session_id: str,
    controller: MemGPTController,
    tools: ToolRegistry,
    model: Any,
    max_steps: int = 5,
) -> dict[str, Any]:
    graph = build_research_graph(
        controller=controller, tools=tools, model=model, max_steps=max_steps
    )
    state: ResearchState = {"session_id": session_id, "question": question}
    final = await graph.ainvoke(state)
    return dict(final)
