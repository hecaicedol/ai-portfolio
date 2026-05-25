"""Meta-agent that turns a natural-language goal into a DAG of MCP tool
calls. The model is injected (production: ChatAnthropic; tests:
ScriptedPlannerLLM) so the same code path runs without API keys."""
from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from planner.dag_parser import DAG
from planner.workflow_memory import WorkflowMemory


PLANNER_SYSTEM_PROMPT = """You are a Workflow Planner. Given a user goal in
natural language and a list of available tools (each with its actions and
parameter schemas), produce a DAG of tasks that achieves the goal.

Available tools and actions are injected at runtime.

Output format (strict JSON):
{
  "goal": "...",
  "nodes": [
    {
      "id": "n1",
      "name": "human-readable step name",
      "tool": "github|jira|slack|gdrive",
      "action": "<one of the tool's actions>",
      "params": {...},          // can use {{nX.field}} to reference upstream output
      "requires_approval": bool, // true for any node that writes to a shared system
      "depends_on": ["n2", ...]
    }
  ],
  "estimated_duration_minutes": int
}

Rules:
- Mark `requires_approval: true` for any write/destructive action.
- Reuse the structure of past successful workflows when relevant ones are provided.
- Do NOT invent tools or actions outside the available list.
- Reply with ONE JSON object — no prose before or after.
"""


def _extract_json(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```\s*$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


class PlannerAgent:
    """Plans a DAG for a goal, optionally seeded with similar past workflows."""

    MAX_JSON_RETRIES: int = 2

    def __init__(
        self,
        *,
        model: Any,
        workflow_memory: WorkflowMemory,
        tool_catalogue: dict[str, Any],
    ) -> None:
        self.model = model
        self.workflow_memory = workflow_memory
        self.tool_catalogue = tool_catalogue

    async def plan(self, goal: str) -> DAG:
        similar = await self.workflow_memory.find_similar(goal, k=3)
        catalogue_text = json.dumps(self.tool_catalogue, indent=2, sort_keys=True)
        if similar:
            seed = "\n".join(
                f"- goal={s['goal']!r} · {len(s['dag']['nodes'])} nodes · sim={s.get('similarity', 0):.2f}"
                for s in similar
            )
        else:
            seed = "(no similar past workflows)"

        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            sys_msg: Any = SystemMessage(content=PLANNER_SYSTEM_PROMPT)
            user_factory = lambda body: HumanMessage(content=body)
        except ImportError:  # pragma: no cover
            sys_msg = {"role": "system", "content": PLANNER_SYSTEM_PROMPT}
            user_factory = lambda body: {"role": "user", "content": body}

        body = (
            f"<goal>{goal}</goal>\n\n"
            f"<tool_catalogue>\n{catalogue_text}\n</tool_catalogue>\n\n"
            f"<similar_past_workflows>\n{seed}\n</similar_past_workflows>"
        )

        last_error: Exception | None = None
        for _ in range(self.MAX_JSON_RETRIES + 1):
            response = await self.model.ainvoke([sys_msg, user_factory(body)])
            try:
                payload = _extract_json(response.content)
                return DAG(**payload)
            except (json.JSONDecodeError, ValidationError, KeyError, TypeError) as exc:
                last_error = exc
                continue
        raise RuntimeError(
            f"PlannerAgent: invalid JSON after {self.MAX_JSON_RETRIES + 1} attempts "
            f"(last error: {last_error})"
        )
