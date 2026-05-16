from typing import Any
from planner.dag_parser import DAG


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
- Reuse the structure of past successful workflows if relevant ones are provided.
- Do NOT invent tools or actions outside the available list.
"""


class PlannerAgent:
    """
    Meta-agent that:
      1. Queries workflow_memory.find_similar(goal, k=3) — past successful DAGs.
      2. Constructs a tool catalogue from MCP server introspection.
      3. Calls Claude with structured output (instructor) → DAG (Pydantic-validated).
      4. Returns DAG; caller persists & asks for approval.
    """

    def __init__(self, *, model: str, api_key: str, workflow_memory, tool_catalogue: dict[str, Any]) -> None:
        self.model = model
        self.api_key = api_key
        self.workflow_memory = workflow_memory
        self.tool_catalogue = tool_catalogue

    async def plan(self, goal: str) -> DAG:
        raise NotImplementedError
