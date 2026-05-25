"""Tests for PlannerAgent — turns a natural-language goal into a
Pydantic-validated DAG."""
from __future__ import annotations

import pytest

from planner.planner_agent import PlannerAgent
from planner.workflow_memory import InMemoryWorkflowMemory
from planner.dag_parser import DAG, DAGNode
from tests.conftest import ScriptedLLM, fake_embed


CATALOGUE = {
    "github": {"actions": ["open_pr", "comment_pr"]},
    "slack":  {"actions": ["send_message"]},
}


@pytest.mark.asyncio
async def test_plan_parses_clean_json_into_a_dag():
    llm = ScriptedLLM(planner_responses=[{
        "goal": "open a PR and notify the team",
        "nodes": [
            {"id": "n1", "name": "open PR", "tool": "github", "action": "open_pr",
             "params": {"title": "fix bug"}, "requires_approval": True, "depends_on": []},
            {"id": "n2", "name": "notify Slack", "tool": "slack", "action": "send_message",
             "params": {"channel": "#eng", "message": "PR opened"}, "requires_approval": False,
             "depends_on": ["n1"]},
        ],
        "estimated_duration_minutes": 5,
    }])
    mem = InMemoryWorkflowMemory(embed=fake_embed)
    planner = PlannerAgent(model=llm, workflow_memory=mem, tool_catalogue=CATALOGUE)
    dag = await planner.plan("open a PR and notify the team")
    assert isinstance(dag, DAG)
    assert [n.id for n in dag.nodes] == ["n1", "n2"]
    assert dag.nodes[0].requires_approval is True


@pytest.mark.asyncio
async def test_plan_seeds_prompt_with_similar_past_workflows():
    mem = InMemoryWorkflowMemory(embed=fake_embed)
    # Pre-seed memory with a past PR workflow
    past = DAG(goal="open a pull request for the auth fix", nodes=[
        DAGNode(id="x1", name="open", tool="github", action="open_pr"),
    ])
    await mem.save(goal=past.goal, dag=past, metrics={})

    llm = ScriptedLLM(planner_responses=[{
        "goal": "open a pull request for the new bug fix",
        "nodes": [{"id": "n1", "name": "open", "tool": "github", "action": "open_pr",
                   "params": {}, "requires_approval": True, "depends_on": []}],
        "estimated_duration_minutes": 2,
    }])
    planner = PlannerAgent(model=llm, workflow_memory=mem, tool_catalogue=CATALOGUE)
    await planner.plan("open a pull request for the new bug fix")

    # The planner should have shown the past workflow to the LLM
    planner_call = next(c for c in llm.calls if c[0] == "planner")
    body = planner_call[1]
    assert "open a pull request for the auth fix" in body
    assert "similar_past_workflows" in body


@pytest.mark.asyncio
async def test_plan_retries_on_invalid_json():
    llm = ScriptedLLM(planner_responses=[
        "not json at all",
        {"goal": "x", "nodes": [
            {"id": "n1", "name": "n1", "tool": "github", "action": "open_pr",
             "params": {}, "requires_approval": False, "depends_on": []},
        ], "estimated_duration_minutes": 1},
    ])
    mem = InMemoryWorkflowMemory(embed=fake_embed)
    planner = PlannerAgent(model=llm, workflow_memory=mem, tool_catalogue=CATALOGUE)
    dag = await planner.plan("x")
    assert len(dag.nodes) == 1
