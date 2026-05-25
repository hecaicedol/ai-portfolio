"""End-to-end tests for the DAGExecutor.

Wires a DAG against FakeMCPClients + InMemoryHITLBroker + ScriptedLLM
replanner + InMemoryWorkflowMemory and verifies:

- Linear DAGs execute and persist to memory
- Parallel layers fan out via asyncio.gather (one shared client sees
  both calls)
- HITL approval pauses a node until resolve(); rejection short-circuits
- Param interpolation `{{n1.field}}` flows upstream outputs into params
- A failed node triggers the replanner and the splice keeps downstream
  intact
"""
from __future__ import annotations

import asyncio
import uuid

import pytest

from executor.engine import DAGExecutor, _interpolate, _splice_replacements
from executor.hitl import InMemoryHITLBroker
from executor.replanner import Replanner
from planner.dag_parser import DAG, DAGNode
from planner.workflow_memory import InMemoryWorkflowMemory
from tests.conftest import FakeMCPClient, ScriptedLLM, fake_embed


CATALOGUE = {"github": {"actions": ["open_pr", "comment_pr"]},
             "slack":  {"actions": ["send_message"]}}


@pytest.mark.asyncio
async def test_linear_dag_executes_and_persists_to_memory():
    dag = DAG(goal="open a PR", nodes=[
        DAGNode(id="n1", name="open", tool="github", action="open_pr",
                params={"title": "fix bug"}),
        DAGNode(id="n2", name="notify", tool="slack", action="send_message",
                params={"channel": "#eng"}, depends_on=["n1"]),
    ])
    mem = InMemoryWorkflowMemory(embed=fake_embed)
    exe = DAGExecutor(
        mcp_clients={
            "github": FakeMCPClient(tool="github", outputs={"open_pr": {"pr_number": 42}}),
            "slack":  FakeMCPClient(tool="slack",  outputs={"send_message": {"ok": True}}),
        },
        replanner=None,
        hitl=InMemoryHITLBroker(),
        workflow_memory=mem,
    )
    result = await exe.run(workflow_id="w-lin", dag=dag)
    assert result["failed"] == []
    assert result["completed"]["n1"]["output"] == {"pr_number": 42}
    # Memory should hold this workflow now
    recent = await mem.recent()
    assert recent and recent[0]["goal"] == "open a PR"


@pytest.mark.asyncio
async def test_parallel_layer_fans_out():
    dag = DAG(goal="multi-notify", nodes=[
        DAGNode(id="n1", name="open", tool="github", action="open_pr"),
        DAGNode(id="n2", name="notify A", tool="slack", action="send_message",
                params={"channel": "#a"}, depends_on=["n1"]),
        DAGNode(id="n3", name="notify B", tool="slack", action="send_message",
                params={"channel": "#b"}, depends_on=["n1"]),
    ])
    slack = FakeMCPClient(tool="slack", outputs={"send_message": {"ok": True}})
    exe = DAGExecutor(
        mcp_clients={
            "github": FakeMCPClient(tool="github", outputs={"open_pr": {"pr_number": 7}}),
            "slack":  slack,
        },
        replanner=None,
        hitl=InMemoryHITLBroker(),
        workflow_memory=InMemoryWorkflowMemory(embed=fake_embed),
    )
    await exe.run(workflow_id="w-par", dag=dag)
    # The shared slack client saw both notifications
    assert len(slack.calls) == 2
    assert {c[1]["channel"] for c in slack.calls} == {"#a", "#b"}


@pytest.mark.asyncio
async def test_hitl_approval_pauses_then_resumes():
    dag = DAG(goal="approve me", nodes=[
        DAGNode(id="n1", name="risky write", tool="github", action="open_pr",
                params={"title": "production hotfix"}, requires_approval=True),
    ])
    hitl = InMemoryHITLBroker()
    exe = DAGExecutor(
        mcp_clients={"github": FakeMCPClient(tool="github", outputs={"open_pr": {"pr_number": 99}})},
        replanner=None, hitl=hitl,
        workflow_memory=InMemoryWorkflowMemory(embed=fake_embed),
    )
    runner = asyncio.create_task(exe.run(workflow_id="w-hitl", dag=dag))
    # Give the executor a chance to reach the HITL gate
    for _ in range(50):
        await asyncio.sleep(0)
        if await hitl.pending("w-hitl"):
            break
    pending = await hitl.pending("w-hitl")
    assert pending and pending[0]["node_id"] == "n1"
    # Approve, with an edited param
    await hitl.resolve(workflow_id="w-hitl", node_id="n1", approved=True,
                       edited_params={"title": "edited title"})
    result = await runner
    assert result["failed"] == []
    assert result["completed"]["n1"]["params"] == {"title": "edited title"}


@pytest.mark.asyncio
async def test_hitl_rejection_marks_node_failed():
    dag = DAG(goal="reject me", nodes=[
        DAGNode(id="n1", name="risky write", tool="github", action="open_pr",
                requires_approval=True),
    ])
    hitl = InMemoryHITLBroker()
    exe = DAGExecutor(
        mcp_clients={"github": FakeMCPClient(tool="github", outputs={"open_pr": {}})},
        replanner=None, hitl=hitl,
        workflow_memory=InMemoryWorkflowMemory(embed=fake_embed),
    )
    runner = asyncio.create_task(exe.run(workflow_id="w-rej", dag=dag))
    for _ in range(50):
        await asyncio.sleep(0)
        if await hitl.pending("w-rej"):
            break
    await hitl.resolve(workflow_id="w-rej", node_id="n1", approved=False)
    result = await runner
    assert result["failed"] == ["n1"]
    assert "n1" not in result["completed"]


@pytest.mark.asyncio
async def test_param_interpolation_resolves_upstream_output():
    dag = DAG(goal="cite the PR", nodes=[
        DAGNode(id="n1", name="open", tool="github", action="open_pr"),
        DAGNode(id="n2", name="notify", tool="slack", action="send_message",
                params={"channel": "#eng", "message": "PR #{{n1.pr_number}} opened"},
                depends_on=["n1"]),
    ])
    slack = FakeMCPClient(tool="slack", outputs={"send_message": {"ok": True}})
    exe = DAGExecutor(
        mcp_clients={
            "github": FakeMCPClient(tool="github", outputs={"open_pr": {"pr_number": 42}}),
            "slack":  slack,
        },
        replanner=None, hitl=InMemoryHITLBroker(),
        workflow_memory=InMemoryWorkflowMemory(embed=fake_embed),
    )
    await exe.run(workflow_id="w-interp", dag=dag)
    sent_msg = slack.calls[0][1]["message"]
    assert sent_msg == "PR #42 opened"


@pytest.mark.asyncio
async def test_failed_node_triggers_replan_and_downstream_continues():
    dag = DAG(goal="resilient", nodes=[
        DAGNode(id="n1", name="risky", tool="github", action="open_pr"),
        DAGNode(id="n2", name="downstream", tool="slack", action="send_message",
                params={"channel": "#eng"}, depends_on=["n1"]),
    ])
    github_client = FakeMCPClient(
        tool="github",
        outputs={"comment_pr": {"comment_id": 1}},
        errors={"open_pr": RuntimeError("rate limited")},
    )
    slack_client = FakeMCPClient(tool="slack", outputs={"send_message": {"ok": True}})
    replanner_llm = ScriptedLLM(replanner_responses=[{
        "nodes": [
            {"id": "r1", "name": "fall back to comment_pr", "tool": "github",
             "action": "comment_pr", "params": {"body": "auto-opened by retry"},
             "requires_approval": False, "depends_on": []},
        ],
    }])
    exe = DAGExecutor(
        mcp_clients={"github": github_client, "slack": slack_client},
        replanner=Replanner(model=replanner_llm, tool_catalogue=CATALOGUE),
        hitl=InMemoryHITLBroker(),
        workflow_memory=InMemoryWorkflowMemory(embed=fake_embed),
    )
    result = await exe.run(workflow_id="w-replan", dag=dag)
    assert result["failed"] == []
    # The original n1 was replaced by r1, and r1 ran successfully
    completed_ids = set(result["completed"].keys())
    assert "r1" in completed_ids
    assert "n2" in completed_ids
    assert "n1" not in completed_ids
    # The spliced DAG no longer mentions n1
    dag_ids = {n["id"] for n in result["dag"]["nodes"]}
    assert "n1" not in dag_ids
    assert "r1" in dag_ids


def test_interpolate_preserves_non_string_types_when_whole_value_is_a_reference():
    completed = {"n1": {"output": {"pr_number": 42, "title": "fix bug"}}}
    # Whole string is a reference → returns the raw int
    assert _interpolate("{{n1.pr_number}}", completed) == 42
    # Embedded reference → returns a string
    assert _interpolate("PR #{{n1.pr_number}} is open", completed) == "PR #42 is open"
    # Dict of params recurses. A list element that is a SINGLE reference
    # preserves its original type (int 42) — same rule as the top-level
    # string case.
    out = _interpolate(
        {"title": "{{n1.title}}", "labels": ["bug", "{{n1.pr_number}}"]},
        completed,
    )
    assert out == {"title": "fix bug", "labels": ["bug", 42]}


def test_splice_replacements_rewires_downstream_deps():
    failed = DAGNode(id="n2", name="failed", tool="github", action="open_pr", depends_on=["n1"])
    dag = DAG(goal="x", nodes=[
        DAGNode(id="n1", name="upstream", tool="github", action="open_pr"),
        failed,
        DAGNode(id="n3", name="downstream", tool="slack", action="send_message",
                depends_on=["n2"]),
    ])
    replacements = [
        DAGNode(id="r1", name="r1", tool="github", action="open_pr"),
        DAGNode(id="r2", name="r2", tool="github", action="comment_pr", depends_on=["r1"]),
    ]
    _splice_replacements(dag, failed, replacements)
    by_id = {n.id: n for n in dag.nodes}
    assert "n2" not in by_id
    # r1 inherits failed.depends_on
    assert by_id["r1"].depends_on == ["n1"]
    # n3 used to depend on n2; should now depend on the terminal replacement r2
    assert by_id["n3"].depends_on == ["r2"]
