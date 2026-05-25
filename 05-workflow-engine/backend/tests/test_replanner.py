"""Tests for the Replanner. The splice math is tested in
test_executor.py — this file only checks the JSON contract."""
from __future__ import annotations

import pytest

from executor.replanner import Replanner
from planner.dag_parser import DAG, DAGNode
from tests.conftest import ScriptedLLM


CATALOGUE = {"github": {"actions": ["open_pr"]}, "slack": {"actions": ["send_message"]}}


@pytest.mark.asyncio
async def test_replan_returns_replacement_nodes():
    failed = DAGNode(id="n2", name="open PR", tool="github", action="open_pr", depends_on=["n1"])
    dag = DAG(goal="x", nodes=[
        DAGNode(id="n1", name="prep", tool="github", action="open_pr"),
        failed,
    ])
    llm = ScriptedLLM(replanner_responses=[{
        "nodes": [
            {"id": "r1", "name": "retry via slack", "tool": "slack",
             "action": "send_message", "params": {}, "requires_approval": False, "depends_on": []},
        ],
    }])
    replanner = Replanner(model=llm, tool_catalogue=CATALOGUE)
    out = await replanner.replan_node(failed_node=failed, dag=dag, error="rate limited")
    assert [n.id for n in out] == ["r1"]
    # The failed node and error were both visible to the LLM
    body = llm.calls[0][1]
    assert "rate limited" in body
    assert "n2" in body


@pytest.mark.asyncio
async def test_replan_raises_when_llm_returns_empty():
    failed = DAGNode(id="n2", name="n2", tool="github", action="open_pr")
    dag = DAG(goal="x", nodes=[failed])
    llm = ScriptedLLM(replanner_responses=[{"nodes": []}, {"nodes": []}, {"nodes": []}])
    replanner = Replanner(model=llm, tool_catalogue=CATALOGUE)
    with pytest.raises(RuntimeError, match="invalid JSON / empty replacement"):
        await replanner.replan_node(failed_node=failed, dag=dag, error="boom")
