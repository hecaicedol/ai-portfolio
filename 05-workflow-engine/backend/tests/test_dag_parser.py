"""Tests for the DAG parser + topological layering."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from planner.dag_parser import DAG, DAGNode, topological_layers


def _node(nid: str, deps: list[str] | None = None, tool: str = "github", action: str = "open_pr") -> DAGNode:
    return DAGNode(id=nid, name=nid, tool=tool, action=action, depends_on=deps or [])


def test_dag_round_trips_valid():
    dag = DAG(goal="ship feature", nodes=[
        _node("n1"),
        _node("n2", ["n1"]),
        _node("n3", ["n1"]),
        _node("n4", ["n2", "n3"]),
    ])
    assert dag.goal == "ship feature"
    assert len(dag.nodes) == 4


def test_dag_rejects_duplicate_ids():
    with pytest.raises(ValidationError, match="duplicate node ids"):
        DAG(goal="x", nodes=[_node("n1"), _node("n1")])


def test_dag_rejects_unknown_dependency():
    with pytest.raises(ValidationError, match="depends on unknown node"):
        DAG(goal="x", nodes=[_node("n1", ["ghost"])])


def test_dag_rejects_cycle():
    with pytest.raises(ValidationError, match="cycle"):
        DAG(goal="x", nodes=[
            _node("n1", ["n2"]),
            _node("n2", ["n1"]),
        ])


def test_topological_layers_groups_parallel_nodes():
    # n2, n3 can both run after n1 in the SAME layer
    dag = DAG(goal="x", nodes=[
        _node("n1"),
        _node("n2", ["n1"]),
        _node("n3", ["n1"]),
        _node("n4", ["n2", "n3"]),
    ])
    layers = topological_layers(dag)
    assert layers[0] == ["n1"]
    assert set(layers[1]) == {"n2", "n3"}
    assert layers[2] == ["n4"]
