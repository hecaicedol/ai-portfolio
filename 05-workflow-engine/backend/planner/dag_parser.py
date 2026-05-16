from typing import Any
from pydantic import BaseModel, Field, model_validator


class DAGNode(BaseModel):
    id: str
    name: str
    tool: str
    action: str
    params: dict[str, Any] = Field(default_factory=dict)
    requires_approval: bool = False
    depends_on: list[str] = Field(default_factory=list)


class DAG(BaseModel):
    goal: str
    nodes: list[DAGNode]
    estimated_duration_minutes: int = 5

    @model_validator(mode="after")
    def _validate(self) -> "DAG":
        ids = {n.id for n in self.nodes}
        if len(ids) != len(self.nodes):
            raise ValueError("duplicate node ids")
        for n in self.nodes:
            for dep in n.depends_on:
                if dep not in ids:
                    raise ValueError(f"node {n.id} depends on unknown node {dep}")
        if _has_cycle({n.id: n.depends_on for n in self.nodes}):
            raise ValueError("DAG contains a cycle")
        return self


def _has_cycle(adj: dict[str, list[str]]) -> bool:
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {k: WHITE for k in adj}

    def dfs(u: str) -> bool:
        color[u] = GRAY
        for v in adj.get(u, []):
            if color.get(v, WHITE) == GRAY:
                return True
            if color.get(v, WHITE) == WHITE and dfs(v):
                return True
        color[u] = BLACK
        return False

    return any(color[u] == WHITE and dfs(u) for u in adj)


def topological_layers(dag: DAG) -> list[list[str]]:
    """Return groups of node ids that can execute in parallel (Kahn's algorithm)."""
    indeg = {n.id: len(n.depends_on) for n in dag.nodes}
    by_id = {n.id: n for n in dag.nodes}
    layers: list[list[str]] = []
    ready = [nid for nid, d in indeg.items() if d == 0]
    seen: set[str] = set()
    while ready:
        layers.append(ready)
        seen.update(ready)
        next_ready: list[str] = []
        for nid in ready:
            for other in dag.nodes:
                if nid in other.depends_on and other.id not in seen:
                    indeg[other.id] -= 1
                    if indeg[other.id] == 0:
                        next_ready.append(other.id)
        ready = next_ready
    return layers
