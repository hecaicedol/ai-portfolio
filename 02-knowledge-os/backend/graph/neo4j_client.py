"""Graph store backends.

The `GraphStore` Protocol is what `GraphRAGEngine` and `StalenessAgent`
talk to. Two implementations live here:

  * `InMemoryGraphStore` — pure-Python dicts + cosine for vector search.
    Used by tests and the no-budget demos.
  * `Neo4jClient` — production backend, lazy-imports the `neo4j` driver
    so the test suite runs without it installed.

Schema (both backends):
  nodes have an `id` (string), a `label` (entity type), a `properties`
  dict, an `embedding` (optional list[float]), and an `updated_at`
  timestamp. Relationships have `source_id`, `target_id`, a `type`
  string (UPPER_SNAKE_CASE verb), and a `properties` dict.
"""
from __future__ import annotations

import math
import time
from typing import Any, Awaitable, Callable, Protocol


class GraphStore(Protocol):
    async def ensure_vector_index(self, label: str, property_name: str = "embedding",
                                  dimensions: int = 1024) -> None: ...
    async def upsert_node(self, *, label: str, properties: dict[str, Any],
                          embedding: list[float] | None = None) -> str: ...
    async def upsert_relationship(self, *, from_id: str, to_id: str, rel_type: str,
                                  properties: dict[str, Any] | None = None) -> None: ...
    async def vector_search(self, *, label: str, embedding: list[float],
                            k: int = 10) -> list[dict[str, Any]]: ...
    async def traverse(self, *, start_node_ids: list[str],
                       max_hops: int = 2) -> list[dict[str, Any]]: ...
    async def snapshot(self, limit: int = 200) -> dict[str, Any]: ...
    async def stale_nodes(self, days: int = 30) -> list[dict[str, Any]]: ...
    async def close(self) -> None: ...


def _cosine(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


class InMemoryGraphStore:
    """Process-local store. Survives across requests in the same uvicorn
    worker but not across restarts. Used by tests and demos."""

    def __init__(self) -> None:
        # id -> {label, properties, embedding, updated_at}
        self._nodes: dict[str, dict[str, Any]] = {}
        # list of {source_id, target_id, type, properties}
        self._edges: list[dict[str, Any]] = []
        # label -> {"property_name", "dimensions"}
        self._indexes: dict[str, dict[str, Any]] = {}

    async def ensure_vector_index(self, label, property_name="embedding", dimensions=1024) -> None:
        self._indexes[label] = {"property_name": property_name, "dimensions": dimensions}

    async def upsert_node(self, *, label, properties, embedding=None) -> str:
        node_id = properties.get("id")
        if not node_id:
            raise ValueError("properties must include an 'id' field")
        existing = self._nodes.get(node_id)
        if existing:
            existing["label"] = label
            existing["properties"].update(properties)
            if embedding is not None:
                existing["embedding"] = list(embedding)
            existing["updated_at"] = time.time()
        else:
            self._nodes[node_id] = {
                "id": node_id,
                "label": label,
                "properties": dict(properties),
                "embedding": list(embedding) if embedding else None,
                "updated_at": time.time(),
            }
        return node_id

    async def upsert_relationship(self, *, from_id, to_id, rel_type, properties=None) -> None:
        props = dict(properties or {})
        for edge in self._edges:
            if (edge["source_id"], edge["target_id"], edge["type"]) == (from_id, to_id, rel_type):
                edge["properties"].update(props)
                edge["updated_at"] = time.time()
                return
        self._edges.append({
            "source_id": from_id, "target_id": to_id, "type": rel_type,
            "properties": props, "updated_at": time.time(),
        })

    async def vector_search(self, *, label, embedding, k=10) -> list[dict[str, Any]]:
        scored: list[tuple[float, dict[str, Any]]] = []
        for node in self._nodes.values():
            if node["label"] != label or not node.get("embedding"):
                continue
            scored.append((_cosine(embedding, node["embedding"]), node))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "id": n["id"],
                "label": n["label"],
                "properties": dict(n["properties"]),
                "score": s,
            }
            for s, n in scored[:k]
        ]

    async def traverse(self, *, start_node_ids, max_hops=2) -> list[dict[str, Any]]:
        if max_hops < 0:
            raise ValueError("max_hops must be >= 0")
        visited: set[str] = set(start_node_ids)
        frontier: set[str] = set(start_node_ids)
        edges_walked: list[dict[str, Any]] = []
        for _ in range(max_hops):
            next_frontier: set[str] = set()
            for nid in frontier:
                for edge in self._edges:
                    if edge["source_id"] == nid and edge["target_id"] not in visited:
                        next_frontier.add(edge["target_id"])
                        edges_walked.append(dict(edge))
                    elif edge["target_id"] == nid and edge["source_id"] not in visited:
                        next_frontier.add(edge["source_id"])
                        edges_walked.append(dict(edge))
            if not next_frontier:
                break
            visited.update(next_frontier)
            frontier = next_frontier
        nodes = [
            {"id": nid, "label": self._nodes[nid]["label"], "properties": dict(self._nodes[nid]["properties"])}
            for nid in visited if nid in self._nodes
        ]
        return [{"nodes": nodes, "edges": edges_walked}]

    async def snapshot(self, limit=200) -> dict[str, Any]:
        nodes = [
            {"id": n["id"], "label": n["label"], "properties": dict(n["properties"])}
            for n in list(self._nodes.values())[:limit]
        ]
        node_ids = {n["id"] for n in nodes}
        edges = [
            {"source_id": e["source_id"], "target_id": e["target_id"], "type": e["type"]}
            for e in self._edges
            if e["source_id"] in node_ids and e["target_id"] in node_ids
        ]
        return {"nodes": nodes, "edges": edges}

    async def stale_nodes(self, days=30) -> list[dict[str, Any]]:
        cutoff = time.time() - days * 86_400
        out = []
        for n in self._nodes.values():
            if n["updated_at"] < cutoff:
                out.append({
                    "id": n["id"], "label": n["label"],
                    "properties": dict(n["properties"]),
                    "updated_at": n["updated_at"],
                })
        return out

    async def close(self) -> None:
        return None


class Neo4jClient:
    """Production backend. Lazy-imports the `neo4j` driver so the dev test
    suite runs without it installed."""

    def __init__(self, uri: str, user: str, password: str) -> None:
        from neo4j import AsyncGraphDatabase  # lazy
        self._driver = AsyncGraphDatabase.driver(uri, auth=(user, password))

    async def close(self) -> None:
        await self._driver.close()

    async def ensure_vector_index(self, label, property_name="embedding", dimensions=1024) -> None:
        raise NotImplementedError("Neo4j vector index — Slice 2")

    async def upsert_node(self, *, label, properties, embedding=None) -> str:
        raise NotImplementedError

    async def upsert_relationship(self, *, from_id, to_id, rel_type, properties=None) -> None:
        raise NotImplementedError

    async def vector_search(self, *, label, embedding, k=10) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def traverse(self, *, start_node_ids, max_hops=2) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def cypher(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        async with self._driver.session() as session:
            result = await session.run(query, params or {})
            return [r.data() async for r in result]

    async def snapshot(self, limit=200) -> dict[str, Any]:
        raise NotImplementedError

    async def stale_nodes(self, days=30) -> list[dict[str, Any]]:
        raise NotImplementedError
