from typing import Any

from neo4j import AsyncGraphDatabase, AsyncDriver


class Neo4jClient:
    """
    Thin async wrapper over the Neo4j driver, exposing:
      - upsert_node(label, properties, embedding)
      - upsert_relationship(from_id, to_id, rel_type, properties)
      - vector_search(label, embedding, k)
      - cypher(query, params)
      - traverse(start_node_ids, max_hops)
      - snapshot(limit) — for frontend force-graph
    """

    def __init__(self, uri: str, user: str, password: str) -> None:
        self._driver: AsyncDriver = AsyncGraphDatabase.driver(uri, auth=(user, password))

    async def close(self) -> None:
        await self._driver.close()

    async def ensure_vector_index(self, label: str, property_name: str = "embedding", dimensions: int = 1024) -> None:
        """Create a Neo4j 5+ vector index on `label.property_name` if it doesn't exist."""
        raise NotImplementedError

    async def upsert_node(self, *, label: str, properties: dict[str, Any], embedding: list[float] | None = None) -> str:
        """MERGE node by `id` property; set props; set embedding. Returns node id."""
        raise NotImplementedError

    async def upsert_relationship(
        self, *, from_id: str, to_id: str, rel_type: str, properties: dict[str, Any] | None = None
    ) -> None:
        raise NotImplementedError

    async def vector_search(self, *, label: str, embedding: list[float], k: int = 10) -> list[dict[str, Any]]:
        """Top-k nodes by cosine similarity on `embedding` property."""
        raise NotImplementedError

    async def traverse(self, *, start_node_ids: list[str], max_hops: int = 2) -> list[dict[str, Any]]:
        """Return all nodes reachable within `max_hops`, with the connecting relationships."""
        raise NotImplementedError

    async def cypher(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        async with self._driver.session() as session:
            result = await session.run(query, params or {})
            return [r.data() async for r in result]

    async def snapshot(self, limit: int = 200) -> dict[str, Any]:
        """Return {'nodes': [...], 'edges': [...]} for force-graph rendering."""
        raise NotImplementedError

    async def stale_nodes(self, days: int = 30) -> list[dict[str, Any]]:
        """Nodes whose `updated_at` is older than `days` days."""
        raise NotImplementedError
