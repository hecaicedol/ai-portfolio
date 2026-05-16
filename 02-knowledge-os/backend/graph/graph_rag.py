from pydantic import BaseModel

from graph.neo4j_client import Neo4jClient


class GraphRAGAnswer(BaseModel):
    answer: str
    reasoning_path: list[str]
    cited_nodes: list[dict]
    confidence: float


class GraphRAGEngine:
    """
    Query pipeline:
      1. Extract query entities with Claude (structured Pydantic output).
      2. Vector-match each query entity to nodes in Neo4j (label-scoped).
      3. Traverse the graph up to `max_hops` from matched nodes; collect
         neighbors + relationships.
      4. Build an enriched context block:
           - matched nodes (with properties)
           - relationship edges (labels + properties)
           - one-hop neighbors
      5. Call Claude to synthesize the answer, requiring it to cite node ids.
      6. Return GraphRAGAnswer with reasoning_path = ordered list of node ids
         the engine traversed.
    """

    def __init__(self, *, neo4j: Neo4jClient, anthropic_api_key: str, model: str) -> None:
        self.neo4j = neo4j
        self.anthropic_api_key = anthropic_api_key
        self.model = model

    async def query(self, question: str, max_hops: int = 2) -> GraphRAGAnswer:
        raise NotImplementedError
