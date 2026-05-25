"""GraphRAG query engine.

Pipeline:
  1. Extract query entities (delegated to EntityExtractor).
  2. For each query entity, vector-match against `GraphStore` (label-
     scoped) to find seed nodes.
  3. Traverse the graph up to `max_hops` from the seeds.
  4. Fuse the vector hits with the graph-traversal hits via RRF.
  5. Build a context block and call the synthesizer model.
  6. Return a GraphRAGAnswer that cites the fused node ids.

The model is injected (production: ChatAnthropic; tests: ScriptedLLM)
so the same code path runs without API keys.
"""
from __future__ import annotations

import json
from typing import Any, Awaitable, Callable

from pydantic import BaseModel

from graph.neo4j_client import GraphStore
from graph.rrf import reciprocal_rank_fusion
from ingestion.entity_extractor import EntityExtractor


EmbedFn = Callable[[str], Awaitable[list[float]]]


SYNTH_SYSTEM_PROMPT = """You answer a question using ONLY the knowledge-graph
context provided in the user message. Cite the node ids you used as
support — they appear inside <nodes> entries.

Reply with ONE JSON object — no prose before or after — of the form:

{
  "answer": "<short answer>",
  "cited_node_ids": ["<id>", "<id>", ...],
  "confidence": <float in [0, 1]>
}

If the context does not contain enough information, set
confidence < 0.4 and explain in the answer what was missing.
"""


class GraphRAGAnswer(BaseModel):
    answer: str
    reasoning_path: list[str]
    cited_nodes: list[dict[str, Any]]
    confidence: float


class GraphRAGEngine:
    def __init__(
        self,
        *,
        graph: GraphStore,
        extractor: EntityExtractor,
        synthesizer_model: Any,
        embed: EmbedFn,
        entity_labels: list[str] | None = None,
    ) -> None:
        self.graph = graph
        self.extractor = extractor
        self.synthesizer_model = synthesizer_model
        self.embed = embed
        # Labels to search when matching query entities to graph nodes.
        self.entity_labels = list(entity_labels or [
            "person", "organization", "project", "concept", "event",
            "document", "date", "amount",
        ])

    async def query(self, question: str, max_hops: int = 2) -> GraphRAGAnswer:
        # 1. Extract query entities
        extracted = await self.extractor.extract(
            document_text=question, document_id="__query__",
        )
        # 2. Vector-match each query entity across the configured labels
        vector_hits: list[dict[str, Any]] = []
        seen: set[str] = set()
        for entity in extracted.entities:
            emb = await self.embed(entity.name)
            for label in self.entity_labels:
                hits = await self.graph.vector_search(label=label, embedding=emb, k=5)
                for h in hits:
                    if h["id"] not in seen:
                        seen.add(h["id"])
                        vector_hits.append(h)
        # Fallback: if extractor returned no entities, embed the whole question
        if not vector_hits:
            emb = await self.embed(question)
            for label in self.entity_labels:
                for h in await self.graph.vector_search(label=label, embedding=emb, k=5):
                    if h["id"] not in seen:
                        seen.add(h["id"])
                        vector_hits.append(h)

        vector_ranked = [h["id"] for h in vector_hits]

        # 3. Traverse the graph from the top vector hits
        seed_ids = vector_ranked[:5]
        traversal = await self.graph.traverse(start_node_ids=seed_ids, max_hops=max_hops)
        traversal_nodes: list[dict[str, Any]] = []
        if traversal:
            traversal_nodes = traversal[0].get("nodes", [])
        # Order traversal nodes by hop distance from seeds via BFS — preserve
        # the seed order first, then their neighbors.
        traversal_ranked = [n["id"] for n in traversal_nodes]
        # Move seeds to the front to give them rank-1 weight
        traversal_ranked = (
            [nid for nid in seed_ids if nid in traversal_ranked]
            + [nid for nid in traversal_ranked if nid not in seed_ids]
        )

        # 4. RRF fuse the two lists
        fused = reciprocal_rank_fusion([vector_ranked, traversal_ranked])
        fused_ids = [doc_id for doc_id, _ in fused]

        # 5. Build context + call synthesizer
        node_index = {n["id"]: n for n in traversal_nodes}
        for h in vector_hits:
            node_index.setdefault(h["id"], h)
        context_nodes = [node_index[nid] for nid in fused_ids if nid in node_index]
        edges_walked = traversal[0].get("edges", []) if traversal else []

        synth_payload = {
            "question": question,
            "nodes": [
                {"id": n["id"], "label": n["label"], "properties": n["properties"]}
                for n in context_nodes
            ],
            "edges": [
                {"source_id": e["source_id"], "target_id": e["target_id"], "type": e["type"]}
                for e in edges_walked
            ],
        }
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            sys_msg: Any = SystemMessage(content=SYNTH_SYSTEM_PROMPT)
            user_factory = lambda body: HumanMessage(content=body)
        except ImportError:  # pragma: no cover
            sys_msg = {"role": "system", "content": SYNTH_SYSTEM_PROMPT}
            user_factory = lambda body: {"role": "user", "content": body}

        response = await self.synthesizer_model.ainvoke([
            sys_msg, user_factory(json.dumps(synth_payload, indent=2)),
        ])
        from ingestion.entity_extractor import _extract_json
        payload = _extract_json(response.content)
        cited_ids = [str(i) for i in payload.get("cited_node_ids", [])]
        cited_nodes = [node_index[i] for i in cited_ids if i in node_index]
        return GraphRAGAnswer(
            answer=str(payload.get("answer", "")),
            reasoning_path=[n["id"] for n in context_nodes],
            cited_nodes=cited_nodes,
            confidence=float(payload.get("confidence", 0.5)),
        )
