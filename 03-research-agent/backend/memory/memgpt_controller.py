from typing import Any

from memory.working_memory import WorkingMemory


class MemGPTController:
    """
    Orchestrates the three memory tiers.

    Responsibilities:
      - Add a new piece of info to working memory (auto-archives evicted entries to episodic).
      - Retrieve from any tier (working / episodic / semantic).
      - Build a single `get_full_context(query)` string for the LLM,
        combining working memory + relevant episodic + relevant semantic.
      - Consolidate at session end: extract durable facts → semantic memory.
    """

    def __init__(self, *, working: WorkingMemory, episodic, semantic, session_id: str) -> None:
        self.working = working
        self.episodic = episodic
        self.semantic = semantic
        self.session_id = session_id

    async def remember(self, *, key: str, content: str, kind: str) -> None:
        evicted = self.working.add(key=key, content=content, kind=kind)
        for e in evicted:
            await self.episodic.archive(
                session_id=self.session_id, kind=e.kind, content=e.content
            )

    async def get_full_context(self, query: str, *, episodic_k: int = 3, semantic_k: int = 5) -> str:
        working_ctx = self.working.get_context()
        episodic_hits = await self.episodic.retrieve_relevant_sessions(query, k=episodic_k)
        semantic_hits = await self.semantic.retrieve_relevant_knowledge(query, k=semantic_k)

        return (
            "# Working memory\n"
            f"{working_ctx}\n\n"
            "# Relevant past sessions\n"
            + "\n".join(f"- {h.get('summary')}" for h in episodic_hits)
            + "\n\n# Relevant durable knowledge\n"
            + "\n".join(f"- ({h.get('confidence')}) {h.get('fact')} [src: {h.get('source')}]" for h in semantic_hits)
        )

    async def consolidate(self, *, summary: str, key_findings: list[dict[str, Any]]) -> None:
        await self.episodic.save_session(
            session_id=self.session_id, summary=summary, key_findings=key_findings
        )
        for finding in key_findings:
            await self.semantic.upsert_fact(
                fact=finding["fact"],
                source=finding.get("source", ""),
                confidence=finding.get("confidence", 0.5),
            )
