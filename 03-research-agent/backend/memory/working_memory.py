from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import tiktoken


@dataclass
class WorkingMemoryEntry:
    key: str
    content: str
    kind: str  # 'goal' | 'plan' | 'snippet' | 'tool_output' | 'note'
    tokens: int


class WorkingMemory:
    """
    Bounded, in-context memory. When adding an entry would exceed `max_tokens`,
    the oldest entries are evicted FIFO (returned to the caller so the controller
    can archive them to episodic memory).
    """

    def __init__(self, max_tokens: int = 12_000, model_for_tokens: str = "cl100k_base") -> None:
        self.max_tokens = max_tokens
        self._enc = tiktoken.get_encoding(model_for_tokens)
        self._entries: OrderedDict[str, WorkingMemoryEntry] = OrderedDict()

    def _count(self, text: str) -> int:
        return len(self._enc.encode(text))

    def used(self) -> int:
        return sum(e.tokens for e in self._entries.values())

    def add(self, *, key: str, content: str, kind: str) -> list[WorkingMemoryEntry]:
        tokens = self._count(content)
        evicted: list[WorkingMemoryEntry] = []
        entry = WorkingMemoryEntry(key=key, content=content, kind=kind, tokens=tokens)
        self._entries[key] = entry
        while self.used() > self.max_tokens and self._entries:
            _, oldest = self._entries.popitem(last=False)
            evicted.append(oldest)
        return evicted

    def get_context(self) -> str:
        parts = []
        for e in self._entries.values():
            parts.append(f"## [{e.kind.upper()}] {e.key}\n{e.content}")
        return "\n\n".join(parts)

    def snapshot(self) -> list[dict[str, Any]]:
        return [
            {"key": e.key, "kind": e.kind, "tokens": e.tokens, "preview": e.content[:200]}
            for e in self._entries.values()
        ]
