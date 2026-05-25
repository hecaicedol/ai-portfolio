"""Human-in-the-loop approval broker.

Two backends, one Protocol:

  * `InMemoryHITLBroker` — asyncio.Event per (workflow, node). Used by
    tests and single-process deployments.
  * `RedisHITLBroker` — pub/sub backend for multi-worker deployments
    (stubbed for Slice 2).
"""
from __future__ import annotations

import asyncio
from typing import Any, Protocol


_Key = tuple[str, str]


class HITLBroker(Protocol):
    async def request(self, *, workflow_id: str, node_id: str, payload: dict[str, Any]) -> None: ...
    async def wait_for(self, *, workflow_id: str, node_id: str) -> dict[str, Any]: ...
    async def resolve(self, *, workflow_id: str, node_id: str, approved: bool,
                      edited_params: dict[str, Any] | None = None) -> None: ...
    async def pending(self, workflow_id: str | None = None) -> list[dict[str, Any]]: ...


class InMemoryHITLBroker:
    def __init__(self) -> None:
        self._events: dict[_Key, asyncio.Event] = {}
        self._results: dict[_Key, dict[str, Any]] = {}
        self._payloads: dict[_Key, dict[str, Any]] = {}

    async def request(self, *, workflow_id: str, node_id: str, payload: dict[str, Any]) -> None:
        key = (workflow_id, node_id)
        self._events[key] = asyncio.Event()
        self._payloads[key] = dict(payload)

    async def wait_for(self, *, workflow_id: str, node_id: str) -> dict[str, Any]:
        key = (workflow_id, node_id)
        ev = self._events.get(key)
        if ev is None:
            raise RuntimeError(f"wait_for called without prior request for {key}")
        await ev.wait()
        return self._results.pop(key, {"approved": False, "edited_params": None})

    async def resolve(
        self,
        *,
        workflow_id: str,
        node_id: str,
        approved: bool,
        edited_params: dict[str, Any] | None = None,
    ) -> None:
        key = (workflow_id, node_id)
        if key not in self._events:
            raise RuntimeError(f"resolve called with no pending request for {key}")
        self._results[key] = {"approved": approved, "edited_params": edited_params}
        self._payloads.pop(key, None)
        self._events[key].set()

    async def pending(self, workflow_id: str | None = None) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for (wf, nid), payload in self._payloads.items():
            if workflow_id is None or wf == workflow_id:
                out.append({"workflow_id": wf, "node_id": nid, "payload": dict(payload)})
        return out


class RedisHITLBroker:
    """Pub/sub backend. Stubbed for Slice 2."""

    def __init__(self, redis: Any) -> None:
        self.redis = redis

    async def request(self, *, workflow_id, node_id, payload):
        raise NotImplementedError("RedisHITLBroker is Slice 2")

    async def wait_for(self, *, workflow_id, node_id):
        raise NotImplementedError

    async def resolve(self, *, workflow_id, node_id, approved, edited_params=None):
        raise NotImplementedError

    async def pending(self, workflow_id=None):
        raise NotImplementedError
