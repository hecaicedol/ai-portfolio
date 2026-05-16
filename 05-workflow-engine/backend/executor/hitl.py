from typing import Any


class HITLBroker:
    """
    Tracks pending Human-in-the-Loop approvals through Redis.

    Keys:
      hitl:{workflow_id}:{node_id} = {'state': 'pending'|'approved'|'rejected', 'edited_params': {...}}

    Pubsub channel `hitl:{workflow_id}` notifies the executor when an approval lands,
    so the corresponding asyncio.Event can be set and the blocked task can proceed.
    """

    def __init__(self, redis) -> None:
        self.redis = redis

    async def request(self, *, workflow_id: str, node_id: str, payload: dict[str, Any]) -> None:
        raise NotImplementedError

    async def wait_for(self, *, workflow_id: str, node_id: str) -> dict[str, Any]:
        """Awaits resolve(); returns {'approved': bool, 'edited_params': dict | None}."""
        raise NotImplementedError

    async def resolve(self, *, workflow_id: str, node_id: str, approved: bool, edited_params: dict | None = None) -> None:
        raise NotImplementedError
