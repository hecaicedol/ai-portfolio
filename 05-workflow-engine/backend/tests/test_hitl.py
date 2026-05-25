"""Tests for InMemoryHITLBroker — the gate that pauses a node until a
human approves, rejects, or edits its params."""
from __future__ import annotations

import asyncio

import pytest

from executor.hitl import InMemoryHITLBroker


@pytest.mark.asyncio
async def test_request_then_resolve_unblocks_wait_for():
    broker = InMemoryHITLBroker()
    await broker.request(workflow_id="w1", node_id="n1", payload={"x": 1})
    # wait_for blocks until resolve; race them
    waiter = asyncio.create_task(broker.wait_for(workflow_id="w1", node_id="n1"))
    await asyncio.sleep(0)  # let waiter actually start awaiting
    assert not waiter.done()
    await broker.resolve(workflow_id="w1", node_id="n1", approved=True, edited_params={"x": 2})
    result = await waiter
    assert result == {"approved": True, "edited_params": {"x": 2}}


@pytest.mark.asyncio
async def test_rejection_returns_approved_false():
    broker = InMemoryHITLBroker()
    await broker.request(workflow_id="w1", node_id="n1", payload={})
    waiter = asyncio.create_task(broker.wait_for(workflow_id="w1", node_id="n1"))
    await broker.resolve(workflow_id="w1", node_id="n1", approved=False)
    result = await waiter
    assert result["approved"] is False
    assert result["edited_params"] is None


@pytest.mark.asyncio
async def test_pending_lists_open_requests_only():
    broker = InMemoryHITLBroker()
    await broker.request(workflow_id="w1", node_id="n1", payload={"a": 1})
    await broker.request(workflow_id="w1", node_id="n2", payload={"a": 2})
    pending = await broker.pending("w1")
    assert {p["node_id"] for p in pending} == {"n1", "n2"}
    # Resolve one and verify it leaves the pending set
    waiter = asyncio.create_task(broker.wait_for(workflow_id="w1", node_id="n1"))
    await broker.resolve(workflow_id="w1", node_id="n1", approved=True)
    await waiter
    pending = await broker.pending("w1")
    assert {p["node_id"] for p in pending} == {"n2"}
