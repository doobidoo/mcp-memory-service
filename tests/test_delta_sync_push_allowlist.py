"""MCP_SYNC_PUSH_ALLOWED_AGENTS must not reject events the hub already holds (#1522).

A spoke pushes every event in its log, including the ones it pulled from the hub. The hub
stamps its own events `unknown` when it has no MCP_AGENT_ID, so an allow-list naming only the
spokes rejected the first batch carrying a pulled event, and the spoke's push stalled on every
cycle. These tests drive the real POST /api/sync/events handler against real sqlite-vec storage.
"""
import os

import pytest
from fastapi import HTTPException

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.storage.sync.apply import apply_remote_event
from mcp_memory_service.storage.sync.orchestrator import _read_local_events_since
from mcp_memory_service.utils.hashing import generate_content_hash
from mcp_memory_service.web.api.sync_events import IngestEventsRequest, ingest_sync_events


@pytest.fixture(autouse=True)
def _eventlog_on(monkeypatch):
    monkeypatch.setenv("MCP_SYNC_EVENTLOG", "on")


async def _store(storage, content):
    h = generate_content_hash(content)
    ok, msg = await storage.store(Memory(content=content, content_hash=h, tags=["t"], memory_type="note"))
    assert ok, msg
    return h


async def _hub_and_spoke(temp_db_path, monkeypatch):
    hub = SqliteVecMemoryStorage(os.path.join(temp_db_path, "hub.db"))
    await hub.initialize()
    spoke = SqliteVecMemoryStorage(os.path.join(temp_db_path, "spoke.db"))
    await spoke.initialize()

    # The hub writes with no agent id, as a hub started per the guide does.
    monkeypatch.delenv("MCP_AGENT_ID", raising=False)
    await _store(hub, "Memory the hub wrote itself via the hybrid queue")
    hub_events = _read_local_events_since(hub, 0, 100)
    assert [e["agent_id"] for e in hub_events] == ["unknown"]

    # The spoke pulls it, then writes its own memory as `laptop`.
    for ev in hub_events:
        assert apply_remote_event(spoke, ev).applied
    monkeypatch.setenv("MCP_AGENT_ID", "laptop")
    await _store(spoke, "Memory the laptop wrote")
    return hub, spoke


@pytest.mark.asyncio
async def test_allowlist_accepts_events_the_hub_already_holds(temp_db_path, monkeypatch):
    hub, spoke = await _hub_and_spoke(temp_db_path, monkeypatch)
    monkeypatch.setenv("MCP_SYNC_PUSH_ALLOWED_AGENTS", "laptop")

    batch = _read_local_events_since(spoke, 0, 100)
    assert sorted(e["agent_id"] for e in batch) == ["laptop", "unknown"]

    resp = await ingest_sync_events(IngestEventsRequest(events=batch), True, hub)

    assert resp.failed == 0
    assert resp.applied == 1  # the laptop's memory
    content = hub.conn.execute(
        "SELECT content FROM memories WHERE content_hash = ?",
        (generate_content_hash("Memory the laptop wrote"),),
    ).fetchone()
    assert content is not None


@pytest.mark.asyncio
async def test_allowlist_still_rejects_new_events_from_unlisted_agents(temp_db_path, monkeypatch):
    hub, spoke = await _hub_and_spoke(temp_db_path, monkeypatch)
    monkeypatch.setenv("MCP_SYNC_PUSH_ALLOWED_AGENTS", "desktop")

    batch = _read_local_events_since(spoke, 0, 100)
    with pytest.raises(HTTPException) as exc:
        await ingest_sync_events(IngestEventsRequest(events=batch), True, hub)

    assert exc.value.status_code == 403
    assert "laptop" in exc.value.detail
    assert hub.conn.execute(
        "SELECT COUNT(*) FROM memories WHERE content_hash = ?",
        (generate_content_hash("Memory the laptop wrote"),),
    ).fetchone()[0] == 0


@pytest.mark.asyncio
async def test_allowlist_rejects_padded_copy_of_a_held_identity(temp_db_path, monkeypatch):
    """A padded agent_id must not borrow a held identity's exemption: apply would see it as new."""
    hub, spoke = await _hub_and_spoke(temp_db_path, monkeypatch)
    monkeypatch.setenv("MCP_SYNC_PUSH_ALLOWED_AGENTS", "laptop")

    held = _read_local_events_since(hub, 0, 100)[0]
    forged = dict(held, agent_id=" unknown ", op="delete", payload={"deleted_at": 1.0})

    with pytest.raises(HTTPException) as exc:
        await ingest_sync_events(IngestEventsRequest(events=[forged]), True, hub)

    assert exc.value.status_code == 403
    assert hub.conn.execute(
        "SELECT deleted_at FROM memories WHERE content_hash = ?", (held["content_hash"],)
    ).fetchone()[0] is None
