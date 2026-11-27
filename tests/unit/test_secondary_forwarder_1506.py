"""A hybrid hub forwards delta-sync events it applied to its secondary (#1506).

``apply_remote_event`` writes through the primary's SQLite connection, so a spoke's
create, edit and delete never reached the hybrid queue and Cloudflare stayed a partial
copy. These tests run the real pieces: a SQLite-vec primary, a CloudflareStorage whose
D1 calls run on an in-memory SQLite database, and ``forward_to_secondary`` tailing the
hub's ``sync_events`` with a persisted cursor.
"""

import json
import os
import shutil
import tempfile
import time

import pytest
import pytest_asyncio

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.hybrid import BackgroundSyncService, HybridMemoryStorage
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.storage.sync.apply import apply_remote_event
from mcp_memory_service.storage.sync.forwarder import (
    forward_to_secondary,
    secondary_id_for,
)
from mcp_memory_service.storage.sync.orchestrator import get_push_cursor
from mcp_memory_service.utils.hashing import generate_content_hash
from tests.unit.test_cloudflare_store_scope import _FakeD1, cloudflare_storage, d1  # noqa: F401


@pytest.fixture(autouse=True)
def _eventlog_on(monkeypatch):
    monkeypatch.setenv("MCP_SYNC_EVENTLOG", "on")


def _build_hub(tmp, cloudflare_storage):  # noqa: F811
    hub = HybridMemoryStorage(sqlite_db_path=os.path.join(tmp, "hub.db"))
    hub.secondary = cloudflare_storage
    hub.sync_service = BackgroundSyncService(hub.primary, cloudflare_storage)
    return hub


@pytest_asyncio.fixture
async def hub(cloudflare_storage, d1):  # noqa: F811
    await cloudflare_storage._initialize_d1_schema()
    tmp = tempfile.mkdtemp(prefix="mcp-test-forwarder-")
    hub = _build_hub(tmp, cloudflare_storage)
    await hub.primary.initialize()
    # First call pins the cursor at the end of the (empty) log, as it would on upgrade.
    assert await forward_to_secondary(hub) is not None
    yield hub
    await hub.primary.close()
    shutil.rmtree(tmp, ignore_errors=True)


@pytest_asyncio.fixture
async def spoke():
    tmp = tempfile.mkdtemp(prefix="mcp-test-forwarder-spoke-")
    storage = SqliteVecMemoryStorage(os.path.join(tmp, "spoke.db"))
    await storage.initialize()
    yield storage
    await storage.close()
    shutil.rmtree(tmp, ignore_errors=True)


def _memory(content):
    now = time.time()
    return Memory(content=content, content_hash=generate_content_hash(content),
                  tags=["spoke"], created_at=now, updated_at=now)


def _events(storage):
    cur = storage.conn.execute(
        "SELECT seq, schema_version, event_id, op, content_hash, agent_id, hlc_physical, "
        "hlc_logical, embedding_model, embedding_dim, payload FROM sync_events ORDER BY seq"
    )
    return [
        {"seq": r[0], "schema_version": r[1] or 1, "event_id": r[2], "op": r[3],
         "content_hash": r[4], "agent_id": r[5], "hlc_physical": r[6], "hlc_logical": r[7],
         "embedding_model": r[8], "embedding_dim": r[9],
         "payload": json.loads(r[10]) if r[10] else {}}
        for r in cur.fetchall()
    ]


def _hub_applies_spoke_log(spoke, hub):
    for event in _events(spoke):
        apply_remote_event(hub, event)


async def _spoke_writes_reach_hub(spoke, hub, memory):
    ok, msg = await spoke.store(memory, skip_semantic_dedup=True)
    assert ok, msg
    _hub_applies_spoke_log(spoke, hub)


@pytest.mark.asyncio
async def test_spoke_create_reaches_cloudflare(hub, spoke, d1):  # noqa: F811
    memory = _memory("A spoke wrote this and the hub applied it.")
    await _spoke_writes_reach_hub(spoke, hub, memory)
    assert d1.row(memory.content_hash) is None  # the bug: applied on the hub only

    result = await forward_to_secondary(hub)

    assert result.events_failed == 0 and result.events_pushed >= 1
    assert d1.row(memory.content_hash)["deleted_at"] is None


@pytest.mark.asyncio
async def test_spoke_delete_reaches_cloudflare(hub, spoke, d1):  # noqa: F811
    memory = _memory("A spoke wrote this and then deleted it.")
    await _spoke_writes_reach_hub(spoke, hub, memory)
    await forward_to_secondary(hub)
    assert d1.row(memory.content_hash) is not None

    ok, msg = await spoke.delete(memory.content_hash)
    assert ok, msg
    _hub_applies_spoke_log(spoke, hub)
    await forward_to_secondary(hub)

    assert d1.row(memory.content_hash)["deleted_at"] is not None


@pytest.mark.asyncio
async def test_spoke_edit_reaches_cloudflare(hub, spoke, d1):  # noqa: F811
    memory = _memory("A spoke wrote this and then retagged it.")
    await _spoke_writes_reach_hub(spoke, hub, memory)
    await forward_to_secondary(hub)

    ok, msg = await spoke.update_memory_metadata(memory.content_hash, {"tags": ["spoke", "edited"]})
    assert ok, msg
    _hub_applies_spoke_log(spoke, hub)
    result = await forward_to_secondary(hub)

    assert result.events_failed == 0 and result.events_pushed == 1
    tags = {r[0] for r in d1.db.execute(
        "SELECT t.name FROM memory_tags mt JOIN tags t ON t.id = mt.tag_id "
        "JOIN memories m ON m.id = mt.memory_id WHERE m.content_hash = ?",
        (memory.content_hash,),
    ).fetchall()}
    assert tags == {"spoke", "edited"}


@pytest.mark.asyncio
async def test_restart_between_apply_and_forward_still_delivers(
    hub, spoke, d1, cloudflare_storage  # noqa: F811
):
    memory = _memory("Applied, then the hub restarted before it forwarded.")
    await _spoke_writes_reach_hub(spoke, hub, memory)
    sid = secondary_id_for(cloudflare_storage)
    path = hub.primary.db_path
    cursor_before = get_push_cursor(hub, sid)
    await hub.primary.close()

    reborn = _build_hub(os.path.dirname(path), cloudflare_storage)
    await reborn.primary.initialize()
    try:
        assert get_push_cursor(reborn, sid) == cursor_before
        result = await forward_to_secondary(reborn)
        assert result.events_failed == 0
        assert d1.row(memory.content_hash)["deleted_at"] is None
    finally:
        await reborn.primary.close()


@pytest.mark.asyncio
async def test_scheduled_cycle_on_a_hub_without_peers_forwards(hub, spoke, d1):  # noqa: F811
    from mcp_memory_service.consolidation.scheduler import ConsolidationScheduler

    class _Consolidator:
        health_monitor = None

        def __init__(self, storage):
            self.storage = storage

    memory = _memory("The hub has no peers; the scheduled cycle still forwards.")
    await _spoke_writes_reach_hub(spoke, hub, memory)
    scheduler = ConsolidationScheduler(_Consolidator(hub), schedule_config={}, enabled=True)

    await scheduler._run_sync_cycle()  # MCP_SYNC_PEERS unset: a hub

    assert d1.row(memory.content_hash)["deleted_at"] is None


@pytest.mark.asyncio
async def test_first_run_does_not_replay_history_to_the_secondary(
    hub, spoke, d1, cloudflare_storage  # noqa: F811
):
    old = _memory("Applied long before the forwarder existed.")
    await _spoke_writes_reach_hub(spoke, hub, old)
    sid = secondary_id_for(cloudflare_storage)
    hub.primary.conn.execute("DELETE FROM push_cursor WHERE peer_id = ?", (sid,))
    hub.primary.conn.commit()  # an upgraded hub: history in the log, no cursor yet

    await forward_to_secondary(hub)

    end = hub.primary.conn.execute("SELECT MAX(seq) FROM sync_events").fetchone()[0]
    assert get_push_cursor(hub, sid) == end
    assert d1.row(old.content_hash) is None  # the drift scan owns the backlog


@pytest.mark.asyncio
async def test_forwarded_events_leave_an_identity_row(hub, spoke, d1):  # noqa: F811
    memory = _memory("The hub keeps the identity of what it forwarded.")
    await _spoke_writes_reach_hub(spoke, hub, memory)
    await forward_to_secondary(hub)

    rows = hub.primary.conn.execute(
        "SELECT op, content_hash FROM secondary_forwarded"
    ).fetchall()
    assert ("create", memory.content_hash) in [tuple(r) for r in rows]


@pytest.mark.asyncio
async def test_refused_forward_does_not_advance_the_cursor(
    hub, spoke, d1, cloudflare_storage  # noqa: F811
):
    memory = _memory("The secondary is down when this arrives.")
    await _spoke_writes_reach_hub(spoke, hub, memory)
    sid = secondary_id_for(cloudflare_storage)
    before = get_push_cursor(hub, sid)

    async def _down(*args, **kwargs):
        raise RuntimeError("D1 unavailable")

    original = cloudflare_storage.store
    cloudflare_storage.store = _down
    try:
        result = await forward_to_secondary(hub)
    finally:
        cloudflare_storage.store = original

    assert result.events_failed == 1
    assert get_push_cursor(hub, sid) == before
    assert (await forward_to_secondary(hub)).events_failed == 0
    assert d1.row(memory.content_hash) is not None
