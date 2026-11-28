"""A hub's pull from its secondary must not re-apply what the hub itself forwarded (#1506, part 2).

Part 1 forwards spoke events to the secondary. D1 stamps the forwarded write with its
own clock (``updated_at`` on an edit that changes tags, ``deleted_at`` on a delete), so
the secondary's copy looks NEWER than the hub's row whenever the forward ran later than
the spoke's write. The pull then reads the hub's own echo as a remote change:

* edit:   the drift scan sees ``cf.updated_at > local.updated_at`` and writes it back
* delete: the tombstone is newer than a re-create the spoke made before the hub
          forwarded the delete, so the pull deletes the live re-created row
* create: hash dedup already covers it, pinned so all three op types are covered

The forward is delayed by moving the hub's clock (D1 stamps and ``forwarded_at``),
not by sleeping.
"""

import os
import shutil
import tempfile
import time as _time

import pytest
import pytest_asyncio

from mcp_memory_service.storage import cloudflare as cloudflare_module
from mcp_memory_service.storage import hybrid as hybrid_module
from mcp_memory_service.storage.sync import forwarder as forwarder_module
from mcp_memory_service.storage.sync.forwarder import forward_to_secondary
from tests.unit.test_cloudflare_store_scope import _FakeD1, cloudflare_storage, d1  # noqa: F401
from tests.unit.test_secondary_forwarder_1506 import (  # noqa: F401
    _build_hub,
    _hub_applies_spoke_log,
    _memory,
    _spoke_writes_reach_hub,
    spoke,
)

FORWARD_LAG = 30.0  # seconds between the spoke's write and the hub forwarding it


@pytest.fixture(autouse=True)
def _eventlog_on(monkeypatch):
    monkeypatch.setenv("MCP_SYNC_EVENTLOG", "on")
    # Set only when the configured backend is hybrid; the test process is not.
    monkeypatch.setattr(hybrid_module.app_config, "HYBRID_DRIFT_BATCH_SIZE", 100, raising=False)


@pytest_asyncio.fixture
async def hub(cloudflare_storage, d1):  # noqa: F811
    await cloudflare_storage._initialize_d1_schema()
    tmp = tempfile.mkdtemp(prefix="mcp-test-echo-")
    hub = _build_hub(tmp, cloudflare_storage)
    await hub.primary.initialize()
    hub.sync_service.drift_check_enabled = True
    assert await forward_to_secondary(hub) is not None
    yield hub
    await hub.primary.close()
    shutil.rmtree(tmp, ignore_errors=True)


class _LaggedClock:
    """Stands in for the ``time`` module inside cloudflare.py: D1 stamps a later time."""

    def __getattr__(self, name):
        return getattr(_time, name)

    @staticmethod
    def time():
        return _time.time() + FORWARD_LAG


async def _forward_late(hub, monkeypatch):
    # One clock on the hub: D1 stamps and ``forwarded_at`` both run FORWARD_LAG ahead.
    for module in (cloudflare_module, forwarder_module):
        monkeypatch.setattr(module, "time", _LaggedClock())
    try:
        result = await forward_to_secondary(hub)
    finally:
        for module in (cloudflare_module, forwarder_module):
            monkeypatch.setattr(module, "time", _time)
    assert result.events_failed == 0
    return result


def _hub_seq(hub):
    return hub.primary.conn.execute("SELECT COALESCE(MAX(seq), 0) FROM sync_events").fetchone()[0]


async def _pull(hub):
    """Both pull paths a hybrid hub runs against its secondary."""
    await hub._sync_memories_from_cloudflare(
        sync_type="manual", broadcast_sse=False, enable_drift_check=False
    )
    hub.sync_service.last_drift_check_time = _time.time() - 3600
    return await hub.sync_service._detect_and_sync_drift()


@pytest.mark.asyncio
async def test_forwarded_create_is_not_applied_again_on_pull(hub, spoke, d1, monkeypatch):  # noqa: F811
    memory = _memory("A spoke created this; the hub forwarded it late.")
    await _spoke_writes_reach_hub(spoke, hub, memory)
    await _forward_late(hub, monkeypatch)
    seq = _hub_seq(hub)

    stats = await _pull(hub)

    assert stats["checked"] >= 1, stats  # the drift scan really looked at the row
    assert stats["synced"] == 0
    assert _hub_seq(hub) == seq  # the pull wrote nothing on the hub


@pytest.mark.asyncio
async def test_forwarded_edit_is_not_applied_again_on_pull(hub, spoke, d1, monkeypatch):  # noqa: F811
    memory = _memory("A spoke retagged this; the hub forwarded the edit late.")
    await _spoke_writes_reach_hub(spoke, hub, memory)
    await forward_to_secondary(hub)
    ok, msg = await spoke.update_memory_metadata(memory.content_hash, {"tags": ["spoke", "edited"]})
    assert ok, msg
    _hub_applies_spoke_log(spoke, hub)
    local_before = await hub.primary.get_by_hash(memory.content_hash)
    await _forward_late(hub, monkeypatch)
    seq = _hub_seq(hub)

    stats = await _pull(hub)

    assert stats["checked"] >= 1, stats
    assert stats["synced"] == 0
    assert _hub_seq(hub) == seq  # no second update_metadata event fanned out to spokes
    local_after = await hub.primary.get_by_hash(memory.content_hash)
    assert local_after.updated_at == local_before.updated_at


@pytest.mark.asyncio
async def test_forwarded_delete_is_not_applied_again_on_pull(hub, spoke, d1, monkeypatch):  # noqa: F811
    memory = _memory("A spoke deleted this and wrote it again before the hub forwarded the delete.")
    await _spoke_writes_reach_hub(spoke, hub, memory)
    await forward_to_secondary(hub)
    ok, msg = await spoke.delete(memory.content_hash)
    assert ok, msg
    _hub_applies_spoke_log(spoke, hub)
    ok, msg = await spoke.store(_memory(memory.content), skip_semantic_dedup=True)
    assert ok, msg  # re-created on the spoke, not yet pushed to the hub
    await _forward_late(hub, monkeypatch)  # D1 tombstone is newer than the re-create
    _hub_applies_spoke_log(spoke, hub)
    state = await hub.primary.get_sync_state_for_hash(memory.content_hash)
    assert state is not None and state[1] is None  # live again on the hub

    await _pull(hub)

    state = await hub.primary.get_sync_state_for_hash(memory.content_hash)
    assert state is not None and state[1] is None  # the hub's own delete did not come back


class _LaterClock(_LaggedClock):
    @staticmethod
    def time():
        return _time.time() + 2 * FORWARD_LAG


@pytest.mark.asyncio
async def test_outside_edit_on_the_secondary_after_a_forward_still_applies(
    hub, spoke, d1, cloudflare_storage, monkeypatch  # noqa: F811
):
    memory = _memory("Forwarded by the hub, then retagged on D1 by another machine.")
    await _spoke_writes_reach_hub(spoke, hub, memory)
    await _forward_late(hub, monkeypatch)
    monkeypatch.setattr(cloudflare_module, "time", _LaterClock())
    ok, msg = await cloudflare_storage.update_memory_metadata(
        memory.content_hash, {"tags": ["spoke", "from-elsewhere"]}
    )
    monkeypatch.setattr(cloudflare_module, "time", _time)
    assert ok, msg

    stats = await _pull(hub)

    assert stats["synced"] == 1
    local = await hub.primary.get_by_hash(memory.content_hash)
    assert set(local.tags) == {"spoke", "from-elsewhere"}


@pytest.mark.asyncio
async def test_outside_delete_on_the_secondary_after_a_forward_still_applies(
    hub, spoke, d1, cloudflare_storage, monkeypatch  # noqa: F811
):
    memory = _memory("Forwarded by the hub, then deleted on D1 by another machine.")
    await _spoke_writes_reach_hub(spoke, hub, memory)
    await _forward_late(hub, monkeypatch)
    monkeypatch.setattr(cloudflare_module, "time", _LaterClock())
    ok, msg = await cloudflare_storage.delete(memory.content_hash)
    monkeypatch.setattr(cloudflare_module, "time", _time)
    assert ok, msg

    await _pull(hub)

    state = await hub.primary.get_sync_state_for_hash(memory.content_hash)
    assert state is None or state[1] is not None  # the outside delete reached the hub


@pytest.mark.asyncio
async def test_delete_echo_is_recognised_after_a_crash_before_the_hub_recorded_it(
    hub, spoke, d1, monkeypatch  # noqa: F811
):
    memory = _memory("D1 took the delete, then the hub died before writing it down.")
    await _spoke_writes_reach_hub(spoke, hub, memory)
    await forward_to_secondary(hub)
    ok, msg = await spoke.delete(memory.content_hash)
    assert ok, msg
    _hub_applies_spoke_log(spoke, hub)
    ok, msg = await spoke.store(_memory(memory.content), skip_semantic_dedup=True)
    assert ok, msg

    def _crash(self, event):
        raise RuntimeError("hub died after the secondary acknowledged")

    record = forwarder_module.SecondaryForwarder._record
    monkeypatch.setattr(forwarder_module.SecondaryForwarder, "_record", _crash)
    with pytest.raises(RuntimeError):
        await _forward_late(hub, monkeypatch)
    monkeypatch.setattr(forwarder_module.SecondaryForwarder, "_record", record)
    assert d1.row(memory.content_hash)["deleted_at"] is not None  # D1 has the tombstone
    _hub_applies_spoke_log(spoke, hub)  # the re-create arrives; pull runs before a retry

    await _pull(hub)

    state = await hub.primary.get_sync_state_for_hash(memory.content_hash)
    assert state is not None and state[1] is None


def test_a_busy_database_does_not_switch_the_guard_off(cloudflare_storage):  # noqa: F811
    import sqlite3

    class _Busy:
        def execute(self, *args):
            raise sqlite3.OperationalError("database is locked")

    class _Primary:
        conn = _Busy()

    with pytest.raises(sqlite3.OperationalError):
        forwarder_module.forwarded_echoes(_Primary(), cloudflare_storage, {"h": 1.0}, "delete")


def test_migration_024_gives_part_1_rows_a_window():
    import pathlib
    import sqlite3

    migrations = pathlib.Path(forwarder_module.__file__).parents[1] / "migrations"
    db = sqlite3.connect(":memory:")
    db.executescript((migrations / "023_add_secondary_forwarded.sql").read_text(encoding="utf-8"))
    db.execute("INSERT INTO secondary_forwarded VALUES ('s', 'a', 'e', 'h', 'delete', 1000.0)")
    db.executescript((migrations / "024_add_secondary_forwarded_window.sql").read_text(encoding="utf-8"))

    class _Primary:
        conn = db

    class _Secondary:
        pass

    db.execute("UPDATE secondary_forwarded SET secondary_id = ?", (forwarder_module.secondary_id_for(_Secondary()),))
    assert db.execute("SELECT started_at, acked FROM secondary_forwarded").fetchone() == (700.0, 1)
    assert forwarder_module.forwarded_echoes(_Primary(), _Secondary(), {"h": 999.0}, "delete") == {"h"}
    assert forwarder_module.forwarded_echoes(_Primary(), _Secondary(), {"h": 1001.0}, "delete") == set()


@pytest.mark.asyncio
async def test_a_failure_to_open_the_window_counts_as_a_failed_forward(
    hub, spoke, d1, cloudflare_storage, monkeypatch  # noqa: F811
):
    memory = _memory("The hub database is busy when the forward starts.")
    await _spoke_writes_reach_hub(spoke, hub, memory)

    def _busy(self, event):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(forwarder_module.SecondaryForwarder, "_begin", _busy)
    result = await forward_to_secondary(hub)

    assert result.events_failed == 1
    assert d1.row(memory.content_hash) is None  # nothing reached D1 without a window
