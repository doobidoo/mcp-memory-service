"""A hybrid hub forwards the delta-sync events it applies to its secondary (#1506).

``apply_remote_event`` writes the primary's SQLite directly, so a create, edit or
delete pushed by a spoke never reached the hybrid queue and Cloudflare stayed a
partial copy. These tests apply real events to a hybrid whose primary is
SQLite-vec and whose secondary is a CloudflareStorage over an in-memory D1, then
run the sync service's periodic cycle, which is where the forwarding happens.
"""

import os
import shutil
import tempfile
import time

import pytest
import pytest_asyncio

from mcp_memory_service.storage import hybrid as hybrid_module
from mcp_memory_service.storage.hybrid import BackgroundSyncService, HybridMemoryStorage
from mcp_memory_service.storage.sync.apply import apply_remote_event
from mcp_memory_service.utils.hashing import generate_content_hash
from tests.unit.test_cloudflare_store_scope import _FakeD1, cloudflare_storage, d1  # noqa: F401


@pytest_asyncio.fixture
async def hub(cloudflare_storage, d1, monkeypatch):  # noqa: F811
    monkeypatch.setenv("MCP_AGENT_ID", "hub")
    monkeypatch.setenv("MCP_SYNC_EVENTLOG", "true")
    monkeypatch.setenv("MCP_SEMANTIC_DEDUP_ENABLED", "false")
    await cloudflare_storage._initialize_d1_schema()
    tmp = tempfile.mkdtemp(prefix="mcp-test-hybrid-forward-")
    storage = HybridMemoryStorage(sqlite_db_path=os.path.join(tmp, "primary.db"))
    await storage.primary.initialize()
    storage.secondary = cloudflare_storage
    # Not started: the tests call the periodic cycle themselves.
    storage.sync_service = BackgroundSyncService(storage.primary, cloudflare_storage)
    # The config module leaves the drift settings unset unless the backend is hybrid.
    storage.sync_service.drift_check_enabled = True
    storage.sync_service.drift_check_interval = 3600
    monkeypatch.setattr(hybrid_module.app_config, "HYBRID_DRIFT_BATCH_SIZE", 100, raising=False)
    yield storage
    await storage.primary.close()
    shutil.rmtree(tmp, ignore_errors=True)


NOW = time.time()


def _event(op, content_hash, payload, n, *, hlc):
    return {
        "seq": n, "agent_id": "spoke", "event_id": f"e{n}", "op": op,
        "content_hash": content_hash, "hlc_physical": hlc, "hlc_logical": 0,
        "embedding_model": None, "embedding_dim": None, "payload": payload,
    }


def _create(content, n, *, store="default", tags=("orig",)):
    h = generate_content_hash(content)
    payload = {
        "content_hash": h, "content": content, "memory_type": "note", "tags": list(tags),
        "created_at": NOW, "updated_at": NOW, "metadata": {}, "store": store,
    }
    return h, _event("create", h, payload, n, hlc=10 * n)


def _edit(h, n, tags):
    payload = {"content_hash": h, "updates": {"tags": tags}, "updated_at": time.time() - 30}
    return _event("update_metadata", h, payload, n, hlc=10 * n)


def _delete(h, n):
    payload = {"content_hash": h, "deleted_at": time.time() - 30}
    return _event("delete", h, payload, n, hlc=10 * n)


def _apply(hub, event):
    result = apply_remote_event(hub, event)
    assert result.applied and result.materialized, result.reason


def _d1_tags(d1, h):  # noqa: F811
    return sorted(r[0] for r in d1.db.execute(
        "SELECT t.name FROM tags t JOIN memory_tags mt ON t.id = mt.tag_id "
        "JOIN memories m ON m.id = mt.memory_id WHERE m.content_hash = ?", (h,)))


def _event_count(hub):
    return hub.primary.conn.execute("SELECT COUNT(*) FROM sync_events").fetchone()[0]


def _writes(d1):  # noqa: F811
    return [s for s in d1.statements if s.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE"))]


async def _drift_and_forward_again(hub, d1):  # noqa: F811
    """What a later cycle does after the forward: pull from the secondary, forward again."""
    events_before, writes_before = _event_count(hub), len(_writes(d1))
    hub.sync_service.last_drift_check_time = 0  # look at every recent secondary row
    await hub.sync_service._detect_and_sync_drift()
    await hub.sync_service._periodic_sync()
    assert _event_count(hub) == events_before, "the secondary's copy was applied to the hub again"
    assert len(_writes(d1)) == writes_before, "the hub forwarded the same event twice"


@pytest.mark.asyncio
async def test_spoke_create_reaches_the_secondary_in_its_store(hub, d1):  # noqa: F811
    h, create = _create("A spoke wrote this in store b.", 1, store="b")
    _apply(hub, create)

    await hub.sync_service._periodic_sync()

    row = d1.row(h)
    assert row is not None and row["store"] == "b" and row["deleted_at"] is None
    assert _d1_tags(d1, h) == ["orig"]
    await _drift_and_forward_again(hub, d1)


@pytest.mark.asyncio
async def test_spoke_edit_reaches_the_secondary(hub, d1):  # noqa: F811
    h, create = _create("A spoke created this, then edited it.", 1)
    _apply(hub, create)
    await hub.sync_service._periodic_sync()

    _apply(hub, _edit(h, 2, ["orig", "edited"]))
    await hub.sync_service._periodic_sync()

    assert _d1_tags(d1, h) == ["edited", "orig"]
    await _drift_and_forward_again(hub, d1)


@pytest.mark.asyncio
async def test_spoke_delete_reaches_the_secondary(hub, d1):  # noqa: F811
    h, create = _create("A spoke created this, then deleted it.", 1)
    _apply(hub, create)
    await hub.sync_service._periodic_sync()
    assert d1.row(h)["deleted_at"] is None

    _apply(hub, _delete(h, 2))
    await hub.sync_service._periodic_sync()

    assert d1.row(h)["deleted_at"] is not None
    await _drift_and_forward_again(hub, d1)


@pytest.mark.asyncio
async def test_restart_between_apply_and_forward_still_delivers(hub, d1, cloudflare_storage):  # noqa: F811
    h, create = _create("Applied on the hub, hub restarted before the push.", 1)
    _apply(hub, create)
    # The old service never ran a cycle. A restart builds a new one with an empty
    # queue; the event is only in the hub's log.
    hub.sync_service = BackgroundSyncService(hub.primary, cloudflare_storage)

    await hub.sync_service._periodic_sync()

    assert d1.row(h) is not None


@pytest.mark.asyncio
async def test_failed_push_is_retried_by_the_next_cycle(hub, d1, cloudflare_storage, monkeypatch):  # noqa: F811
    h, create = _create("The secondary is down when this arrives.", 1)
    _apply(hub, create)
    real_store = cloudflare_storage.store

    async def down(*args, **kwargs):
        raise ConnectionError("secondary unreachable")

    monkeypatch.setattr(cloudflare_storage, "store", down)
    await hub.sync_service._periodic_sync()
    assert d1.row(h) is None

    monkeypatch.setattr(cloudflare_storage, "store", real_store)
    await hub.sync_service._periodic_sync()

    assert d1.row(h) is not None


@pytest.mark.asyncio
async def test_stale_secondary_copy_converges_whatever_the_last_op_was(hub, d1):  # noqa: F811
    h, create = _create("The secondary holds an older copy of this.", 1)
    _apply(hub, create)
    await hub.sync_service._periodic_sync()
    _apply(hub, _edit(h, 2, ["orig", "edited"]))

    # A batch can end on a store, for example a delete followed by a recreate of the
    # same content. The copy is compared by value, so the op does not decide.
    await hub.sync_service._forward_one(h, "store")

    assert _d1_tags(d1, h) == ["edited", "orig"]


@pytest.mark.asyncio
async def test_removed_metadata_key_reaches_the_secondary(hub, d1):  # noqa: F811
    h, create = _create("A spoke created this with two metadata keys.", 1)
    create["payload"]["metadata"] = {"a": 1, "b": 2}
    _apply(hub, create)
    await hub.sync_service._periodic_sync()
    assert (await hub.secondary.get_by_hash(h)).metadata.get("b") == 2

    hub.primary.conn.execute(
        "UPDATE memories SET metadata = ? WHERE content_hash = ?", ('{"a": 1}', h))
    hub.primary.conn.commit()
    await hub.sync_service._forward_one(h, "update_metadata")

    assert "b" not in (await hub.secondary.get_by_hash(h)).metadata


@pytest.mark.asyncio
async def test_failed_forward_does_not_skip_the_drift_check(hub, d1, cloudflare_storage, monkeypatch):  # noqa: F811
    h, create = _create("The secondary refuses this one write.", 1)
    _apply(hub, create)

    async def refuse(*args, **kwargs):
        raise RuntimeError("store refused")

    drift_runs = []

    async def drift(*args, **kwargs):
        drift_runs.append(1)
        return {}

    monkeypatch.setattr(cloudflare_storage, "store", refuse)
    monkeypatch.setattr(hub.sync_service, "_detect_and_sync_drift", drift)
    hub.sync_service.last_drift_check_time = 0
    await hub.sync_service._periodic_sync()

    assert d1.row(h) is None
    assert drift_runs == [1], "a failed forward skipped the drift check"
    assert hub.sync_service.sync_stats["forward_failures"] == 1
