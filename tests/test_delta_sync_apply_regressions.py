"""
Regression tests for delta-sync apply (Phase 4) — the materialization bugs Greptile
flagged on PR #1489. Each test fails against the pre-fix apply and passes after it.

Covered findings:
- update_metadata must merge, not overwrite with empty content (P1 "Metadata changes erase memories")
- replay with altered payload must not rewrite the memory (P1 security "Replays overwrite recorded memories")
- a failed materialization must report applied=False (P1 "Failed writes get skipped")
- accepting a remote event must advance the saved HLC clock (P1 "Later edits get older clocks")
- synced tags must be stored CSV so existing readers match them (P1 "Synced tags stop matching")
- create events must preserve store/created_at/updated_at (P1 "Stores and dates change")
"""

import os
import json
import time
import tempfile

import pytest
import pytest_asyncio

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash
from mcp_memory_service.storage.sync.apply import apply_remote_event


@pytest_asyncio.fixture
async def store(monkeypatch):
    monkeypatch.setenv("MCP_AGENT_ID", "beta")
    monkeypatch.setenv("MCP_SYNC_EVENTLOG", "true")
    monkeypatch.setenv("MCP_SEMANTIC_DEDUP_ENABLED", "false")
    with tempfile.TemporaryDirectory() as tmp:
        s = SqliteVecMemoryStorage(os.path.join(tmp, "b.db"))
        await s.initialize()
        yield s
        await s.close()


def _create_event(content_hash, *, content, tags=None, store="default",
                  created_at=1000.0, updated_at=1000.0, agent="alpha", event_id="e1",
                  hlc_physical=10, hlc_logical=0, memory_type="note"):
    return {
        "seq": 1, "agent_id": agent, "event_id": event_id, "op": "create",
        "content_hash": content_hash, "hlc_physical": hlc_physical, "hlc_logical": hlc_logical,
        "embedding_model": None, "embedding_dim": None,
        "payload": {
            "content_hash": content_hash, "content": content, "memory_type": memory_type,
            "tags": tags or [], "created_at": created_at, "updated_at": updated_at,
            "metadata": {}, "store": store,
        },
    }


@pytest.mark.asyncio
async def test_update_metadata_does_not_erase_memory(store):
    """P1: update_metadata carries only `updates`; applying it must merge, not blank the row."""
    content = "A real memory that must survive a metadata update"
    h = generate_content_hash(content)
    apply_remote_event(store, _create_event(h, content=content, tags=["keep"]))

    # update_metadata event: ONLY updates, no top-level content
    upd = {
        "seq": 2, "agent_id": "alpha", "event_id": "e2", "op": "update_metadata",
        "content_hash": h, "hlc_physical": 20, "hlc_logical": 0,
        "embedding_model": None, "embedding_dim": None,
        "payload": {"content_hash": h, "updates": {"tags": ["keep", "added"]}, "updated_at": 2000.0},
    }
    res = apply_remote_event(store, upd)
    assert res.applied, res.reason

    mem = await store.get_by_hash(h)
    assert mem is not None, "memory must still exist after update_metadata"
    assert mem.content == content, "content must NOT be erased by update_metadata"
    assert set(mem.tags) == {"keep", "added"}, f"tags should be merged, got {mem.tags}"


@pytest.mark.asyncio
async def test_replay_with_altered_payload_is_rejected(store):
    """P1 security: same (agent_id, event_id) with different payload must not rewrite the memory."""
    content = "Original content for the winning event"
    h = generate_content_hash(content)
    apply_remote_event(store, _create_event(h, content=content, event_id="dup1"))

    forged = _create_event(h, content="FORGED replacement content", event_id="dup1")
    res = apply_remote_event(store, forged)
    assert res.applied is False, "replay with altered payload must be rejected"

    mem = await store.get_by_hash(h)
    assert mem.content == content, "memory content must remain the original, not the forged replay"


@pytest.mark.asyncio
async def test_identical_duplicate_is_idempotent(store):
    """A re-pulled identical event (same identity AND payload) is a benign no-op, counted applied."""
    content = "Idempotent event content"
    h = generate_content_hash(content)
    ev = _create_event(h, content=content, event_id="idem1")
    first = apply_remote_event(store, ev)
    assert first.applied and first.materialized
    second = apply_remote_event(store, dict(ev))
    assert second.applied is True, "identical duplicate must count as applied (idempotent resume)"
    assert second.materialized is False, "identical duplicate must not re-materialize"


@pytest.mark.asyncio
async def test_failed_create_reports_not_applied(store):
    """P1: a create with no content fails to materialize → applied must be False (sender retries)."""
    h = generate_content_hash("missing-content-hash-probe")
    ev = _create_event(h, content="")  # empty content → materialization returns False
    res = apply_remote_event(store, ev)
    assert res.applied is False, "failed materialization must not report applied=True"
    assert res.materialized is False


@pytest.mark.asyncio
async def test_synced_tags_are_csv_and_matchable(store):
    """P1: tags must be stored CSV (not a JSON array) so the comma-splitting readers match them."""
    content = "Memory whose tags must stay matchable after sync"
    h = generate_content_hash(content)
    apply_remote_event(store, _create_event(h, content=content, tags=["work", "urgent"]))

    raw = store.conn.execute("SELECT tags FROM memories WHERE content_hash = ?", (h,)).fetchone()[0]
    assert raw == "work,urgent", f"tags must be CSV like store(), got {raw!r}"
    # and the read path returns them as a clean list
    mem = await store.get_by_hash(h)
    assert set(mem.tags) == {"work", "urgent"}


@pytest.mark.asyncio
async def test_create_preserves_store_and_timestamps(store):
    """P1: create events must keep their store and created_at/updated_at, not default+now()."""
    content = "Old memory from a named store"
    h = generate_content_hash(content)
    apply_remote_event(store, _create_event(
        h, content=content, store="projects", created_at=1234.0, updated_at=5678.0,
    ))
    row = store.conn.execute(
        "SELECT store, created_at, updated_at FROM memories WHERE content_hash = ?", (h,)
    ).fetchone()
    assert row[0] == "projects", f"store must be preserved, got {row[0]!r}"
    assert float(row[1]) == 1234.0, f"created_at must be preserved, got {row[1]}"
    assert float(row[2]) == 5678.0, f"updated_at must be preserved, got {row[2]}"


@pytest.mark.asyncio
async def test_accepting_remote_event_advances_saved_hlc(store):
    """P1: accepting a remote event whose clock is ahead must bump the saved last_hlc."""
    content = "Remote event with a clock ahead of ours"
    h = generate_content_hash(content)
    apply_remote_event(store, _create_event(h, content=content, hlc_physical=999999, hlc_logical=5))

    saved = dict(store.conn.execute(
        "SELECT key, value FROM metadata WHERE key IN ('sync_hlc_physical','sync_hlc_logical')"
    ).fetchall())
    assert int(saved.get("sync_hlc_physical", 0)) >= 999999, "saved HLC physical must advance to the accepted clock"


def _update_metadata_event(content_hash, *, updates, event_id, agent="alpha",
                           hlc_physical, hlc_logical=0, updated_at=2000.0):
    return {
        "agent_id": agent, "event_id": event_id, "op": "update_metadata",
        "content_hash": content_hash, "hlc_physical": hlc_physical, "hlc_logical": hlc_logical,
        "embedding_model": None, "embedding_dim": None,
        "payload": {"content_hash": content_hash, "updates": updates, "updated_at": updated_at},
    }


async def _mk_store(tmp_name):
    import tempfile as _tf
    d = _tf.mkdtemp()
    s = SqliteVecMemoryStorage(os.path.join(d, tmp_name))
    await s.initialize()
    return s


@pytest.mark.asyncio
async def test_update_metadata_converges_regardless_of_order(monkeypatch):
    """P1 convergence (Greptile): two spokes editing DIFFERENT fields must reach the SAME row
    no matter the arrival order. One event sets tags (older HLC), another sets memory_type
    (newer HLC). Peer A receives oldest-first, peer B newest-first; both must keep BOTH edits."""
    monkeypatch.setenv("MCP_AGENT_ID", "beta")
    monkeypatch.setenv("MCP_SYNC_EVENTLOG", "true")
    monkeypatch.setenv("MCP_SEMANTIC_DEDUP_ENABLED", "false")

    content = "Shared memory edited by two spokes on different fields"
    h = generate_content_hash(content)
    create = _create_event(h, content=content, tags=["base"], memory_type="note")
    ev_tags = _update_metadata_event(h, updates={"tags": ["base", "work"]},
                                     event_id="u-tags", hlc_physical=20)
    ev_type = _update_metadata_event(h, updates={"memory_type": "reference"},
                                     event_id="u-type", hlc_physical=30)

    # Peer A: oldest-first (tags then type). Peer B: newest-first (type then tags).
    sa = await _mk_store("a.db")
    sb = await _mk_store("b.db")
    try:
        for ev in (create, ev_tags, ev_type):
            apply_remote_event(sa, dict(ev))
        for ev in (create, ev_type, ev_tags):
            apply_remote_event(sb, dict(ev))

        ma = await sa.get_by_hash(h)
        mb = await sb.get_by_hash(h)
        # Both edits survive on BOTH peers, and the two rows are identical (convergence).
        assert set(ma.tags) == {"base", "work"}, f"peer A lost the tags edit: {ma.tags}"
        assert ma.memory_type == "reference", f"peer A lost the type edit: {ma.memory_type}"
        assert set(mb.tags) == set(ma.tags), f"peers diverged on tags: {ma.tags} vs {mb.tags}"
        assert mb.memory_type == ma.memory_type, f"peers diverged on type: {ma.memory_type} vs {mb.memory_type}"
    finally:
        await sa.close()
        await sb.close()


@pytest.mark.asyncio
async def test_apply_acquires_connection_lock(store):
    """P1 concurrency (Greptile): apply_remote_event must take the storage _conn_lock so it
    cannot commit over an in-flight local write's savepoint. We assert the lock is held during
    the apply by observing it is NOT acquirable from another thread mid-apply."""
    import threading

    # Ensure the lock exists (mirrors base.py lazy init).
    if not hasattr(store, "_conn_lock") or store._conn_lock is None:
        store._conn_lock = threading.Lock()

    held_during_apply = {"value": None}
    real_materialize = None
    from mcp_memory_service.storage.sync import apply as apply_mod

    def _probe(*args, **kwargs):
        # While inside apply (which must hold the lock), a non-blocking acquire must FAIL.
        got = store._conn_lock.acquire(blocking=False)
        held_during_apply["value"] = not got  # True means the lock was already held (good)
        if got:
            store._conn_lock.release()
        return real_materialize(*args, **kwargs)

    real_materialize = apply_mod._materialize_event
    apply_mod._materialize_event = _probe
    try:
        content = "Event used to observe the connection lock during apply"
        h = generate_content_hash(content)
        apply_remote_event(store, _create_event(h, content=content))
    finally:
        apply_mod._materialize_event = real_materialize

    assert held_during_apply["value"] is True, "apply_remote_event must hold _conn_lock during the write"


def _delete_event(content_hash, *, event_id, agent="alpha", hlc_physical, hlc_logical=0, deleted_at=1500.0):
    return {
        "agent_id": agent, "event_id": event_id, "op": "delete",
        "content_hash": content_hash, "hlc_physical": hlc_physical, "hlc_logical": hlc_logical,
        "embedding_model": None, "embedding_dim": None,
        "payload": {"content_hash": content_hash, "deleted_at": deleted_at},
    }


@pytest.mark.asyncio
async def test_old_edit_does_not_restore_recreated_memory_field(store):
    """P1 (Greptile apply:315): after delete+recreate, an OLD update_metadata must not restore
    a field from the memory's previous life. Create(type=note,hlc10) → old edit(type=note,hlc12)
    → delete(hlc20) → recreate(type=reference,hlc30) → later tags-only edit(hlc40). memory_type
    must stay 'reference' (the recreate seed wins over the pre-recreate edit)."""
    content = "Memory that gets deleted and recreated"
    h = generate_content_hash(content)
    apply_remote_event(store, _create_event(h, content=content, memory_type="note",
                                            event_id="c1", hlc_physical=10))
    apply_remote_event(store, _update_metadata_event(h, updates={"memory_type": "note"},
                                                     event_id="old-edit", hlc_physical=12))
    apply_remote_event(store, _delete_event(h, event_id="d1", hlc_physical=20))
    apply_remote_event(store, _create_event(h, content=content, memory_type="reference",
                                            event_id="c2", hlc_physical=30))
    # a later tags-only edit must NOT drag memory_type back to the pre-recreate 'note'
    apply_remote_event(store, _update_metadata_event(h, updates={"tags": ["fresh"]},
                                                     event_id="tags-edit", hlc_physical=40))

    rowm = store.conn.execute(
        "SELECT memory_type, tags, deleted_at FROM memories WHERE content_hash = ?", (h,)
    ).fetchone()
    assert rowm is not None and rowm[2] is None, "recreated memory must be live"
    assert rowm[0] == "reference", f"recreated field must survive, got {rowm[0]}"
    assert set((rowm[1] or "").split(",")) == {"fresh"}


@pytest.mark.asyncio
async def test_equal_clock_edits_follow_resolver_tiebreak(store):
    """P1 (Greptile apply:328): equal-HLC edits to the same field must pick the SAME winner the
    resolver picks (smaller agent_id/event_id wins), not the opposite."""
    content = "Memory edited by two agents at the same clock"
    h = generate_content_hash(content)
    apply_remote_event(store, _create_event(h, content=content, tags=["base"], event_id="c0", hlc_physical=10))
    # Two update_metadata on the SAME field (tags), SAME hlc, different agents.
    apply_remote_event(store, _update_metadata_event(h, updates={"tags": ["from-alpha"]},
                                                     event_id="e-a", agent="alpha", hlc_physical=20))
    apply_remote_event(store, _update_metadata_event(h, updates={"tags": ["from-omega"]},
                                                     event_id="e-b", agent="omega", hlc_physical=20))
    rowm = store.conn.execute("SELECT tags FROM memories WHERE content_hash = ?", (h,)).fetchone()
    # resolver: smaller agent_id wins on equal clock → 'alpha' < 'omega' → from-alpha
    assert set((rowm[0] or "").split(",")) == {"from-alpha"}, \
        f"equal-clock tie must follow resolver (alpha), got {rowm[0]}"


@pytest.mark.asyncio
async def test_local_metadata_key_survives_remote_tags_only_edit(store):
    """P1 (Greptile apply:334): a locally-saved metadata key (e.g. quality_score, written via
    update_memory_metadata) must NOT be erased when a remote tags-only update_metadata arrives.
    The metadata column is merged per-key, not overwritten with the create's whole dict."""
    content = "Memory whose local quality_score must survive sync"
    h = generate_content_hash(content)
    # Remote create (no quality_score in metadata).
    apply_remote_event(store, _create_event(h, content=content, tags=["base"],
                                            event_id="c1", hlc_physical=10))
    # Local write records a custom metadata key (quality endpoint path).
    ok, _ = await store.update_memory_metadata(h, {"quality_score": 0.87})
    assert ok

    # Remote tags-only edit arrives with a HIGHER clock.
    apply_remote_event(store, _update_metadata_event(h, updates={"tags": ["base", "synced"]},
                                                     event_id="u-tags", hlc_physical=50))

    rowm = store.conn.execute(
        "SELECT tags, metadata FROM memories WHERE content_hash = ?", (h,)
    ).fetchone()
    assert set((rowm[0] or "").split(",")) == {"base", "synced"}, f"tags should sync, got {rowm[0]}"
    md = json.loads(rowm[1]) if rowm[1] else {}
    assert md.get("quality_score") == 0.87, f"local quality_score must survive the sync, got {md}"
