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
