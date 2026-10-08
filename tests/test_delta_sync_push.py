"""
Tests for delta-sync Phase 4c: outbound push (spoke -> hub).

Covers CA1-CA6 from spec-delta-sync-fase4c.md (ADR-0027).
MCP_SYNC_EVENTLOG forced on so stores produce events to push.
"""

import pytest
import pytest_asyncio
import tempfile
import os
import shutil
from typing import Dict, Any, List

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash

from mcp_memory_service.storage.sync.orchestrator import (
    push_to_peer, PushResult, get_push_cursor, advance_push_cursor,
)
from mcp_memory_service.storage.sync.apply import apply_remote_event


@pytest.fixture(autouse=True)
def _eventlog_on(monkeypatch):
    monkeypatch.setenv("MCP_SYNC_EVENTLOG", "on")


@pytest_asyncio.fixture
async def two_stores():
    d = tempfile.mkdtemp(prefix="phase4c-push-")
    src = SqliteVecMemoryStorage(os.path.join(d, "src.db"))
    await src.initialize()
    dst = SqliteVecMemoryStorage(os.path.join(d, "dst.db"))
    await dst.initialize()
    yield src, dst
    shutil.rmtree(d, ignore_errors=True)


class IngestPeerAdapter:
    """Simulates the hub POST /api/sync/events: applies pushed events to dst storage."""
    def __init__(self, dst_storage, allowed=None, fail_event_ids=None):
        self.dst = dst_storage
        self.allowed = allowed
        self.fail_event_ids = fail_event_ids or set()

    async def push_events(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        for ev in events:
            aid = (ev.get("agent_id") or "").strip()
            if not aid:
                return {"results": [], "applied": 0, "skipped": 0, "failed": len(events), "error": "missing agent_id"}
            if self.allowed is not None and aid not in self.allowed:
                return {"results": [], "applied": 0, "skipped": 0, "failed": len(events), "error": f"not allowed: {aid}"}
        results = []
        applied = skipped = failed = 0
        stop = False
        for ev in events:
            eid = ev.get("event_id", "")
            if stop:
                # hub applies in seq order and STOPS at the first failure; remaining
                # events are not applied (will be re-pushed). Report them as failed.
                results.append({"event_id": eid, "status": "failed"}); failed += 1
                continue
            if eid in self.fail_event_ids:
                results.append({"event_id": eid, "status": "failed"}); failed += 1
                stop = True
                continue
            res = apply_remote_event(self.dst, ev)
            if res.applied and res.materialized:
                results.append({"event_id": eid, "status": "applied"}); applied += 1
            elif res.applied:
                results.append({"event_id": eid, "status": "skipped_duplicate"}); skipped += 1
            else:
                results.append({"event_id": eid, "status": "failed"}); failed += 1
                stop = True
        return {"results": results, "applied": applied, "skipped": skipped, "failed": failed}


async def _store(storage, content, tags=("push",)):
    h = generate_content_hash(content)
    await storage.store(Memory(content=content, content_hash=h, tags=list(tags), memory_type="note"))
    return h


@pytest.mark.asyncio
async def test_ca4_push_materializes_on_peer(two_stores):
    src, dst = two_stores
    await _store(src, "Alpha unique content for push one")
    await _store(src, "Beta distinct content for push two")
    await _store(src, "Gamma separate content for push three")
    res = await push_to_peer(src, IngestPeerAdapter(dst), peer_id="hub", limit=100)
    assert isinstance(res, PushResult)
    assert res.events_pushed == 3, f"expected 3 pushed, got {res.events_pushed}"
    assert res.events_failed == 0
    for frag in ("Alpha unique", "Beta distinct", "Gamma separate"):
        row = dst.conn.execute("SELECT content FROM memories WHERE content LIKE ?", (f"%{frag}%",)).fetchone()
        assert row is not None, f"{frag} not materialized on peer"


@pytest.mark.asyncio
async def test_ca1_push_idempotent(two_stores):
    src, dst = two_stores
    await _store(src, "Idempotent push probe single")
    await push_to_peer(src, IngestPeerAdapter(dst), peer_id="hub")
    before = dst.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    advance_push_cursor(src, "hub", 0)
    res2 = await push_to_peer(src, IngestPeerAdapter(dst), peer_id="hub")
    after = dst.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    assert after == before, "re-push must not duplicate on peer"
    assert res2.events_failed == 0


@pytest.mark.asyncio
async def test_ca5_failure_does_not_advance_past_unapplied(two_stores):
    src, dst = two_stores
    h1 = await _store(src, "First event ok for resume test")
    await _store(src, "Second event will fail for resume test")
    h3 = await _store(src, "Third event blocked after failure")
    eid2 = src.conn.execute("SELECT event_id FROM sync_events WHERE seq=2").fetchone()[0]
    peer = IngestPeerAdapter(dst, fail_event_ids={eid2})
    res = await push_to_peer(src, peer, peer_id="hub", limit=100)
    assert get_push_cursor(src, "hub") == 1, f"cursor should be 1, got {get_push_cursor(src,'hub')}"
    assert res.events_failed >= 1
    assert dst.conn.execute("SELECT 1 FROM memories WHERE content_hash=?", (h1,)).fetchone() is not None
    assert dst.conn.execute("SELECT 1 FROM memories WHERE content_hash=?", (h3,)).fetchone() is None


@pytest.mark.asyncio
async def test_ca6_push_cursor_independent(two_stores):
    src, dst = two_stores
    assert get_push_cursor(src, "hub") == 0
    advance_push_cursor(src, "hub", 7)
    assert get_push_cursor(src, "hub") == 7
    sc = src.conn.execute("SELECT COUNT(*) FROM sync_cursor").fetchone()[0]
    assert sc == 0, "push must not write sync_cursor"


@pytest.mark.asyncio
async def test_ca3_allowlist_rejects_foreign_agent(two_stores):
    src, dst = two_stores
    await _store(src, "Event authored locally for allowlist test")
    peer_block = IngestPeerAdapter(dst, allowed={"zero"})
    src.conn.execute("UPDATE sync_events SET agent_id='intruder'")
    src.conn.commit()
    res = await push_to_peer(src, peer_block, peer_id="hub")
    assert dst.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 0
    assert get_push_cursor(src, "hub") == 0
    assert res.events_pushed == 0
