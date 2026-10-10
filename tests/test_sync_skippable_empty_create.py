"""Tests for the delta-sync 'poison event' deadlock fix.

Context (2026-10-10, PR #1499): a handful of hub memories had their content
cleared (quarantine->unquarantine). The feed clobbered the good payload content
with the empty table value, and the apply path fail-stopped the WHOLE feed on
the first such event — a few bad events blocked thousands of good ones.

Fixes under test:
  1. apply_remote_event marks an unmaterializable empty-content create as
     ApplyResult(applied=False, skippable=True) instead of a hard failure.
  2. sync_from_peer advances past a skippable event instead of fail-stopping,
     so later events still apply and the cursor advances (REAL path, not a copy).
  3. apply_remote_event repairs a previously-recorded empty-content create when
     the fixed feed resends the same identity with real content (upgrade case).
"""
import json
from typing import Any, Dict, List, Tuple

import pytest
import pytest_asyncio

from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.storage.sync.apply import ApplyResult, apply_remote_event
from mcp_memory_service.storage.sync.orchestrator import sync_from_peer


# --- Unit contract on ApplyResult -------------------------------------------

def test_apply_result_defaults_skippable_false():
    r = ApplyResult(applied=False, materialized=False, reason="Materialization failed")
    assert r.skippable is False


def test_apply_result_can_signal_skippable():
    r = ApplyResult(applied=False, materialized=False,
                    reason="skipped", skippable=True)
    assert r.applied is False and r.skippable is True


# --- Real-path fixtures ------------------------------------------------------

@pytest_asyncio.fixture
async def storage(tmp_path):
    db = str(tmp_path / "local.db")
    st = SqliteVecMemoryStorage(db)
    await st.initialize()
    yield st
    try:
        st.conn.close()
    except Exception:
        pass


class ScriptedPeer:
    """Peer adapter that yields a fixed, pre-built list of events (one page)."""

    def __init__(self, events: List[Dict[str, Any]]):
        self._events = events

    async def get_events_since(self, since_seq: int, limit: int) -> Tuple[List[Dict[str, Any]], int, bool]:
        page = [e for e in self._events if e["seq"] > since_seq][:limit]
        if not page:
            return [], since_seq, False
        return page, page[-1]["seq"], False


def _create_event(seq: int, eid: str, content: str, chash: str) -> Dict[str, Any]:
    return {
        "seq": seq, "agent_id": "peer", "event_id": eid, "op": "create",
        "content_hash": chash, "hlc_physical": 1000 + seq, "hlc_logical": 0,
        "embedding_model": None, "embedding_dim": None,
        "payload": {"content_hash": chash, "content": content, "tags": [],
                    "memory_type": "note", "metadata": {}},
    }


# --- 2: skippable empty create must NOT fail-stop the real pull --------------

@pytest.mark.asyncio
async def test_sync_from_peer_advances_past_empty_create(storage):
    """A poison empty-content create in the middle of the feed must be skipped,
    the later good event applied, and the cursor advanced past both."""
    events = [
        _create_event(1, "ev-good-1", "a real memory with enough content here", "h1"),
        _create_event(2, "ev-poison", "", "h2"),             # empty content -> skippable
        _create_event(3, "ev-good-2", "another real memory with content", "h3"),
    ]
    result = await sync_from_peer(storage, ScriptedPeer(events), "peer")

    # cursor advanced to the last event (did not stall on the poison one)
    row = storage.conn.execute(
        "SELECT last_seq_seen FROM sync_cursor WHERE peer_id = ?", ("peer",)
    ).fetchone()
    assert row is not None and row[0] == 3, "cursor must advance past the skipped event"

    # the two good memories materialized; the poison one did not
    def _has(chash):
        return storage.conn.execute(
            "SELECT 1 FROM memories WHERE content_hash = ? AND length(content) > 0", (chash,)
        ).fetchone() is not None
    assert _has("h1") and _has("h3"), "good events must materialize"
    assert not _has("h2"), "the empty-content event must not materialize a row"


# --- 3: content-repair upgrade case (previously stalled receiver) ------------

@pytest.mark.asyncio
async def test_apply_repairs_previously_empty_create(storage):
    """A receiver that recorded the empty-content create (pre-fix) must accept the
    SAME identity resent with real content (repair), not reject it as a replay."""
    chash = "hrepair"
    empty = _create_event(1, "ev-x", "", chash)
    # First the empty one lands (simulates the pre-fix stalled state).
    r1 = apply_remote_event(storage, empty)
    assert r1.skippable is True and r1.applied is False

    # The fixed feed resends the SAME identity with real content.
    repaired = _create_event(1, "ev-x", "the recovered real content for this memory", chash)
    r2 = apply_remote_event(storage, repaired)
    assert r2.applied is True and r2.materialized is True, "repair must apply, not reject"

    got = storage.conn.execute(
        "SELECT content FROM memories WHERE content_hash = ?", (chash,)
    ).fetchone()
    assert got and got[0] == "the recovered real content for this memory"


@pytest.mark.asyncio
async def test_apply_still_rejects_genuine_tampering(storage):
    """A same-identity replay that changes NON-empty content stays rejected
    (anti-replay protection must not be weakened by the repair path)."""
    chash = "htamper"
    original = _create_event(1, "ev-y", "the original authoritative content", chash)
    r1 = apply_remote_event(storage, original)
    assert r1.applied is True

    tampered = _create_event(1, "ev-y", "DIFFERENT tampered content injected", chash)
    r2 = apply_remote_event(storage, tampered)
    assert r2.applied is False and r2.skippable is False, "tampering must still be rejected"
    got = storage.conn.execute(
        "SELECT content FROM memories WHERE content_hash = ?", (chash,)
    ).fetchone()
    assert got[0] == "the original authoritative content", "stored content must not change"
