"""
RED tests for delta-sync audit fixes (#1522) — PROVE the bugs exist BEFORE the fix.

These tests must FAIL against current code (v12) and pass only after the fixes are implemented.
Each test targets a specific bug from the 1522 audit findings.

Bugs tested:
- #2: _materialize_event resurrects tombstones (INV-2a, INV-2b)
- #3: sync_from_peer never calls bootstrap_from_peer (INV-3a) 
- #4: embedding_pending never gets cleared (INV-4a)

VERIFICATION DISCIPLINE: These tests must FAIL for the right reasons (behavior bugs),
not incidental errors (imports, setup). Run and paste the actual FAILED outputs.
"""

import os
import json
import time
import tempfile
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Tuple
from unittest.mock import Mock, AsyncMock, patch

import pytest
import pytest_asyncio

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash
from mcp_memory_service.storage.sync.apply import apply_remote_event, _materialize_event
from mcp_memory_service.storage.sync.orchestrator import sync_from_peer
from mcp_memory_service.storage.sync.bootstrap import bootstrap_from_peer


@pytest_asyncio.fixture
async def store():
    """Create a clean test storage instance."""
    with tempfile.TemporaryDirectory() as tmp:
        s = SqliteVecMemoryStorage(os.path.join(tmp, "test.db"))
        await s.initialize()
        yield s
        await s.close()


def _create_event(content_hash, *, content, op="create", tags=None, store="default",
                  created_at=1000.0, updated_at=1000.0, agent="alpha", event_id="e1",
                  hlc_physical=10, hlc_logical=0, memory_type="note"):
    """Helper to create a sync event."""
    return {
        "seq": 1, "agent_id": agent, "event_id": event_id, "op": op,
        "content_hash": content_hash, "hlc_physical": hlc_physical, "hlc_logical": hlc_logical,
        "embedding_model": None, "embedding_dim": None,
        "payload": {
            "content_hash": content_hash, "content": content, "memory_type": memory_type,
            "tags": tags or [], "created_at": created_at, "updated_at": updated_at,
            "metadata": {}, "store": store,
        },
    }


def _create_delete_event(content_hash, *, deleted_at=2000.0, agent="alpha", event_id="del1",
                         hlc_physical=20, hlc_logical=0):
    """Helper to create a delete event."""
    return {
        "seq": 2, "agent_id": agent, "event_id": event_id, "op": "delete",
        "content_hash": content_hash, "hlc_physical": hlc_physical, "hlc_logical": hlc_logical,
        "embedding_model": None, "embedding_dim": None,
        "payload": {"deleted_at": deleted_at},
    }


class FakePeerAdapter:
    """Mock peer adapter for testing sync_from_peer."""
    
    def __init__(self, events, has_baseline=False):
        self.events = events
        self.has_baseline = has_baseline
        self.get_baseline_called = False
        self.get_events_since_called = False
        
    async def get_events_since(self, since_seq: int, limit: int) -> Tuple[List[Dict[str, Any]], int, bool]:
        """Return events after since_seq."""
        self.get_events_since_called = True
        # Filter events after since_seq
        filtered = [e for e in self.events if e.get("seq", 0) > since_seq]
        if not filtered:
            return [], since_seq, False
        # Return first batch
        batch = filtered[:limit]
        next_seq = max(e.get("seq", 0) for e in batch)
        has_more = len(filtered) > len(batch)
        return batch, next_seq, has_more
        
    async def get_baseline(self):
        """Return baseline if this peer supports it."""
        self.get_baseline_called = True
        if not self.has_baseline:
            raise AttributeError("get_baseline")
        return self.events, max(e.get("seq", 0) for e in self.events)


@pytest.mark.asyncio
async def test_create_does_not_resurrect_tombstone(store):
    """
    INV-2a (#2, 3-STATE model, Henry Ponto 2 2026-10-16): a v2 create whose recreate_of is
    the EMPTY sentinel "" (a producer that NEVER saw the delete) over a tombstone, with
    HLC > delete, must NOT resurrect it — the DELETE WINS.

    This replaces the old bare-create "stays dead" case (which contradicted absence=revive):
    now a BARE create (key absent, v1 peer) revives, and the case that stays dead is the
    EMPTY sentinel — a v2 peer that concurrently re-created without having seen another
    peer's delete. That is the real concurrent hole the 3-state model closes.
    """
    content = "Memory that should stay dead"
    h = generate_content_hash(content)

    # Step 1: Create memory normally
    create_event = _create_event(h, content=content, hlc_physical=10)
    result = apply_remote_event(store, create_event)
    assert result.applied, f"Initial create failed: {result.reason}"

    # Verify it exists and appears in search
    memory = await store.get_by_hash(h)
    assert memory is not None, "Memory should exist after create"

    # Check deleted_at directly from database (Memory model doesn't expose this field)
    cursor = store.conn.execute("SELECT deleted_at FROM memories WHERE content_hash = ?", (h,))
    row = cursor.fetchone()
    assert row is not None, "Memory row should exist"
    assert row[0] is None, "Memory should not be deleted initially"

    search_results = await store.search("Memory that should", n_results=5)
    assert len(search_results) > 0, "Memory should appear in search initially"

    # Step 2: Delete the memory (deleted_at NOT NULL). The CURRENT tombstone is 'del-current'.
    delete_event = _create_delete_event(h, deleted_at=2000.0, hlc_physical=20, event_id="del-current")
    result = apply_remote_event(store, delete_event)
    assert result.applied, f"Delete failed: {result.reason}"

    # Verify tombstone exists
    cursor = store.conn.execute("SELECT deleted_at FROM memories WHERE content_hash = ?", (h,))
    row = cursor.fetchone()
    assert row is not None, "Memory row should still exist"
    assert row[0] is not None, "Memory should have tombstone (deleted_at NOT NULL)"

    # Verify it doesn't appear in search
    search_results = await store.search("Memory that should", n_results=5)
    assert len(search_results) == 0, "Deleted memory should not appear in search"

    # Step 3: Apply a v2 create with HIGHER HLC and an EMPTY recreate_of "" (the producer
    # never saw the delete). This must NOT resurrect — the delete wins (3-state).
    resurrect_event = _create_event_empty(
        h, content=content, hlc_physical=30, event_id="resurrect",
    )
    result = apply_remote_event(store, resurrect_event)
    assert result.applied, f"Later create failed: {result.reason}"

    # CRITICAL ASSERTION: Memory must REMAIN deleted (tombstone preserved)
    cursor = store.conn.execute("SELECT deleted_at FROM memories WHERE content_hash = ?", (h,))
    row = cursor.fetchone()
    assert row is not None, "Memory row should exist"
    assert row[0] is not None, "Tombstone must be preserved - an EMPTY recreate_of must not revive"

    # CRITICAL ASSERTION: Memory must NOT appear in search
    search_results = await store.search("Memory that should", n_results=5)
    assert len(search_results) == 0, "A v2 empty recreate_of must not resurrect the tombstone"


@pytest.mark.asyncio
async def test_legit_create_without_tombstone_materializes(store):
    """
    INV-2b (guard against regression): create without prior tombstone materializes normally.
    
    This test should PASS today and continue passing after the fix.
    It proves the fix doesn't break legitimate creates.
    """
    content = "Fresh memory with no tombstone"
    h = generate_content_hash(content)
    
    # Create memory for first time (no prior tombstone)
    create_event = _create_event(h, content=content, hlc_physical=10)
    result = apply_remote_event(store, create_event)
    assert result.applied, f"Create failed: {result.reason}"
    
    # Verify it materializes normally
    memory = await store.get_by_hash(h)
    assert memory is not None, "Memory should exist after create"
    
    # Check deleted_at directly from database
    cursor = store.conn.execute("SELECT deleted_at FROM memories WHERE content_hash = ?", (h,))
    row = cursor.fetchone()
    assert row is not None, "Memory row should exist"
    assert row[0] is None, "Memory should not be deleted"
    
    assert memory.content == content, "Content should be preserved"
    
    # Verify it appears in search
    search_results = await store.search("Fresh memory", n_results=5)
    assert len(search_results) > 0, "Memory should appear in search"


@pytest.mark.asyncio  
async def test_first_sync_calls_bootstrap(store):
    """
    INV-3a (#3): sync_from_peer with no cursor + peer supporting get_baseline must call bootstrap.
    
    TODAY THIS FAILS: orchestrator.py:69-75 never calls bootstrap_from_peer.
    """
    content = "Bootstrap baseline memory"
    h = generate_content_hash(content)
    baseline_events = [_create_event(h, content=content, hlc_physical=10)]
    
    # Create peer that supports baseline
    peer = FakePeerAdapter(baseline_events, has_baseline=True)
    
    # Mock bootstrap_from_peer where the orchestrator looks it up (it imports the name
    # into its own namespace: `from .bootstrap import bootstrap_from_peer`).
    with patch('mcp_memory_service.storage.sync.orchestrator.bootstrap_from_peer') as mock_bootstrap:
        mock_bootstrap.return_value = {"memories_installed": 1, "watermark": 10}
        
        # Sync from peer with no existing cursor (first sync)
        result = await sync_from_peer(store, peer, "test_peer")
        
        # CRITICAL ASSERTION: bootstrap must have been called
        assert mock_bootstrap.called, "bootstrap_from_peer should be called on first sync with baseline-capable peer"
        
        # Verify bootstrap was called with correct args
        call_args = mock_bootstrap.call_args[0]
        assert call_args[0] == store, "bootstrap should be called with correct storage"
        assert call_args[1] == peer, "bootstrap should be called with correct peer"
        assert call_args[2] == "test_peer", "bootstrap should be called with correct peer_id"


@pytest.mark.asyncio
async def test_existing_cursor_skips_bootstrap(store):
    """
    INV-3a complement: sync_from_peer with existing cursor must NOT call bootstrap.
    
    This should PASS today (negative test - we don't expect bootstrap to be called).
    """
    content = "Regular sync memory"
    h = generate_content_hash(content)
    sync_events = [_create_event(h, content=content, hlc_physical=10)]
    
    # Create existing sync cursor for this peer
    store.conn.execute(
        "INSERT OR REPLACE INTO sync_cursor (peer_id, last_seq_seen, updated_at) VALUES (?, ?, ?)",
        ("existing_peer", 5, time.time())
    )
    store.conn.commit()
    
    # Create peer that supports baseline 
    peer = FakePeerAdapter(sync_events, has_baseline=True)
    
    # Mock bootstrap_from_peer where the orchestrator looks it up (same target as the
    # positive test) — patching the source module would make this guard test vacuous.
    with patch('mcp_memory_service.storage.sync.orchestrator.bootstrap_from_peer') as mock_bootstrap:
        # Sync from peer with existing cursor 
        result = await sync_from_peer(store, peer, "existing_peer")
        
        # ASSERTION: bootstrap must NOT be called when cursor exists
        assert not mock_bootstrap.called, "bootstrap_from_peer should not be called when cursor exists"


@pytest.mark.asyncio
async def test_first_sync_installs_baseline_and_resumes_at_watermark(store):
    """INV-3a (deeper, per Greptile): a real bootstrap (no mock) on first sync must install
    the baseline AND park the cursor at the peer's watermark, so the following pull resumes
    from the watermark — not from 0. Proves the cursor re-read is load-bearing."""
    base_hash = generate_content_hash("baseline memory")
    later_hash = generate_content_hash("post-baseline edit")
    # Baseline carries one memory at seq 5 (watermark 5); a later event sits at seq 6.
    baseline = [_create_event(base_hash, content="baseline memory", event_id="b1",
                              hlc_physical=10)]
    baseline[0]["seq"] = 5
    later = _create_event(later_hash, content="post-baseline edit", event_id="l1",
                          hlc_physical=20)
    later["seq"] = 6
    # Peer serves baseline (watermark 5) and, on get_events_since, only the later event.
    peer = FakePeerAdapter([baseline[0], later], has_baseline=True)

    await sync_from_peer(store, peer, "wm_peer")

    # The baseline memory is installed...
    assert await store.get_by_hash(base_hash) is not None, "baseline memory must be installed"
    # ...the later event (seq 6 > watermark 5) is pulled...
    assert await store.get_by_hash(later_hash) is not None, "post-watermark event must be pulled"
    # ...and the cursor is at the watermark-or-beyond, never reset to 0.
    cur = store.conn.execute(
        "SELECT last_seq_seen FROM sync_cursor WHERE peer_id = ?", ("wm_peer",)
    ).fetchone()
    assert cur is not None and cur[0] >= 5, f"cursor must resume at the watermark, got {cur}"


@pytest.mark.asyncio
async def test_first_sync_falls_back_when_baseline_fails(store):
    """INV-3a fallback (per Greptile): if the baseline fetch fails, first sync must fall back
    to pull-from-0 and still converge, not abort."""
    h = generate_content_hash("fallback memory")
    ev = _create_event(h, content="fallback memory", event_id="f1", hlc_physical=10)
    ev["seq"] = 1

    class FailingBaselinePeer(FakePeerAdapter):
        async def get_baseline(self):
            self.get_baseline_called = True
            raise RuntimeError("baseline endpoint down")

    peer = FailingBaselinePeer([ev], has_baseline=True)
    # Must not raise; falls back to pull-from-0 and applies the event.
    await sync_from_peer(store, peer, "fb_peer")
    assert await store.get_by_hash(h) is not None, "event must be pulled via fallback-from-0"


@pytest.mark.asyncio
async def test_reembed_respects_named_store(store):
    """#4/#1521 (per Greptile): the repair must write each vector into the memory's OWN store
    partition, not 'default'. A memory in a named store must be found by a search scoped to
    that store after repair."""
    import importlib.util
    content = "Memory living in a named store"
    h = generate_content_hash(content)
    # Insert into a NAMED store with embedding_pending=1 and no vector.
    store.conn.execute("""
        INSERT INTO memories
        (content_hash, content, tags, memory_type, metadata, created_at, created_at_iso,
         updated_at, updated_at_iso, deleted_at, embedding_pending, store, superseded_by)
        VALUES (?, ?, '', 'note', '{}', ?, ?, ?, ?, NULL, 1, 'projectX', NULL)
    """, (h, content, 1000.0, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(1000.0)),
          1000.0, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(1000.0))))
    store.conn.commit()

    _repair_path = Path(__file__).resolve().parents[2] / "scripts" / "maintenance" / "repair_missing_embeddings_onnx.py"
    _spec = importlib.util.spec_from_file_location("repair_missing_embeddings_onnx", _repair_path)
    _repair = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_repair)
    await _repair.reembed_missing_embeddings(store)

    # The vector must land in the 'projectX' partition, not 'default'.
    rowid = store.conn.execute("SELECT id FROM memories WHERE content_hash = ?", (h,)).fetchone()[0]
    vec_store = store.conn.execute(
        "SELECT store FROM memory_embeddings WHERE rowid = ?", (rowid,)
    ).fetchone()
    assert vec_store is not None and vec_store[0] == "projectX", \
        f"vector must be written into the memory's own store partition, got {vec_store}"
    # And the flag is cleared.
    pend = store.conn.execute(
        "SELECT embedding_pending FROM memories WHERE content_hash = ?", (h,)
    ).fetchone()[0]
    assert pend == 0, "embedding_pending must be cleared after repair"


@pytest.mark.asyncio
async def test_reembed_clears_pending(store):
    """
    INV-4a (#4): re-embedding a memory with embedding_pending=1 must clear the flag and make it searchable.
    
    TODAY THIS FAILS: nothing in the codebase sets embedding_pending=0 after successful re-embed.
    """
    content = "Memory with pending embedding"
    h = generate_content_hash(content)
    
    # Step 1: Create memory directly in DB with embedding_pending=1 (simulate model mismatch)
    store.conn.execute("""
        INSERT INTO memories 
        (content_hash, content, tags, memory_type, metadata, created_at, created_at_iso, 
         updated_at, updated_at_iso, deleted_at, embedding_pending, store, superseded_by)
        VALUES (?, ?, '', 'note', '{}', ?, ?, ?, ?, NULL, 1, 'default', NULL)
    """, (
        h, content, 
        1000.0, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(1000.0)),
        1000.0, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(1000.0))
    ))
    store.conn.commit()
    
    # Verify memory exists but has embedding_pending=1
    cursor = store.conn.execute(
        "SELECT embedding_pending FROM memories WHERE content_hash = ?", (h,)
    )
    row = cursor.fetchone()
    assert row is not None, "Memory should exist"
    assert row[0] == 1, "Memory should have embedding_pending=1"
    
    # Verify it does NOT appear in search (due to pending flag)
    search_results = await store.search("Memory with pending", n_results=5)
    assert len(search_results) == 0, "Memory with embedding_pending=1 should not appear in search"
    
    # Step 2: Re-embed under the matching model via the real maintenance path
    # (#1522 finding #4). The test must exercise the actual re-embed function, not
    # hand-simulate the INSERT — the bug is precisely that the re-embed path never
    # cleared embedding_pending.
    import importlib.util
    _repair_path = Path(__file__).resolve().parents[2] / "scripts" / "maintenance" / "repair_missing_embeddings_onnx.py"
    _spec = importlib.util.spec_from_file_location("repair_missing_embeddings_onnx", _repair_path)
    _repair = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_repair)
    try:
        await _repair.reembed_missing_embeddings(store)
    except Exception as e:
        pytest.fail(f"Re-embed failed: {e}")
    
    # CRITICAL ASSERTION: embedding_pending should be cleared to 0
    cursor = store.conn.execute(
        "SELECT embedding_pending FROM memories WHERE content_hash = ?", (h,)
    )
    row = cursor.fetchone()
    assert row is not None, "Memory should still exist"
    assert row[0] == 0, "embedding_pending should be cleared to 0 after re-embed"
    
    # CRITICAL ASSERTION: Memory should now appear in search 
    search_results = await store.search("Memory with pending", n_results=5)
    assert len(search_results) > 0, "Memory should appear in search after embedding_pending cleared"


# ===========================================================================
# #1522 finding #2 — recreate_of intentionality signal (ADR-0012 v2)
# Henry's final decision (2026-10-16): recreate_of on the create PAYLOAD,
# referencing the delete's event_id; absence = revive (zero regression);
# NO schema_version bump (would fail-stop v1 peers); apply revives only when
# recreate_of names the current tombstone. Sync-path only (REST-path is #1560).
# These are RED until store.py (producer) + apply.py (consumer) implement it.
# ===========================================================================


def _create_event_recreate(content_hash, *, content, recreate_of, agent="alpha",
                           event_id="recreate", hlc_physical=30, store="default"):
    """A create event that legitimately supersedes a named tombstone (recreate_of = delete's event_id)."""
    ev = _create_event(content_hash, content=content, agent=agent, event_id=event_id,
                       hlc_physical=hlc_physical, store=store)
    ev["payload"]["recreate_of"] = recreate_of
    return ev


def _create_event_empty(content_hash, *, content, agent="alpha", event_id="recreate",
                        hlc_physical=30, store="default"):
    """A v2 create whose recreate_of is the EMPTY sentinel "" — a producer that NEVER saw a
    delete. Distinct from a bare create (_create_event, key ABSENT = v1 peer)."""
    ev = _create_event(content_hash, content=content, agent=agent, event_id=event_id,
                       hlc_physical=hlc_physical, store=store)
    ev["payload"]["recreate_of"] = ""
    return ev


@pytest.mark.asyncio
async def test_recreate_signal_discriminates_stale_vs_valid(store):
    """
    INV-2a + INV-2b CONTRAST (the discriminating test, reconciled 2026-10-16): the SAME
    physical sequence (create -> delete -> later create, same hash, HLC > delete) must
    yield OPPOSITE outcomes depending ONLY on the recreate_of signal:
      - recreate_of names a STALE/unrelated delete -> stays DELETED (INV-2a)
      - recreate_of names the CURRENT delete event_id -> REVIVES (INV-2b, legit re-create)
      - a BARE create (absence of recreate_of) also REVIVES (Henry: absence = revive) —
        asserted here too so the discriminator cannot pass by accident on the old bug.

    The genuine discriminator is "stale vs current recreate_of", NOT "bare vs signalled":
    Henry's reconciled rule makes absence revive, so bare and valid share an outcome and
    only a STALE signal preserves the tombstone.
    """
    content = "Discriminator memory"

    # --- Case A: bare create (absence of signal) now REVIVES (absence = revive) ---
    hA = generate_content_hash(content + " A")
    assert apply_remote_event(store, _create_event(hA, content=content + " A", hlc_physical=10, event_id="createA")).applied
    assert apply_remote_event(store, _create_delete_event(hA, deleted_at=2000.0, hlc_physical=20, event_id="delA")).applied
    assert apply_remote_event(store, _create_event(hA, content=content + " A", hlc_physical=30, event_id="bareA")).applied
    rowA = store.conn.execute("SELECT deleted_at FROM memories WHERE content_hash = ?", (hA,)).fetchone()
    assert rowA is not None and rowA[0] is None, \
        "absence = revive: a bare create over a tombstone (HLC>delete, no recreate_of) MUST revive"

    # --- Case B: signalled create naming the CURRENT delete MUST REVIVE ---
    hB = generate_content_hash(content + " B")
    assert apply_remote_event(store, _create_event(hB, content=content + " B", hlc_physical=10, event_id="createB")).applied
    assert apply_remote_event(store, _create_delete_event(hB, deleted_at=2000.0, hlc_physical=20, event_id="delB")).applied
    assert apply_remote_event(store, _create_event_recreate(hB, content=content + " B", recreate_of="delB", hlc_physical=30, event_id="reB")).applied
    rowB = store.conn.execute("SELECT deleted_at FROM memories WHERE content_hash = ?", (hB,)).fetchone()
    assert rowB is not None and rowB[0] is None, \
        "INV-2b: a create carrying recreate_of=<current delete> MUST revive (deleted_at cleared)"

    # --- Case C: signalled create naming a STALE delete MUST STAY DEAD (the discriminator) ---
    hC = generate_content_hash(content + " C")
    assert apply_remote_event(store, _create_event(hC, content=content + " C", hlc_physical=10, event_id="createC")).applied
    assert apply_remote_event(store, _create_delete_event(hC, deleted_at=2000.0, hlc_physical=20, event_id="delC-current")).applied
    assert apply_remote_event(store, _create_event_recreate(hC, content=content + " C", recreate_of="delC-stale", hlc_physical=30, event_id="reC")).applied
    rowC = store.conn.execute("SELECT deleted_at FROM memories WHERE content_hash = ?", (hC,)).fetchone()
    assert rowC is not None and rowC[0] is not None, \
        "INV-2a: a create carrying a STALE recreate_of (not the current delete) MUST stay deleted"

    # --- Case D: v2 EMPTY recreate_of "" (producer never saw the delete) MUST STAY DEAD ---
    # (3-state model, Henry Ponto 2): distinct from Case A (bare/absent = v1 = revive). The
    # EMPTY sentinel means "delete wins" — the real concurrent hole the model closes.
    hD = generate_content_hash(content + " D")
    assert apply_remote_event(store, _create_event(hD, content=content + " D", hlc_physical=10, event_id="createD")).applied
    assert apply_remote_event(store, _create_delete_event(hD, deleted_at=2000.0, hlc_physical=20, event_id="delD")).applied
    assert apply_remote_event(store, _create_event_empty(hD, content=content + " D", hlc_physical=30, event_id="emptyD")).applied
    rowD = store.conn.execute("SELECT deleted_at FROM memories WHERE content_hash = ?", (hD,)).fetchone()
    assert rowD is not None and rowD[0] is not None, \
        "3-state: a v2 EMPTY recreate_of over a tombstone MUST stay dead (delete wins) — unlike bare (Case A)"


@pytest.mark.asyncio
async def test_recreate_of_names_stale_delete_does_not_revive(store):
    """
    recreate_of must name the CURRENT tombstone. A create whose recreate_of points at an
    OLD/unrelated event_id (not the tombstone on the row) must NOT revive — it is not a
    legitimate supersession of the delete the node actually holds.
    RED today: apply has no recreate_of logic at all.
    """
    content = "Memory with a stale recreate_of"
    h = generate_content_hash(content)

    assert apply_remote_event(store, _create_event(h, content=content, hlc_physical=10)).applied
    # current tombstone is del-NEW
    assert apply_remote_event(store, _create_delete_event(h, deleted_at=2000.0, hlc_physical=20, event_id="del-NEW")).applied

    # create names a DIFFERENT (stale) delete event_id
    recreate = _create_event_recreate(h, content=content, recreate_of="del-OLD-unrelated", hlc_physical=30)
    result = apply_remote_event(store, recreate)
    assert result.applied, f"apply should not error: {result.reason}"

    cursor = store.conn.execute("SELECT deleted_at FROM memories WHERE content_hash = ?", (h,))
    row = cursor.fetchone()
    assert row[0] is not None, "Naming a stale/unrelated delete must NOT revive the current tombstone"


@pytest.mark.asyncio
async def test_old_peer_bare_recreate_revives_and_unknown_key_never_fail_stops(store):
    """
    BACK-COMPAT (Henry's explicit ask, REQ-2.5), rewritten to PROVE the older-peer case honestly.

    An older (v1) producer never stamps `recreate_of`. We SIMULATE that peer by emitting a
    re-create over a tombstone with the key STRIPPED from the payload (exactly what a v1 peer
    would send), then applying it. Absence = revive, so the memory must come back ALIVE and the
    apply must not raise/stall — the new key is additive, no schema_version bump.

    Part 2 also guards that an unknown additive key on a FRESH hash (no tombstone) is ignored
    without fail-stop. The old test name/claim ('ignores unknown key') only exercised Part 2;
    the real back-compat guarantee is the bare-recreate revive proven in Part 1.
    """
    # --- Part 1: older peer re-creates over a tombstone with NO recreate_of → revive ---
    content = "Memory a v1 peer re-creates after delete"
    h = generate_content_hash(content)
    assert apply_remote_event(store, _create_event(h, content=content, hlc_physical=10, event_id="c0")).applied
    assert apply_remote_event(store, _create_delete_event(h, deleted_at=2000.0, hlc_physical=20, event_id="d1")).applied
    # v1 re-create: build a create that WOULD carry the key, then strip it to mimic an old peer
    # that does not know recreate_of at all.
    v1_recreate = _create_event(h, content=content, hlc_physical=30, event_id="c2-v1")
    v1_recreate["payload"].pop("recreate_of", None)  # old peer: key absent entirely
    assert "recreate_of" not in v1_recreate["payload"]
    result = apply_remote_event(store, v1_recreate)
    assert result.applied, f"v1 bare re-create must apply without error: {result.reason}"
    row = store.conn.execute("SELECT deleted_at FROM memories WHERE content_hash = ?", (h,)).fetchone()
    assert row is not None and row[0] is None, \
        "older-peer bare re-create over a tombstone MUST revive (absence = revive, back-compat)"

    # --- Part 2: unknown additive key on a FRESH hash must never fail-stop a v1 apply ---
    content2 = "Fresh memory carrying an unknown additive key"
    h2 = generate_content_hash(content2)
    ev = _create_event(h2, content=content2, hlc_physical=10, event_id="fresh")
    ev["payload"]["recreate_of"] = "some-delete-id"  # unknown to a v1 apply; fresh hash → no tombstone
    result = apply_remote_event(store, ev)
    assert result.applied, "an unknown additive key must never fail-stop apply (no schema bump)"
    assert await store.get_by_hash(h2) is not None, "fresh create with an additive key still materializes"


# ===========================================================================
# #1522 P1 CONVERGENCE (Greptile apply:674) — liveness is a function of the
# event SET, not of arrival order. Product decision (Claudio 2026-10-16) =
# variant iii': {C0, D1@HLC20, C2@HLC30 with STALE recreate_of} => DEAD in
# EVERY order; VALID/BARE recreate_of => LIVE in every order. Same deleted_at.
# These FAIL if _reconcile_liveness / recompute_liveness is neutralized.
# ===========================================================================

import itertools


def _apply_set_in_order(store_obj, events_by_name, order):
    """Apply the named events to a storage in the given arrival order."""
    for name in order:
        res = apply_remote_event(store_obj, events_by_name[name])
        assert res.applied, f"event {name} should apply (reason={res.reason})"


async def _final_deleted_at(tmp_factory, events_by_name, order):
    """Apply the event set in `order` on a FRESH storage; return final deleted_at for the hash."""
    s = tmp_factory()
    await s.initialize()
    try:
        _apply_set_in_order(s, events_by_name, order)
        h = events_by_name[order[0]]["content_hash"]
        row = s.conn.execute("SELECT deleted_at FROM memories WHERE content_hash = ?", (h,)).fetchone()
        return row[0] if row is not None else "NOROW"
    finally:
        await s.close()


def _make_convergence_set(recreate_of):
    """Build {C0 create@10, D1 delete@20, C2 re-create@30(recreate_of=...)} for ONE hash."""
    content = "convergence memory"
    h = generate_content_hash(content)
    return h, {
        "C0": _create_event(h, content=content, hlc_physical=10, event_id="C0"),
        "D1": _create_delete_event(h, deleted_at=2000.0, hlc_physical=20, event_id="D1"),
        "C2": (_create_event_recreate(h, content=content, recreate_of=recreate_of,
                                      hlc_physical=30, event_id="C2")
               if recreate_of is not None
               else _create_event(h, content=content, hlc_physical=30, event_id="C2")),
    }


@pytest.fixture
def tmp_store_factory():
    """Yield a factory that builds a fresh SqliteVecMemoryStorage in a throwaway dir."""
    import tempfile as _tf
    dirs = []

    def factory():
        d = _tf.TemporaryDirectory()
        dirs.append(d)
        return SqliteVecMemoryStorage(os.path.join(d.name, "test.db"))

    yield factory
    for d in dirs:
        d.cleanup()


@pytest.mark.asyncio
async def test_converge_recreate_stale__all_orders(tmp_store_factory):
    """STALE recreate_of (names a delete not in the set) => DEAD in ALL 6 arrival orders,
    with the SAME deterministic deleted_at (from D1's payload, never time.time())."""
    _h, events = _make_convergence_set(recreate_of="del-STALE-unrelated")
    outcomes = set()
    for order in itertools.permutations(["C0", "D1", "C2"]):
        outcomes.add(await _final_deleted_at(tmp_store_factory, events, order))
    assert outcomes == {2000.0}, (
        f"stale recreate_of must converge to DEAD(deleted_at=2000.0) in every order, got {outcomes}"
    )


@pytest.mark.asyncio
async def test_converge_recreate_valid__all_orders(tmp_store_factory):
    """VALID recreate_of (names the dominant delete D1) => LIVE (deleted_at None) in ALL orders."""
    _h, events = _make_convergence_set(recreate_of="D1")
    outcomes = set()
    for order in itertools.permutations(["C0", "D1", "C2"]):
        outcomes.add(await _final_deleted_at(tmp_store_factory, events, order))
    assert outcomes == {None}, f"valid recreate_of must converge to LIVE in every order, got {outcomes}"


@pytest.mark.asyncio
async def test_converge_bare_recreate__all_orders(tmp_store_factory):
    """BARE create (no recreate_of), HLC30 > delete@20 => absence = revive => LIVE in ALL orders."""
    _h, events = _make_convergence_set(recreate_of=None)
    outcomes = set()
    for order in itertools.permutations(["C0", "D1", "C2"]):
        outcomes.add(await _final_deleted_at(tmp_store_factory, events, order))
    assert outcomes == {None}, f"bare re-create (absence=revive) must converge to LIVE in every order, got {outcomes}"


@pytest.mark.asyncio
async def test_converge_delete_loses_resolver_still_tombstones(tmp_store_factory):
    """Fonte 1, Order B: a create@30 materializes ALIVE first, then a delete@20 arrives and
    LOSES the resolver. The lost delete must STILL re-impose the tombstone (same result as
    delete-first), because the create carries a STALE recreate_of — proving a losing delete
    reconciles liveness from the set rather than no-op'ing."""
    _h, events = _make_convergence_set(recreate_of="del-STALE-unrelated")
    create_first = await _final_deleted_at(tmp_store_factory, events, ("C0", "C2", "D1"))
    delete_first = await _final_deleted_at(tmp_store_factory, events, ("C0", "D1", "C2"))
    assert create_first == delete_first == 2000.0, (
        f"a delete that loses the resolver must still tombstone (Order B); "
        f"create_first={create_first} delete_first={delete_first}"
    )


@pytest.mark.asyncio
async def test_converge_partial_arrival_monotonic(tmp_store_factory):
    """Partial arrival: {C2 stale} alone, then +D1, then +C0. The final state (full set) is the
    same DEAD as every other order; no intermediate subset resurrects permanently."""
    _h, events = _make_convergence_set(recreate_of="del-STALE-unrelated")
    s = tmp_store_factory()
    await s.initialize()
    try:
        h = events["C2"]["content_hash"]
        # C2 alone: a lone create (no delete in the set yet) is LIVE — nothing to tombstone.
        apply_remote_event(s, events["C2"])
        row = s.conn.execute("SELECT deleted_at FROM memories WHERE content_hash = ?", (h,)).fetchone()
        assert row is not None and row[0] is None, "a lone create with no delete in the set is LIVE"
        # + D1 (the delete the stale recreate_of does NOT name): now a delete exists and the
        # stale signal cannot supersede it → DEAD, deterministic deleted_at.
        apply_remote_event(s, events["D1"])
        row = s.conn.execute("SELECT deleted_at FROM memories WHERE content_hash = ?", (h,)).fetchone()
        assert row is not None and row[0] == 2000.0, f"after +D1 the stale re-create is DEAD, got {row}"
        # + C0: the full set; still DEAD (same final state as all permutations).
        apply_remote_event(s, events["C0"])
        row = s.conn.execute("SELECT deleted_at FROM memories WHERE content_hash = ?", (h,)).fetchone()
        assert row is not None and row[0] == 2000.0, f"full set converges to DEAD, got {row}"
    finally:
        await s.close()


@pytest.mark.asyncio
async def test_converge_two_deletes_pick_dominant(tmp_store_factory):
    """Fonte 2 (tied/multi deletes): {C0, D_old@15, D1@20 (dominant), C2(recreate_of=D_old STALE)}.
    recreate_of names the NON-dominant (older) delete → STALE → DEAD in every order. Proves the
    dominant-delete election uses the full set, not a partial 'last local delete'."""
    content = "two-delete convergence memory"
    h = generate_content_hash(content)
    events = {
        "C0": _create_event(h, content=content, hlc_physical=10, event_id="C0"),
        "Dold": _create_delete_event(h, deleted_at=1500.0, hlc_physical=15, event_id="D_old"),
        "D1": _create_delete_event(h, deleted_at=2000.0, hlc_physical=20, event_id="D1"),
        "C2": _create_event_recreate(h, content=content, recreate_of="D_old",
                                     hlc_physical=30, event_id="C2"),
    }
    outcomes = set()
    # sample a spread of orders including the two critical ones + a few shuffles
    orders = [
        ("C0", "Dold", "D1", "C2"),
        ("C2", "D1", "Dold", "C0"),
        ("D1", "C2", "C0", "Dold"),
        ("C0", "C2", "Dold", "D1"),
        ("C2", "C0", "D1", "Dold"),
    ]
    for order in orders:
        outcomes.add(await _final_deleted_at(tmp_store_factory, events, order))
    # recreate_of=D_old is STALE (dominant delete is D1@20) → DEAD, deleted_at of the DOMINANT delete.
    assert outcomes == {2000.0}, (
        f"recreate_of naming the non-dominant delete is STALE → DEAD with the dominant delete's "
        f"deleted_at (2000.0) in every order, got {outcomes}"
    )


@pytest.mark.asyncio
async def test_converge_repull_is_idempotent_noop(tmp_store_factory):
    """Re-applying the whole set a 2nd time does not change the state (idempotency must not mask
    or flip the converged outcome)."""
    _h, events = _make_convergence_set(recreate_of="del-STALE-unrelated")
    s = tmp_store_factory()
    await s.initialize()
    try:
        h = events["C2"]["content_hash"]
        order = ("C0", "D1", "C2")
        _apply_set_in_order(s, events, order)
        first = s.conn.execute("SELECT deleted_at FROM memories WHERE content_hash = ?", (h,)).fetchone()[0]
        # second pass: every event is a benign duplicate (idempotent, no rewrite)
        for name in order:
            apply_remote_event(s, events[name])
        second = s.conn.execute("SELECT deleted_at FROM memories WHERE content_hash = ?", (h,)).fetchone()[0]
        assert first == second == 2000.0, f"re-pull must be a no-op; first={first} second={second}"
    finally:
        await s.close()


# ===========================================================================
# #1522 Ponto 2 Henry (3-STATE) — the recreate_of key has THREE meanings:
#   ABSENT (key not in payload) = v1 peer          → revive (back-compat)
#   EMPTY  (key present == "")   = v2 never saw del → DELETE WINS (DEAD)
#   NAMED  (names dominant delete) = valid supers.  → revive
# The EMPTY branch is the real concurrent hole the PR closes. These FAIL if
# the EMPTY branch in recompute_liveness is neutralized.
# ===========================================================================


def _make_concurrent_set():
    """The REAL concurrent case (Henry): C2 is a v2 create with recreate_of="" (this peer
    never saw the delete); D_other is a delete emitted by ANOTHER peer for the SAME hash.
    Same base hash. The EMPTY sentinel must make the delete win in BOTH arrival orders."""
    content = "concurrent v2 empty vs peer delete"
    h = generate_content_hash(content)
    return h, {
        "C2": _create_event_empty(h, content=content, hlc_physical=30, event_id="C2-empty", agent="peerA"),
        "Dother": _create_delete_event(h, deleted_at=2000.0, hlc_physical=20,
                                       event_id="D-other", agent="peerB"),
    }


@pytest.mark.asyncio
async def test_concurrent_v2_empty_recreate_of_vs_peer_delete__both_orders(tmp_store_factory):
    """THE HENRY CASE (load-bearing): a v2 create with recreate_of="" (a peer that never saw
    the delete) vs a delete D_other from ANOTHER peer, for the same hash. In BOTH arrival
    orders (create-first and delete-first, create HLC30 > delete HLC20) the memory must be
    DEAD with the delete's deterministic deleted_at. This is the concurrent hole the PR's
    first (2-state) design did NOT close — the EMPTY branch makes the delete win regardless
    of order/HLC. Neutralizing the EMPTY branch (treat "" as absent) makes this RED."""
    _h, events = _make_concurrent_set()
    create_first = await _final_deleted_at(tmp_store_factory, events, ("C2", "Dother"))
    delete_first = await _final_deleted_at(tmp_store_factory, events, ("Dother", "C2"))
    assert create_first == delete_first == 2000.0, (
        f"v2 empty recreate_of vs a peer's delete must converge to DEAD in BOTH orders; "
        f"create_first={create_first} delete_first={delete_first}"
    )


def _make_empty_convergence_set():
    """{C0 create@10, D1 delete@20, C2 v2-empty-recreate@30} for ONE hash."""
    content = "empty recreate convergence memory"
    h = generate_content_hash(content)
    return h, {
        "C0": _create_event(h, content=content, hlc_physical=10, event_id="C0"),
        "D1": _create_delete_event(h, deleted_at=2000.0, hlc_physical=20, event_id="D1"),
        "C2": _create_event_empty(h, content=content, hlc_physical=30, event_id="C2"),
    }


@pytest.mark.asyncio
async def test_v2_empty_recreate_of_with_delete_stays_dead__all_orders(tmp_store_factory):
    """State 2 (EMPTY + delete present): DEAD in ALL 6 arrival orders, deterministic
    deleted_at from D1's payload. Distinct from the bare (absent) case which revives."""
    _h, events = _make_empty_convergence_set()
    outcomes = set()
    for order in itertools.permutations(["C0", "D1", "C2"]):
        outcomes.add(await _final_deleted_at(tmp_store_factory, events, order))
    assert outcomes == {2000.0}, (
        f"v2 EMPTY recreate_of + delete must converge to DEAD(2000.0) in every order, got {outcomes}"
    )


@pytest.mark.asyncio
async def test_v1_absent_recreate_of_with_delete_revives__all_orders(tmp_store_factory):
    """State 1 (ABSENT + delete present): a v1 bare create (key absent) over a tombstone
    REVIVES in ALL 6 orders (absence = revive, back-compat). This is the explicit CONTRAST
    to the EMPTY case above: same shape, opposite outcome, driven ONLY by key presence."""
    _h, events = _make_convergence_set(recreate_of=None)  # None → _create_event (key ABSENT)
    outcomes = set()
    for order in itertools.permutations(["C0", "D1", "C2"]):
        outcomes.add(await _final_deleted_at(tmp_store_factory, events, order))
    assert outcomes == {None}, (
        f"v1 ABSENT recreate_of + delete must converge to LIVE in every order, got {outcomes}"
    )


@pytest.mark.asyncio
async def test_inv2c_fresh_create_with_empty_recreate_of_materializes_live(store):
    """INV-2c REVISED (Henry 3-state): a FRESH hash (no prior tombstone) whose v2 create
    carries recreate_of="" materializes LIVE. The empty sentinel is INERT without deletes —
    recompute_liveness returns LIVE before ever reading recreate_of (the `if not deletes`
    short-circuit). Proves the "stamp on every create" rule does not block fresh creates."""
    content = "Fresh v2 create carrying empty recreate_of"
    h = generate_content_hash(content)
    ev = _create_event_empty(h, content=content, hlc_physical=10, event_id="fresh-empty")
    assert "recreate_of" in ev["payload"] and ev["payload"]["recreate_of"] == "", \
        "the fixture must carry the empty sentinel"
    result = apply_remote_event(store, ev)
    assert result.applied, f"fresh v2 create must apply: {result.reason}"
    row = store.conn.execute("SELECT deleted_at FROM memories WHERE content_hash = ?", (h,)).fetchone()
    assert row is not None and row[0] is None, \
        "a fresh create with recreate_of='' and NO delete in the set must materialize LIVE (inert sentinel)"
    assert len(await store.search("Fresh v2 create", n_results=5)) > 0, \
        "the fresh live memory must appear in search"


# ===========================================================================
# #1522 P2 — PRODUCER load-bearing tests. The producer (StoreMixin) must stamp
# recreate_of = <event_id of the delete it saw> onto a re-create's payload,
# and a FRESH create (no tombstone) must carry NO recreate_of. Exercised via
# the REAL public methods store() and store_batch() (not the private helper),
# inspecting the emitted create event in sync_events. These FAIL if the
# producer stamping is removed.
# ===========================================================================


def _latest_create_payload(store_obj, content_hash):
    """Return (event_id, payload) of the most-recent create sync_event for a hash."""
    row = store_obj.conn.execute(
        "SELECT event_id, payload FROM sync_events "
        "WHERE content_hash = ? AND op = 'create' "
        "ORDER BY hlc_physical DESC, hlc_logical DESC LIMIT 1",
        (content_hash,),
    ).fetchone()
    assert row is not None, "expected a create sync_event for the hash"
    return row[0], json.loads(row[1])


def _delete_event_id(store_obj, content_hash):
    """Return the event_id of the (single) delete sync_event for a hash.

    Tests here stage exactly one delete per hash, so a plain HLC-desc ORDER BY
    is unambiguous. Production elects the dominant delete via the resolver's
    full _winner_key (HLC -> delete-vs-update -> agent_id -> event_id); with
    tied deletes this helper would diverge, which never happens in these tests.
    """
    row = store_obj.conn.execute(
        "SELECT event_id FROM sync_events WHERE content_hash = ? AND op = 'delete' "
        "ORDER BY hlc_physical DESC, hlc_logical DESC LIMIT 1",
        (content_hash,),
    ).fetchone()
    assert row is not None, "expected a delete sync_event for the hash"
    return row[0]


@pytest.mark.asyncio
async def test_producer_store_stamps_recreate_of_on_recreate(store):
    """store() (Henry 3-state): a FRESH create carries recreate_of == "" (explicit empty
    sentinel — "stamp on every create"); a re-create over a tombstone carries the delete's
    event_id. Load-bearing: remove the producer stamp and both assertions flip."""
    with patch.dict(os.environ, {"MCP_SYNC_EVENTLOG": "on"}):
        content = "producer recreate via store()"
        h = generate_content_hash(content)
        m = Memory(content=content, content_hash=h, tags=["p"], memory_type="note")
        ok, _ = await store.store(m)
        assert ok, "initial store must succeed"

        # Fresh create: key PRESENT and EMPTY (v2 stamps "" on every create).
        _eid, fresh_payload = _latest_create_payload(store, h)
        assert "recreate_of" in fresh_payload, \
            "a v2 fresh create must stamp the recreate_of key (3-state model)"
        assert fresh_payload["recreate_of"] == "", \
            f"a fresh create (no tombstone) must carry the EMPTY sentinel, got {fresh_payload.get('recreate_of')!r}"

        # Delete, then re-store the same content (re-create over the tombstone).
        ok, _ = await store.delete(h)
        assert ok, "delete must succeed"
        del_eid = _delete_event_id(store, h)

        m2 = Memory(content=content, content_hash=h, tags=["p"], memory_type="note")
        ok, _ = await store.store(m2)
        assert ok, "re-store must succeed"

        _eid2, recreate_payload = _latest_create_payload(store, h)
        assert recreate_payload.get("recreate_of") == del_eid, (
            f"re-create via store() must stamp recreate_of = delete's event_id {del_eid}, "
            f"got {recreate_payload.get('recreate_of')}"
        )


@pytest.mark.asyncio
async def test_producer_store_batch_stamps_recreate_of_on_recreate(store):
    """store_batch() (Henry 3-state): same contract via the batch public method —
    fresh create carries "", re-create over a tombstone carries the delete's event_id."""
    with patch.dict(os.environ, {"MCP_SYNC_EVENTLOG": "on"}):
        content = "producer recreate via store_batch()"
        h = generate_content_hash(content)
        m = Memory(content=content, content_hash=h, tags=["b"], memory_type="note")
        res = await store.store_batch([m])
        assert res and res[0][0], "initial store_batch must succeed"

        _eid, fresh_payload = _latest_create_payload(store, h)
        assert fresh_payload.get("recreate_of") == "", \
            "a fresh store_batch create must carry the EMPTY sentinel (stamp on every create)"

        ok, _ = await store.delete(h)
        assert ok, "delete must succeed"
        del_eid = _delete_event_id(store, h)

        m2 = Memory(content=content, content_hash=h, tags=["b"], memory_type="note")
        res2 = await store.store_batch([m2])
        assert res2 and res2[0][0], "re-store_batch must succeed"

        _eid2, recreate_payload = _latest_create_payload(store, h)
        assert recreate_payload.get("recreate_of") == del_eid, (
            f"re-create via store_batch() must stamp recreate_of = delete's event_id {del_eid}, "
            f"got {recreate_payload.get('recreate_of')}"
        )


@pytest.mark.asyncio
async def test_producer_fresh_create_has_empty_recreate_of(store):
    """A first-ever create of a hash (no prior tombstone) carries recreate_of == "" —
    both via store() and store_batch(). This is "stamp on every create" (Henry Ponto 2):
    a v2 producer that never saw a delete emits the EMPTY sentinel, NOT an absent key, so
    apply can tell it apart from a v1 peer."""
    with patch.dict(os.environ, {"MCP_SYNC_EVENTLOG": "on"}):
        c1 = "fresh via store"
        h1 = generate_content_hash(c1)
        await store.store(Memory(content=c1, content_hash=h1, memory_type="note"))
        _e, p1 = _latest_create_payload(store, h1)
        assert p1.get("recreate_of") == "", "fresh store() create must carry the empty sentinel"

        c2 = "fresh via batch"
        h2 = generate_content_hash(c2)
        await store.store_batch([Memory(content=c2, content_hash=h2, memory_type="note")])
        _e2, p2 = _latest_create_payload(store, h2)
        assert p2.get("recreate_of") == "", "fresh store_batch() create must carry the empty sentinel"


# ===========================================================================
# #1564 REGRESSION — purge erases the tombstone row BEFORE the producer stamps.
# _recreate_of_for (store.py:60) guards on `SELECT 1 FROM memories WHERE
# content_hash=? AND deleted_at IS NOT NULL` (store.py:89-90 `if tomb is None:
# return None`). purge_deleted / _purge_tombstone (delete.py:270 / store.py:36)
# HARD-DELETES that soft-deleted row but NEVER touches sync_events, so the delete
# event is still the latest for the hash. A subsequent re-store therefore stamps
# recreate_of="" (EMPTY — "I never saw a delete") instead of the delete's
# event_id (NAMED). apply.py reads EMPTY as delete-wins → origin LIVE, peer DEAD,
# permanent divergence. The producer must stamp the delete it CAN STILL SEE in
# sync_events even after the tombstone row is gone.
#
# prove-fix: the assert below is the SAME assertion that turns GREEN once the
# tombstone guard stops being the sole gate (the delete still lives in
# sync_events). RED now (vem ""), GREEN after the fix.
# ===========================================================================


@pytest.mark.asyncio
async def test_producer_recreate_after_purge_still_stamps_recreate_of(store):
    """#1564: store() a hash, delete it, PURGE the tombstone row (sync_events intact),
    then re-store. The re-create MUST stamp recreate_of = the delete's event_id — the
    delete is still the latest event for the hash in sync_events. Today _recreate_of_for
    short-circuits on the missing tombstone row and the producer emits "" (EMPTY), which
    apply treats as delete-wins → divergence. FAILS today (got ""), passes after the fix."""
    with patch.dict(os.environ, {"MCP_SYNC_EVENTLOG": "on"}):
        content = "producer recreate AFTER purge via store()"
        h = generate_content_hash(content)

        m = Memory(content=content, content_hash=h, tags=["p"], memory_type="note")
        ok, _ = await store.store(m)
        assert ok, "initial store must succeed"

        ok, _ = await store.delete(h)
        assert ok, "delete must succeed"
        del_eid = _delete_event_id(store, h)

        # Purge the tombstone row — sync_events (incl. the delete) is untouched.
        store._purge_tombstone(h)
        store.conn.commit()
        tomb = store.conn.execute(
            "SELECT 1 FROM memories WHERE content_hash = ? AND deleted_at IS NOT NULL LIMIT 1",
            (h,),
        ).fetchone()
        assert tomb is None, "precondition: the tombstone ROW must be gone after purge"
        assert _delete_event_id(store, h) == del_eid, \
            "precondition: the delete sync_event must survive the purge"

        # Re-create the same content after the purge.
        m2 = Memory(content=content, content_hash=h, tags=["p"], memory_type="note")
        ok, _ = await store.store(m2)
        assert ok, "re-store must succeed"

        _eid2, recreate_payload = _latest_create_payload(store, h)
        assert recreate_payload.get("recreate_of") == del_eid, (
            f"re-create via store() after purge must stamp recreate_of = delete's event_id "
            f"{del_eid!r} (delete still in sync_events), got {recreate_payload.get('recreate_of')!r} "
            f"— EMPTY means apply resolves it as delete-wins → divergence (#1564)"
        )


@pytest.mark.asyncio
async def test_producer_recreate_after_purge_still_stamps_recreate_of_batch(store):
    """#1564 via store_batch(): same contract — purge the tombstone row, re-create in a
    batch, and the re-create MUST carry recreate_of = the delete's event_id. The batch path
    shares _recreate_of_for, so it regresses identically. FAILS today (got ""), GREEN after."""
    with patch.dict(os.environ, {"MCP_SYNC_EVENTLOG": "on"}):
        content = "producer recreate AFTER purge via store_batch()"
        h = generate_content_hash(content)

        m = Memory(content=content, content_hash=h, tags=["b"], memory_type="note")
        res = await store.store_batch([m])
        assert res and res[0][0], "initial store_batch must succeed"

        ok, _ = await store.delete(h)
        assert ok, "delete must succeed"
        del_eid = _delete_event_id(store, h)

        store._purge_tombstone(h)
        store.conn.commit()
        tomb = store.conn.execute(
            "SELECT 1 FROM memories WHERE content_hash = ? AND deleted_at IS NOT NULL LIMIT 1",
            (h,),
        ).fetchone()
        assert tomb is None, "precondition: the tombstone ROW must be gone after purge"
        assert _delete_event_id(store, h) == del_eid, \
            "precondition: the delete sync_event must survive the purge"

        m2 = Memory(content=content, content_hash=h, tags=["b"], memory_type="note")
        res2 = await store.store_batch([m2])
        assert res2 and res2[0][0], "re-store_batch must succeed"

        _eid2, recreate_payload = _latest_create_payload(store, h)
        assert recreate_payload.get("recreate_of") == del_eid, (
            f"re-create via store_batch() after purge must stamp recreate_of = delete's event_id "
            f"{del_eid!r} (delete still in sync_events), got {recreate_payload.get('recreate_of')!r} "
            f"— EMPTY means apply resolves it as delete-wins → divergence (#1564)"
        )


# ===========================================================================
# #1522 FINAL rule — P1-A (order gate) + P1-B (bootstrap HLC anchor), seven G0
# 2026-10-16, spec §6. The NAMED branch of recompute_liveness must:
#   P1-A: revive ONLY if the create STRICTLY supersedes the named delete under
#         _winner_key (cwin.key() < dtop.key()); naming a dominant delete it does
#         NOT supersede => DEAD.
#   P1-B: match the named delete by raw event_id OR, when the id was rewritten to
#         'bl-...' (bootstrap), by the stable (hlc_physical, hlc_logical, agent_id)
#         anchor carried as recreate_of_hlc.
# prove-fix: (a) neutralize the order gate -> named_recreate_older RED;
#            (b) neutralize the HLC fallback -> bootstrapped_peer_revives RED.
# ===========================================================================


def _create_event_recreate_anchored(content_hash, *, content, recreate_of, recreate_of_hlc,
                                     agent="alpha", event_id="recreate", hlc_physical=30,
                                     hlc_logical=0, store="default"):
    """A create naming a delete by event_id AND carrying the stable HLC anchor (P1-B).

    recreate_of_hlc = [hlc_physical, hlc_logical, agent_id] of the delete it supersedes."""
    ev = _create_event(content_hash, content=content, agent=agent, event_id=event_id,
                       hlc_physical=hlc_physical, hlc_logical=hlc_logical, store=store)
    ev["payload"]["recreate_of"] = recreate_of
    ev["payload"]["recreate_of_hlc"] = recreate_of_hlc
    return ev


# ---- P1-A: order gate on the NAMED branch -------------------------------------

@pytest.mark.asyncio
async def test_named_recreate_older_than_delete_stays_dead__both_orders(tmp_store_factory):
    """P1-A: {D@HLC40, C@HLC30(recreate_of=D.event_id)}. The create NAMES the dominant delete
    but is OLDER (HLC30 < HLC40), so it does NOT supersede it — DEAD in BOTH arrival orders.
    FURA HOJE sem o gate (name-match puro reviveria). Load-bearing for the order gate."""
    content = "named older than delete"
    h = generate_content_hash(content)
    events = {
        "C0": _create_event(h, content=content, hlc_physical=10, event_id="C0"),
        "D": _create_delete_event(h, deleted_at=2000.0, hlc_physical=40, event_id="D40"),
        "C": _create_event_recreate(h, content=content, recreate_of="D40",
                                    hlc_physical=30, event_id="C30"),
    }
    outcomes = set()
    for order in itertools.permutations(["C0", "D", "C"]):
        outcomes.add(await _final_deleted_at(tmp_store_factory, events, order))
    # NEVER alive: the memory is dead in every order — either a deterministic tombstone
    # (2000.0, orders where a row materialized) or NOROW (delete-dominant orders where no
    # create ever won the resolver, so nothing materialized — also "not alive"). The gate is
    # load-bearing: with it neutralized these orders flip to None (alive). None must NOT appear.
    assert None not in outcomes, (
        f"a NAMED re-create older than the delete it names must NEVER revive (P1-A order gate); "
        f"got {outcomes}"
    )
    assert outcomes <= {2000.0, "NOROW"}, f"unexpected non-dead outcome: {outcomes}"


@pytest.mark.asyncio
async def test_named_recreate_newer_than_delete_revives__both_orders(tmp_store_factory):
    """P1-A guard: {D@HLC30, C@HLC40(recreate_of=D.event_id)}. The create NAMES the dominant
    delete AND supersedes it (HLC40 > HLC30) — LIVE in BOTH orders. Anti-regression for the
    gate (must not kill legitimate re-creates)."""
    content = "named newer than delete"
    h = generate_content_hash(content)
    events = {
        "D": _create_delete_event(h, deleted_at=2000.0, hlc_physical=30, event_id="D30"),
        "C": _create_event_recreate(h, content=content, recreate_of="D30",
                                    hlc_physical=40, event_id="C40"),
    }
    outcomes = set()
    for order in itertools.permutations(["D", "C"]):
        outcomes.add(await _final_deleted_at(tmp_store_factory, events, order))
    assert outcomes == {None}, (
        f"a NAMED re-create newer than the delete it names must REVIVE in both orders, got {outcomes}"
    )


@pytest.mark.asyncio
async def test_named_recreate_ties_delete_hlc_stays_dead(tmp_store_factory):
    """P1-A border: {D@HLC30(agentA), C@HLC30(recreate_of=D.event_id)} — IDENTICAL HLC. In an
    HLC tie the delete wins (_winner_key comp.2: delete<create), so the create does NOT
    supersede → DEAD in both orders. Proves the gate uses the FULL total order, not HLC-only."""
    content = "named ties delete hlc"
    h = generate_content_hash(content)
    events = {
        "C0": _create_event(h, content=content, hlc_physical=5, event_id="C0"),
        "D": _create_delete_event(h, deleted_at=2000.0, hlc_physical=30, hlc_logical=0,
                                  agent="agentA", event_id="D30"),
        "C": _create_event_recreate(h, content=content, recreate_of="D30",
                                    hlc_physical=30, event_id="C30"),
    }
    # pin the create's hlc_logical to 0 and agent to agentA so the ONLY tie-break is op (delete wins)
    events["C"]["hlc_logical"] = 0
    events["C"]["agent_id"] = "agentA"
    events["C"]["payload"]["recreate_of"] = "D30"
    outcomes = set()
    for order in itertools.permutations(["C0", "D", "C"]):
        outcomes.add(await _final_deleted_at(tmp_store_factory, events, order))
    assert None not in outcomes, (
        f"on an HLC tie the delete wins (_winner_key comp.2), so the NAMED re-create must "
        f"NEVER revive (P1-A border); got {outcomes}"
    )
    assert outcomes <= {2000.0, "NOROW"}, f"unexpected non-dead outcome: {outcomes}"


# ---- P1-B: bootstrap HLC anchor on the NAMED branch ---------------------------

def _make_bootstrapped_set(*, create_hlc, recreate_of_hlc, recreate_of="orig-del-id",
                           del_agent="origin", del_hlc_physical=20, del_hlc_logical=0):
    """Simulate a BOOTSTRAPPED peer: the delete exists ONLY as 'bl-...' (event_id rewritten by
    the baseline) but with its ORIGINAL HLC+agent preserved. The incoming create names the
    ORIGINAL delete id (which no longer matches 'bl-...') and carries the stable anchor."""
    content = "bootstrapped re-create"
    h = generate_content_hash(content)
    return h, {
        # the original incarnation (so the row materializes; a re-create implies a prior create)
        "C0": _create_event(h, content=content, hlc_physical=10, event_id="C0"),
        # delete as the baseline left it: synthetic 'bl-...' id, REAL hlc+agent preserved
        "Dbl": _create_delete_event(h, deleted_at=2000.0, hlc_physical=del_hlc_physical,
                                    hlc_logical=del_hlc_logical, agent=del_agent,
                                    event_id="bl-rewritten-id"),
        # the incoming re-create names the ORIGINAL id (!= 'bl-...') + the stable anchor
        "C2": _create_event_recreate_anchored(
            h, content=content, recreate_of=recreate_of, recreate_of_hlc=recreate_of_hlc,
            hlc_physical=create_hlc, event_id="C2-recreate"),
    }


@pytest.mark.asyncio
async def test_bootstrapped_peer_revives_on_recreate_via_hlc_anchor(tmp_store_factory):
    """P1-B: on a bootstrapped peer the delete is 'bl-...' (id rewritten) but its HLC+agent are
    preserved. A re-create that names the ORIGINAL id (no longer matching) but carries
    recreate_of_hlc = [delete HLC+agent] and SUPERSEDES the delete (C@30 > D@20) must REVIVE —
    names_delete matches by the HLC anchor fallback. FURA HOJE (name-match puro → DEAD).
    Load-bearing for the HLC fallback."""
    _h, events = _make_bootstrapped_set(create_hlc=30,
                                       recreate_of_hlc=[20, 0, "origin"])
    outcomes = set()
    for order in itertools.permutations(["C0", "Dbl", "C2"]):
        outcomes.add(await _final_deleted_at(tmp_store_factory, events, order))
    assert outcomes == {None}, (
        f"a bootstrapped peer must REVIVE via the HLC anchor fallback when the create supersedes "
        f"the 'bl-...' delete, got {outcomes}"
    )


@pytest.mark.asyncio
async def test_bootstrapped_peer_keeps_dead_on_stale_recreate(tmp_store_factory):
    """P1-B guard: the anchor points at a delete that is NOT the dominant one (wrong HLC), so
    names_delete returns False → DEAD. The HLC fallback must not become a stale-revive hole."""
    # dominant delete is at HLC20/origin; the create's anchor points at HLC99/ghost (no such
    # dominant delete) → stale.
    _h, events = _make_bootstrapped_set(create_hlc=30,
                                       recreate_of_hlc=[99, 0, "ghost-agent"])
    outcomes = set()
    for order in itertools.permutations(["C0", "Dbl", "C2"]):
        outcomes.add(await _final_deleted_at(tmp_store_factory, events, order))
    assert outcomes == {2000.0}, (
        f"an anchor pointing at a NON-dominant/absent delete is stale → DEAD, got {outcomes}"
    )


@pytest.mark.asyncio
async def test_bootstrapped_peer_keeps_dead_on_named_older_recreate(tmp_store_factory):
    """P1-A + P1-B together: on a bootstrapped peer the anchor MATCHES the dominant delete
    (HLC20/origin) but the create is OLDER (C@15 < D@20) → does NOT supersede → DEAD. The
    order gate holds even on the bootstrap/anchor path."""
    _h, events = _make_bootstrapped_set(create_hlc=15,
                                       recreate_of_hlc=[20, 0, "origin"])
    outcomes = set()
    for order in itertools.permutations(["C0", "Dbl", "C2"]):
        outcomes.add(await _final_deleted_at(tmp_store_factory, events, order))
    assert None not in outcomes, (
        f"a bootstrapped re-create that names the delete via anchor but does NOT supersede it "
        f"(older HLC) must NEVER revive (P1-A on the bootstrap path); got {outcomes}"
    )
    assert outcomes <= {2000.0, "NOROW"}, f"unexpected non-dead outcome: {outcomes}"


# ---- P1-B: producer stamps the anchor -----------------------------------------

@pytest.mark.asyncio
async def test_producer_stamps_recreate_of_hlc_on_recreate(store):
    """P1-B producer: store()→delete→store(), the 2nd create's payload carries recreate_of_hlc
    == [delete.hlc_physical, delete.hlc_logical, delete.agent_id] alongside recreate_of ==
    delete.event_id. Load-bearing: remove the anchor stamp and this flips."""
    with patch.dict(os.environ, {"MCP_SYNC_EVENTLOG": "on"}):
        content = "producer stamps anchor"
        h = generate_content_hash(content)
        assert (await store.store(Memory(content=content, content_hash=h, memory_type="note")))[0]
        assert (await store.delete(h))[0], "delete must succeed"

        del_row = store.conn.execute(
            "SELECT event_id, hlc_physical, hlc_logical, agent_id FROM sync_events "
            "WHERE content_hash = ? AND op = 'delete' "
            "ORDER BY hlc_physical DESC, hlc_logical DESC LIMIT 1",
            (h,),
        ).fetchone()
        del_eid, dph, dpl, dag = del_row

        assert (await store.store(Memory(content=content, content_hash=h, memory_type="note")))[0]
        _eid, recreate_payload = _latest_create_payload(store, h)
        assert recreate_payload.get("recreate_of") == del_eid, \
            f"re-create must name the delete event_id {del_eid}"
        assert recreate_payload.get("recreate_of_hlc") == [dph, dpl, dag], (
            f"re-create must stamp recreate_of_hlc = delete anchor [{dph}, {dpl}, {dag}], "
            f"got {recreate_payload.get('recreate_of_hlc')}"
        )


@pytest.mark.asyncio
async def test_producer_omits_recreate_of_hlc_on_empty(store):
    """P1-B guard: a fresh create of an unseen hash carries recreate_of == "" and NO
    recreate_of_hlc key (the anchor exists only on the NAMED path)."""
    with patch.dict(os.environ, {"MCP_SYNC_EVENTLOG": "on"}):
        content = "producer fresh omits anchor"
        h = generate_content_hash(content)
        assert (await store.store(Memory(content=content, content_hash=h, memory_type="note")))[0]
        _eid, fresh_payload = _latest_create_payload(store, h)
        assert fresh_payload.get("recreate_of") == "", "fresh create carries the empty sentinel"
        assert "recreate_of_hlc" not in fresh_payload, (
            f"a fresh create must NOT carry recreate_of_hlc (anchor is NAMED-only), "
            f"got {fresh_payload.get('recreate_of_hlc')!r}"
        )
