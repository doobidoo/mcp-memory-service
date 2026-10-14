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
@pytest.mark.xfail(
    reason="#1522 finding #2: known bug. An absolute tombstone-preserve fix breaks legitimate "
    "re-create (test_old_edit_does_not_restore_recreated_memory_field). HLC alone cannot tell "
    "a malicious clock-ahead resurrection from a legitimate re-create. Correct fix needs an "
    "intentionality signal in the event — tracked as ADR-0012 v2 (separate item). Kept as a "
    "living RED marker, not a CI blocker.",
    strict=False,
)
async def test_create_does_not_resurrect_tombstone(store):
    """
    INV-2a (#2 CRITICAL): create with HLC > delete must NOT resurrect a tombstone.
    
    TODAY THIS FAILS: _materialize_event hard-codes deleted_at=NULL (apply.py:658)
    regardless of existing tombstone, so the memory gets resurrected and appears in search.
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
    
    # Step 2: Delete the memory (deleted_at NOT NULL)
    delete_event = _create_delete_event(h, deleted_at=2000.0, hlc_physical=20)
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
    
    # Step 3: Apply create with HIGHER HLC (this should NOT resurrect)
    resurrect_event = _create_event(h, content=content, hlc_physical=30, event_id="resurrect")
    result = apply_remote_event(store, resurrect_event)
    assert result.applied, f"Later create failed: {result.reason}"
    
    # CRITICAL ASSERTION: Memory must REMAIN deleted (tombstone preserved)
    cursor = store.conn.execute("SELECT deleted_at FROM memories WHERE content_hash = ?", (h,))
    row = cursor.fetchone()
    assert row is not None, "Memory row should exist"
    assert row[0] is not None, "Tombstone must be preserved - memory should stay deleted"
    
    # CRITICAL ASSERTION: Memory must NOT appear in search
    search_results = await store.search("Memory that should", n_results=5)
    assert len(search_results) == 0, "Resurrected tombstone must not appear in search"


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