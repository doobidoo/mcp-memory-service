"""
Tests for delta-sync Phase 4b: pull orchestration (sync_from_peer).

These tests are designed to FAIL against current code since Phase 4b orchestrator.py
does not exist yet. They test requirements F1-F6, CA1-CA6 from spec-delta-sync-fase4b.md.

ANTI-PATTERN AVOIDED: No pytest.skip, no try/except that makes tests pass,
imports at top will break collection = valid RED, asserts test real behavior.
"""

import pytest
import pytest_asyncio
import tempfile
import os
import json
import time
import sqlite3
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash
from mcp_memory_service.storage.sync.apply import ApplyResult

# This import WILL break collection (ModuleNotFoundError) = valid RED for CA1
from mcp_memory_service.storage.sync.orchestrator import sync_from_peer, SyncResult


class LocalPeerAdapter:
    """
    Test adapter that simulates a peer by reading from source storage's sync_events.
    Implements the same interface as RemoteHTTPStorage.get_events_since.
    """
    
    def __init__(self, source_storage):
        self.source_storage = source_storage
    
    async def get_events_since(self, since_seq: int, limit: int) -> Tuple[List[Dict[str, Any]], int, bool]:
        """
        Get events from source storage's sync_events table with pagination.
        
        Returns: (events, next_seq, has_more)
        """
        # Query sync_events with JOIN to memories to get content
        cursor = self.source_storage.conn.execute("""
            SELECT 
                se.seq, se.agent_id, se.event_id, se.op, se.content_hash,
                se.hlc_physical, se.hlc_logical, se.embedding_model, 
                se.embedding_dim, se.payload, m.content
            FROM sync_events se
            JOIN memories m ON se.content_hash = m.content_hash  
            WHERE se.seq > ?
            ORDER BY se.seq
            LIMIT ?
        """, (since_seq, limit))
        
        rows = cursor.fetchall()
        
        events = []
        next_seq = since_seq
        
        for row in rows:
            seq, agent_id, event_id, op, content_hash, hlc_physical, hlc_logical, embedding_model, embedding_dim, payload_json, content = row
            
            # Parse payload JSON
            payload = json.loads(payload_json) if payload_json else {}
            
            # Add content to payload (enriched event)
            if content:
                payload["content"] = content
            
            event = {
                "seq": seq,
                "agent_id": agent_id,
                "event_id": event_id,
                "op": op,
                "content_hash": content_hash,
                "hlc_physical": hlc_physical,
                "hlc_logical": hlc_logical,
                "embedding_model": embedding_model,
                "embedding_dim": embedding_dim,
                "payload": payload
            }
            events.append(event)
            next_seq = seq
        
        # Check if there are more events beyond this page
        has_more = False
        if events:
            cursor = self.source_storage.conn.execute("""
                SELECT COUNT(*) FROM sync_events WHERE seq > ?
            """, (next_seq,))
            remaining_count = cursor.fetchone()[0]
            has_more = remaining_count > 0
        
        return events, next_seq, has_more


@pytest.fixture
def temp_db_a():
    """Create temporary database A for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "storage_a.db")
        yield db_path


@pytest.fixture  
def temp_db_b():
    """Create temporary database B for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "storage_b.db")
        yield db_path


@pytest_asyncio.fixture
async def storage_a(temp_db_a, monkeypatch):
    """Create and initialize storage A (source) with agent=alpha."""
    monkeypatch.setenv('MCP_AGENT_ID', 'alpha')
    monkeypatch.setenv('MCP_SYNC_EVENTLOG', 'true')
    monkeypatch.setenv('MCP_SEMANTIC_DEDUP_ENABLED', 'false')
    
    storage = SqliteVecMemoryStorage(db_path=temp_db_a)
    await storage.initialize()
    return storage


@pytest_asyncio.fixture
async def storage_b(temp_db_b, monkeypatch):
    """Create and initialize storage B (target) with agent=beta."""  
    monkeypatch.setenv('MCP_AGENT_ID', 'beta')
    monkeypatch.setenv('MCP_SYNC_EVENTLOG', 'true')
    monkeypatch.setenv('MCP_SEMANTIC_DEDUP_ENABLED', 'false')
    
    storage = SqliteVecMemoryStorage(db_path=temp_db_b)
    await storage.initialize()
    return storage


@pytest_asyncio.fixture
async def storage_a_with_memories(storage_a):
    """Storage A with 5 distinct memories to create a feed > 1 page."""
    from mcp_memory_service.utils.hashing import generate_content_hash
    import os
    from unittest.mock import patch
    
    memories = []
    for i in range(5):
        content = f"Test memory {i+1} for sync orchestrator testing"
        memory = Memory(
            content=content,
            content_hash=generate_content_hash(content),
            tags=[f"conv-{i}"],
            metadata={"agent_id": "alpha"}
        )
        # Force agent_id to be "alpha" during store (for sync_events)
        with patch.dict(os.environ, {"MCP_AGENT_ID": "alpha"}):
            result = await storage_a.store(memory)
        memories.append(memory)
        time.sleep(0.01)  # Ensure different HLC values
    
    return storage_a, memories


class TestDeltaSyncOrchestrator:
    """Test suite for delta-sync Phase 4b orchestrator."""

    @pytest.mark.asyncio
    async def test_ca1_sync_from_peer_basic_materialization(self, storage_a_with_memories, storage_b):
        """
        CA1 (F1/F6): sync_from_peer exists in orchestrator.py and materializes memories from A to B.
        
        Expected RED: ModuleNotFoundError on orchestrator import at top of file.
        """
        storage_a, memories = storage_a_with_memories
        
        # Create LocalPeerAdapter over storage A
        adapter_a = LocalPeerAdapter(storage_a)
        
        # Sync from A to B
        result = await sync_from_peer(storage_b, adapter_a, 'peer-A')
        
        # Verify result type - should be SyncResult or similar
        assert isinstance(result, SyncResult), f"Expected SyncResult, got {type(result)}"
        assert result.events_applied == 5, f"Expected 5 events applied, got {result.events_applied}"
        
        # Verify memories materialized in B
        b_cursor = storage_b.conn.execute("SELECT COUNT(*) FROM memories")
        memory_count = b_cursor.fetchone()[0]
        assert memory_count == 5, f"Expected 5 memories in B, got {memory_count}"
        
        # Verify authorship preserved (agent_id should be 'alpha')
        b_cursor = storage_b.conn.execute("SELECT metadata FROM memories")
        metadata_rows = b_cursor.fetchall()
        agent_ids = []
        for row in metadata_rows:
            metadata = json.loads(row[0])
            agent_id = metadata.get("agent_id")
            if agent_id:
                agent_ids.append(agent_id)
        assert agent_ids == ['alpha'] * 5, f"Expected authorship preserved as 'alpha', got {agent_ids}"

    @pytest.mark.asyncio
    async def test_ca2_pagination_complete(self, storage_a_with_memories, storage_b):
        """
        CA2 (F2): With 5 events and limit=2, all 5 should be applied through pagination.
        
        Expected RED: ModuleNotFoundError on orchestrator import.
        """
        storage_a, memories = storage_a_with_memories
        adapter_a = LocalPeerAdapter(storage_a)
        
        # Mock the adapter to use small limit to force pagination
        original_get_events = adapter_a.get_events_since
        
        async def paginated_get_events(since_seq: int, limit: int = 2):
            return await original_get_events(since_seq, 2)  # Force limit=2
        
        adapter_a.get_events_since = paginated_get_events
        
        result = await sync_from_peer(storage_b, adapter_a, 'peer-A')
        
        # Should process 3 pages: 2+2+1 = 5 events
        assert result.pages_processed == 3, f"Expected 3 pages, got {result.pages_processed}"
        assert result.events_applied == 5, f"Expected 5 events applied, got {result.events_applied}"
        
        # All memories should be in B
        b_cursor = storage_b.conn.execute("SELECT COUNT(*) FROM memories") 
        memory_count = b_cursor.fetchone()[0]
        assert memory_count == 5, f"Expected 5 memories, got {memory_count}"

    @pytest.mark.asyncio
    async def test_ca3_cursor_durability_and_resume(self, storage_a_with_memories, storage_b):
        """
        CA3 (F3): Cursor advances per page and resumes correctly from last position.
        
        Expected RED: ModuleNotFoundError on orchestrator import.
        """
        storage_a, memories = storage_a_with_memories
        adapter_a = LocalPeerAdapter(storage_a)
        
        # First sync
        result = await sync_from_peer(storage_b, adapter_a, 'peer-A')
        
        # Check cursor was set to highest seq
        cursor = storage_b.conn.execute("""
            SELECT last_seq_seen FROM sync_cursor WHERE peer_id = 'peer-A'
        """)
        cursor_row = cursor.fetchone()
        assert cursor_row is not None, "Sync cursor not found"
        last_seq = cursor_row[0]
        assert last_seq > 0, f"Expected cursor > 0, got {last_seq}"
        
        # Simulate "resume" - run sync again from same cursor
        result2 = await sync_from_peer(storage_b, adapter_a, 'peer-A')
        
        # Should be no-op (0 new events)
        assert result2.events_applied == 0, f"Expected 0 new events on resume, got {result2.events_applied}"
        
        # Simulate partial resume: manually set cursor to middle (seq=3)
        storage_b.conn.execute("""
            UPDATE sync_cursor SET last_seq_seen = 3 WHERE peer_id = 'peer-A'
        """)
        storage_b.conn.commit()
        
        result3 = await sync_from_peer(storage_b, adapter_a, 'peer-A')
        
        # Should sync only remaining events (seq > 3)
        assert result3.events_applied == 2, f"Expected 2 remaining events, got {result3.events_applied}"

    @pytest.mark.asyncio  
    async def test_ca4_idempotency_no_duplicates(self, storage_a_with_memories, storage_b):
        """
        CA4 (F4): Running sync_from_peer twice produces identical result, zero duplicates.
        
        Expected RED: ModuleNotFoundError on orchestrator import.
        """
        storage_a, memories = storage_a_with_memories
        adapter_a = LocalPeerAdapter(storage_a)
        
        # First sync
        result1 = await sync_from_peer(storage_b, adapter_a, 'peer-A')
        
        # Get counts after first sync
        mem_cursor = storage_b.conn.execute("SELECT COUNT(*) FROM memories")
        memory_count_1 = mem_cursor.fetchone()[0]
        
        evt_cursor = storage_b.conn.execute("SELECT COUNT(*) FROM sync_events")  
        event_count_1 = evt_cursor.fetchone()[0]
        
        # Second sync (identical)
        result2 = await sync_from_peer(storage_b, adapter_a, 'peer-A')
        
        # Get counts after second sync
        mem_cursor = storage_b.conn.execute("SELECT COUNT(*) FROM memories")
        memory_count_2 = mem_cursor.fetchone()[0]
        
        evt_cursor = storage_b.conn.execute("SELECT COUNT(*) FROM sync_events")
        event_count_2 = evt_cursor.fetchone()[0]
        
        # Should be identical - no duplicates
        assert memory_count_1 == memory_count_2, f"Memory count changed: {memory_count_1} -> {memory_count_2}"
        assert event_count_1 == event_count_2, f"Event count changed: {event_count_1} -> {event_count_2}"
        assert result2.events_applied == 0, f"Second sync should apply 0 events, got {result2.events_applied}"

    @pytest.mark.asyncio
    async def test_ca5_late_event_authorship_preserved(self, storage_a, storage_b):
        """
        CA5 (F5): Late event with lower HLC doesn't overwrite; authorship from A preserved in B.
        
        Expected RED: ModuleNotFoundError on orchestrator import.
        """
        from mcp_memory_service.utils.hashing import generate_content_hash
        import os
        from unittest.mock import patch
        
        content = "Shared content for conflict test"
        
        # Store memory in B first (agent=beta)
        memory_b = Memory(
            content=content,
            content_hash=generate_content_hash(content),
            tags=["conv-shared"],
            metadata={"agent_id": "beta"}
        )
        with patch.dict(os.environ, {"MCP_AGENT_ID": "beta"}):
            await storage_b.store(memory_b)
        
        # Get beta's HLC from the event
        b_events = storage_b.conn.execute("""
            SELECT hlc_physical, hlc_logical FROM sync_events ORDER BY seq DESC LIMIT 1
        """).fetchone()
        beta_hlc_physical, beta_hlc_logical = b_events
        
        # Manually create alpha's "late event" with lower HLC 
        alpha_hlc_physical = beta_hlc_physical - 1000  # 1 second earlier
        alpha_hlc_logical = 0
        
        # Insert alpha's late event directly into A's sync_events
        import uuid
        alpha_event_id = str(uuid.uuid4())
        content_hash = generate_content_hash(content)
        
        storage_a.conn.execute("""
            INSERT INTO sync_events 
            (agent_id, event_id, op, content_hash, hlc_physical, hlc_logical,
             embedding_model, embedding_dim, payload, created_at, created_at_iso)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "alpha", alpha_event_id, "create", content_hash, 
            alpha_hlc_physical, alpha_hlc_logical,
            storage_a.embedding_model_name, 384,
            json.dumps({"content": content, "tags": ["conv-shared"], "metadata": {"agent_id": "alpha"}}),
            time.time(), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        ))
        storage_a.conn.commit()
        
        # Also need to add the memory to A so the LocalPeerAdapter can find content
        memory_a = Memory(
            content=content,
            content_hash=content_hash,
            tags=["conv-shared"],
            metadata={"agent_id": "alpha"}
        )
        # Store without triggering another sync event (use metadata to detect)
        storage_a.conn.execute("""
            INSERT OR REPLACE INTO memories
            (content_hash, content, tags, memory_type, metadata,
             created_at, created_at_iso, updated_at, updated_at_iso,
             deleted_at, embedding_pending, store)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 0, 'default')
        """, (
            content_hash, content, json.dumps(["conv-shared"]), "general", 
            json.dumps({"agent_id": "alpha"}),
            time.time(), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            time.time(), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        ))
        storage_a.conn.commit()
        
        # Sync A->B (alpha's late event should not overwrite beta)
        adapter_a = LocalPeerAdapter(storage_a)
        result = await sync_from_peer(storage_b, adapter_a, 'peer-A')
        
        # Memory in B should still have agent_id='beta' (winner)
        # Note: memories table doesn't have agent_id column - it's in metadata
        b_cursor = storage_b.conn.execute("""
            SELECT metadata FROM memories WHERE content = ?
        """, (content,))
        winner_metadata_json = b_cursor.fetchone()[0]
        winner_metadata = json.loads(winner_metadata_json)
        winner_agent = winner_metadata.get("agent_id")
        assert winner_agent == "beta", f"Expected beta to win conflict, got {winner_agent}"
        
        # But event should be recorded with original authorship (alpha)
        evt_cursor = storage_b.conn.execute("""
            SELECT agent_id FROM sync_events WHERE agent_id = 'alpha'
        """)
        alpha_event = evt_cursor.fetchone()
        assert alpha_event is not None, "Alpha's event should be recorded in sync_events"
        assert alpha_event[0] == "alpha", "Event should preserve alpha authorship"

    @pytest.mark.asyncio
    async def test_ca6_anti_fusion_no_background_sync_dependency(self, storage_a, storage_b):
        """
        CA6 (F6): Orchestrator module does NOT import BackgroundSyncService or reference _sync_loop.
        
        Expected RED: ModuleNotFoundError on orchestrator import, but if it exists,
        this test will fail if it imports forbidden symbols.
        """
        # Import the orchestrator module to inspect it
        try:
            import mcp_memory_service.storage.sync.orchestrator as orchestrator_module
            
            # Check that orchestrator.py doesn't import BackgroundSyncService
            orchestrator_source = open(orchestrator_module.__file__).read()
            
            assert 'BackgroundSyncService' not in orchestrator_source, \
                "Orchestrator must NOT import BackgroundSyncService (forbidden fusion)"
            
            assert '_sync_loop' not in orchestrator_source, \
                "Orchestrator must NOT reference _sync_loop (forbidden fusion)"
            
            # Additional check: verify no direct import from hybrid module
            assert 'from mcp_memory_service.storage.hybrid' not in orchestrator_source, \
                "Orchestrator must NOT import from hybrid module"
            
            # Check module attributes don't include forbidden symbols
            module_attrs = dir(orchestrator_module)
            assert 'BackgroundSyncService' not in module_attrs, \
                "BackgroundSyncService found in orchestrator module attributes"
            
        except ImportError:
            # This is the expected RED - orchestrator.py doesn't exist yet
            pytest.fail("ImportError expected - orchestrator.py should not exist yet (valid RED)")


    @pytest.mark.asyncio
    async def test_partial_apply_failure_does_not_advance_cursor_past_unapplied(
        self, storage_a_with_memories, storage_b
    ):
        """§8.5 (tuvok Gate 5 HIGH): if an apply fails mid-page, the cursor must NOT
        advance past the failed event (it would be lost), and a later re-sync must
        recover the remaining events."""
        from unittest.mock import patch
        from mcp_memory_service.storage.sync import orchestrator as _orch

        storage_a, _ = storage_a_with_memories  # 5 distinct events in A
        adapter = LocalPeerAdapter(storage_a)

        # Make the 3rd apply raise.
        real_apply = _orch.apply_remote_event
        calls = {"n": 0}

        def flaky(storage, event):
            calls["n"] += 1
            if calls["n"] == 3:
                raise RuntimeError("boom on 3rd event")
            return real_apply(storage, event)

        with patch.object(_orch, "apply_remote_event", flaky):
            result = await sync_from_peer(storage_b, adapter, "peer-A", limit=10)

        cur = storage_b.conn.execute(
            "SELECT last_seq_seen FROM sync_cursor WHERE peer_id = 'peer-A'"
        ).fetchone()[0]
        # Only the first 2 applied; cursor must be at 2 (the last good), NOT 5.
        assert result.events_applied == 2, f"expected 2 applied before failure, got {result.events_applied}"
        assert result.events_failed >= 1, "the failed event must be counted"
        assert cur == 2, f"cursor must stop at last applied seq (2), not advance past the failure, got {cur}"

        # Re-sync WITHOUT the fault recovers the rest — nothing was lost.
        result2 = await sync_from_peer(storage_b, adapter, "peer-A", limit=10)
        cur2 = storage_b.conn.execute(
            "SELECT last_seq_seen FROM sync_cursor WHERE peer_id = 'peer-A'"
        ).fetchone()[0]
        live = storage_b.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE deleted_at IS NULL"
        ).fetchone()[0]
        assert cur2 == 5 and live == 5, f"re-sync must recover all 5 (cursor={cur2}, live={live})"

    @pytest.mark.asyncio
    async def test_applied_false_mid_page_also_fail_stops(self, storage_a_with_memories, storage_b):
        """§8.5: an apply returning applied=False mid-page must ALSO fail-stop (not just an
        exception) — locks the applied=False branch (tuvok Gate 5 residual LOW)."""
        from unittest.mock import patch
        from mcp_memory_service.storage.sync import orchestrator as _orch
        from mcp_memory_service.storage.sync.apply import ApplyResult

        storage_a, _ = storage_a_with_memories
        adapter = LocalPeerAdapter(storage_a)
        real_apply = _orch.apply_remote_event
        calls = {"n": 0}

        def soft_fail(storage, event):
            calls["n"] += 1
            if calls["n"] == 3:
                return ApplyResult(applied=False, materialized=False, reason="simulated reject")
            return real_apply(storage, event)

        with patch.object(_orch, "apply_remote_event", soft_fail):
            result = await sync_from_peer(storage_b, adapter, "peer-A", limit=10)

        cur = storage_b.conn.execute(
            "SELECT last_seq_seen FROM sync_cursor WHERE peer_id = 'peer-A'"
        ).fetchone()[0]
        assert cur == 2, f"applied=False mid-page must fail-stop at seq 2, got {cur}"
        assert result.events_failed >= 1
