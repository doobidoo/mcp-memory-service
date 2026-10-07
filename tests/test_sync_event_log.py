"""
RED tests for delta-sync event log functionality (Phase 1).

These tests are designed to FAIL until the sync event log implementation
is complete. They validate the CA1-CA6 requirements from the spec.

DO NOT add try/except blocks that make tests PASS when code is missing.
RED done right: import failures or real assertion failures, never false passes.
"""

import pytest
import pytest_asyncio
import tempfile
import os
import shutil
import json
import time
import uuid
from unittest.mock import patch, MagicMock

# Skip tests if sqlite-vec is not available
try:
    import sqlite_vec
    SQLITE_VEC_AVAILABLE = True
except ImportError:
    SQLITE_VEC_AVAILABLE = False

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.utils.hashing import generate_content_hash

if SQLITE_VEC_AVAILABLE:
    from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
    # These imports WILL fail until implemented - that's the RED we expect
    # from mcp_memory_service.storage.sqlite_vec import _append_sync_event

pytestmark = pytest.mark.skipif(not SQLITE_VEC_AVAILABLE, reason="sqlite-vec not available")


class TestSyncEventLog:
    """Test suite for delta-sync event log functionality."""
    
    @pytest_asyncio.fixture
    async def storage(self):
        """Create a test storage instance with initialized database."""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_sync_events.db")
        
        storage = SqliteVecMemoryStorage(db_path)
        await storage.initialize()
        
        yield storage
        
        # Cleanup
        if storage.conn:
            storage.conn.close()
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def sample_memory(self):
        """Create a sample memory for testing."""
        content = "Test memory for sync events"
        return Memory(
            content=content,
            content_hash=generate_content_hash(content),
            tags=["sync", "test"],
            memory_type="note",
            metadata={"agent_id": "test-agent-001", "test": True}
        )

    # CA4 (F1/F6): Event logging with kill-switch
    @pytest.mark.asyncio
    async def test_store_generates_sync_event_when_enabled(self, storage, sample_memory):
        """
        CA4: With MCP_SYNC_EVENTLOG=on, store() should generate exactly 1 sync event.
        Expected to FAIL: sync_events table doesn't exist, no event hooks in store().
        """
        with patch.dict(os.environ, {"MCP_SYNC_EVENTLOG": "on"}):
            success, message = await storage.store(sample_memory)
            
            # Store should succeed
            assert success, f"Store failed: {message}"
            
            # Should have exactly 1 event in sync_events
            cursor = storage.conn.execute("""
                SELECT seq, agent_id, event_id, op, content_hash, payload, created_at 
                FROM sync_events WHERE content_hash = ?
            """, (sample_memory.content_hash,))
            events = cursor.fetchall()
            
            assert len(events) == 1, f"Expected 1 sync event, got {len(events)}"
            
            event = events[0]
            seq, agent_id, event_id, op, content_hash, payload, created_at = event
            
            assert op == "create", f"Expected op=create, got {op}"
            assert content_hash == sample_memory.content_hash
            assert event_id is not None and event_id != ""
            assert agent_id is not None and agent_id != ""
            assert created_at is not None
    
    @pytest.mark.asyncio
    async def test_store_no_sync_event_when_disabled(self, storage, sample_memory):
        """
        CA4: With MCP_SYNC_EVENTLOG=off, store() should generate 0 sync events (back-compat).
        Expected to FAIL: sync_events table doesn't exist.
        """
        with patch.dict(os.environ, {"MCP_SYNC_EVENTLOG": "off"}):
            success, message = await storage.store(sample_memory)
            
            # Store should succeed
            assert success, f"Store failed: {message}"
            
            # Should have 0 events in sync_events
            cursor = storage.conn.execute("""
                SELECT COUNT(*) FROM sync_events WHERE content_hash = ?
            """, (sample_memory.content_hash,))
            count = cursor.fetchone()[0]
            
            assert count == 0, f"Expected 0 sync events, got {count}"
    
    @pytest.mark.asyncio
    async def test_delete_generates_sync_event_when_enabled(self, storage, sample_memory):
        """
        CA4: delete() should generate sync event with op=delete.
        Expected to FAIL: sync_events table doesn't exist, no event hooks in delete().
        """
        # First store the memory
        await storage.store(sample_memory)
        
        with patch.dict(os.environ, {"MCP_SYNC_EVENTLOG": "on"}):
            success, message = await storage.delete(sample_memory.content_hash)
            
            assert success, f"Delete failed: {message}"
            
            # Should have delete event
            cursor = storage.conn.execute("""
                SELECT op FROM sync_events 
                WHERE content_hash = ? AND op = 'delete'
            """, (sample_memory.content_hash,))
            delete_events = cursor.fetchall()
            
            assert len(delete_events) >= 1, "Expected at least 1 delete sync event"
    
    @pytest.mark.asyncio
    async def test_update_memory_metadata_generates_sync_event_when_enabled(self, storage, sample_memory):
        """
        CA4: update_memory_metadata() should generate sync event with op=update_metadata.
        Expected to FAIL: sync_events table doesn't exist, no event hooks in update_memory_metadata().
        """
        # First store the memory
        await storage.store(sample_memory)
        
        with patch.dict(os.environ, {"MCP_SYNC_EVENTLOG": "on"}):
            success, message = await storage.update_memory_metadata(
                sample_memory.content_hash, 
                {"metadata": {"updated": True}}
            )
            
            assert success, f"Update failed: {message}"
            
            # Should have update_metadata event
            cursor = storage.conn.execute("""
                SELECT op FROM sync_events 
                WHERE content_hash = ? AND op = 'update_metadata'
            """, (sample_memory.content_hash,))
            update_events = cursor.fetchall()
            
            assert len(update_events) >= 1, "Expected at least 1 update_metadata sync event"

    # CA1 (§8.1a): Idempotency via UNIQUE(agent_id, event_id) constraint
    @pytest.mark.asyncio
    async def test_duplicate_event_id_constraint(self, storage):
        """
        CA1 (§8.1a): the same (agent_id, event_id) inserted twice yields exactly one row.
        Proves idempotency via UNIQUE(agent_id, event_id) + INSERT OR IGNORE (ADR-0007).
        """
        import json as _json
        fixed_event_id = str(uuid.uuid4())
        agent_id = "test-agent-001"
        content_hash = "test-hash-123"
        payload = _json.dumps({"content_hash": content_hash, "op": "create"})

        # Direct INSERT OR IGNORE twice with the SAME origin key — the second is a no-op.
        for _ in range(2):
            storage.conn.execute(
                """
                INSERT OR IGNORE INTO sync_events
                    (schema_version, agent_id, event_id, op, content_hash, payload, created_at)
                VALUES (1, ?, ?, 'create', ?, ?, ?)
                """,
                (agent_id, fixed_event_id, content_hash, payload, 0.0),
            )
        storage.conn.commit()

        cursor = storage.conn.execute(
            "SELECT COUNT(*) FROM sync_events WHERE agent_id = ? AND event_id = ?",
            (agent_id, fixed_event_id),
        )
        count = cursor.fetchone()[0]
        assert count == 1, f"Expected 1 event after duplicate insert, got {count}"

    # CA2 (§8.1b): Atomicity - memory without event should not happen
    @pytest.mark.asyncio
    async def test_store_atomicity_event_failure_prevents_memory_storage(self, storage, sample_memory):
        """
        CA2 (§8.1b): if the event append fails during store() with the flag ON, the
        memory is NOT stored (rolled back together) — ADR-0008 fail-closed.
        """
        with patch.dict(os.environ, {"MCP_SYNC_EVENTLOG": "on"}):
            # Make the event append blow up; the hosting store() transaction must roll back.
            with patch.object(type(storage), "_append_sync_event",
                              side_effect=RuntimeError("event append failed")):
                try:
                    await storage.store(sample_memory)
                except Exception:
                    pass  # store may propagate or swallow+return False; both are acceptable

            # Invariant: the memory must NOT be persisted (no memory without its event).
            cursor = storage.conn.execute(
                "SELECT COUNT(*) FROM memories WHERE content_hash = ? AND deleted_at IS NULL",
                (sample_memory.content_hash,),
            )
            memory_count = cursor.fetchone()[0]
            assert memory_count == 0, \
                "Memory must not be stored when the sync event append fails (fail-closed, ADR-0008)"

    # CA2b (§8.1b): atomicity for update_memory_metadata (regression — tuvok P1, PR review)
    @pytest.mark.asyncio
    async def test_update_metadata_atomicity_event_failure_rolls_back(self, storage, sample_memory):
        """
        ADR-0008 fail-closed for update_memory_metadata: if the event append fails, the
        metadata UPDATE must be rolled back — never a mutation without its event, and no
        pending UPDATE that leaks into the next commit.
        """
        with patch.dict(os.environ, {"MCP_SYNC_EVENTLOG": "on"}):
            await storage.store(sample_memory)
            # Update fails at the event append.
            with patch.object(type(storage), "_append_sync_event",
                              side_effect=RuntimeError("event append failed")):
                try:
                    await storage.update_memory_metadata(
                        sample_memory.content_hash, {"metadata": {"leaked": True}}
                    )
                except Exception:
                    pass
            # Do an unrelated committing operation to flush any dangling transaction.
            other = Memory(content="unrelated after failed update",
                           content_hash=generate_content_hash("unrelated after failed update"),
                           tags=["x"], memory_type="note")
            await storage.store(other)
            # The failed update must NOT have leaked into memories.
            cursor = storage.conn.execute(
                "SELECT metadata FROM memories WHERE content_hash = ?",
                (sample_memory.content_hash,),
            )
            row = cursor.fetchone()
            meta = row[0] if row else ""
            assert "leaked" not in (meta or ""), \
                "Metadata update must roll back when its sync event append fails (ADR-0008)"

    # CA2c (§8.1b): atomicity for delete
    @pytest.mark.asyncio
    async def test_delete_atomicity_event_failure_rolls_back(self, storage, sample_memory):
        """ADR-0008 fail-closed for delete(): a failed event append must not leave the
        memory soft-deleted without a delete event."""
        with patch.dict(os.environ, {"MCP_SYNC_EVENTLOG": "on"}):
            await storage.store(sample_memory)
            with patch.object(type(storage), "_append_sync_event",
                              side_effect=RuntimeError("event append failed")):
                try:
                    await storage.delete(sample_memory.content_hash)
                except Exception:
                    pass
            # The memory must still be alive (delete rolled back).
            cursor = storage.conn.execute(
                "SELECT COUNT(*) FROM memories WHERE content_hash = ? AND deleted_at IS NULL",
                (sample_memory.content_hash,),
            )
            assert cursor.fetchone()[0] == 1, \
                "Delete must roll back when its sync event append fails (ADR-0008)"

    # CA2d (§8.1b): atomicity for store_batch (tuvok LOW — lock in the P2 fix)
    @pytest.mark.asyncio
    async def test_store_batch_atomicity_event_failure_rolls_back(self, storage):
        """store_batch must stay fail-closed and leave no dangling savepoint when the
        event append raises a non-sqlite error (ADR-0008 / tuvok P2)."""
        mems = [
            Memory(content=f"batch atomicity {i}",
                   content_hash=generate_content_hash(f"batch atomicity {i}"),
                   tags=["batch"], memory_type="note")
            for i in range(3)
        ]
        with patch.dict(os.environ, {"MCP_SYNC_EVENTLOG": "on"}):
            with patch.object(type(storage), "_append_sync_event",
                              side_effect=RuntimeError("event append failed")):
                try:
                    await storage.store_batch(mems)
                except Exception:
                    pass
            # No memory from the batch should be persisted.
            cursor = storage.conn.execute(
                "SELECT COUNT(*) FROM memories WHERE tags LIKE '%batch%' AND deleted_at IS NULL"
            )
            assert cursor.fetchone()[0] == 0, "store_batch must not persist items when append fails"
            # The connection must remain usable (no dangling savepoint): a later store works.
            ok, _ = await storage.store(
                Memory(content="after batch failure",
                       content_hash=generate_content_hash("after batch failure"),
                       tags=["y"], memory_type="note")
            )
            assert ok, "connection must be usable after a failed batch (no dangling savepoint)"

    # CA3 (§8.1c): Tombstone durability - delete event persists after purge
    @pytest.mark.asyncio
    async def test_delete_tombstone_persists_after_purge(self, storage, sample_memory):
        """
        CA3: delete() -> op=delete event; purge_deleted() removes memory but keeps delete event.
        Expected to FAIL: sync_events table doesn't exist, no delete event hooks.
        """
        with patch.dict(os.environ, {"MCP_SYNC_EVENTLOG": "on"}):
            # Store memory
            await storage.store(sample_memory)
            
            # Delete memory
            await storage.delete(sample_memory.content_hash)
            
            # Purge deleted memories (0 days = purge immediately)
            await storage.purge_deleted(0)
            
            # Memory should be gone from memories table
            cursor = storage.conn.execute("""
                SELECT COUNT(*) FROM memories WHERE content_hash = ?
            """, (sample_memory.content_hash,))
            memory_count = cursor.fetchone()[0]
            
            # But delete event should remain in sync_events
            cursor = storage.conn.execute("""
                SELECT COUNT(*) FROM sync_events 
                WHERE content_hash = ? AND op = 'delete'
            """, (sample_memory.content_hash,))
            delete_event_count = cursor.fetchone()[0]
            
            assert memory_count == 0, "Memory should be purged"
            assert delete_event_count >= 1, "Delete event should persist after purge"

    # CA5 (F7): Inventory of operations that generate events
    @pytest.mark.asyncio
    async def test_delete_by_tag_does_not_generate_events(self, storage, sample_memory):
        """
        CA5: delete_by_tag() should NOT generate events (out of scope for Phase 1).
        Expected to FAIL: sync_events table doesn't exist.
        """
        with patch.dict(os.environ, {"MCP_SYNC_EVENTLOG": "on"}):
            # Store memory with tag
            await storage.store(sample_memory)
            
            # Delete by tag
            success, message = await storage.delete_by_tag("sync")
            assert success, f"delete_by_tag failed: {message}"
            
            # Should NOT have generated delete events
            cursor = storage.conn.execute("""
                SELECT COUNT(*) FROM sync_events 
                WHERE content_hash = ? AND op = 'delete'
            """, (sample_memory.content_hash,))
            delete_event_count = cursor.fetchone()[0]
            
            assert delete_event_count == 0, "delete_by_tag should not generate sync events"
    
    @pytest.mark.asyncio 
    async def test_delete_by_timeframe_generates_events(self, storage, sample_memory):
        """
        CA5: delete_by_timeframe() should generate N events (inherits from delete).
        Expected to FAIL: sync_events table doesn't exist, delete_by_timeframe has no event hooks.
        """
        with patch.dict(os.environ, {"MCP_SYNC_EVENTLOG": "on"}):
            # Store memory
            await storage.store(sample_memory)
            
            # Delete by timeframe (should call delete internally)
            from datetime import date, timedelta
            today = date.today()
            future_date = today + timedelta(days=1)
            success, message = await storage.delete_by_timeframe(today, future_date)
            assert success, f"delete_by_timeframe failed: {message}"
            
            # Should have generated delete event
            cursor = storage.conn.execute("""
                SELECT COUNT(*) FROM sync_events 
                WHERE content_hash = ? AND op = 'delete'
            """, (sample_memory.content_hash,))
            delete_event_count = cursor.fetchone()[0]
            
            assert delete_event_count >= 1, "delete_by_timeframe should generate sync events"

    # Boundary test: Check sync_events table structure
    @pytest.mark.asyncio
    async def test_sync_events_table_exists(self, storage):
        """
        Verify sync_events table exists with correct schema.
        Expected to FAIL: migration 015_add_sync_events.sql not implemented.
        """
        # Check table exists
        cursor = storage.conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='sync_events'
        """)
        table = cursor.fetchone()
        assert table is not None, "sync_events table should exist"
        
        # Check table schema
        cursor = storage.conn.execute("PRAGMA table_info(sync_events)")
        columns = cursor.fetchall()
        
        expected_columns = {
            'seq': 'INTEGER',
            'schema_version': 'INT', 
            'agent_id': 'TEXT',
            'event_id': 'TEXT',
            'op': 'TEXT',
            'content_hash': 'TEXT', 
            'payload': 'TEXT',
            'created_at': 'REAL'
        }
        
        actual_columns = {col[1]: col[2] for col in columns}
        
        for col_name, col_type in expected_columns.items():
            assert col_name in actual_columns, f"Column {col_name} should exist"
            # Note: SQLite type matching is flexible, so we don't enforce exact type strings
    
    @pytest.mark.asyncio
    async def test_sync_events_indexes_exist(self, storage):
        """
        Verify required indexes exist on sync_events table.
        Expected to FAIL: migration 015_add_sync_events.sql not implemented.
        """
        cursor = storage.conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='index' AND tbl_name='sync_events'
        """)
        indexes = [row[0] for row in cursor.fetchall()]
        
        # Should have indexes on commonly queried columns
        # The exact index names depend on the migration implementation
        assert len(indexes) >= 1, "Should have at least 1 index on sync_events table"

    # Helper method test
    @pytest.mark.asyncio
    async def test_append_sync_event_helper_function(self, storage):
        """
        Test the _append_sync_event helper directly: with the flag ON it writes one row
        (op/content_hash/event_id/schema_version) in the current transaction, no own commit.
        """
        with patch.dict(os.environ, {"MCP_SYNC_EVENTLOG": "on"}):
            storage._append_sync_event(
                storage.conn,
                op="create",
                content_hash="helper-hash-xyz",
                payload={"content_hash": "helper-hash-xyz", "op": "create"},
            )
            storage.conn.commit()

        cursor = storage.conn.execute(
            "SELECT op, content_hash, event_id, schema_version FROM sync_events WHERE content_hash = ?",
            ("helper-hash-xyz",),
        )
        row = cursor.fetchone()
        assert row is not None, "Helper must insert a sync_events row when enabled"
        assert row[0] == "create"
        assert row[1] == "helper-hash-xyz"
        assert row[2], "event_id must be a non-empty UUID"
        assert row[3] == 1, "schema_version must be stamped (envelope versioning)"

        # Flag OFF → helper is a no-op (back-compat).
        with patch.dict(os.environ, {"MCP_SYNC_EVENTLOG": "off"}):
            storage._append_sync_event(
                storage.conn, op="create", content_hash="helper-off", payload={},
            )
            storage.conn.commit()
        cursor = storage.conn.execute(
            "SELECT COUNT(*) FROM sync_events WHERE content_hash = ?", ("helper-off",),
        )
        assert cursor.fetchone()[0] == 0, "Helper must be a no-op when MCP_SYNC_EVENTLOG is off"

class TestGreptilePhase1Fixes:
    """Regression tests for the Greptile review findings on PR #1478 (delta-sync Phase 1)."""

    @pytest_asyncio.fixture
    async def storage(self):
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_greptile_fixes.db")
        storage = SqliteVecMemoryStorage(db_path)
        await storage.initialize()
        yield storage
        if storage.conn:
            storage.conn.close()
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_null_agent_uses_non_null_sentinel(self, storage):
        """greptile P1: without MCP_AGENT_ID, agent_id must be a non-null sentinel so
        UNIQUE(agent_id, event_id) stays effective (SQLite treats NULL as distinct)."""
        with patch.dict(os.environ, {"MCP_SYNC_EVENTLOG": "on"}, clear=False):
            os.environ.pop("MCP_AGENT_ID", None)
            c = "null agent sentinel probe"
            await storage.store(Memory(content=c, content_hash=generate_content_hash(c),
                                       tags=["x"], memory_type="note"))
            agent = storage.conn.execute("SELECT agent_id FROM sync_events LIMIT 1").fetchone()[0]
            assert agent is not None and agent != "", f"agent_id must be a non-null sentinel, got {agent!r}"

    @pytest.mark.asyncio
    async def test_create_event_payload_carries_content_and_store(self, storage):
        """greptile P1: a create event must carry content and store so a delta reader can
        reconstruct the memory without a separate enrichment step."""
        with patch.dict(os.environ, {"MCP_SYNC_EVENTLOG": "on"}):
            c = "create payload content probe"
            await storage.store(Memory(content=c, content_hash=generate_content_hash(c),
                                       tags=["x"], memory_type="note"))
            payload = json.loads(storage.conn.execute(
                "SELECT payload FROM sync_events WHERE op='create' LIMIT 1").fetchone()[0])
            assert payload.get("content") == c, "create payload must carry content"
            assert "store" in payload, "create payload must carry the store partition"

    @pytest.mark.asyncio
    async def test_update_on_deleted_row_emits_no_event(self, storage):
        """greptile P1: if the row is gone (deleted) when the UPDATE runs, no event is
        emitted (rowcount 0 must not log a mutation that didn't happen)."""
        with patch.dict(os.environ, {"MCP_SYNC_EVENTLOG": "on"}):
            c = "update rowcount probe"
            h = generate_content_hash(c)
            await storage.store(Memory(content=c, content_hash=h, tags=["x"], memory_type="note"))
            await storage.delete(h)  # soft-delete
            before = storage.conn.execute(
                "SELECT COUNT(*) FROM sync_events WHERE op='update_metadata' AND content_hash=?", (h,)
            ).fetchone()[0]
            # Update a now-deleted row — must not emit an update_metadata event.
            await storage.update_memory_metadata(h, {"metadata": {"x": 1}})
            after = storage.conn.execute(
                "SELECT COUNT(*) FROM sync_events WHERE op='update_metadata' AND content_hash=?", (h,)
            ).fetchone()[0]
            assert after == before, "no update_metadata event for an UPDATE that matched no live row"

    @pytest.mark.asyncio
    async def test_batch_commit_failure_marks_all_results_failed(self, storage):
        """greptile P1 (re-review): if the batch commit fails, the transaction is rolled
        back and NO memory is persisted — every result must report failure, not success."""
        from unittest.mock import patch
        with patch.dict(os.environ, {"MCP_SYNC_EVENTLOG": "on", "MCP_SEMANTIC_DEDUP_ENABLED": "false"}):
            mems = [
                Memory(content=f"batch commit fail {i}",
                       content_hash=generate_content_hash(f"batch commit fail {i}"),
                       tags=["bcf"], memory_type="note")
                for i in range(3)
            ]
            # Let the inserts succeed but make the final commit blow up. store_batch runs the
            # commit via self._execute_with_retry(self.conn.commit); wrap that to raise only
            # when it's handed the commit callable (can't patch conn.commit — read-only).
            real_ewr = storage._execute_with_retry
            async def flaky_ewr(op, *a, **k):
                if getattr(op, "__name__", "") == "commit":
                    raise RuntimeError("commit failed")
                return await real_ewr(op, *a, **k)
            with patch.object(type(storage), "_execute_with_retry", flaky_ewr):
                results = await storage.store_batch(mems)
            assert all(not ok for ok, _ in results), f"all results must be failure, got {results}"
            live = storage.conn.execute(
                "SELECT COUNT(*) FROM memories WHERE tags LIKE '%bcf%' AND deleted_at IS NULL"
            ).fetchone()[0]
            assert live == 0, "no memory may persist when the batch commit fails"
