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
        """greptile P1: the batch is all-or-nothing. If any item fails inside the explicit
        transaction, the whole batch rolls back and NO memory is persisted — every result
        reports failure (a caller is never told a memory was stored when it wasn't)."""
        from unittest.mock import patch
        with patch.dict(os.environ, {"MCP_SYNC_EVENTLOG": "on", "MCP_SEMANTIC_DEDUP_ENABLED": "false"}):
            mems = [
                Memory(content=f"batch atomic {i}",
                       content_hash=generate_content_hash(f"batch atomic {i}"),
                       tags=["bcf"], memory_type="note")
                for i in range(3)
            ]
            # Force a failure on the 2nd item's event append (fail-closed path), inside the
            # batch's explicit transaction. The whole batch must roll back.
            real_append = type(storage)._append_sync_event
            calls = {"n": 0}
            def boom_on_second(self, conn, op, content_hash, payload):
                calls["n"] += 1
                if calls["n"] == 2:
                    raise RuntimeError("event append failed on item 2")
                return real_append(self, conn, op, content_hash, payload)
            with patch.object(type(storage), "_append_sync_event", boom_on_second):
                results = await storage.store_batch(mems)
            assert all(not ok for ok, _ in results), f"all results must be failure, got {results}"
            live = storage.conn.execute(
                "SELECT COUNT(*) FROM memories WHERE tags LIKE '%bcf%' AND deleted_at IS NULL"
            ).fetchone()[0]
            assert live == 0, "all-or-nothing: no memory may persist when any batch item fails"
            # Connection is usable afterwards (no dangling transaction).
            ok, _ = await storage.store(Memory(content="after atomic batch",
                                               content_hash=generate_content_hash("after atomic batch"),
                                               tags=["y"], memory_type="note"))
            assert ok, "connection must be usable after a rolled-back batch"

    @pytest.mark.asyncio
    async def test_batch_final_commit_failure_rolls_back_all(self, storage):
        """greptile P2: cover the final-commit failure specifically — all inserts succeed
        but the commit inside batch_insert fails; the batch must roll back (nothing
        persisted) and report all items failed."""
        from unittest.mock import patch
        with patch.dict(os.environ, {"MCP_SYNC_EVENTLOG": "on", "MCP_SEMANTIC_DEDUP_ENABLED": "false"}):
            mems = [
                Memory(content=f"batch final commit {i}",
                       content_hash=generate_content_hash(f"batch final commit {i}"),
                       tags=["bfc"], memory_type="note")
                for i in range(3)
            ]
            # conn.commit is read-only to patch; wrap the connection in a thin proxy that
            # delegates everything but raises on commit().
            real_conn = storage.conn
            class _CommitFailsConn:
                def __init__(self, c): self._c = c
                def commit(self): raise RuntimeError("commit failed")
                def __getattr__(self, name): return getattr(self._c, name)
            storage.conn = _CommitFailsConn(real_conn)
            try:
                results = await storage.store_batch(mems)
            finally:
                storage.conn = real_conn
            assert all(not ok for ok, _ in results), f"all results must be failure, got {results}"
            live = storage.conn.execute(
                "SELECT COUNT(*) FROM memories WHERE tags LIKE '%bfc%' AND deleted_at IS NULL"
            ).fetchone()[0]
            assert live == 0, "final-commit failure must roll the whole batch back"

    # CA7 (F8): retry-safety under a REAL competing writer holding the SQLite lock.
    @pytest.mark.asyncio
    async def test_batch_locked_by_competing_writer_then_retry_succeeds(self, storage):
        """ducanhnguyen223 review (store.py:232): this host runs several concurrent agents
        against the SAME sqlite_vec.db (single-writer). A transient `database is locked`
        mid-batch must NOT corrupt state or poison the connection.

        Reproduces the finding with a REAL competing writer (not a mocked exception):
        writer A opens BEGIN IMMEDIATE on a second WAL connection and holds the write lock;
        store_batch (B) opens its own BEGIN, then its first INSERT fails with
        'database is locked'; _execute_with_retry backs off; A commits (releases) during the
        backoff; B retries. Without the rollback-before-BEGIN fix (commit 7650a06) the retry
        would hit 'cannot start a transaction within a transaction' (not a lock error, so not
        retried) and the batch would fail.

        Proves, from the client's standpoint:
          - the batch COMPLETES despite the transient lock (all items stored);
          - nothing is half-applied — each stored memory has exactly its create event;
          - the connection stays usable afterwards (no lingering transaction).
        """
        import sqlite3 as _sqlite3
        import threading

        # Make the lock surface fast: a long busy_timeout would just block instead of
        # raising 'database is locked', so the retry path would never be exercised.
        storage.conn.execute("PRAGMA busy_timeout=200")

        with patch.dict(os.environ, {"MCP_SYNC_EVENTLOG": "on", "MCP_SEMANTIC_DEDUP_ENABLED": "false"}):
            mems = [
                Memory(content=f"locked batch {i}",
                       content_hash=generate_content_hash(f"locked batch {i}"),
                       tags=["lbk"], memory_type="note")
                for i in range(3)
            ]

            lock_acquired = threading.Event()
            release_writer = threading.Event()
            busy_seen = threading.Event()
            writer_error = {}

            def competing_writer():
                """Hold the write lock on a second real connection, then release it."""
                w = _sqlite3.connect(storage.db_path, timeout=30, check_same_thread=False)
                try:
                    w.execute("PRAGMA journal_mode=WAL")
                    w.execute("PRAGMA busy_timeout=30000")
                    # Grab the write lock immediately and keep a real pending write.
                    w.execute("BEGIN IMMEDIATE")
                    w.execute(
                        "UPDATE memories SET updated_at = updated_at WHERE 1=0"
                    )
                    lock_acquired.set()
                    # Deterministic hand-off: hold the lock until the batch has ACTUALLY hit
                    # the busy error (release_writer is set by the proxy below on the first
                    # 'database is locked'), not after a wall-clock timer. A timer could fire
                    # before the batch's first write (e.g. slow embedding setup), letting the
                    # batch succeed without ever locking — a green test that proves nothing
                    # (greptile P2). The timeout is only a safety net so a bug can't hang.
                    release_writer.wait(timeout=10)
                    w.commit()
                except Exception as e:  # pragma: no cover - surfaced via assertion below
                    writer_error["err"] = e
                finally:
                    w.close()

            t = threading.Thread(target=competing_writer, daemon=True)
            t.start()
            assert lock_acquired.wait(timeout=5), "competing writer failed to acquire the lock"

            # Proxy the storage connection so that the FIRST 'database is locked' raised by a
            # write deterministically releases the competing writer. This couples the release
            # to the real occurrence of the lock (causal), removing the timing race. It also
            # counts BEGINs: batch_insert issues one BEGIN per attempt, so >=2 proves a retry
            # actually happened (not just that the batch returned success) — greptile P2.
            real_conn = storage.conn
            begins = {"n": 0}

            class _ReleaseOnBusyConn:
                def __init__(self, c):
                    self._c = c

                def execute(self, sql, *a, **k):
                    if isinstance(sql, str) and sql.strip().upper().startswith("BEGIN"):
                        begins["n"] += 1
                    try:
                        return self._c.execute(sql, *a, **k)
                    except _sqlite3.OperationalError as e:
                        msg = str(e).lower()
                        if ("locked" in msg or "busy" in msg) and not busy_seen.is_set():
                            busy_seen.set()
                            release_writer.set()
                        raise

                def __getattr__(self, name):
                    return getattr(self._c, name)

            storage.conn = _ReleaseOnBusyConn(real_conn)
            try:
                results = await storage.store_batch(mems)
            finally:
                storage.conn = real_conn

        t.join(timeout=5)
        assert "err" not in writer_error, f"competing writer errored: {writer_error.get('err')}"

        # 0) The test MUST have actually exercised the lock + retry, otherwise a green result
        #    proves nothing (greptile P2).
        assert busy_seen.is_set(), "the batch never hit 'database is locked' — test is vacuous"
        assert begins["n"] >= 2, f"expected the batch to retry (>=2 BEGINs), ran {begins['n']}"

        # 1) Batch completed despite the transient lock — every item stored.
        assert all(ok for ok, _ in results), f"batch must complete after retry, got {results}"

        # 2) Memories actually persisted.
        live = storage.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE tags LIKE '%lbk%' AND deleted_at IS NULL"
        ).fetchone()[0]
        assert live == len(mems), f"expected {len(mems)} memories persisted, got {live}"

        # 3) Nothing half-applied: exactly one create event per stored memory (ADR-0008).
        for m in mems:
            ev = storage.conn.execute(
                "SELECT COUNT(*) FROM sync_events WHERE content_hash = ? AND op = 'create'",
                (m.content_hash,)
            ).fetchone()[0]
            assert ev == 1, f"expected exactly 1 create event for {m.content_hash}, got {ev}"

        # 4) Connection is usable afterwards — no lingering transaction.
        assert not getattr(storage.conn, "in_transaction", False), \
            "connection must not be left inside a transaction"
        # This probe tests that the CONNECTION is usable, not dedup. The storage fixture is
        # initialized before the env patch, so semantic dedup may still be enabled on the
        # instance; the follow-up content is also similar to the locked-batch items, so a
        # plain store() could return (False, "...semantically similar...") and masquerade as
        # a connection failure. Skip dedup for the probe and surface the reason in the assert
        # so a real connection problem is never hidden by a duplicate rejection
        # (ducanhnguyen223, ML Extras failure on 308b1f4).
        ok, reason = await storage.store(
            Memory(content="after locked batch",
                   content_hash=generate_content_hash("after locked batch"),
                   tags=["z"], memory_type="note"),
            skip_semantic_dedup=True,
        )
        assert ok, f"connection must be usable after the lock-retry batch (store reason: {reason})"


class TestDeltaSyncHLC:
    """Test suite for Phase 2 HLC (Hybrid Logical Clock) functionality with DB."""

    @pytest_asyncio.fixture
    async def storage(self):
        """Create a test storage instance with initialized database."""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_hlc_sync.db")

        storage = SqliteVecMemoryStorage(db_path)
        await storage.initialize()

        yield storage

        # Cleanup
        if storage.conn:
            storage.conn.close()
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def sample_memory(self):
        """Create a sample memory for HLC testing."""
        content = "Test memory for HLC sync"
        return Memory(
            content=content,
            content_hash=generate_content_hash(content),
            tags=["hlc", "test"],
            memory_type="note",
            metadata={"agent_id": "test-hlc-agent", "test": True}
        )

    # CA5 (F1/F2 HLC monotonic): HLC columns exist and grow monotonically
    @pytest.mark.asyncio
    async def test_hlc_columns_exist_in_sync_events(self, storage):
        """
        CA5: sync_events table should have hlc_physical and hlc_logical INTEGER columns.
        Expected to FAIL: migration 016 not implemented, columns don't exist.
        """
        # Check hlc_physical column exists
        cursor = storage.conn.execute("PRAGMA table_info(sync_events)")
        columns = {col[1]: col[2] for col in cursor.fetchall()}

        assert 'hlc_physical' in columns, "sync_events should have hlc_physical column"
        assert 'hlc_logical' in columns, "sync_events should have hlc_logical column"
        assert columns['hlc_physical'] == 'INTEGER', "hlc_physical should be INTEGER type"
        assert columns['hlc_logical'] == 'INTEGER', "hlc_logical should be INTEGER type"

    @pytest.mark.asyncio
    async def test_hlc_index_exists(self, storage):
        """
        Check that HLC index (hlc_physical,hlc_logical,agent_id,event_id) exists.
        Expected to FAIL: migration 016 not implemented, index doesn't exist.
        """
        cursor = storage.conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='index' AND tbl_name='sync_events' AND sql LIKE '%hlc_physical%'
        """)
        hlc_indexes = cursor.fetchall()

        assert len(hlc_indexes) >= 1, "Should have at least one index on hlc_physical"

    @pytest.mark.asyncio
    async def test_store_generates_hlc_when_enabled(self, storage, sample_memory):
        """
        CA5 (F1): With MCP_SYNC_EVENTLOG=on, store() should generate HLC values.
        Expected to FAIL: _append_sync_event not calculating/storing HLC.
        """
        with patch.dict(os.environ, {"MCP_SYNC_EVENTLOG": "on"}):
            success, message = await storage.store(sample_memory)
            assert success, f"Store failed: {message}"

            # Should have HLC values in sync_events
            cursor = storage.conn.execute("""
                SELECT hlc_physical, hlc_logical, created_at
                FROM sync_events WHERE content_hash = ?
            """, (sample_memory.content_hash,))
            row = cursor.fetchone()

            assert row is not None, "Should have sync event"
            hlc_physical, hlc_logical, created_at = row

            assert hlc_physical is not None, "hlc_physical must not be NULL"
            assert hlc_logical is not None, "hlc_logical must not be NULL"
            assert isinstance(hlc_physical, int), "hlc_physical must be INTEGER"
            assert isinstance(hlc_logical, int), "hlc_logical must be INTEGER"
            assert hlc_physical > 0, "hlc_physical should be epoch milliseconds > 0"
            assert hlc_logical >= 0, "hlc_logical should be >= 0"

    @pytest.mark.asyncio
    async def test_hlc_monotonic_across_events(self, storage):
        """
        CA5 (F2): Multiple stores should generate monotonically increasing HLC.
        Expected to FAIL: HLC calculation not implemented in _append_sync_event.
        """
        with patch.dict(os.environ, {"MCP_SYNC_EVENTLOG": "on"}):
            memories = []
            for i in range(3):
                content = f"HLC monotonic test {i}"
                memory = Memory(
                    content=content,
                    content_hash=generate_content_hash(content),
                    tags=["monotonic"],
                    memory_type="note"
                )
                memories.append(memory)

                # skip_semantic_dedup: the full suite warms the embedding model, so
                # near-duplicate content would be dropped and the event count would be
                # non-deterministic. We assert on event count, so bypass dedup (same as
                # the Phase 1 lock-retry test).
                await storage.store(memory, skip_semantic_dedup=True)
                # Small delay to ensure different physical time if HLC uses wall clock
                time.sleep(0.001)

            # Get all HLC values in order
            cursor = storage.conn.execute("""
                SELECT hlc_physical, hlc_logical, content_hash
                FROM sync_events
                WHERE content_hash IN (?, ?, ?)
                ORDER BY seq
            """, tuple(m.content_hash for m in memories))

            hlc_values = cursor.fetchall()
            assert len(hlc_values) == 3, "Should have 3 events"

            # Check monotonic property: (physical, logical) should increase
            for i in range(1, 3):
                prev_physical, prev_logical, _ = hlc_values[i-1]
                curr_physical, curr_logical, _ = hlc_values[i]

                # Either physical increased, or physical same and logical increased
                hlc_increased = (
                    curr_physical > prev_physical or
                    (curr_physical == prev_physical and curr_logical > prev_logical)
                )

                assert hlc_increased, f"HLC must be monotonic: event {i-1} ({prev_physical},{prev_logical}) -> event {i} ({curr_physical},{curr_logical})"

    @pytest.mark.asyncio
    async def test_hlc_logical_increments_within_same_physical_ms(self, storage):
        """
        CA5 (F1): when two events land in the SAME physical millisecond, the logical
        counter MUST increment (kills the `last_logical+1`→`last_logical` mutant; tuvok P2).
        Freeze wall-clock so physical is identical across both stores.
        """
        with patch.dict(os.environ, {"MCP_SYNC_EVENTLOG": "on"}):
            frozen_ms = 1_700_000_000.0  # fixed epoch seconds → identical physical ms
            with patch("time.time", return_value=frozen_ms):
                hashes = []
                for i in range(3):
                    c = f"same-ms event {i}"
                    h = generate_content_hash(c)
                    hashes.append(h)
                    await storage.store(Memory(content=c, content_hash=h,
                                               tags=["same-ms"], memory_type="note"),
                                        skip_semantic_dedup=True)
            cursor = storage.conn.execute(
                "SELECT hlc_physical, hlc_logical FROM sync_events WHERE content_hash IN (?,?,?) ORDER BY seq",
                tuple(hashes),
            )
            rows = cursor.fetchall()
            assert len(rows) == 3
            # All same physical; logical must strictly increase 0,1,2...
            physicals = {r[0] for r in rows}
            assert len(physicals) == 1, "physical must be identical (frozen clock)"
            logicals = [r[1] for r in rows]
            assert logicals == sorted(logicals) and len(set(logicals)) == 3, \
                f"logical must strictly increment within the same physical ms, got {logicals}"

    @pytest.mark.asyncio
    async def test_last_hlc_persisted_in_metadata(self, storage, sample_memory):
        """
        CA5 (F2): last_hlc should be persisted in metadata table.
        Expected to FAIL: last_hlc persistence not implemented.
        """
        with patch.dict(os.environ, {"MCP_SYNC_EVENTLOG": "on"}):
            await storage.store(sample_memory)

            # Check last_hlc in metadata
            cursor = storage.conn.execute("""
                SELECT key, value FROM metadata
                WHERE key IN ('sync_hlc_physical', 'sync_hlc_logical')
                ORDER BY key
            """)
            metadata_hlc = cursor.fetchall()

            assert len(metadata_hlc) == 2, "Should have both sync_hlc_physical and sync_hlc_logical"

            logical_entry = next((entry for entry in metadata_hlc if entry[0] == 'sync_hlc_logical'), None)
            physical_entry = next((entry for entry in metadata_hlc if entry[0] == 'sync_hlc_physical'), None)

            assert physical_entry is not None, "sync_hlc_physical should exist in metadata"
            assert logical_entry is not None, "sync_hlc_logical should exist in metadata"

            # Values should be integers stored as strings
            physical_val = int(physical_entry[1])
            logical_val = int(logical_entry[1])

            assert physical_val > 0, "sync_hlc_physical should be > 0"
            assert logical_val >= 0, "sync_hlc_logical should be >= 0"

    @pytest.mark.asyncio
    async def test_hlc_no_regression_after_restart(self, storage):
        """
        CA5 (F2): Reopening storage should not regress HLC (monotonic across restarts).
        Expected to FAIL: last_hlc not loaded from metadata on initialization.
        """
        with patch.dict(os.environ, {"MCP_SYNC_EVENTLOG": "on"}):
            # Store first memory
            content1 = "Before restart"
            memory1 = Memory(
                content=content1,
                content_hash=generate_content_hash(content1),
                tags=["restart"],
                memory_type="note"
            )
            await storage.store(memory1, skip_semantic_dedup=True)

            # Get the HLC value
            cursor = storage.conn.execute("""
                SELECT hlc_physical, hlc_logical FROM sync_events
                WHERE content_hash = ?
            """, (memory1.content_hash,))
            first_physical, first_logical = cursor.fetchone()

            # Close and reopen storage (simulate restart)
            db_path = storage.db_path
            storage.conn.close()

            new_storage = SqliteVecMemoryStorage(db_path)
            await new_storage.initialize()

            # Store another memory - HLC should not regress
            content2 = "After restart"
            memory2 = Memory(
                content=content2,
                content_hash=generate_content_hash(content2),
                tags=["restart"],
                memory_type="note"
            )
            await new_storage.store(memory2, skip_semantic_dedup=True)

            # Get the new HLC value
            cursor = new_storage.conn.execute("""
                SELECT hlc_physical, hlc_logical FROM sync_events
                WHERE content_hash = ?
            """, (memory2.content_hash,))
            second_physical, second_logical = cursor.fetchone()

            # HLC should not regress
            hlc_increased = (
                second_physical > first_physical or
                (second_physical == first_physical and second_logical > first_logical)
            )

            assert hlc_increased, f"HLC must not regress after restart: ({first_physical},{first_logical}) -> ({second_physical},{second_logical})"

            # Cleanup
            new_storage.conn.close()

    @pytest.mark.asyncio
    async def test_ca6_backfill_phase1_events(self, storage):
        """
        CA6 (NF1 backfill): Phase 1 events without HLC get backfilled at STARTUP.

        Backfill lives in _seed_last_hlc_on_first_boot (startup), NOT in the write hot
        path (_append_sync_event only reads last_hlc from metadata — Greptile P2). So we
        insert a NULL-HLC Phase 1 event, then reopen the storage (simulating a restart)
        to trigger the seed/backfill, and assert the row was repaired.
        """
        import json as _json

        content_hash = "phase1-backfill-test"
        event_id = str(uuid.uuid4())
        created_at = time.time()

        # Insert event WITHOUT hlc_physical/hlc_logical (they would be NULL)
        storage.conn.execute("""
            INSERT INTO sync_events
            (schema_version, agent_id, event_id, op, content_hash, payload, created_at, seq)
            VALUES (1, ?, ?, 'create', ?, ?, ?, ?)
        """, (
            "backfill-agent", event_id, content_hash,
            _json.dumps({"content_hash": content_hash, "op": "create"}),
            created_at, None  # seq will be auto-generated
        ))
        storage.conn.commit()

        # Get the seq value that was generated and confirm HLC starts NULL
        cursor = storage.conn.execute("""
            SELECT seq, hlc_physical, hlc_logical FROM sync_events
            WHERE content_hash = ?
        """, (content_hash,))
        seq, hlc_physical, hlc_logical = cursor.fetchone()
        assert hlc_physical is None, "HLC should be NULL before backfill"
        assert hlc_logical is None, "HLC should be NULL before backfill"

        # Trigger backfill by reopening the storage (restart → _seed_last_hlc_on_first_boot).
        db_path = storage.db_path
        storage.conn.close()
        new_storage = SqliteVecMemoryStorage(db_path)
        await new_storage.initialize()

        # Now the Phase 1 event must be repaired deterministically.
        cursor = new_storage.conn.execute("""
            SELECT seq, hlc_physical, hlc_logical FROM sync_events
            WHERE content_hash = ?
        """, (content_hash,))
        seq, hlc_physical, hlc_logical = cursor.fetchone()

        expected_physical = int(created_at * 1000)
        expected_logical = seq

        assert hlc_physical == expected_physical, f"Backfilled hlc_physical should be {expected_physical}, got {hlc_physical}"
        assert hlc_logical == expected_logical, f"Backfilled hlc_logical should be {expected_logical}, got {hlc_logical}"

    @pytest.mark.asyncio
    async def test_ca7_tombstone_no_resurrect(self, storage):
        """
        CA7 (§8.1c): a delete beats a *replayed old create* (one whose HLC is LOWER
        than the delete). This is a resolver/apply invariant for events arriving out of
        order (Phase 4 transport), NOT a block on legitimately re-creating a memory
        locally — a new local create has a HIGHER HLC and must win (ADR-0012). We assert
        the invariant at the resolver level, which is where Phase 2 owns it.
        """
        from mcp_memory_service.storage.sync.resolver import resolve, EventView

        content_hash = "ca7-hash"
        # A delete at HLC (100,0) and an OLD create replayed at HLC (50,0) — create is older.
        delete_evt = EventView(hlc_physical=100, hlc_logical=0, agent_id="A",
                               event_id="evt-delete", op="delete", content_hash=content_hash)
        old_create = EventView(hlc_physical=50, hlc_logical=0, agent_id="A",
                               event_id="evt-create-old", op="create", content_hash=content_hash)
        # The replayed old create must NOT win over the later delete (no resurrection).
        assert resolve(delete_evt, old_create) is delete_evt
        assert resolve(old_create, delete_evt) is delete_evt  # commutative

        # Conversely, a NEWER create (legitimate re-creation, higher HLC) DOES win —
        # this is the behavior §8.1c must NOT block (ADR-0012).
        new_create = EventView(hlc_physical=150, hlc_logical=0, agent_id="A",
                               event_id="evt-create-new", op="create", content_hash=content_hash)
        assert resolve(delete_evt, new_create) is new_create

    @pytest.mark.asyncio
    async def test_startup_backfill_does_not_regress_last_hlc(self, storage):
        """
        Greptile P1: a backfill on an already-seeded DB must reconcile last_hlc to the
        MAX observed clock, so the next event cannot receive an older HLC.

        Scenario: seed last_hlc low (physical=10), then inject a Phase 1 event whose
        backfilled clock (created_at*1000) is far higher. After restart (seed +
        reconcile), last_hlc must be >= the backfilled event, and the next real store
        must produce a strictly greater HLC — never a regression.
        """
        import json as _json

        # Force a low persisted last_hlc to simulate a stale seed.
        storage.conn.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES ('sync_hlc_physical', '10')")
        storage.conn.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES ('sync_hlc_logical', '0')")

        # Inject a Phase 1 event with NO HLC whose created_at maps to a high physical clock.
        high_created_at = 1_900_000_000.0  # *1000 => 1_900_000_000_000, far above 10
        storage.conn.execute("""
            INSERT INTO sync_events
            (schema_version, agent_id, event_id, op, content_hash, payload, created_at, seq)
            VALUES (1, ?, ?, 'create', ?, ?, ?, ?)
        """, (
            "p1-agent", str(uuid.uuid4()), "p1-regress-hash",
            _json.dumps({"content_hash": "p1-regress-hash", "op": "create"}),
            high_created_at, None,
        ))
        storage.conn.commit()

        # Restart → _seed_last_hlc_on_first_boot backfills AND reconciles last_hlc.
        db_path = storage.db_path
        storage.conn.close()
        new_storage = SqliteVecMemoryStorage(db_path)
        await new_storage.initialize()

        prev = dict(new_storage.conn.execute(
            "SELECT key, value FROM metadata WHERE key IN ('sync_hlc_physical', 'sync_hlc_logical')"
        ).fetchall())
        reconciled_physical = int(prev["sync_hlc_physical"])
        backfilled_physical = int(high_created_at * 1000)
        assert reconciled_physical >= backfilled_physical, (
            f"last_hlc must reconcile up to the backfilled clock, got {reconciled_physical} "
            f"< {backfilled_physical} (would hand the next event an older HLC)"
        )

        # The next real store must not regress below the backfilled event.
        with patch.dict(os.environ, {"MCP_SYNC_EVENTLOG": "on"}):
            c = "post-restart store"
            await new_storage.store(
                Memory(content=c, content_hash=generate_content_hash(c),
                       tags=["regress"], memory_type="note"),
                skip_semantic_dedup=True,
            )
        row = new_storage.conn.execute(
            "SELECT MAX(hlc_physical) FROM sync_events WHERE content_hash = ?",
            (generate_content_hash("post-restart store"),),
        ).fetchone()
        assert row[0] >= backfilled_physical, (
            f"new event HLC {row[0]} regressed below backfilled {backfilled_physical}"
        )
