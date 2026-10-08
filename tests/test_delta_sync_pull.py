"""
Tests for delta-sync Phase 4a: PULL transport (feed + apply + cursor).

These tests are designed to FAIL against current code since Phase 4a is not implemented yet.
They test requirements F1-F7, CA1-CA7 from spec-delta-sync-fase4a.md.

ANTI-PATTERN AVOIDED: No pytest.skip, no try/except that makes tests pass,
imports at top will break collection = valid RED, asserts test real behavior.
"""

import pytest
import pytest_asyncio
import tempfile
import os
import json
import time
from typing import Dict, Any, List, Tuple
from fastapi.testclient import TestClient

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash
from mcp_memory_service.web.dependencies import set_storage

# These imports WILL break collection (ModuleNotFoundError) = valid RED
from mcp_memory_service.storage.sync.apply import apply_remote_event, advance_sync_cursor


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
    
    storage = SqliteVecMemoryStorage(temp_db_a)
    await storage.initialize()
    yield storage
    await storage.close()


@pytest_asyncio.fixture
async def storage_b(temp_db_b, monkeypatch):
    """Create and initialize storage B (target) with agent=beta."""
    monkeypatch.setenv('MCP_AGENT_ID', 'beta')
    monkeypatch.setenv('MCP_SYNC_EVENTLOG', 'true')
    monkeypatch.setenv('MCP_SEMANTIC_DEDUP_ENABLED', 'false')
    
    storage = SqliteVecMemoryStorage(temp_db_b)
    await storage.initialize()
    yield storage
    await storage.close()


@pytest.fixture
def test_app_with_storage(storage_a):
    """Create FastAPI test app with storage A for endpoint testing."""
    from mcp_memory_service.web.app import create_app
    
    app = create_app()
    set_storage(storage_a)
    
    # Override auth for testing
    from mcp_memory_service.web.oauth.middleware import require_read_access, require_write_access
    app.dependency_overrides[require_read_access] = lambda: True
    app.dependency_overrides[require_write_access] = lambda: True
    
    return TestClient(app)


class TestSyncCursorTable:
    """Test CA7 (NF1): Migration 018 creates sync_cursor table."""
    
    @pytest_asyncio.fixture
    async def fresh_storage(self, temp_db_a):
        """Storage for migration testing."""
        storage = SqliteVecMemoryStorage(temp_db_a)
        await storage.initialize()
        yield storage
        await storage.close()
    
    async def test_migration_018_creates_sync_cursor_table(self, fresh_storage):
        """
        CA7: Migration 018 should create sync_cursor table with correct schema.
        
        Expected to FAIL: table sync_cursor does not exist yet.
        """
        # Test that sync_cursor table exists with correct columns
        cursor = fresh_storage.conn.execute("PRAGMA table_info(sync_cursor)")
        columns = cursor.fetchall()
        
        # Should have columns: peer_id, last_seq_seen, last_hlc_physical, last_hlc_logical, updated_at
        assert len(columns) > 0, "sync_cursor table should exist"
        
        column_names = [col[1] for col in columns]  # Column name is index 1 in PRAGMA result
        assert 'peer_id' in column_names, "sync_cursor should have peer_id column"
        assert 'last_seq_seen' in column_names, "sync_cursor should have last_seq_seen column"
        assert 'last_hlc_physical' in column_names, "sync_cursor should have last_hlc_physical column"
        assert 'last_hlc_logical' in column_names, "sync_cursor should have last_hlc_logical column"
        assert 'updated_at' in column_names, "sync_cursor should have updated_at column"


class TestSyncEventsEndpoint:
    """Test CA1 (F1): GET /api/sync/events endpoint."""
    
    async def test_sync_events_endpoint_exists(self, test_app_with_storage, storage_a):
        """
        CA1: GET /api/sync/events should return events > since_seq with correct format.
        
        Expected to FAIL: endpoint does not exist yet (404).
        """
        # First store some memories to generate events
        from mcp_memory_service.utils.hashing import generate_content_hash
        
        content1 = "Test memory 1"
        memory1 = Memory(
            content=content1,
            content_hash=generate_content_hash(content1),
            tags=["test"]
        )
        content2 = "Test memory 2"
        memory2 = Memory(
            content=content2,
            content_hash=generate_content_hash(content2),
            tags=["test"]
        )
        content3 = "Test memory 3"
        memory3 = Memory(
            content=content3,
            content_hash=generate_content_hash(content3),
            tags=["test"]
        )
        
        await storage_a.store(memory1)
        await storage_a.store(memory2)
        await storage_a.store(memory3)
        
        # Test the endpoint
        response = test_app_with_storage.get("/api/sync/events?since_seq=0&limit=100")
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        
        data = response.json()
        
        # Check response structure
        assert "events" in data, "Response should have events field"
        assert "next_seq" in data, "Response should have next_seq field"
        assert "has_more" in data, "Response should have has_more field"
        
        events = data["events"]
        assert len(events) == 3, f"Should have 3 events, got {len(events)}"
        
        # Check event structure
        for event in events:
            assert "seq" in event, "Event should have seq field"
            assert "event_id" in event, "Event should have event_id field"
            assert "op" in event, "Event should have op field"
            assert "content_hash" in event, "Event should have content_hash field"
            assert "agent_id" in event, "Event should have agent_id field"
            assert "hlc_physical" in event, "Event should have hlc_physical field"
            assert "hlc_logical" in event, "Event should have hlc_logical field"
            assert "embedding_model" in event, "Event should have embedding_model field"
            assert "embedding_dim" in event, "Event should have embedding_dim field"
            assert "payload" in event, "Event should have payload field"
        
        # Check that create events are enriched with content
        create_events = [e for e in events if e["op"] == "create"]
        for event in create_events:
            assert "content" in event["payload"], "Create events should have content in payload"


class TestDeltaSyncApply:
    """Test CA2-CA6: apply_remote_event functionality."""
    
    async def test_idempotent_apply_ca2(self, storage_a, storage_b):
        """
        CA2 (F3): Applying the same feed twice should be idempotent.
        
        Expected to FAIL: apply_remote_event function does not exist yet.
        """
        # Store 3 memories in A
        from mcp_memory_service.utils.hashing import generate_content_hash
        
        content1 = "Memory 1"
        memory1 = Memory(
            content=content1,
            content_hash=generate_content_hash(content1),
            tags=["test"]
        )
        content2 = "Memory 2" 
        memory2 = Memory(
            content=content2,
            content_hash=generate_content_hash(content2),
            tags=["test"]
        )
        content3 = "Memory 3"
        memory3 = Memory(
            content=content3,
            content_hash=generate_content_hash(content3),
            tags=["test"]
        )
        
        await storage_a.store(memory1)
        await storage_a.store(memory2)
        await storage_a.store(memory3)
        
        # Read events from A (simulate feed)
        cursor_a = storage_a.conn.execute("""
            SELECT seq, agent_id, event_id, op, content_hash, hlc_physical, hlc_logical,
                   embedding_model, embedding_dim, payload
            FROM sync_events 
            ORDER BY seq
        """)
        events = []
        for row in cursor_a.fetchall():
            event = {
                'seq': row[0],
                'agent_id': row[1], 
                'event_id': row[2],
                'op': row[3],
                'content_hash': row[4],
                'hlc_physical': row[5],
                'hlc_logical': row[6],
                'embedding_model': row[7],
                'embedding_dim': row[8],
                'payload': json.loads(row[9]) if row[9] else {}
            }
            
            # Enrich create events with content from memories table
            if event['op'] == 'create':
                memory_cursor = storage_a.conn.execute(
                    "SELECT content FROM memories WHERE content_hash = ?",
                    (event['content_hash'],)
                )
                memory_row = memory_cursor.fetchone()
                if memory_row:
                    event['payload']['content'] = memory_row[0]
                    
            events.append(event)
        
        # Apply events to B (first time)
        for event in events:
            apply_remote_event(storage_b, event)
        
        # Count records in B after first apply
        sync_events_count_1 = storage_b.conn.execute("SELECT COUNT(*) FROM sync_events").fetchone()[0]
        memories_count_1 = storage_b.conn.execute("SELECT COUNT(*) FROM memories WHERE deleted_at IS NULL").fetchone()[0]
        
        # Apply the SAME events to B (second time) - should be idempotent
        for event in events:
            apply_remote_event(storage_b, event)
            
        # Count records in B after second apply
        sync_events_count_2 = storage_b.conn.execute("SELECT COUNT(*) FROM sync_events").fetchone()[0]
        memories_count_2 = storage_b.conn.execute("SELECT COUNT(*) FROM memories WHERE deleted_at IS NULL").fetchone()[0]
        
        # Should be identical (idempotent)
        assert sync_events_count_1 == sync_events_count_2, f"sync_events count changed: {sync_events_count_1} -> {sync_events_count_2}"
        assert memories_count_1 == memories_count_2, f"memories count changed: {memories_count_1} -> {memories_count_2}"
        assert sync_events_count_1 == 3, f"Expected 3 sync_events, got {sync_events_count_1}"
        assert memories_count_1 == 3, f"Expected 3 memories, got {memories_count_1}"
    
    async def test_authorship_preserved_ca5(self, storage_a, storage_b):
        """
        CA5 (F7): Events applied to B should preserve agent_id=alpha from A.
        
        Expected to FAIL: apply_remote_event does not preserve agent_id yet.
        """
        # Store memory in A (agent=alpha). Force the env at store time: fixture order can
        # leave MCP_AGENT_ID=beta in the process env (storage_b's monkeypatch), and the
        # event's agent_id is read from os.getenv at _append_sync_event time.
        import os as _os
        from unittest.mock import patch as _patch
        content = "Test memory"
        memory = Memory(
            content=content,
            content_hash=generate_content_hash(content),
            tags=["test"]
        )
        with _patch.dict(_os.environ, {"MCP_AGENT_ID": "alpha"}):
            await storage_a.store(memory)
        
        # Get event from A
        cursor_a = storage_a.conn.execute("""
            SELECT agent_id, event_id, op, content_hash, hlc_physical, hlc_logical,
                   embedding_model, embedding_dim, payload
            FROM sync_events 
            LIMIT 1
        """)
        row = cursor_a.fetchone()
        
        event = {
            'seq': 1,  # Will be different in B
            'agent_id': row[0],  # Should be 'alpha'
            'event_id': row[1],
            'op': row[2], 
            'content_hash': row[3],
            'hlc_physical': row[4],
            'hlc_logical': row[5],
            'embedding_model': row[6],
            'embedding_dim': row[7],
            'payload': {'content': content, 'tags': ['test']}
        }
        
        # Apply to B (agent=beta)
        apply_remote_event(storage_b, event)
        
        # Check that B preserves agent_id=alpha
        cursor_b = storage_b.conn.execute("SELECT agent_id FROM sync_events LIMIT 1")
        stored_agent_id = cursor_b.fetchone()[0]
        
        assert stored_agent_id == 'alpha', f"Expected agent_id='alpha', got '{stored_agent_id}'"
    
    async def test_resolver_and_late_events_ca3(self, storage_b):
        """
        CA3 (F4): a winning event (higher HLC) materializes; a LATE event (lower HLC)
        over the same content_hash is recorded but does NOT overwrite the materialized
        winner. Exercises the resolver inside the apply (ADR-0010/0012), not just a
        single call.
        """
        content_hash = generate_content_hash("ca3 resolver content")
        base = {
            'op': 'create', 'content_hash': content_hash, 'agent_id': 'alpha',
            'embedding_model': storage_b.embedding_model_name, 'embedding_dim': 384,
        }
        # Winner: higher HLC, materializes "WINNER CONTENT".
        winner = {**base, 'event_id': 'evt-winner', 'hlc_physical': 2000, 'hlc_logical': 0,
                  'payload': {'content': 'WINNER CONTENT', 'tags': ['ca3']}}
        apply_remote_event(storage_b, winner)
        # Late event: LOWER HLC, different content — must NOT overwrite the winner.
        late = {**base, 'op': 'update_metadata', 'event_id': 'evt-late',
                'hlc_physical': 1000, 'hlc_logical': 0,
                'payload': {'content': 'LATE CONTENT', 'tags': ['ca3-late']}}
        apply_remote_event(storage_b, late)

        row = storage_b.conn.execute(
            "SELECT content FROM memories WHERE content_hash = ?", (content_hash,)
        ).fetchone()
        assert row is not None, "winner must be materialized"
        assert row[0] == 'WINNER CONTENT', \
            f"late event (lower HLC) must NOT overwrite the materialized winner, got {row[0]!r}"
        # Both events are recorded (idempotent log), even though late lost.
        n = storage_b.conn.execute(
            "SELECT COUNT(*) FROM sync_events WHERE content_hash = ?", (content_hash,)
        ).fetchone()[0]
        assert n == 2, f"both events recorded in the log, got {n}"

    async def test_materialization_ca6(self, storage_a, storage_b):
        """
        CA6 (F5): Materialization should create searchable memories, handle embedding_pending.
        
        Expected to FAIL: materialization logic not implemented in apply yet.
        """
        # Store memory in A
        content = "Test content for search"
        memory = Memory(
            content=content,
            content_hash=generate_content_hash(content),
            tags=["searchable"]
        )
        await storage_a.store(memory)
        content_hash = memory.content_hash
        
        # Get create event from A
        cursor_a = storage_a.conn.execute("""
            SELECT agent_id, event_id, op, content_hash, hlc_physical, hlc_logical,
                   embedding_model, embedding_dim, payload
            FROM sync_events 
            WHERE content_hash = ?
        """, (content_hash,))
        row = cursor_a.fetchone()
        
        event = {
            'seq': 1,
            'agent_id': row[0],
            'event_id': row[1],
            'op': row[2],
            'content_hash': row[3], 
            'hlc_physical': row[4],
            'hlc_logical': row[5],
            'embedding_model': row[6],
            'embedding_dim': row[7],
            'payload': {'content': 'Test content for search', 'tags': ['searchable']}
        }
        
        # Apply to B
        apply_remote_event(storage_b, event)
        
        # Memory should be materialized and searchable in B
        materialized = await storage_b.get_by_hash(content_hash)
        assert materialized is not None, "Memory should be materialized in B"
        assert materialized.content == "Test content for search", "Content should match"
        
        # Test search works
        # Search must find the materialized memory. Query by a term from the CONTENT
        # ("Test content for search"), not by the tag "searchable" — semantic search
        # ranks on content, and the point of CA6 is that the apply generated a usable
        # embedding (memory is searchable), which the content query proves.
        search_results = await storage_b.retrieve("content for search", n_results=10)
        found = any(r.memory.content_hash == content_hash for r in search_results)
        assert found, "Materialized memory must be searchable in B (apply generated its embedding)"
    
    async def test_cursor_advance_ca4(self, storage_a, storage_b):
        """
        CA4 (F6): Cursor should advance after successful batch apply.
        
        Expected to FAIL: advance_sync_cursor function does not exist yet.
        """
        # Store memory in A to create event
        content = "Test memory"
        memory = Memory(
            content=content,
            content_hash=generate_content_hash(content),
            tags=["test"]
        )
        await storage_a.store(memory)
        
        # Get event from A
        cursor_a = storage_a.conn.execute("SELECT seq, agent_id, event_id, op, content_hash, hlc_physical, hlc_logical, embedding_model, embedding_dim, payload FROM sync_events LIMIT 1")
        row = cursor_a.fetchone()
        
        event = {
            'seq': row[0],  # This will be the seq from A
            'agent_id': row[1],
            'event_id': row[2],
            'op': row[3],
            'content_hash': row[4],
            'hlc_physical': row[5],
            'hlc_logical': row[6],
            'embedding_model': row[7], 
            'embedding_dim': row[8],
            'payload': {'content': content, 'tags': ['test']}
        }
        
        # Apply event
        apply_remote_event(storage_b, event)
        
        # Advance cursor for peer 'alpha' to seq 1 
        advance_sync_cursor(storage_b, 'alpha', 1)
        
        # Check cursor was advanced
        cursor_b = storage_b.conn.execute(
            "SELECT last_seq_seen FROM sync_cursor WHERE peer_id = ?", 
            ('alpha',)
        )
        cursor_row = cursor_b.fetchone()
        assert cursor_row is not None, "Cursor should exist for peer alpha"
        assert cursor_row[0] == 1, f"Cursor should be advanced to seq 1, got {cursor_row[0]}"
        
        # Re-pull from same cursor should be no-op
        # (This would be tested by checking no duplicate records are created)