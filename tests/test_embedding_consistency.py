"""
RED tests for delta-sync Phase 3: embedding consistency functionality.

These tests are designed to FAIL until the embedding consistency implementation
is complete. They validate the CA1-CA5 requirements from spec-delta-sync-fase3.md.

DO NOT add try/except blocks that make tests PASS when code is missing.
RED done right: import failures or real assertion failures, never false passes.

CRITICAL NUANCE (per ADR-0017): Phase 3 LOCAL has no live producer of
embedding_pending (store is atomic, content immutable per row). Tests drive
the flag DIRECTLY (UPDATE SQL or helper) - they do NOT invent fake content
updates to create a producer.
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

pytestmark = pytest.mark.skipif(not SQLITE_VEC_AVAILABLE, reason="sqlite-vec not available")


class TestEmbeddingConsistency:
    """Test suite for delta-sync Phase 3 embedding consistency functionality."""
    
    @pytest_asyncio.fixture
    async def storage(self):
        """Create a test storage instance with initialized database."""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_embedding_consistency.db")
        
        storage = SqliteVecMemoryStorage(db_path)
        await storage.initialize()
        
        yield storage
        
        # Cleanup
        if storage.conn:
            storage.conn.close()
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def sample_memory(self):
        """Create a sample memory for testing embedding consistency."""
        content = "Test memory for embedding consistency validation"
        return Memory(
            content=content,
            content_hash=generate_content_hash(content),
            tags=["embedding", "consistency", "test"],
            memory_type="note",
            metadata={"agent_id": "test-agent-embedding", "test": True}
        )

    def mark_embedding_pending(self, storage, content_hash: str, pending: int = 1):
        """
        Helper to directly set embedding_pending flag via SQL.
        This simulates the state that Phase 4 apply or model change would create.
        Per ADR-0017, we drive the flag directly - no fake content updates.
        """
        storage.conn.execute(
            "UPDATE memories SET embedding_pending = ? WHERE content_hash = ?",
            (pending, content_hash)
        )
        storage.conn.commit()

    # CA1 (F3/F5): pending memories invisible to retrieve/search, visible when cleared
    @pytest.mark.asyncio
    async def test_pending_memory_excluded_from_retrieve(self, storage, sample_memory):
        """
        CA1: Store memory → visible in retrieve. Mark embedding_pending=1 → NOT in retrieve.
        Clear flag → visible again.
        Expected to FAIL: embedding_pending column doesn't exist, retrieve doesn't exclude pending.
        """
        # Store memory - should be visible
        success, message = await storage.store(sample_memory)
        assert success, f"Store failed: {message}"
        
        # Should be visible in retrieve
        results = await storage.retrieve(sample_memory.content, n_results=5)
        assert len(results) >= 1, "Stored memory should be visible in retrieve"
        assert any(r.memory.content_hash == sample_memory.content_hash for r in results), \
            "Stored memory should be found by retrieve"
        
        # Mark as embedding_pending=1 (simulates Phase 4/model-change state)
        self.mark_embedding_pending(storage, sample_memory.content_hash, 1)
        
        # Should NOT be visible in retrieve
        results = await storage.retrieve(sample_memory.content, n_results=5)
        assert not any(r.memory.content_hash == sample_memory.content_hash for r in results), \
            "Pending memory should be excluded from retrieve"
        
        # Clear pending flag (embedding_pending=0)
        self.mark_embedding_pending(storage, sample_memory.content_hash, 0)
        
        # Should be visible again
        results = await storage.retrieve(sample_memory.content, n_results=5)
        assert any(r.memory.content_hash == sample_memory.content_hash for r in results), \
            "Memory should be visible again after clearing pending flag"

    @pytest.mark.asyncio
    async def test_pending_memory_excluded_from_search_by_tag(self, storage, sample_memory):
        """
        CA1: Store memory → visible in search_by_tag. Mark embedding_pending=1 → NOT in search.
        Clear flag → visible again.
        Expected to FAIL: embedding_pending column doesn't exist, search_by_tag doesn't exclude pending.
        """
        # Store memory - should be visible
        success, message = await storage.store(sample_memory)
        assert success, f"Store failed: {message}"
        
        # Should be visible in search_by_tag
        results = await storage.search_by_tag(["embedding"])
        assert len(results) >= 1, "Stored memory should be visible in search_by_tag"
        assert any(r.content_hash == sample_memory.content_hash for r in results), \
            "Stored memory should be found by search_by_tag"
        
        # Mark as embedding_pending=1
        self.mark_embedding_pending(storage, sample_memory.content_hash, 1)
        
        # Should NOT be visible in search_by_tag
        results = await storage.search_by_tag(["embedding"])
        assert not any(r.content_hash == sample_memory.content_hash for r in results), \
            "Pending memory should be excluded from search_by_tag"
        
        # Clear pending flag
        self.mark_embedding_pending(storage, sample_memory.content_hash, 0)
        
        # Should be visible again
        results = await storage.search_by_tag(["embedding"])
        assert any(r.content_hash == sample_memory.content_hash for r in results), \
            "Memory should be visible again after clearing pending flag"

    # CA2 (F1): create event stamped with embedding_model/embedding_dim
    @pytest.mark.asyncio
    async def test_sync_event_includes_embedding_metadata(self, storage, sample_memory):
        """
        CA2: With MCP_SYNC_EVENTLOG=on, store() should stamp create event with
        embedding_model (active model name) and embedding_dim (>0).
        Expected to FAIL: sync_events table missing embedding_model/embedding_dim columns.
        """
        with patch.dict(os.environ, {"MCP_SYNC_EVENTLOG": "on"}):
            success, message = await storage.store(sample_memory)
            assert success, f"Store failed: {message}"
            
            # Query sync event with embedding metadata
            cursor = storage.conn.execute("""
                SELECT embedding_model, embedding_dim, op, content_hash
                FROM sync_events 
                WHERE content_hash = ? AND op = 'create'
            """, (sample_memory.content_hash,))
            events = cursor.fetchall()
            
            assert len(events) == 1, f"Expected 1 create event, got {len(events)}"
            
            embedding_model, embedding_dim, op, content_hash = events[0]
            
            # Should have embedding metadata from active model
            assert embedding_model is not None, "embedding_model should not be NULL"
            assert embedding_model != "", "embedding_model should not be empty"
            assert embedding_dim is not None, "embedding_dim should not be NULL"
            assert embedding_dim > 0, f"embedding_dim should be > 0, got {embedding_dim}"
            
            # Provenance must be HONEST: when the hash fallback is active (e.g. the minimal
            # CI job with model downloads disabled), the event must carry the distinct
            # '__hash_fallback__::<dim>' identity — NOT the configured model name (Greptile P1).
            # Otherwise it must match the storage's active embedding model.
            if getattr(storage, "embedding_backend_degraded", False):
                assert embedding_model == f"__hash_fallback__::{storage.embedding_dimension}", \
                    f"degraded backend must stamp hash-fallback identity, got {embedding_model}"
            else:
                assert embedding_model == storage.embedding_model_name, \
                    f"Event embedding_model {embedding_model} should match storage model {storage.embedding_model_name}"

    # CA3 (F4 honesty): store() atomic path never produces embedding_pending
    @pytest.mark.asyncio
    async def test_atomic_store_leaves_embedding_pending_zero(self, storage, sample_memory):
        """
        CA3: Normal store() via atomic path should leave embedding_pending=0.
        Per ADR-0017, local atomic store never produces pending state.
        Expected to FAIL: embedding_pending column doesn't exist.
        """
        success, message = await storage.store(sample_memory)
        assert success, f"Store failed: {message}"
        
        # Check embedding_pending is 0 (consistent)
        cursor = storage.conn.execute("""
            SELECT embedding_pending FROM memories WHERE content_hash = ?
        """, (sample_memory.content_hash,))
        rows = cursor.fetchall()
        
        assert len(rows) == 1, f"Expected 1 memory row, got {len(rows)}"
        embedding_pending = rows[0][0]
        
        assert embedding_pending == 0, \
            f"Atomic store should leave embedding_pending=0, got {embedding_pending}"

    # CA4 (F3 enumeration): pending memories invisible in ALL read sites
    @pytest.mark.asyncio
    async def test_pending_memory_excluded_from_all_read_sites(self, storage, sample_memory):
        """
        CA4: Ensure pending memory is invisible in retrieve AND search_by_tag
        (covering all read sites that join memories table).
        Expected to FAIL: embedding_pending column doesn't exist, read methods don't exclude pending.
        """
        # Store and verify visible
        success, message = await storage.store(sample_memory)
        assert success, f"Store failed: {message}"
        
        # Verify visible in both read methods
        retrieve_results = await storage.retrieve(sample_memory.content, n_results=5)
        search_results = await storage.search_by_tag(["embedding"])
        
        assert any(r.memory.content_hash == sample_memory.content_hash for r in retrieve_results), \
            "Memory should be visible in retrieve before pending"
        assert any(r.content_hash == sample_memory.content_hash for r in search_results), \
            "Memory should be visible in search_by_tag before pending"
        
        # Mark as pending
        self.mark_embedding_pending(storage, sample_memory.content_hash, 1)
        
        # Should be invisible in ALL read methods
        retrieve_results = await storage.retrieve(sample_memory.content, n_results=5)
        search_results = await storage.search_by_tag(["embedding"])
        
        assert not any(r.memory.content_hash == sample_memory.content_hash for r in retrieve_results), \
            "Pending memory should be excluded from retrieve"
        assert not any(r.content_hash == sample_memory.content_hash for r in search_results), \
            "Pending memory should be excluded from search_by_tag"

    # CA4b (nuance): pending memory STILL visible in list_content_hashes (sync needs it)
    @pytest.mark.asyncio
    async def test_pending_memory_visible_in_sync_paths(self, storage, sample_memory):
        """
        CA4b: Critical nuance - pending memory should be EXCLUDED from search/retrieve
        but STILL VISIBLE in list_content_hashes (sync reconciliation needs to see everything).
        Expected to FAIL: embedding_pending column doesn't exist.
        """
        # Store memory
        success, message = await storage.store(sample_memory)
        assert success, f"Store failed: {message}"
        
        # Get the memory ID for list_content_hashes check
        cursor = storage.conn.execute("""
            SELECT id FROM memories WHERE content_hash = ?
        """, (sample_memory.content_hash,))
        memory_id = cursor.fetchone()[0]
        
        # Verify visible in sync path before pending
        hashes_page = await storage.list_content_hashes_page(after_id=0, limit=1000)
        hash_pairs = [(id_val, hash_val) for id_val, hash_val in hashes_page if hash_val == sample_memory.content_hash]
        assert len(hash_pairs) == 1, "Memory should be visible in list_content_hashes before pending"
        
        # Mark as pending
        self.mark_embedding_pending(storage, sample_memory.content_hash, 1)
        
        # Should be EXCLUDED from search (already tested above, but verify for clarity)
        search_results = await storage.search_by_tag(["embedding"])
        assert not any(r.content_hash == sample_memory.content_hash for r in search_results), \
            "Pending memory should be excluded from search_by_tag"
        
        # Should STILL be visible in list_content_hashes (sync path)
        hashes_page = await storage.list_content_hashes_page(after_id=0, limit=1000)
        hash_pairs = [(id_val, hash_val) for id_val, hash_val in hashes_page if hash_val == sample_memory.content_hash]
        assert len(hash_pairs) == 1, \
            "Pending memory should STILL be visible in list_content_hashes (sync needs to see it)"

    # CA5 (NF1): migration 017 creates schema and defaults
    @pytest.mark.asyncio
    async def test_migration_creates_embedding_consistency_schema(self, storage):
        """
        CA5: After initialize(), schema should have embedding_pending column on memories
        and embedding_model/embedding_dim columns on sync_events. Existing rows default to 0.
        Expected to FAIL: columns don't exist.
        """
        # Check memories table has embedding_pending column
        cursor = storage.conn.execute("PRAGMA table_info(memories)")
        columns = cursor.fetchall()
        
        embedding_pending_col = None
        for col in columns:
            if col[1] == 'embedding_pending':  # col[1] is column name
                embedding_pending_col = col
                break
        
        assert embedding_pending_col is not None, \
            "memories table should have embedding_pending column"
        assert embedding_pending_col[3] == 1, \
            "embedding_pending should be NOT NULL"  # col[3] is notnull flag
        assert embedding_pending_col[4] == '0', \
            "embedding_pending should default to 0"  # col[4] is default value
        
        # Check sync_events table has embedding metadata columns
        cursor = storage.conn.execute("PRAGMA table_info(sync_events)")
        columns = cursor.fetchall()
        
        has_embedding_model = any(col[1] == 'embedding_model' for col in columns)
        has_embedding_dim = any(col[1] == 'embedding_dim' for col in columns)
        
        assert has_embedding_model, "sync_events table should have embedding_model column"
        assert has_embedding_dim, "sync_events table should have embedding_dim column"
        
        # Check index exists for embedding_pending
        cursor = storage.conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type = 'index' AND tbl_name = 'memories' 
            AND sql LIKE '%embedding_pending%'
        """)
        indices = cursor.fetchall()
        
        assert len(indices) > 0, "Should have index on embedding_pending column"

    @pytest.mark.asyncio
    async def test_existing_memories_default_to_consistent_state(self, storage, sample_memory):
        """
        Additional verification: memories stored after migration should default to
        embedding_pending=0 (consistent state).
        Expected to FAIL: column doesn't exist.
        """
        success, message = await storage.store(sample_memory)
        assert success, f"Store failed: {message}"
        
        # Verify default state is consistent (0)
        cursor = storage.conn.execute("""
            SELECT embedding_pending FROM memories WHERE content_hash = ?
        """, (sample_memory.content_hash,))
        rows = cursor.fetchall()
        
        assert len(rows) == 1, "Should find exactly one memory"
        assert rows[0][0] == 0, "New memory should default to embedding_pending=0"

class TestEmbeddingPendingAllSearchSurfaces:
    """CA4 extended: pending must be excluded from recall() and the BM25/hybrid search
    too — the surfaces the first pass missed (tuvok Gate 5)."""

    @pytest_asyncio.fixture
    async def storage(self):
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_embedding_surfaces.db")
        storage = SqliteVecMemoryStorage(db_path)
        await storage.initialize()
        yield storage
        if storage.conn:
            storage.conn.close()
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_pending_excluded_from_recall(self, storage):
        from mcp_memory_service.models.memory import Memory
        from mcp_memory_service.utils.hashing import generate_content_hash
        c = "recall pending exclusion probe unique xyz"
        h = generate_content_hash(c)
        await storage.store(Memory(content=c, content_hash=h, tags=["recallp"], memory_type="note"))
        # visible before
        res = await storage.recall(query=c, n_results=10)
        assert any(r.memory.content_hash == h for r in res), "should be visible before pending"
        # mark pending
        storage.conn.execute("UPDATE memories SET embedding_pending = 1 WHERE content_hash = ?", (h,))
        storage.conn.commit()
        res2 = await storage.recall(query=c, n_results=10)
        assert not any(r.memory.content_hash == h for r in res2), \
            "recall() must exclude embedding_pending memories (tuvok P1)"

    @pytest.mark.asyncio
    async def test_pending_excluded_from_recall_time_based(self, storage):
        from mcp_memory_service.models.memory import Memory
        from mcp_memory_service.utils.hashing import generate_content_hash
        c = "recall time pending probe unique abc"
        h = generate_content_hash(c)
        await storage.store(Memory(content=c, content_hash=h, tags=["recalltp"], memory_type="note"))
        storage.conn.execute("UPDATE memories SET embedding_pending = 1 WHERE content_hash = ?", (h,))
        storage.conn.commit()
        # time-only recall (no query) must also exclude
        res = await storage.recall(query=None, n_results=50)
        assert not any(r.memory.content_hash == h for r in res), \
            "time-based recall() must exclude pending (tuvok P1)"

    @pytest.mark.asyncio
    async def test_pending_excluded_from_search_memories_time_only(self, storage):
        """search_memories() time-only path (Issue #374 optimized route via
        get_memories_by_time_range) must ALSO exclude pending. This is the public MCP
        search surface the first Phase-3 pass missed — a pending row leaked through it
        (tuvok Gate-5 finding, proven empirically). Regression lock for the fix."""
        from mcp_memory_service.models.memory import Memory
        from mcp_memory_service.utils.hashing import generate_content_hash
        c = "search_memories time-only pending probe unique qrs"
        h = generate_content_hash(c)
        await storage.store(Memory(content=c, content_hash=h, tags=["smto"], memory_type="note"))
        # Time-only public search (query=None + after/before → optimized time-range path).
        # Returns a dict with a 'memories' list.
        before = await storage.search_memories(query=None, after="2000-01-01", before="2100-01-01", limit=100)
        assert any(m["content_hash"] == h for m in before["memories"]), "should be visible before pending"
        # mark pending (ADR-0017: tests drive the flag directly)
        storage.conn.execute("UPDATE memories SET embedding_pending = 1 WHERE content_hash = ?", (h,))
        storage.conn.commit()
        after = await storage.search_memories(query=None, after="2000-01-01", before="2100-01-01", limit=100)
        assert not any(m["content_hash"] == h for m in after["memories"]), \
            "search_memories() time-only must exclude embedding_pending (tuvok P1 leak)"

    @pytest.mark.asyncio
    async def test_get_memories_by_time_range_default_sees_pending(self, storage):
        """The low-level getter defaults to exclude_pending=False so internal callers
        (delete, consolidation) still see every row; only the public search passes True."""
        import time as _time
        from mcp_memory_service.models.memory import Memory
        from mcp_memory_service.utils.hashing import generate_content_hash
        c = "time-range default-sees-pending probe unique tuv"
        h = generate_content_hash(c)
        await storage.store(Memory(content=c, content_hash=h, tags=["trdp"], memory_type="note"))
        storage.conn.execute("UPDATE memories SET embedding_pending = 1 WHERE content_hash = ?", (h,))
        storage.conn.commit()
        now = _time.time()
        default = await storage.get_memories_by_time_range(0.0, now + 60)
        assert any(m.content_hash == h for m in default), \
            "default (exclude_pending=False) must still see pending for internal callers"
        excluded = await storage.get_memories_by_time_range(0.0, now + 60, exclude_pending=True)
        assert not any(m.content_hash == h for m in excluded), \
            "exclude_pending=True must drop pending"

    @pytest.mark.asyncio
    async def test_pending_excluded_from_bm25_search(self, storage):
        """BM25/keyword search must also exclude pending (tuvok P1).

        Greptile P2: asserting only ABSENCE is a false-positive trap — a broken BM25
        (returns [] on any failure) would pass. So we also assert the row IS found
        before marking pending and again after clearing it: a broken search fails those.
        """
        from mcp_memory_service.models.memory import Memory
        from mcp_memory_service.utils.hashing import generate_content_hash
        if not hasattr(storage, "_search_bm25"):
            import pytest as _pt
            _pt.skip("backend has no _search_bm25")
        c = "bm25 keyword pending probe palavra unica"
        h = generate_content_hash(c)
        ok, _msg = await storage.store(Memory(content=c, content_hash=h, tags=["bm25p"], memory_type="note"))
        assert ok, "store must succeed for the test to be meaningful"

        def _hash(x):
            # _search_bm25 returns List[Tuple[content_hash, score]]
            if isinstance(x, (tuple, list)):
                return x[0]
            return getattr(x, "content_hash", None) or getattr(getattr(x, "memory", None), "content_hash", None)

        # 1. BM25 finds it BEFORE pending (proves the query actually works)
        before = await storage._search_bm25("palavra", n_results=10)
        assert any(_hash(x) == h for x in before), "BM25 must find the row before pending (else the search is broken)"

        # 2. mark pending → BM25 must EXCLUDE it
        storage.conn.execute("UPDATE memories SET embedding_pending = 1 WHERE content_hash = ?", (h,))
        storage.conn.commit()
        during = await storage._search_bm25("palavra", n_results=10)
        assert not any(_hash(x) == h for x in during), "BM25 search must exclude pending"

        # 3. clear pending → BM25 finds it again (proves exclusion was the cause, not a broken query)
        storage.conn.execute("UPDATE memories SET embedding_pending = 0 WHERE content_hash = ?", (h,))
        storage.conn.commit()
        after = await storage._search_bm25("palavra", n_results=10)
        assert any(_hash(x) == h for x in after), "BM25 must find the row again after clearing pending"

    @pytest.mark.asyncio
    async def test_pending_still_visible_to_list_content_hashes(self, storage):
        """Sync listing must STILL see pending (reconciliation needs it) — CA4b for this class."""
        from mcp_memory_service.models.memory import Memory
        from mcp_memory_service.utils.hashing import generate_content_hash
        c = "sync still sees pending probe"
        h = generate_content_hash(c)
        await storage.store(Memory(content=c, content_hash=h, tags=["syncp"], memory_type="note"))
        storage.conn.execute("UPDATE memories SET embedding_pending = 1 WHERE content_hash = ?", (h,))
        storage.conn.commit()
        hashes = await storage.list_content_hashes()
        assert h in hashes, "list_content_hashes must still see pending (sync reconciliation)"

    @pytest.mark.asyncio
    async def test_get_all_memories_exclude_pending_param(self, storage):
        """get_all_memories(exclude_pending=True) hides pending; default (False) keeps it
        (sync/listing). Closes the coverage gap for the exclude_pending branch."""
        from mcp_memory_service.models.memory import Memory
        from mcp_memory_service.utils.hashing import generate_content_hash
        c = "get_all exclude_pending probe unique"
        h = generate_content_hash(c)
        await storage.store(Memory(content=c, content_hash=h, tags=["gap"], memory_type="note"))
        storage.conn.execute("UPDATE memories SET embedding_pending = 1 WHERE content_hash = ?", (h,))
        storage.conn.commit()
        default = await storage.get_all_memories(tags=["gap"])
        assert any(m.content_hash == h for m in default), "default must still see pending (sync)"
        search = await storage.get_all_memories(tags=["gap"], exclude_pending=True)
        assert not any(m.content_hash == h for m in search), "exclude_pending=True must hide pending"
