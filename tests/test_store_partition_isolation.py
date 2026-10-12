"""
RED tests for store partition isolation bugs #1465 and #1466.

These tests expose bugs where delete_memories and search_memories operations
cross store boundaries when they should be isolated to the specified store.

BUG #1465: delete_memories accepts store parameter but does not propagate it to
           actual delete operations, causing deletions to cross stores.

BUG #1466: search_memories only honors store parameter in semantic and ranked modes;
           other modes (hybrid/quality_boost/exact/tag-only/time-only) leak
           memories from other stores.
"""

import os
import tempfile
import pytest

try:
    import sqlite_vec  # noqa: F401
    SQLITE_VEC_AVAILABLE = True
except ImportError:
    SQLITE_VEC_AVAILABLE = False

if SQLITE_VEC_AVAILABLE:
    from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
    from mcp_memory_service.models.memory import Memory

pytestmark = pytest.mark.skipif(not SQLITE_VEC_AVAILABLE, reason="sqlite-vec not available")

# Enable hash embeddings for testing
os.environ['MCP_MEMORY_ALLOW_HASH_EMBEDDINGS'] = '1'


@pytest.fixture
async def storage():
    """Create temporary SqliteVecMemoryStorage for testing."""
    d = tempfile.mkdtemp()
    s = SqliteVecMemoryStorage(os.path.join(d, "test_store_isolation.db"))
    await s.initialize()
    yield s
    await s.close()


class TestDeleteMemoriesStoreIsolation:
    """Tests for bug #1465: delete_memories crossing store boundaries."""

    @pytest.mark.asyncio
    async def test_delete_by_hash_respects_store(self, storage):
        """Delete by content_hash should only affect memories in specified store."""
        # Create memory with same content in two different stores
        content = "Shared content across stores"
        hash_val = "shared_hash_123"
        
        mem_default = Memory(content=content, content_hash=hash_val, tags=["default"])
        mem_docs = Memory(content=content, content_hash=hash_val, tags=["docs"])
        
        # Store in different stores
        await storage.store(mem_default, store="default")
        await storage.store(mem_docs, store="docs")
        
        # Verify both memories exist in their respective stores
        cursor = storage.conn.execute("SELECT COUNT(*) FROM memories WHERE store = 'default'")
        default_count = cursor.fetchone()[0]
        cursor = storage.conn.execute("SELECT COUNT(*) FROM memories WHERE store = 'docs'")
        docs_count = cursor.fetchone()[0]
        assert default_count == 1
        assert docs_count == 1
        
        # Delete from docs store only
        result = await storage.delete_memories(content_hash=hash_val, store="docs")
        assert result["success"] is True
        assert result["deleted_count"] == 1
        
        # Memory in default store should still exist (this will FAIL due to bug #1465)
        cursor = storage.conn.execute("SELECT COUNT(*) FROM memories WHERE store = 'default'")
        default_remaining = cursor.fetchone()[0]
        cursor = storage.conn.execute("SELECT COUNT(*) FROM memories WHERE store = 'docs'")
        docs_remaining = cursor.fetchone()[0]
        
        assert default_remaining == 1, "Memory in default store should not be deleted"
        assert docs_remaining == 0, "Memory in docs store should be deleted"

    @pytest.mark.asyncio
    async def test_delete_by_tag_respects_store(self, storage):
        """Delete by tag should only affect memories in specified store."""
        # Create memories with same tag in different stores
        mem1 = Memory(content="Content 1", content_hash="hash1", tags=["shared_tag"])
        mem2 = Memory(content="Content 2", content_hash="hash2", tags=["shared_tag"])
        
        # Store in different stores
        await storage.store(mem1, store="default")
        await storage.store(mem2, store="docs")
        
        # Delete by tag from docs store only
        result = await storage.delete_memories(tags=["shared_tag"], store="docs")
        assert result["success"] is True
        assert result["deleted_count"] == 1
        
        # Memory in default store should still exist (this will FAIL due to bug #1465)
        cursor = storage.conn.execute("SELECT COUNT(*) FROM memories WHERE store = 'default'")
        default_remaining = cursor.fetchone()[0]
        cursor = storage.conn.execute("SELECT COUNT(*) FROM memories WHERE store = 'docs'")
        docs_remaining = cursor.fetchone()[0]
        
        assert default_remaining == 1, "Memory in default store should not be deleted"
        assert docs_remaining == 0, "Memory in docs store should be deleted"


class TestSearchMemoriesStoreIsolation:
    """Tests for bug #1466: search_memories leaking across stores."""

    @pytest.mark.asyncio
    async def test_search_quality_boost_scoped_to_store(self, storage):
        """Search with quality_boost should only return memories from specified store."""
        # Create memories in different stores
        mem_default = Memory(content="Important content about databases", content_hash="hash_default", tags=["important"])
        mem_docs = Memory(content="Important content about databases", content_hash="hash_docs", tags=["important"])
        
        await storage.store(mem_default, store="default")
        await storage.store(mem_docs, store="docs")
        
        # Search with quality_boost in docs store only
        results = await storage.search_memories(
            query="databases",
            store="docs",
            mode="semantic",
            quality_boost=0.3
        )
        
        # Should only return memory from docs store (this will FAIL due to bug #1466)
        assert results["total"] >= 1, "Should find at least one memory"
        
        # Check that all returned memories are from docs store by checking database
        returned_hashes = [m.memory.content_hash for m in results["memories"]]
        for hash_val in returned_hashes:
            cursor = storage.conn.execute("SELECT store FROM memories WHERE content_hash = ?", (hash_val,))
            row = cursor.fetchone()
            assert row[0] == "docs", f"Memory {hash_val} should be from docs store, but is from {row[0]}"

    @pytest.mark.asyncio
    async def test_search_exact_mode_scoped_to_store(self, storage):
        """Search in exact mode should only return memories from specified store."""
        # Create memories in different stores
        mem_default = Memory(content="Exact match content", content_hash="hash_default", tags=["test"])
        mem_docs = Memory(content="Exact match content", content_hash="hash_docs", tags=["test"])
        
        await storage.store(mem_default, store="default")
        await storage.store(mem_docs, store="docs")
        
        # Search in exact mode in docs store only
        results = await storage.search_memories(
            query="Exact match",
            store="docs",
            mode="exact"
        )
        
        # Should only return memory from docs store (this will FAIL due to bug #1466)
        assert results["total"] >= 1, "Should find at least one memory"
        
        # Check that all returned memories are from docs store by checking database
        returned_hashes = [m.memory.content_hash for m in results["memories"]]
        for hash_val in returned_hashes:
            cursor = storage.conn.execute("SELECT store FROM memories WHERE content_hash = ?", (hash_val,))
            row = cursor.fetchone()
            assert row[0] == "docs", f"Memory {hash_val} should be from docs store, but is from {row[0]}"

    @pytest.mark.asyncio
    async def test_search_time_only_scoped_to_store(self, storage):
        """Search by time range should only return memories from specified store."""
        # Create memories in different stores at same time
        mem_default = Memory(content="Time-based content", content_hash="hash_default", tags=["time"])
        mem_docs = Memory(content="Time-based content", content_hash="hash_docs", tags=["time"])
        
        await storage.store(mem_default, store="default")
        await storage.store(mem_docs, store="docs")
        
        # Search by time range in docs store only (no query, time-only path)
        results = await storage.search_memories(
            time_expr="last week",
            store="docs"
        )
        
        # Should only return memory from docs store (this will FAIL due to bug #1466)
        assert results["total"] >= 1, "Should find at least one memory"
        
        # Check that all returned memories are from docs store by checking database
        returned_hashes = [m.memory.content_hash for m in results["memories"]]
        docs_count = 0
        default_count = 0
        
        for hash_val in returned_hashes:
            cursor = storage.conn.execute("SELECT store FROM memories WHERE content_hash = ?", (hash_val,))
            row = cursor.fetchone()
            if row[0] == "docs":
                docs_count += 1
            elif row[0] == "default":
                default_count += 1
        
        assert docs_count >= 1, "Should find memory from docs store"
        assert default_count == 0, "Should not find memory from default store"