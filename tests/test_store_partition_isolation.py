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
        """Delete by content_hash should only affect memories in specified store (#1465).

        NOTE: usa content_hash DISTINTO por store (cenário real — o hash deriva do
        conteúdo). O caso de MESMO content_hash em 2 stores esbarra no constraint
        UNIQUE(content_hash) global — dívida mais profunda, FORA do escopo da issue
        #1465 (ver docs/_fork: achado do UNIQUE global).
        """
        mem_default = Memory(content="conteudo em default", content_hash="hash_default_1465", tags=["default"])
        mem_docs = Memory(content="conteudo em docs", content_hash="hash_docs_1465", tags=["docs"])

        # Store in different stores
        await storage.store(mem_default, store="default")
        await storage.store(mem_docs, store="docs")

        # Verify both memories exist in their respective stores
        cursor = storage.conn.execute("SELECT COUNT(*) FROM memories WHERE store = 'default' AND deleted_at IS NULL")
        default_count = cursor.fetchone()[0]
        cursor = storage.conn.execute("SELECT COUNT(*) FROM memories WHERE store = 'docs' AND deleted_at IS NULL")
        docs_count = cursor.fetchone()[0]
        assert default_count == 1
        assert docs_count == 1

        # Delete from docs store only
        result = await storage.delete_memories(content_hash="hash_docs_1465", store="docs")
        assert result["success"] is True
        assert result["deleted_count"] == 1

        # Memory in default store should still exist (RED before #1465 fix)
        cursor = storage.conn.execute("SELECT COUNT(*) FROM memories WHERE store = 'default' AND deleted_at IS NULL")
        default_remaining = cursor.fetchone()[0]
        cursor = storage.conn.execute("SELECT COUNT(*) FROM memories WHERE store = 'docs' AND deleted_at IS NULL")
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
        
        # Memory in default store should still exist (RED before #1465 fix)
        cursor = storage.conn.execute("SELECT COUNT(*) FROM memories WHERE store = 'default' AND deleted_at IS NULL")
        default_remaining = cursor.fetchone()[0]
        cursor = storage.conn.execute("SELECT COUNT(*) FROM memories WHERE store = 'docs' AND deleted_at IS NULL")
        docs_remaining = cursor.fetchone()[0]
        
        assert default_remaining == 1, "Memory in default store should not be deleted"
        assert docs_remaining == 0, "Memory in docs store should be deleted"


class TestSearchMemoriesStoreIsolation:
    """Tests for bug #1466: search_memories leaking across stores."""

    @pytest.mark.asyncio
    async def test_search_quality_boost_scoped_to_store(self, storage):
        """Search with quality_boost should only return memories from specified store (#1466).

        Conteúdo DISTINTO por store (evita a dedup semântica, que não é store-aware —
        achado separado), mas ambos contêm 'databases' para que um vazamento apareça.
        """
        await storage.store(Memory(content="databases tuning in the default store", content_hash="qb_default", tags=["important"]), store="default")
        await storage.store(Memory(content="databases indexing in the docs store", content_hash="qb_docs", tags=["important"]), store="docs")

        results = await storage.search_memories(query="databases", store="docs", mode="semantic", quality_boost=0.3)

        returned_hashes = [m["content_hash"] for m in results["memories"]]
        for hash_val in returned_hashes:
            row = storage.conn.execute("SELECT store FROM memories WHERE content_hash = ?", (hash_val,)).fetchone()
            assert row[0] == "docs", f"Memory {hash_val} should be from docs store, but is from {row[0]}"
        assert "qb_default" not in returned_hashes, "quality_boost must not leak default-store memory"

    @pytest.mark.asyncio
    async def test_search_exact_mode_scoped_to_store(self, storage):
        """Search in exact mode should only return memories from specified store (#1466)."""
        await storage.store(Memory(content="exact phrase alpha default", content_hash="ex_default", tags=["test"]), store="default")
        await storage.store(Memory(content="exact phrase alpha docs", content_hash="ex_docs", tags=["test"]), store="docs")

        results = await storage.search_memories(query="exact phrase alpha", store="docs", mode="exact")

        returned_hashes = [m["content_hash"] for m in results["memories"]]
        assert returned_hashes, "exact search should find the docs memory"
        for hash_val in returned_hashes:
            row = storage.conn.execute("SELECT store FROM memories WHERE content_hash = ?", (hash_val,)).fetchone()
            assert row[0] == "docs", f"Memory {hash_val} should be from docs store, but is from {row[0]}"
        assert "ex_default" not in returned_hashes, "exact mode must not leak default-store memory"

    @pytest.mark.asyncio
    async def test_search_time_only_scoped_to_store(self, storage):
        """Time-range search should only return memories from specified store (#1466).

        Usa get_memories_by_time_range direto (start/end explícitos) — não depende do
        parser de 'last week', que não resolve no ambiente de teste (hash embeddings).
        """
        import time as _time
        await storage.store(Memory(content="time content default", content_hash="tm_default", tags=["time"]), store="default")
        await storage.store(Memory(content="time content docs", content_hash="tm_docs", tags=["time"]), store="docs")

        start = 0.0
        end = _time.time() + 3600
        memories = await storage.get_memories_by_time_range(start, end, exclude_pending=True, store="docs")

        hashes = [m.content_hash for m in memories]
        assert "tm_docs" in hashes, "time-range should find the docs memory"
        assert "tm_default" not in hashes, "time-range must not leak default-store memory"