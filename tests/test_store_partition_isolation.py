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
import shutil
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
    shutil.rmtree(d, ignore_errors=True)


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

        # Stronger (P2): deleting a hash scoped to the WRONG store must NOT delete it.
        # The old unscoped implementation would delete regardless of store — this proves scoping.
        await storage.store(Memory(content="wrong store guard content", content_hash="wrongstore_hash", tags=["g"]), store="docs")
        res_wrong = await storage.delete_memories(content_hash="wrongstore_hash", store="default")
        still = storage.conn.execute("SELECT COUNT(*) FROM memories WHERE content_hash='wrongstore_hash' AND deleted_at IS NULL").fetchone()[0]
        assert still == 1, "Deleting a docs hash scoped to 'default' must NOT delete it"
        assert res_wrong["deleted_count"] == 0, "Wrong-store delete must report 0 deleted"

    @pytest.mark.asyncio
    async def test_delete_by_tag_respects_store(self, storage):
        """Delete by tag should only affect memories in specified store (#1465).

        Conteúdos deliberadamente distintos (não-similares) para que a dedup
        semântica — que não é store-aware — não rejeite o segundo store.
        """
        mem1 = Memory(content="Postgres replication lag runbook for the default partition", content_hash="tag1_default", tags=["shared_tag_1465"])
        mem2 = Memory(content="Kubernetes ingress TLS renewal notes kept in the docs partition", content_hash="tag2_docs", tags=["shared_tag_1465"])

        # Store in different stores
        ok1, _ = await storage.store(mem1, store="default")
        ok2, _ = await storage.store(mem2, store="docs")
        assert ok1 and ok2, "both memories must be stored (distinct content avoids dedup)"

        # Delete by tag from docs store only
        result = await storage.delete_memories(tags=["shared_tag_1465"], store="docs")
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
        # P2: must actually FIND the docs memory — an empty result must NOT pass this test.
        assert "qb_docs" in returned_hashes, "quality_boost search must return the docs-store memory (not empty)"
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

    @pytest.mark.asyncio
    async def test_search_tag_only_scoped_to_store(self, storage):
        """Tag-only search (no query) must be scoped to the store (#1466, P1 #4).

        The tag-only branch loads via get_all_memories, which previously defaulted
        to the 'default' store regardless of the requested partition.
        """
        await storage.store(Memory(content="alpha note in default partition", content_hash="to_default", tags=["scoped_tag"]), store="default")
        await storage.store(Memory(content="beta note in docs partition", content_hash="to_docs", tags=["scoped_tag"]), store="docs")

        results = await storage.search_memories(tags=["scoped_tag"], store="docs")

        returned_hashes = [m["content_hash"] for m in results["memories"]]
        assert "to_docs" in returned_hashes, "tag-only search must return the docs memory"
        assert "to_default" not in returned_hashes, "tag-only search must not leak default-store memory"