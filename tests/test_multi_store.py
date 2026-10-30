"""Tests for multi-store partition key support (Issue #57)."""
import pytest
import os
import uuid

os.environ.setdefault('MCP_MEMORY_STORAGE_BACKEND', 'sqlite_vec')

from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.models.memory import Memory


@pytest.fixture
async def storage(tmp_path):
    db_path = str(tmp_path / "test_multistore.db")
    s = SqliteVecMemoryStorage(db_path)
    await s.initialize()
    yield s
    await s.close()


class TestMultiStoreSchema:
    """Phase 1: schema correctly has partition key."""

    @pytest.mark.asyncio
    async def test_vec0_has_partition_key(self, storage):
        """memory_embeddings table includes store partition key."""
        cursor = storage.conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='memory_embeddings'"
        )
        row = cursor.fetchone()
        assert row is not None
        assert 'partition' in row[0].lower() or 'store' in row[0].lower()

    @pytest.mark.asyncio
    async def test_memories_has_store_column(self, storage):
        """memories table has store column with default."""
        cursor = storage.conn.execute("PRAGMA table_info(memories)")
        columns = {row[1]: row[4] for row in cursor.fetchall()}
        assert 'store' in columns
        assert columns['store'] == "'default'" or 'default' in str(columns['store'])

    @pytest.mark.asyncio
    async def test_store_memory_with_default_store(self, storage):
        """Storing memory without explicit store uses 'default'."""
        mem = Memory(content="Test content for default store", content_hash="hash_default_1", tags=["test"])
        result = await storage.store(mem)
        assert result[0] is True
        cursor = storage.conn.execute(
            "SELECT store FROM memories WHERE content='Test content for default store'"
        )
        row = cursor.fetchone()
        assert row[0] == 'default'

    @pytest.mark.asyncio
    async def test_store_memory_with_custom_store(self, storage):
        """Storing memory with explicit store='docs'."""
        mem = Memory(content="Doc chunk about APIs", content_hash="hash_docs_1", tags=["docs"])
        result = await storage.store(mem, store="docs")
        assert result[0] is True
        cursor = storage.conn.execute(
            "SELECT store FROM memories WHERE content='Doc chunk about APIs'"
        )
        row = cursor.fetchone()
        assert row[0] == 'docs'

    @pytest.mark.asyncio
    async def test_search_respects_store_isolation(self, storage):
        """Search in store A does not return results from store B."""
        mem_work = Memory(content="Python is great for scripting", content_hash="hash_work_1", tags=["test"])
        mem_docs = Memory(content="Python API documentation reference", content_hash="hash_docs_2", tags=["test"])
        await storage.store(mem_work, store="work")
        await storage.store(mem_docs, store="docs")

        # Search in 'work' store should only find the first
        results = await storage.retrieve("Python scripting", n_results=10, store="work")
        contents = [r.memory.content for r in results]
        assert any("scripting" in c for c in contents)
        assert not any("documentation reference" in c for c in contents)

    @pytest.mark.asyncio
    async def test_search_federation_all_stores(self, storage):
        """Search with store=None returns from all stores."""
        mem_a = Memory(content="Unique alpha content xyz", content_hash="hash_fed_1", tags=["test"])
        mem_b = Memory(content="Unique beta content xyz", content_hash="hash_fed_2", tags=["test"])
        await storage.store(mem_a, store="work")
        await storage.store(mem_b, store="docs")

        # Federation: store=None searches all
        results = await storage.retrieve("unique content xyz", n_results=10, store=None)
        contents = [r.memory.content for r in results]
        assert len(contents) >= 2


class TestMultiStoreMigration:
    """Blocker 2: atomic migration."""

    @pytest.mark.asyncio
    async def test_migration_creates_partition_table(self, storage):
        """Fresh DB gets partition key table."""
        row = storage.conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='memory_embeddings'"
        ).fetchone()
        assert row is not None
        assert 'partition' in row[0].lower()

    @pytest.mark.asyncio
    async def test_existing_memories_survive_migration(self, storage):
        """Memories stored before migration are accessible after."""
        await storage.store(Memory(
            content="Pre-migration content survives",
            content_hash="hash_premigration_1",
            tags=["test"],
        ))
        results = await storage.retrieve("Pre-migration content", n_results=5, store="default")
        assert len(results) > 0


class TestMultiStoreHybridPassthrough:
    """Blocker 1: HybridMemoryStorage forwards store kwarg."""

    @pytest.mark.asyncio
    async def test_hybrid_store_forwards_store(self, tmp_path):
        """HybridMemoryStorage.store() passes store to primary."""
        from mcp_memory_service.storage.hybrid import HybridMemoryStorage
        db_path = str(tmp_path / "test_hybrid.db")
        hybrid = HybridMemoryStorage(sqlite_db_path=db_path)
        await hybrid.initialize()
        try:
            mem = Memory(content="Hybrid store test", content_hash="hash_hybrid_1", tags=["test"])
            success, _ = await hybrid.store(mem, store="docs")
            assert success
            # Verify primary got the store value
            cursor = hybrid.primary.conn.execute(
                "SELECT store FROM memories WHERE content_hash='hash_hybrid_1'"
            )
            row = cursor.fetchone()
            assert row[0] == "docs"
        finally:
            await hybrid.close()

    @pytest.mark.asyncio
    async def test_hybrid_retrieve_forwards_store(self, tmp_path):
        """HybridMemoryStorage.retrieve() passes store to primary."""
        from mcp_memory_service.storage.hybrid import HybridMemoryStorage
        db_path = str(tmp_path / "test_hybrid_ret.db")
        hybrid = HybridMemoryStorage(sqlite_db_path=db_path)
        await hybrid.initialize()
        try:
            mem = Memory(content="Only in docs store for retrieve", content_hash="hash_hybrid_2", tags=["test"])
            await hybrid.store(mem, store="docs")
            # Retrieve from 'default' should NOT find it
            results = await hybrid.retrieve("Only in docs store", n_results=5, store="default")
            assert all("Only in docs store" not in r.memory.content for r in results)
            # Retrieve from 'docs' SHOULD find it
            results = await hybrid.retrieve("Only in docs store", n_results=5, store="docs")
            assert any("Only in docs store" in r.memory.content for r in results)
        finally:
            await hybrid.close()

    @pytest.mark.asyncio
    async def test_hybrid_count_forwards_store(self, tmp_path):
        """HybridMemoryStorage.count_all_memories() passes store."""
        from mcp_memory_service.storage.hybrid import HybridMemoryStorage
        db_path = str(tmp_path / "test_hybrid_count.db")
        hybrid = HybridMemoryStorage(sqlite_db_path=db_path)
        await hybrid.initialize()
        try:
            mem = Memory(content="Count store test", content_hash="hash_hybrid_3", tags=["test"])
            await hybrid.store(mem, store="isolated")
            count_default = await hybrid.count_all_memories(store="default")
            count_isolated = await hybrid.count_all_memories(store="isolated")
            assert count_isolated >= 1
            assert count_default < count_isolated or count_default == 0
        finally:
            await hybrid.close()


class TestMultiStoreToolWiring:
    """Phase 2: store param flows through tool handlers."""

    @pytest.mark.asyncio
    async def test_store_memory_accepts_store_param(self):
        """memory_store tool accepts store parameter."""
        from mcp_memory_service.server import MemoryServer
        server = MemoryServer()
        result = await server.call_tool("memory_store", {
            "content": "API docs for OAuth flow",
            "metadata": {"tags": "docs,oauth"},
            "store": "docs"
        })
        text = result[0].text
        assert "error" not in text.lower() or "success" in text.lower()

    @pytest.mark.asyncio
    async def test_search_with_store_param(self):
        """memory_search tool respects store parameter."""
        from mcp_memory_service.server import MemoryServer
        server = MemoryServer()
        # The default store is shared with every earlier test in this process
        # (and semantic search returns its nearest matches whatever they are),
        # so assert that *this* memory is absent, not that the results are
        # empty. The marker leads the content because results show only a
        # short preview of each memory, and stays out of the query because
        # the response echoes the query.
        marker = f"msiso-{uuid.uuid4().hex[:12]}"
        query = "unique multi-store test content alpha"
        await server.call_tool("memory_store", {
            "content": f"{marker} {query}",
            "metadata": {"tags": "test"},
            "store": "isolated"
        })
        # Search in different store should not find it
        result = await server.call_tool("memory_search", {
            "query": query,
            "store": "default"
        })
        assert marker not in result[0].text
        # Control: the same search in its own store does find it, so the
        # absence above is not just a search that matched nothing.
        result = await server.call_tool("memory_search", {
            "query": query,
            "store": "isolated"
        })
        assert marker in result[0].text


async def _build_pre_62_db(db_path):
    """Create a database in the pre-#62 state: a vec0 `memory_embeddings`
    table WITHOUT the store partition key (plus its shadow tables and one row),
    so re-opening triggers the multi-store migration on an existing DB.

    Returns the original embedding rowids for later verification.
    """
    s = SqliteVecMemoryStorage(db_path)
    await s.initialize()
    await s.store(Memory(content="legacy chunk before #62", content_hash="legacy_pre62_1", tags=["t"]))
    rows = s.conn.execute("SELECT rowid, content_embedding FROM memory_embeddings").fetchall()
    dim = s.embedding_dimension
    # Downgrade the table to the pre-#62 shape (no partition key). vec0 DROP
    # removes the partitioned shadow tables; the fresh CREATE makes old-style ones.
    s.conn.execute("DROP TABLE memory_embeddings")
    s.conn.execute(
        f"CREATE VIRTUAL TABLE memory_embeddings USING vec0("
        f"content_embedding FLOAT[{dim}] distance_metric=cosine)"
    )
    for rid, emb in rows:
        s.conn.execute(
            "INSERT INTO memory_embeddings (rowid, content_embedding) VALUES (?, ?)",
            (rid, emb),
        )
    s.conn.commit()
    await s.close()
    return rows


class TestMultiStoreMigrationExistingDB:
    """Issue #68: multi-store vec0 migration must not abort on an existing DB."""

    @pytest.mark.asyncio
    async def test_migration_on_existing_pre62_db_succeeds(self, tmp_path):
        """Re-opening a pre-#62 database runs the migration without aborting,
        preserves embeddings, and leaves writes working."""
        db_path = str(tmp_path / "pre62.db")
        original = await _build_pre_62_db(db_path)
        assert len(original) == 1

        # Re-open: this is where the migration runs. Pre-fix it raised
        # RuntimeError("Multi-store vec0 migration failed (aborting): ...
        # memory_embeddings_info already exists").
        s2 = SqliteVecMemoryStorage(db_path)
        await s2.initialize()
        try:
            sql = s2.conn.execute(
                "SELECT sql FROM sqlite_master WHERE name='memory_embeddings'"
            ).fetchone()[0]
            assert 'partition' in sql.lower(), "partition key not added by migration"

            count = s2.conn.execute("SELECT COUNT(*) FROM memory_embeddings").fetchone()[0]
            assert count == 1, "existing embedding lost during migration"

            res = await s2.store(
                Memory(content="written after migration", content_hash="post_migration_1", tags=["t"])
            )
            assert res[0] is True, "writes still broken after migration"
        finally:
            await s2.close()

    @pytest.mark.asyncio
    async def test_migration_is_idempotent(self, tmp_path):
        """Running initialize() twice on a migrated DB is a no-op, not a re-abort."""
        db_path = str(tmp_path / "pre62_idem.db")
        await _build_pre_62_db(db_path)
        s2 = SqliteVecMemoryStorage(db_path)
        await s2.initialize()
        await s2.close()
        # Second open must also succeed cleanly.
        s3 = SqliteVecMemoryStorage(db_path)
        await s3.initialize()
        try:
            sql = s3.conn.execute(
                "SELECT sql FROM sqlite_master WHERE name='memory_embeddings'"
            ).fetchone()[0]
            assert 'partition' in sql.lower()
        finally:
            await s3.close()
