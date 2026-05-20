"""Tests for entity-centric memory grouping (auto-link by shared entities)."""

import os
import pytest
import tempfile

from mcp_memory_service.reasoning.entity_linker import EntityLinker, is_entity_linking_enabled
from mcp_memory_service.storage.graph import GraphStorage


@pytest.fixture
def graph_db():
    """Create a temporary GraphStorage instance."""
    tmp = tempfile.mkdtemp()
    db_path = os.path.join(tmp, "test_graph.db")
    gs = GraphStorage(db_path)
    # Ensure the memory_graph table exists
    conn = gs._create_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_graph (
            source_hash TEXT NOT NULL,
            target_hash TEXT NOT NULL,
            similarity REAL DEFAULT 0.0,
            connection_types TEXT DEFAULT '[]',
            metadata TEXT DEFAULT '{}',
            created_at REAL,
            relationship_type TEXT DEFAULT 'related',
            PRIMARY KEY (source_hash, target_hash, relationship_type)
        )
    """)
    conn.commit()
    conn.close()
    gs._connection = None  # Force reconnect via _get_connection
    yield gs


class TestEntityLinker:
    @pytest.mark.asyncio
    async def test_link_two_memories_shared_entity(self, graph_db):
        """Two memories sharing an entity get a shares_entity edge."""
        # Memory A has entity 'backend'
        await graph_db.store_entity_link("hash_a", "backend", "tag")
        # Memory B is stored with entity 'backend'
        await graph_db.store_entity_link("hash_b", "backend", "tag")

        linker = EntityLinker()
        count = await linker.link_by_entities("hash_b", ["backend"], graph_db)

        assert count == 1
        # Verify edge exists
        assoc = await graph_db.get_association("hash_a", "hash_b")
        assert assoc is not None

    @pytest.mark.asyncio
    async def test_no_duplicate_edges(self, graph_db):
        """Calling link_by_entities twice doesn't create duplicate edges."""
        await graph_db.store_entity_link("hash_a", "api", "tag")
        await graph_db.store_entity_link("hash_b", "api", "tag")

        linker = EntityLinker()
        count1 = await linker.link_by_entities("hash_b", ["api"], graph_db)
        count2 = await linker.link_by_entities("hash_b", ["api"], graph_db)

        # Second call uses INSERT OR REPLACE, so still returns 1 (upsert)
        # but no actual duplicate rows
        assert count1 == 1
        assert count2 == 1

        # Verify only one logical edge (bidirectional stored as 2 rows for symmetric)
        conn = await graph_db._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) as cnt FROM memory_graph WHERE relationship_type='shares_entity'"
        )
        row = cursor.fetchone()
        # shares_entity is symmetric → 2 rows (A→B and B→A)
        assert row["cnt"] == 2
        cursor.close()

    @pytest.mark.asyncio
    async def test_no_self_link(self, graph_db):
        """A memory should not link to itself."""
        await graph_db.store_entity_link("hash_a", "python", "tag")

        linker = EntityLinker()
        count = await linker.link_by_entities("hash_a", ["python"], graph_db)

        assert count == 0

    @pytest.mark.asyncio
    async def test_multiple_entities_multiple_links(self, graph_db):
        """Multiple shared entities create edges to different memories."""
        await graph_db.store_entity_link("hash_a", "backend", "tag")
        await graph_db.store_entity_link("hash_b", "frontend", "tag")
        await graph_db.store_entity_link("hash_c", "backend", "tag")
        await graph_db.store_entity_link("hash_c", "frontend", "tag")

        linker = EntityLinker()
        # hash_c shares 'backend' with hash_a and 'frontend' with hash_b
        count = await linker.link_by_entities("hash_c", ["backend", "frontend"], graph_db)

        assert count == 2

    @pytest.mark.asyncio
    async def test_entity_search_filter(self, graph_db):
        """find_memories_by_entity returns all memories linked to an entity."""
        await graph_db.store_entity_link("hash_a", "docker", "tag")
        await graph_db.store_entity_link("hash_b", "docker", "tag")
        await graph_db.store_entity_link("hash_c", "kubernetes", "tag")

        results = await graph_db.find_memories_by_entity("docker")
        assert set(results) == {"hash_a", "hash_b"}

    @pytest.mark.asyncio
    async def test_empty_entities_no_op(self, graph_db):
        """Empty entity list returns 0 edges."""
        linker = EntityLinker()
        count = await linker.link_by_entities("hash_x", [], graph_db)
        assert count == 0


class TestEntityLinkingOptIn:
    def test_disabled_by_default(self, monkeypatch):
        """Entity linking is disabled when env var is not set."""
        monkeypatch.delenv("MCP_ENTITY_LINKING_ENABLED", raising=False)
        assert is_entity_linking_enabled() is False

    def test_enabled_with_true(self, monkeypatch):
        monkeypatch.setenv("MCP_ENTITY_LINKING_ENABLED", "true")
        assert is_entity_linking_enabled() is True

    def test_enabled_with_1(self, monkeypatch):
        monkeypatch.setenv("MCP_ENTITY_LINKING_ENABLED", "1")
        assert is_entity_linking_enabled() is True

    def test_disabled_with_false(self, monkeypatch):
        monkeypatch.setenv("MCP_ENTITY_LINKING_ENABLED", "false")
        assert is_entity_linking_enabled() is False
