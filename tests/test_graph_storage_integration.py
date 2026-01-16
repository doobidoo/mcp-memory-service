"""
Integration tests for GraphStorage with typed relationships

Tests the complete typed relationship workflow end-to-end.
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
import importlib.util

# Load GraphStorage module directly to avoid heavy imports
graph_path = Path(__file__).parent.parent / "src" / "mcp_memory_service" / "storage" / "graph.py"
spec = importlib.util.spec_from_file_location("graph", graph_path)
graph_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(graph_module)

GraphStorage = graph_module.GraphStorage


@pytest.fixture
async def graph_storage():
    """Create temporary GraphStorage for testing"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
        db_path = tmp.name

    storage = GraphStorage(db_path)

    # Initialize schema
    conn = await storage._get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_graph (
            source_hash TEXT NOT NULL,
            target_hash TEXT NOT NULL,
            similarity REAL NOT NULL,
            connection_types TEXT NOT NULL,
            metadata TEXT,
            created_at REAL NOT NULL,
            relationship_type TEXT DEFAULT 'related',
            PRIMARY KEY (source_hash, target_hash)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_graph_relationship ON memory_graph(relationship_type)")
    conn.commit()

    yield storage

    # Cleanup
    if storage._connection:
        storage._connection.close()
    os.unlink(db_path)


class TestBurst33StoreAssociationWithType:
    """Tests for Burst 3.3: store_association with relationship_type"""

    @pytest.mark.asyncio
    async def test_store_with_relationship_type(self, graph_storage):
        """Should store association with specified relationship type"""
        result = await graph_storage.store_association(
            source_hash="hash1",
            target_hash="hash2",
            similarity=0.85,
            connection_types=["semantic"],
            relationship_type="causes"
        )
        assert result is True

        # Verify stored value
        conn = await graph_storage._get_connection()
        cursor = conn.execute("SELECT relationship_type FROM memory_graph WHERE source_hash = 'hash1'")
        row = cursor.fetchone()
        assert row['relationship_type'] == "causes"

    @pytest.mark.asyncio
    async def test_store_defaults_to_related(self, graph_storage):
        """Should default to 'related' when relationship_type not specified"""
        result = await graph_storage.store_association(
            source_hash="hash1",
            target_hash="hash2",
            similarity=0.85,
            connection_types=["semantic"]
        )
        assert result is True

        conn = await graph_storage._get_connection()
        cursor = conn.execute("SELECT relationship_type FROM memory_graph WHERE source_hash = 'hash1'")
        row = cursor.fetchone()
        assert row['relationship_type'] == "related"


class TestBurst36GetRelationshipTypes:
    """Tests for Burst 3.6: get_relationship_types method"""

    @pytest.mark.asyncio
    async def test_get_relationship_types_returns_counts(self, graph_storage):
        """Should return count of each relationship type"""
        # Store multiple associations with different types
        await graph_storage.store_association("hash1", "hash2", 0.8, ["semantic"], relationship_type="causes")
        await graph_storage.store_association("hash1", "hash3", 0.7, ["semantic"], relationship_type="causes")
        await graph_storage.store_association("hash1", "hash4", 0.9, ["semantic"], relationship_type="fixes")
        await graph_storage.store_association("hash1", "hash5", 0.6, ["semantic"], relationship_type="related")

        types = await graph_storage.get_relationship_types("hash1")

        assert types["causes"] == 2
        assert types["fixes"] == 1
        assert types["related"] == 1

    @pytest.mark.asyncio
    async def test_empty_dict_for_no_associations(self, graph_storage):
        """Should return empty dict for memory with no associations"""
        types = await graph_storage.get_relationship_types("nonexistent")
        assert types == {}


class TestBackwardCompatibility:
    """Tests to ensure backward compatibility is maintained"""

    @pytest.mark.asyncio
    async def test_existing_code_works_without_changes(self, graph_storage):
        """Existing code should work without modification"""
        # Call store_association without relationship_type parameter
        result = await graph_storage.store_association(
            "hash1", "hash2", 0.85, ["semantic"]
        )
        assert result is True

        # Call find_connected without new parameters
        connected = await graph_storage.find_connected("hash1", max_hops=2)
        assert len(connected) >= 0  # Should not error
