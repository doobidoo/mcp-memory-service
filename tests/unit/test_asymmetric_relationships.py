"""Tests for asymmetric relationship semantic correctness."""
import sys
from pathlib import Path
import pytest
import tempfile
import shutil
import os
import sqlite3
import importlib.util

# Load GraphStorage module directly without importing the package to avoid numpy issues
repo_root = Path(__file__).parent.parent.parent
graph_path = repo_root / "src" / "mcp_memory_service" / "storage" / "graph.py"
ontology_path = repo_root / "src" / "mcp_memory_service" / "models" / "ontology.py"

# Load ontology module first (dependency of graph)
spec = importlib.util.spec_from_file_location("ontology", ontology_path)
ontology_module = importlib.util.module_from_spec(spec)
sys.modules['mcp_memory_service.models.ontology'] = ontology_module
spec.loader.exec_module(ontology_module)

# Load graph module
spec = importlib.util.spec_from_file_location("graph", graph_path)
graph_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(graph_module)

GraphStorage = graph_module.GraphStorage


@pytest.fixture
def temp_graph_db():
    """Create a temporary database with graph table for testing."""
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_graph.db")

    # Initialize database with graph schema
    conn = sqlite3.connect(db_path)
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

    # Create indexes for performance
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_graph_source
        ON memory_graph(source_hash)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_graph_target
        ON memory_graph(target_hash)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_graph_relationship
        ON memory_graph(relationship_type)
    """)

    conn.commit()
    conn.close()

    yield db_path

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
async def graph_storage(temp_graph_db):
    """Create GraphStorage instance with initialized database."""
    storage = GraphStorage(temp_graph_db)
    # Ensure connection is initialized
    await storage._get_connection()
    yield storage


class TestAsymmetricSemantics:
    """Verify semantic correctness of directional relationships."""

    @pytest.mark.asyncio
    async def test_causes_directionality(self, graph_storage):
        """A causes B should NOT imply B causes A."""
        await graph_storage.store_association(
            "decision_a", "error_b", 0.9, ["causal"],
            relationship_type="causes"
        )

        # Forward direction exists
        causes = await graph_storage.find_connected(
            "decision_a", relationship_type="causes", direction="outgoing"
        )
        assert "error_b" in [h for h, _ in causes], "decision_a should cause error_b"

        # Reverse direction does NOT exist
        reverse = await graph_storage.find_connected(
            "error_b", relationship_type="causes", direction="outgoing"
        )
        assert "decision_a" not in [h for h, _ in reverse], "error_b should NOT cause decision_a"

    @pytest.mark.asyncio
    async def test_fixes_directionality(self, graph_storage):
        """A fixes B should NOT imply B fixes A."""
        await graph_storage.store_association(
            "decision_fix", "error_bug", 0.95, ["remediation"],
            relationship_type="fixes"
        )

        # Forward direction exists
        fixes = await graph_storage.find_connected(
            "decision_fix", relationship_type="fixes", direction="outgoing"
        )
        assert "error_bug" in [h for h, _ in fixes], "decision_fix should fix error_bug"

        # Reverse does NOT exist
        reverse = await graph_storage.find_connected(
            "error_bug", relationship_type="fixes", direction="outgoing"
        )
        assert "decision_fix" not in [h for h, _ in reverse], "error_bug should NOT fix decision_fix"

    @pytest.mark.asyncio
    async def test_supports_directionality(self, graph_storage):
        """A supports B should NOT imply B supports A."""
        await graph_storage.store_association(
            "learning_a", "decision_b", 0.85, ["reinforcement"],
            relationship_type="supports"
        )

        # Forward direction exists
        supports = await graph_storage.find_connected(
            "learning_a", relationship_type="supports", direction="outgoing"
        )
        assert "decision_b" in [h for h, _ in supports], "learning_a should support decision_b"

        # Reverse does NOT exist
        reverse = await graph_storage.find_connected(
            "decision_b", relationship_type="supports", direction="outgoing"
        )
        assert "learning_a" not in [h for h, _ in reverse], "decision_b should NOT support learning_a"

    @pytest.mark.asyncio
    async def test_follows_directionality(self, graph_storage):
        """A follows B should NOT imply B follows A."""
        await graph_storage.store_association(
            "observation_a", "observation_b", 0.80, ["temporal"],
            relationship_type="follows"
        )

        # Forward direction exists
        follows = await graph_storage.find_connected(
            "observation_a", relationship_type="follows", direction="outgoing"
        )
        assert "observation_b" in [h for h, _ in follows], "observation_a should follow observation_b"

        # Reverse does NOT exist
        reverse = await graph_storage.find_connected(
            "observation_b", relationship_type="follows", direction="outgoing"
        )
        assert "observation_a" not in [h for h, _ in reverse], "observation_b should NOT follow observation_a"

    @pytest.mark.asyncio
    async def test_contradicts_bidirectional(self, graph_storage):
        """A contradicts B SHOULD imply B contradicts A (symmetric)."""
        await graph_storage.store_association(
            "learning_a", "learning_b", 0.85, ["semantic"],
            relationship_type="contradicts"
        )

        # Both directions should work
        from_a = await graph_storage.find_connected(
            "learning_a", relationship_type="contradicts"
        )
        from_b = await graph_storage.find_connected(
            "learning_b", relationship_type="contradicts"
        )

        assert "learning_b" in [h for h, _ in from_a], "learning_a should contradict learning_b"
        assert "learning_a" in [h for h, _ in from_b], "learning_b should contradict learning_a"

    @pytest.mark.asyncio
    async def test_related_bidirectional(self, graph_storage):
        """A related B SHOULD imply B related A (symmetric)."""
        await graph_storage.store_association(
            "observation_a", "observation_b", 0.75, ["semantic"],
            relationship_type="related"
        )

        # Both directions should work
        from_a = await graph_storage.find_connected(
            "observation_a", relationship_type="related"
        )
        from_b = await graph_storage.find_connected(
            "observation_b", relationship_type="related"
        )

        assert "observation_b" in [h for h, _ in from_a], "observation_a should be related to observation_b"
        assert "observation_a" in [h for h, _ in from_b], "observation_b should be related to observation_a"

    @pytest.mark.asyncio
    async def test_incoming_query_for_asymmetric(self, graph_storage):
        """Incoming queries should work for asymmetric relationships."""
        await graph_storage.store_association(
            "decision_a", "error_b", 0.90, ["causal"],
            relationship_type="causes"
        )

        # Can query incoming to find the cause
        incoming = await graph_storage.find_connected(
            "error_b", relationship_type="causes", direction="incoming"
        )
        assert "decision_a" in [h for h, _ in incoming], "Should find decision_a as incoming cause"

        # Reverse incoming query should be empty
        reverse_incoming = await graph_storage.find_connected(
            "decision_a", relationship_type="causes", direction="incoming"
        )
        assert "error_b" not in [h for h, _ in reverse_incoming], "error_b should not be an incoming cause"

    @pytest.mark.asyncio
    async def test_direction_both_with_asymmetric(self, graph_storage):
        """Direction='both' should work correctly with asymmetric relationships."""
        await graph_storage.store_association(
            "decision_a", "error_b", 0.90, ["causal"],
            relationship_type="causes"
        )

        # Query from decision_a with direction="both"
        from_a = await graph_storage.find_connected(
            "decision_a", relationship_type="causes", direction="both"
        )
        assert "error_b" in [h for h, _ in from_a], "Should find error_b from decision_a"

        # Query from error_b with direction="both"
        from_b = await graph_storage.find_connected(
            "error_b", relationship_type="causes", direction="both"
        )
        assert "decision_a" in [h for h, _ in from_b], "Should find decision_a from error_b"
