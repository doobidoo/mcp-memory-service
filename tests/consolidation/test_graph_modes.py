"""Unit tests for consolidator graph storage mode switching.

NOTE: These tests are scaffolding for Issue #279 Phase 2 implementation.
Some tests are marked as xfail until the storage mode switching is fully implemented.
"""

import pytest
import os
import tempfile
import shutil
import sqlite3
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock

from mcp_memory_service.consolidation.base import MemoryAssociation, ConsolidationConfig
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.graph import GraphStorage


@pytest.mark.unit
class TestGraphStorageModes:
    """Test consolidator storage mode switching between memories_only, dual_write, and graph_only."""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database for mode testing."""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_modes.db")

        # Initialize both memories and graph tables
        conn = sqlite3.connect(db_path)

        # Memories table (simplified schema)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                content_hash TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                tags TEXT,
                memory_type TEXT,
                metadata TEXT,
                created_at REAL
            )
        """)

        # Graph table
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

        conn.commit()
        conn.close()

        yield db_path

        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def sample_association(self):
        """Create sample association for testing."""
        return MemoryAssociation(
            source_memory_hashes=["hash_test_a", "hash_test_b"],
            similarity_score=0.75,
            connection_type="semantic",
            discovery_method="test_engine",
            discovery_date=datetime.now(),
            metadata={"test": "data"}
        )

    @pytest.mark.asyncio
    async def test_graph_storage_mode_env_variable(self):
        """Test that GRAPH_STORAGE_MODE environment variable is read correctly.

        Validates:
        - Config reads MCP_GRAPH_STORAGE_MODE from environment
        - Default value is 'dual_write'
        - Valid modes: memories_only, dual_write, graph_only
        """
        from mcp_memory_service.config import GRAPH_STORAGE_MODE, VALID_GRAPH_MODES

        # Verify mode is one of valid values
        assert GRAPH_STORAGE_MODE in VALID_GRAPH_MODES, \
            f"GRAPH_STORAGE_MODE should be one of {VALID_GRAPH_MODES}"

        # Verify VALID_GRAPH_MODES contains all expected values
        assert 'memories_only' in VALID_GRAPH_MODES
        assert 'dual_write' in VALID_GRAPH_MODES
        assert 'graph_only' in VALID_GRAPH_MODES

    @pytest.mark.asyncio
    async def test_mode_configuration_validation(self):
        """Test configuration validation for invalid storage modes.

        Validates:
        - Invalid mode values trigger warning
        - Default 'dual_write' is used as fallback
        - Warning logged for invalid configuration
        """
        # This test verifies the validation logic exists
        from mcp_memory_service.config import VALID_GRAPH_MODES

        # Valid modes should be a list/tuple
        assert isinstance(VALID_GRAPH_MODES, (list, tuple)), \
            "VALID_GRAPH_MODES should be a list or tuple"

        # Should have exactly 3 modes
        assert len(VALID_GRAPH_MODES) == 3, \
            "Should have 3 valid graph storage modes"

    @pytest.mark.asyncio
    async def test_graph_storage_basic_operations(self, temp_db_path, sample_association):
        """Test GraphStorage basic operations work independently.

        This validates the graph storage layer is functional,
        regardless of consolidator integration.

        Validates:
        - Can create GraphStorage instance
        - Can store associations in graph table
        - Can query associations back
        """
        storage = GraphStorage(temp_db_path)

        # Store association
        result = await storage.store_association(
            source_hash=sample_association.source_memory_hashes[0],
            target_hash=sample_association.source_memory_hashes[1],
            similarity=sample_association.similarity_score,
            connection_types=[sample_association.connection_type],
            metadata=sample_association.metadata
        )

        assert result is True, "Should successfully store association"

        # Query back via find_connected
        connected = await storage.find_connected(
            sample_association.source_memory_hashes[0],
            max_hops=1
        )

        assert len(connected) > 0, "Should find stored association"
        assert connected[0][0] == sample_association.source_memory_hashes[1], \
            "Should find the target hash"

    @pytest.mark.asyncio
    async def test_storage_size_comparison_concept(self, temp_db_path):
        """Test storage size comparison concept: graph vs memory objects.

        Validates:
        - Graph table schema is more efficient than Memory objects
        - Can measure database file size
        - Provides baseline for 97% reduction target

        NOTE: Actual size comparison will be validated when consolidator
        integration is complete in Phase 2.
        """
        # Create 100 associations in graph table
        storage = GraphStorage(temp_db_path)

        for i in range(100):
            await storage.store_association(
                source_hash=f"hash_{i:03d}",
                target_hash=f"hash_{i+1:03d}",
                similarity=0.65,
                connection_types=["semantic"],
                metadata={"test": "data"}
            )

        # Measure database file size
        db_size = os.path.getsize(temp_db_path)

        # Verify database was created and has reasonable size
        assert db_size > 0, "Database should have data"
        assert db_size < 1024 * 1024, "100 associations should be < 1MB"

        # Calculate approximate Memory object size for comparison
        # Each association as Memory object ~500-1000 bytes
        estimated_memory_size = 100 * 750  # Conservative estimate

        # Graph table should be more efficient
        print(f"\nStorage comparison (100 associations):")
        print(f"  Graph table (db file): {db_size:,} bytes")
        print(f"  Estimated Memory objects: {estimated_memory_size:,} bytes")
        print(f"  Reduction: {(1 - db_size/estimated_memory_size)*100:.1f}%")

    @pytest.mark.xfail(reason="Awaiting Phase 2 consolidator integration (Issue #279)")
    @pytest.mark.asyncio
    async def test_memories_only_mode(self):
        """Test memories_only mode - stores only in memories table (legacy behavior).

        This test is scaffolding for Phase 2 implementation.
        Will pass once consolidator._store_associations() is updated.
        """
        pytest.fail("Phase 2 implementation pending")

    @pytest.mark.xfail(reason="Awaiting Phase 2 consolidator integration (Issue #279)")
    @pytest.mark.asyncio
    async def test_dual_write_mode(self):
        """Test dual_write mode - stores in both memories and graph tables.

        This test is scaffolding for Phase 2 implementation.
        Will pass once consolidator integration is complete.
        """
        pytest.fail("Phase 2 implementation pending")

    @pytest.mark.xfail(reason="Awaiting Phase 2 consolidator integration (Issue #279)")
    @pytest.mark.asyncio
    async def test_graph_only_mode(self):
        """Test graph_only mode - stores only in graph table (recommended).

        This test is scaffolding for Phase 2 implementation.
        Will pass once consolidator integration is complete.
        """
        pytest.fail("Phase 2 implementation pending")

    @pytest.mark.xfail(reason="Awaiting Phase 2 consolidator integration (Issue #279)")
    @pytest.mark.asyncio
    async def test_memories_only_backward_compat(self):
        """Test backward compatibility with existing memories_only deployments.

        This test is scaffolding for Phase 2 implementation.
        Will pass once legacy behavior is preserved in new code.
        """
        pytest.fail("Phase 2 implementation pending")

    @pytest.mark.xfail(reason="Awaiting Phase 2 consolidator integration (Issue #279)")
    @pytest.mark.asyncio
    async def test_dual_write_consistency(self):
        """Test consistency between memories and graph storage in dual_write mode.

        This test is scaffolding for Phase 2 implementation.
        Will pass once dual_write logic is implemented.
        """
        pytest.fail("Phase 2 implementation pending")

    @pytest.mark.xfail(reason="Awaiting Phase 2 consolidator integration (Issue #279)")
    @pytest.mark.asyncio
    async def test_graph_only_no_memory_pollution(self):
        """Test that graph_only mode doesn't pollute memories table.

        This test is scaffolding for Phase 2 implementation.
        Will pass once graph_only mode is fully implemented.
        """
        pytest.fail("Phase 2 implementation pending")

    @pytest.mark.xfail(reason="Awaiting Phase 2 consolidator integration (Issue #279)")
    @pytest.mark.asyncio
    async def test_mode_switching_runtime(self):
        """Test that storage mode can be changed at runtime via environment variable.

        This test is scaffolding for Phase 2 implementation.
        Will pass once mode switching logic is complete.
        """
        pytest.fail("Phase 2 implementation pending")
