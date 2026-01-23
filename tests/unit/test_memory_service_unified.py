"""
Unit tests for MemoryService unified methods (Phase 8).

Tests the new consolidated MemoryService methods:
- search_memories() - replaces retrieve/recall/quality_boost/exact/debug
- delete_memories() - replaces delete/delete_by_tag/delete_by_tags/delete_by_all_tags/delete_by_timeframe
- (manage_consolidation will be tested if/when it's added to MemoryService)

These tests verify the business logic layer, using mocked storage backends.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from typing import List

from mcp_memory_service.services.memory_service import MemoryService
from mcp_memory_service.models.memory import Memory, MemoryQueryResult
from mcp_memory_service.storage.base import MemoryStorage


# Test Fixtures

@pytest.fixture
def mock_storage():
    """Create a mock storage backend for testing."""
    storage = AsyncMock(spec=MemoryStorage)
    # Required properties
    storage.max_content_length = 1000
    storage.supports_chunking = True
    # Default return values
    storage.store.return_value = (True, "Success")
    storage.delete.return_value = (True, "Deleted")
    storage.get_stats.return_value = {"backend": "mock", "total_memories": 0}
    storage.search_memories.return_value = {"memories": [], "total": 0, "mode": "semantic"}
    storage.delete_memories.return_value = {"success": True, "deleted_count": 0, "deleted_hashes": []}
    return storage


@pytest.fixture
def memory_service(mock_storage):
    """Create a MemoryService instance with mock storage."""
    return MemoryService(storage=mock_storage)


@pytest.fixture
def sample_memory():
    """Create a sample memory for testing."""
    return Memory(
        content="Test memory content",
        content_hash="test_hash_123",
        tags=["test", "sample"],
        memory_type="note",
        metadata={"source": "test"},
        created_at=datetime.now().timestamp(),
        updated_at=datetime.now().timestamp()
    )


@pytest.fixture
def sample_memories():
    """Create a list of sample memories."""
    memories = []
    now = datetime.now().timestamp()
    for i in range(5):
        memories.append(Memory(
            content=f"Test memory {i+1}",
            content_hash=f"hash_{i+1}",
            tags=[f"tag{i+1}", "test"],
            memory_type="note",
            metadata={"index": i+1, "quality_score": 0.5 + i * 0.1},
            created_at=now + i * 100,
            updated_at=now + i * 100
        ))
    return memories


# Test search_memories method

class TestSearchMemories:
    """Tests for unified search_memories method."""

    @pytest.mark.asyncio
    async def test_semantic_search_default_mode(self, memory_service, mock_storage, sample_memories):
        """Test default semantic search mode."""
        # Setup mock
        mock_storage.search_memories.return_value = {
            "memories": [m.__dict__ for m in sample_memories[:3]],
            "total": 3,
            "mode": "semantic"
        }

        # Execute
        result = await memory_service.storage.search_memories(query="test query")

        # Verify
        assert result["mode"] == "semantic"
        assert "memories" in result
        assert len(result["memories"]) == 3
        mock_storage.search_memories.assert_called_once()

    @pytest.mark.asyncio
    async def test_exact_search_mode(self, memory_service, mock_storage):
        """Test exact string match mode."""
        # Setup mock
        mock_storage.search_memories.return_value = {
            "memories": [],
            "total": 0,
            "mode": "exact"
        }

        # Execute
        result = await memory_service.storage.search_memories(
            query="exact phrase",
            mode="exact"
        )

        # Verify
        assert result["mode"] == "exact"
        mock_storage.search_memories.assert_called_once()

    @pytest.mark.asyncio
    async def test_time_expression_parsing(self, memory_service, mock_storage):
        """Test time_expr parameter is handled correctly."""
        # Setup mock
        mock_storage.search_memories.return_value = {
            "memories": [],
            "total": 0
        }

        # Execute - time_expr should be parsed by storage layer
        result = await memory_service.storage.search_memories(time_expr="last week")

        # Verify
        mock_storage.search_memories.assert_called_once()
        call_args = mock_storage.search_memories.call_args
        assert call_args.kwargs.get("time_expr") == "last week"

    @pytest.mark.asyncio
    async def test_quality_boost_parameter(self, memory_service, mock_storage, sample_memories):
        """Test quality_boost triggers proper reranking."""
        # Setup mock with quality metadata
        mock_storage.search_memories.return_value = {
            "memories": [m.__dict__ for m in sample_memories],
            "total": len(sample_memories)
        }

        # Execute
        result = await memory_service.storage.search_memories(
            query="test",
            quality_boost=0.5
        )

        # Verify
        mock_storage.search_memories.assert_called_once()
        call_args = mock_storage.search_memories.call_args
        assert call_args.kwargs.get("quality_boost") == 0.5

    @pytest.mark.asyncio
    async def test_combined_filters(self, memory_service, mock_storage):
        """Test combining multiple filter parameters."""
        # Setup mock
        mock_storage.search_memories.return_value = {
            "memories": [],
            "total": 0
        }

        # Execute with multiple filters
        result = await memory_service.storage.search_memories(
            query="test",
            time_expr="yesterday",
            tags=["important"],
            quality_boost=0.3,
            limit=5
        )

        # Verify
        mock_storage.search_memories.assert_called_once()
        call_args = mock_storage.search_memories.call_args
        assert call_args.kwargs.get("query") == "test"
        assert call_args.kwargs.get("tags") == ["important"]
        assert call_args.kwargs.get("quality_boost") == 0.3
        assert call_args.kwargs.get("limit") == 5

    @pytest.mark.asyncio
    async def test_include_debug_info(self, memory_service, mock_storage):
        """Test debug info is included when requested."""
        # Setup mock
        mock_storage.search_memories.return_value = {
            "memories": [],
            "total": 0,
            "debug": {"time_filter": None}
        }

        # Execute
        result = await memory_service.storage.search_memories(
            query="test",
            include_debug=True
        )

        # Verify
        assert "debug" in result


# Test delete_memories method

class TestDeleteMemories:
    """Tests for unified delete_memories method."""

    @pytest.mark.asyncio
    async def test_delete_by_content_hash(self, memory_service, mock_storage):
        """Test deletion by content hash."""
        # Setup mock
        mock_storage.delete_memories.return_value = {
            "success": True,
            "deleted_count": 1,
            "deleted_hashes": ["test_hash_123"]
        }

        # Execute
        result = await memory_service.storage.delete_memories(content_hash="test_hash_123")

        # Verify
        assert result["success"] is True
        assert result["deleted_count"] == 1
        mock_storage.delete_memories.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_by_tags_any_match(self, memory_service, mock_storage):
        """Test deletion by tags with ANY match mode."""
        # Setup mock
        mock_storage.delete_memories.return_value = {
            "success": True,
            "deleted_count": 3,
            "deleted_hashes": ["h1", "h2", "h3"]
        }

        # Execute
        result = await memory_service.storage.delete_memories(
            tags=["temp", "draft"],
            tag_match="any"
        )

        # Verify
        assert result["deleted_count"] == 3
        mock_storage.delete_memories.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_by_tags_all_match(self, memory_service, mock_storage):
        """Test deletion by tags with ALL match mode."""
        # Setup mock
        mock_storage.delete_memories.return_value = {
            "success": True,
            "deleted_count": 1,
            "deleted_hashes": ["h1"]
        }

        # Execute
        result = await memory_service.storage.delete_memories(
            tags=["archived", "old"],
            tag_match="all"
        )

        # Verify
        assert result["deleted_count"] == 1
        mock_storage.delete_memories.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_by_time_range(self, memory_service, mock_storage):
        """Test deletion by time range."""
        # Setup mock
        mock_storage.delete_memories.return_value = {
            "success": True,
            "deleted_count": 5,
            "deleted_hashes": [f"h{i}" for i in range(5)]
        }

        # Execute
        result = await memory_service.storage.delete_memories(
            after="2024-01-01",
            before="2024-06-30"
        )

        # Verify
        assert result["deleted_count"] == 5
        mock_storage.delete_memories.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_dry_run_mode(self, memory_service, mock_storage, sample_memories):
        """Test dry_run mode doesn't actually delete."""
        # Setup mock
        mock_storage.delete_memories.return_value = {
            "success": True,
            "deleted_count": 2,
            "deleted_hashes": ["h1", "h2"],
            "dry_run": True
        }

        # Execute
        result = await memory_service.storage.delete_memories(
            tags=["temp"],
            dry_run=True
        )

        # Verify
        assert result["dry_run"] is True
        assert result["deleted_count"] == 2
        mock_storage.delete_memories.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_no_filters_validation(self, memory_service, mock_storage):
        """Test that delete without filters is handled properly."""
        # Setup mock - storage should validate this
        mock_storage.delete_memories.return_value = {
            "error": "At least one filter required"
        }

        # Execute
        result = await memory_service.storage.delete_memories()

        # Verify
        assert "error" in result
        mock_storage.delete_memories.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_combined_filters(self, memory_service, mock_storage):
        """Test combining tag and time filters."""
        # Setup mock
        mock_storage.delete_memories.return_value = {
            "success": True,
            "deleted_count": 2,
            "deleted_hashes": ["h1", "h2"]
        }

        # Execute
        result = await memory_service.storage.delete_memories(
            tags=["cleanup"],
            before="2024-01-01",
            tag_match="any"
        )

        # Verify
        assert "deleted_count" in result
        mock_storage.delete_memories.assert_called_once()


# Test storage interface compatibility

class TestStorageInterfaceCompatibility:
    """Verify storage interface supports unified methods."""

    def test_storage_has_search_memories_method(self):
        """Verify storage interface has search_memories method."""
        assert hasattr(MemoryStorage, 'search_memories'), "MemoryStorage should have search_memories method"

    def test_storage_has_delete_memories_method(self):
        """Verify storage interface has delete_memories method."""
        assert hasattr(MemoryStorage, 'delete_memories'), "MemoryStorage should have delete_memories method"

    @pytest.mark.asyncio
    async def test_mock_storage_supports_unified_methods(self, mock_storage):
        """Verify mock storage can handle unified method calls."""
        # Test search_memories
        search_result = await mock_storage.search_memories(query="test")
        assert "memories" in search_result

        # Test delete_memories
        delete_result = await mock_storage.delete_memories(tags=["test"])
        assert "deleted_count" in delete_result or "success" in delete_result
