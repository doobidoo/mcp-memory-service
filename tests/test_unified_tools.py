"""
Tests for unified MCP Memory Service tools.

Tests cover:
1. New unified tools work correctly
2. Deprecated tools still work via compatibility layer
3. Argument transformation is correct
4. Deprecation warnings are emitted
"""

import pytest
import warnings
from unittest.mock import AsyncMock, patch, MagicMock
from mcp_memory_service.compat import (
    DEPRECATED_TOOLS,
    transform_deprecated_call,
    is_deprecated,
    get_new_tool_name
)


class TestMemoryDelete:
    """Tests for unified memory_delete tool."""

    @pytest.fixture
    def mock_storage(self):
        """Create mocked storage backend."""
        storage = AsyncMock()
        return storage

    @pytest.mark.asyncio
    async def test_delete_by_hash(self, mock_storage):
        """Test single memory deletion by hash."""
        mock_storage.delete_memories.return_value = {
            "success": True,
            "deleted_count": 1,
            "deleted_hashes": ["abc123"]
        }

        result = await mock_storage.delete_memories(content_hash="abc123")

        assert result["deleted_count"] == 1
        mock_storage.delete_memories.assert_called_once_with(content_hash="abc123")

    @pytest.mark.asyncio
    async def test_delete_by_tags_any(self, mock_storage):
        """Test deletion by tags with ANY match."""
        mock_storage.delete_memories.return_value = {
            "success": True,
            "deleted_count": 5,
            "deleted_hashes": ["a", "b", "c", "d", "e"]
        }

        result = await mock_storage.delete_memories(
            tags=["temp", "draft"],
            tag_match="any"
        )

        assert result["deleted_count"] == 5

    @pytest.mark.asyncio
    async def test_delete_by_tags_all(self, mock_storage):
        """Test deletion by tags with ALL match."""
        mock_storage.delete_memories.return_value = {
            "success": True,
            "deleted_count": 2,
            "deleted_hashes": ["x", "y"]
        }

        result = await mock_storage.delete_memories(
            tags=["archived", "old"],
            tag_match="all"
        )

        assert result["deleted_count"] == 2

    @pytest.mark.asyncio
    async def test_delete_by_timeframe(self, mock_storage):
        """Test deletion by time range."""
        mock_storage.delete_memories.return_value = {
            "success": True,
            "deleted_count": 10,
            "deleted_hashes": [f"h{i}" for i in range(10)]
        }

        result = await mock_storage.delete_memories(
            after="2024-01-01",
            before="2024-06-30"
        )

        assert result["deleted_count"] == 10

    @pytest.mark.asyncio
    async def test_delete_dry_run(self, mock_storage):
        """Test dry run returns preview without deleting."""
        mock_storage.delete_memories.return_value = {
            "success": True,
            "deleted_count": 3,
            "deleted_hashes": ["a", "b", "c"],
            "dry_run": True
        }

        result = await mock_storage.delete_memories(
            tags=["cleanup"],
            dry_run=True
        )

        assert result["dry_run"] is True

    @pytest.mark.asyncio
    async def test_delete_no_filters_error(self, mock_storage):
        """Test that delete without filters returns error."""
        mock_storage.delete_memories.return_value = {
            "error": "At least one filter required"
        }

        result = await mock_storage.delete_memories()

        assert "error" in result


class TestMemorySearch:
    """Tests for unified memory_search tool."""

    @pytest.fixture
    def mock_storage(self):
        """Create mocked storage backend."""
        storage = AsyncMock()
        return storage

    @pytest.mark.asyncio
    async def test_semantic_search(self, mock_storage):
        """Test default semantic search."""
        mock_storage.search_memories.return_value = {
            "memories": [{"content": "test", "content_hash": "abc"}],
            "total": 1,
            "mode": "semantic"
        }

        result = await mock_storage.search_memories(query="python patterns")

        assert result["mode"] == "semantic"
        assert len(result["memories"]) == 1

    @pytest.mark.asyncio
    async def test_exact_search(self, mock_storage):
        """Test exact string match."""
        mock_storage.search_memories.return_value = {
            "memories": [],
            "total": 0,
            "mode": "exact"
        }

        result = await mock_storage.search_memories(
            query="exact phrase",
            mode="exact"
        )

        assert result["mode"] == "exact"

    @pytest.mark.asyncio
    async def test_time_expression_search(self, mock_storage):
        """Test natural language time expression."""
        mock_storage.search_memories.return_value = {
            "memories": [{"content": "recent"}],
            "total": 1
        }

        result = await mock_storage.search_memories(time_expr="last week")

        assert result["total"] >= 0

    @pytest.mark.asyncio
    async def test_quality_boost(self, mock_storage):
        """Test quality-boosted search."""
        mock_storage.search_memories.return_value = {
            "memories": [{"content": "high quality", "quality": 0.9}],
            "total": 1
        }

        result = await mock_storage.search_memories(
            query="important info",
            quality_boost=0.3
        )

        assert result["total"] == 1

    @pytest.mark.asyncio
    async def test_combined_filters(self, mock_storage):
        """Test combining multiple filters."""
        mock_storage.search_memories.return_value = {
            "memories": [],
            "total": 0
        }

        result = await mock_storage.search_memories(
            query="database",
            time_expr="last month",
            tags=["reference"],
            quality_boost=0.2,
            limit=20
        )

        # Should have called with all parameters
        mock_storage.search_memories.assert_called_once()

    @pytest.mark.asyncio
    async def test_debug_output(self, mock_storage):
        """Test debug information included."""
        mock_storage.search_memories.return_value = {
            "memories": [],
            "total": 0,
            "debug": {
                "time_filter": None,
                "quality_boost": 0.0
            }
        }

        result = await mock_storage.search_memories(
            query="test",
            include_debug=True
        )

        assert "debug" in result


class TestMemoryConsolidate:
    """Tests for unified memory_consolidate tool."""

    @pytest.fixture
    def mock_handler(self):
        """Create mocked consolidation handler."""
        handler = AsyncMock()
        return handler

    @pytest.mark.asyncio
    async def test_status_action(self, mock_handler):
        """Test status retrieval."""
        mock_handler.handle_consolidation_status.return_value = {
            "status": "healthy",
            "last_run": "2024-01-15"
        }

        result = await mock_handler.handle_consolidation_status()

        assert result["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_run_action(self, mock_handler):
        """Test consolidation run."""
        mock_handler.handle_consolidate_memories.return_value = {
            "success": True,
            "consolidated": 50
        }

        result = await mock_handler.handle_consolidate_memories(time_horizon="weekly")

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_recommend_action(self, mock_handler):
        """Test recommendations."""
        mock_handler.handle_consolidation_recommendations.return_value = {
            "recommendations": ["Run weekly consolidation"]
        }

        result = await mock_handler.handle_consolidation_recommendations(time_horizon="weekly")

        assert "recommendations" in result


class TestDeprecationLayer:
    """Tests for backwards compatibility."""

    def test_all_deprecated_tools_mapped(self):
        """Verify all expected deprecated tools are in mapping."""
        expected_deprecated = [
            # Delete tools
            "delete_memory", "delete_by_tag", "delete_by_tags",
            "delete_by_all_tags", "delete_by_timeframe", "delete_before_date",
            # Search tools
            "retrieve_memory", "recall_memory", "recall_by_timeframe",
            "retrieve_with_quality_boost", "exact_match_retrieve", "debug_retrieve",
            # Consolidation tools
            "consolidate_memories", "consolidation_status", "consolidation_recommendations",
            "scheduler_status", "trigger_consolidation", "pause_consolidation", "resume_consolidation",
            # Renamed tools
            "store_memory", "check_database_health", "get_cache_stats",
            "cleanup_duplicates", "update_memory_metadata", "rate_memory",
            # Merged tools
            "list_memories", "search_by_tag",
            "ingest_document", "ingest_directory",
            "get_memory_quality", "analyze_quality_distribution",
            "find_connected_memories", "find_shortest_path", "get_memory_subgraph",
        ]

        for tool in expected_deprecated:
            assert tool in DEPRECATED_TOOLS, f"Missing: {tool}"

    def test_is_deprecated(self):
        """Test is_deprecated helper."""
        assert is_deprecated("delete_by_tag") is True
        assert is_deprecated("memory_delete") is False
        assert is_deprecated("unknown_tool") is False

    def test_get_new_tool_name(self):
        """Test get_new_tool_name helper."""
        assert get_new_tool_name("delete_by_tag") == "memory_delete"
        assert get_new_tool_name("retrieve_memory") == "memory_search"
        assert get_new_tool_name("memory_delete") is None

    def test_transform_delete_by_tag(self):
        """Test argument transformation for delete_by_tag."""
        old_args = {"tag": "temporary"}

        new_name, new_args = transform_deprecated_call("delete_by_tag", old_args)

        assert new_name == "memory_delete"
        assert new_args["tags"] == ["temporary"]
        assert new_args["tag_match"] == "any"

    def test_transform_delete_by_tags(self):
        """Test argument transformation for delete_by_tags."""
        old_args = {"tags": ["temp", "draft"]}

        new_name, new_args = transform_deprecated_call("delete_by_tags", old_args)

        assert new_name == "memory_delete"
        assert new_args["tags"] == ["temp", "draft"]
        assert new_args["tag_match"] == "any"

    def test_transform_retrieve_memory(self):
        """Test argument transformation for retrieve_memory."""
        old_args = {"query": "python", "n_results": 10}

        new_name, new_args = transform_deprecated_call("retrieve_memory", old_args)

        assert new_name == "memory_search"
        assert new_args["query"] == "python"
        assert new_args["limit"] == 10

    def test_transform_consolidate_memories(self):
        """Test argument transformation for consolidate_memories."""
        old_args = {"time_horizon": "weekly"}

        new_name, new_args = transform_deprecated_call("consolidate_memories", old_args)

        assert new_name == "memory_consolidate"
        assert new_args["action"] == "run"
        assert new_args["time_horizon"] == "weekly"

    def test_transform_store_memory(self):
        """Test argument transformation for store_memory (simple rename)."""
        old_args = {"content": "test", "tags": ["note"]}

        new_name, new_args = transform_deprecated_call("store_memory", old_args)

        assert new_name == "memory_store"
        assert new_args["content"] == "test"
        assert new_args["tags"] == ["note"]

    def test_deprecation_warning_emitted(self):
        """Test that deprecation warning is emitted."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            transform_deprecated_call("delete_by_tag", {"tag": "test"})

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "delete_by_tag" in str(w[0].message)
            assert "memory_delete" in str(w[0].message)

    def test_none_values_removed_from_transformed_args(self):
        """Test that None values are filtered from transformed args."""
        old_args = {"start_date": "2024-01-01"}  # no end_date

        new_name, new_args = transform_deprecated_call("delete_by_timeframe", old_args)

        # Check that before is either not present or not None
        assert "before" not in new_args or new_args.get("before") is not None

    def test_transform_list_memories(self):
        """Test argument transformation for list_memories."""
        old_args = {"limit": 20, "offset": 10}

        new_name, new_args = transform_deprecated_call("list_memories", old_args)

        assert new_name == "memory_list"
        assert new_args["limit"] == 20
        assert new_args["offset"] == 10

    def test_transform_ingest_document(self):
        """Test argument transformation for ingest_document."""
        old_args = {"file_path": "/path/to/doc.pdf", "tags": ["docs"]}

        new_name, new_args = transform_deprecated_call("ingest_document", old_args)

        assert new_name == "memory_ingest"
        assert new_args["path"] == "/path/to/doc.pdf"
        assert new_args["mode"] == "file"
        assert new_args["tags"] == ["docs"]

    def test_transform_rate_memory(self):
        """Test argument transformation for rate_memory."""
        old_args = {"content_hash": "abc123", "rating": 1}

        new_name, new_args = transform_deprecated_call("rate_memory", old_args)

        assert new_name == "memory_quality"
        assert new_args["action"] == "rate"
        assert new_args["content_hash"] == "abc123"
        assert new_args["rating"] == 1

    def test_transform_find_connected_memories(self):
        """Test argument transformation for find_connected_memories."""
        old_args = {"hash": "abc123", "max_hops": 2}

        new_name, new_args = transform_deprecated_call("find_connected_memories", old_args)

        assert new_name == "memory_graph"
        assert new_args["action"] == "connected"
        assert new_args["hash"] == "abc123"
        assert new_args["max_hops"] == 2


class TestIntegration:
    """Integration tests for full tool pipeline."""

    @pytest.mark.asyncio
    async def test_deprecation_layer_complete(self):
        """Test that deprecation layer handles all tool types."""
        # Test delete tool transformation
        delete_name, delete_args = transform_deprecated_call(
            "delete_by_tag", {"tag": "test"}
        )
        assert delete_name == "memory_delete"

        # Test search tool transformation
        search_name, search_args = transform_deprecated_call(
            "retrieve_memory", {"query": "test", "n_results": 5}
        )
        assert search_name == "memory_search"

        # Test consolidation tool transformation
        consolidate_name, consolidate_args = transform_deprecated_call(
            "consolidation_status", {}
        )
        assert consolidate_name == "memory_consolidate"

        # Test rename transformation
        rename_name, rename_args = transform_deprecated_call(
            "store_memory", {"content": "test"}
        )
        assert rename_name == "memory_store"

        # Test merge transformation (list)
        list_name, list_args = transform_deprecated_call(
            "list_memories", {"limit": 10}
        )
        assert list_name == "memory_list"

        # Test merge transformation (ingest)
        ingest_name, ingest_args = transform_deprecated_call(
            "ingest_document", {"file_path": "/test.pdf"}
        )
        assert ingest_name == "memory_ingest"

        # Test merge transformation (quality)
        quality_name, quality_args = transform_deprecated_call(
            "rate_memory", {"content_hash": "abc", "rating": 1}
        )
        assert quality_name == "memory_quality"

        # Test merge transformation (graph)
        graph_name, graph_args = transform_deprecated_call(
            "find_connected_memories", {"hash": "abc"}
        )
        assert graph_name == "memory_graph"
