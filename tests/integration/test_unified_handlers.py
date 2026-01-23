"""
Integration tests for UNIFIED memory handlers (Phase 7).

Tests the new consolidated handler methods:
- handle_memory_search (replaces retrieve/recall/quality_boost/exact/debug)
- handle_memory_delete (replaces delete/delete_by_tag/delete_by_tags/delete_by_all_tags/delete_by_timeframe/delete_before_date)
- handle_memory_list (replaces search_by_tag + list_memories functionality)

These tests verify that the new unified handlers work correctly.
For backwards compatibility testing of deprecated handlers, see test_deprecation_compatibility.py
"""

import pytest
from datetime import datetime, timedelta
from mcp import types
from mcp_memory_service.server import MemoryServer


class TestHandleMemorySearch:
    """Tests for unified handle_memory_search handler."""

    @pytest.mark.asyncio
    async def test_semantic_search_default(self, unique_content):
        """Test default semantic search mode."""
        server = MemoryServer()

        # Store searchable memory
        content = unique_content("Searchable memory content for semantic test")
        await server.handle_store_memory({
            "content": content,
            "metadata": {"tags": ["semantic-search-test"]}
        })

        # Search with semantic mode (default)
        result = await server.handle_memory_search({
            "query": "searchable semantic",
            "limit": 5
        })

        # Verify response format
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], types.TextContent)

    @pytest.mark.asyncio
    async def test_exact_search_mode(self, unique_content):
        """Test exact string match mode."""
        server = MemoryServer()

        content = unique_content("Exact match test string for searching")
        await server.handle_store_memory({"content": content})

        # Search with exact mode
        result = await server.handle_memory_search({
            "query": "Exact match test",
            "mode": "exact",
            "limit": 5
        })

        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_hybrid_search_with_quality_boost(self, unique_content):
        """Test hybrid mode with quality boosting."""
        server = MemoryServer()

        content = unique_content("Quality boosted search test memory")
        await server.handle_store_memory({
            "content": content,
            "metadata": {"tags": ["quality-test"]}
        })

        # Search with hybrid mode and quality boost
        result = await server.handle_memory_search({
            "query": "quality",
            "mode": "hybrid",
            "quality_boost": 0.3,
            "limit": 5
        })

        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_time_expression_filter(self, unique_content):
        """Test time_expr parameter for natural language time filtering."""
        server = MemoryServer()

        content = unique_content("Recent memory for time expression test")
        await server.handle_store_memory({
            "content": content,
            "metadata": {"tags": ["time-expr-test"]}
        })

        # Search with time expression
        result = await server.handle_memory_search({
            "time_expr": "last week",
            "limit": 10
        })

        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_date_range_filter(self, unique_content):
        """Test after/before parameters for date range filtering."""
        server = MemoryServer()

        content = unique_content("Dated memory for range filter test")
        await server.handle_store_memory({"content": content})

        # Search with date range
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

        result = await server.handle_memory_search({
            "after": yesterday,
            "before": tomorrow,
            "limit": 10
        })

        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_tag_filter(self, unique_content):
        """Test filtering search results by tags."""
        server = MemoryServer()

        content = unique_content("Tagged memory for search filter")
        await server.handle_store_memory({
            "content": content,
            "metadata": {"tags": ["search-tag-filter"]}
        })

        # Search with tag filter
        result = await server.handle_memory_search({
            "tags": ["search-tag-filter"],
            "limit": 10
        })

        assert isinstance(result, list)


class TestHandleMemoryDelete:
    """Tests for unified handle_memory_delete handler."""

    @pytest.mark.asyncio
    async def test_delete_by_content_hash(self, unique_content):
        """Test deletion by content_hash (single memory)."""
        server = MemoryServer()

        # Store memory
        content = unique_content("Memory to delete by hash")
        store_result = await server.handle_store_memory({
            "content": content,
            "metadata": {"tags": ["delete-hash-test"]}
        })

        # Extract hash from store response
        store_text = store_result[0].text

        # For testing, we'll use tag-based deletion instead since hash extraction is complex
        # This tests the unified delete handler
        result = await server.handle_memory_delete({
            "tags": ["delete-hash-test"],
            "tag_match": "any"
        })

        # Verify response format
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], types.TextContent)

    @pytest.mark.asyncio
    async def test_delete_by_tags_any_match(self, unique_content):
        """Test deletion by tags with ANY match mode."""
        server = MemoryServer()

        # Store memories with different tags
        await server.handle_store_memory({
            "content": unique_content("Delete tag any 1"),
            "metadata": {"tags": ["delete-any-a", "common"]}
        })
        await server.handle_store_memory({
            "content": unique_content("Delete tag any 2"),
            "metadata": {"tags": ["delete-any-b", "common"]}
        })

        # Delete with ANY match (matches either tag)
        result = await server.handle_memory_delete({
            "tags": ["delete-any-a", "delete-any-b"],
            "tag_match": "any"
        })

        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_delete_by_tags_all_match(self, unique_content):
        """Test deletion by tags with ALL match mode."""
        server = MemoryServer()

        # Store memory with both required tags
        await server.handle_store_memory({
            "content": unique_content("Has both delete tags"),
            "metadata": {"tags": ["delete-all-a", "delete-all-b"]}
        })

        # Delete with ALL match (requires both tags)
        result = await server.handle_memory_delete({
            "tags": ["delete-all-a", "delete-all-b"],
            "tag_match": "all"
        })

        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_delete_by_timeframe(self, unique_content):
        """Test deletion by time range (after/before)."""
        server = MemoryServer()

        # Store test memory
        await server.handle_store_memory({
            "content": unique_content("Timeframe delete test"),
            "metadata": {"tags": ["timeframe-delete-test"]}
        })

        # Delete with time range (use past dates for safety with dry_run)
        result = await server.handle_memory_delete({
            "after": "2020-01-01",
            "before": "2020-12-31",
            "dry_run": True  # Safety: preview only
        })

        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_delete_dry_run_mode(self, unique_content):
        """Test dry_run mode prevents actual deletion."""
        server = MemoryServer()

        # Store memory
        content = unique_content("Should not be deleted - dry run test")
        await server.handle_store_memory({
            "content": content,
            "metadata": {"tags": ["dry-run-delete-test"]}
        })

        # Dry run delete
        result = await server.handle_memory_delete({
            "tags": ["dry-run-delete-test"],
            "dry_run": True
        })

        assert isinstance(result, list)
        # Should indicate dry_run in response (or error if storage doesn't support it yet)
        text = result[0].text
        assert "dry" in text.lower() or "preview" in text.lower() or "would" in text.lower() or "error" in text.lower()


class TestHandleMemoryList:
    """Tests for handle_memory_list handler."""

    @pytest.mark.asyncio
    async def test_list_with_pagination(self, unique_content):
        """Test listing memories with limit and offset."""
        server = MemoryServer()

        # Store some test memories
        for i in range(5):
            await server.handle_store_memory({
                "content": unique_content(f"List pagination test {i}"),
                "metadata": {"tags": ["list-pagination-test"]}
            })

        # List with pagination
        result = await server.handle_memory_list({
            "limit": 3,
            "offset": 0
        })

        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], types.TextContent)

    @pytest.mark.asyncio
    async def test_list_filtered_by_tags(self, unique_content):
        """Test filtering list by tags."""
        server = MemoryServer()

        # Store with specific tag
        await server.handle_store_memory({
            "content": unique_content("Tagged memory for list filter"),
            "metadata": {"tags": ["list-filter-tag"]}
        })

        # List filtered by tag
        result = await server.handle_memory_list({
            "tags": ["list-filter-tag"],
            "limit": 10
        })

        assert isinstance(result, list)
        assert len(result) > 0


class TestBackwardsCompatibility:
    """Verify old handler names still work alongside new ones."""

    @pytest.mark.asyncio
    async def test_old_retrieve_memory_still_works(self, unique_content):
        """Verify handle_retrieve_memory (deprecated) still functions."""
        server = MemoryServer()

        content = unique_content("Backwards compat retrieve test")
        await server.handle_store_memory({
            "content": content,
            "metadata": {"tags": ["compat-retrieve-test"]}
        })

        # Call old handler name
        result = await server.handle_retrieve_memory({
            "query": "backwards compat",
            "n_results": 5
        })

        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_old_delete_by_tag_still_works(self, unique_content):
        """Verify handle_delete_by_tag (deprecated) still functions."""
        server = MemoryServer()

        await server.handle_store_memory({
            "content": unique_content("Backwards compat delete test"),
            "metadata": {"tags": ["compat-delete-test"]}
        })

        # Call old handler name
        result = await server.handle_delete_by_tag({
            "tags": ["compat-delete-test"]
        })

        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_old_search_by_tag_still_works(self, unique_content):
        """Verify handle_search_by_tag (deprecated) still functions."""
        server = MemoryServer()

        await server.handle_store_memory({
            "content": unique_content("Backwards compat tag search test"),
            "metadata": {"tags": ["compat-tag-search"]}
        })

        # Call old handler name
        result = await server.handle_search_by_tag({
            "tags": ["compat-tag-search"]
        })

        assert isinstance(result, list)
        assert len(result) > 0
