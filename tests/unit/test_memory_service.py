"""
Unit tests for MemoryService business logic.

These tests verify:
1. Input validation and normalization
2. Pagination calculations
3. Error handling paths
4. Data transformation logic

For integration tests with real storage backends, see:
tests/integration/test_storage_integration.py

NOTE: Tests that only verify mocks were called have been removed.
We test BEHAVIOR, not implementation details.
"""

from unittest.mock import AsyncMock

import pytest

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.services.memory_service import MemoryService
from mcp_memory_service.storage.base import MemoryStorage

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_storage():
    """Create a mock storage backend."""
    storage = AsyncMock(spec=MemoryStorage)
    storage.max_content_length = 1000
    storage.supports_chunking = True
    storage.store.return_value = (True, "Success")
    storage.delete.return_value = (True, "Deleted")
    storage.get_stats.return_value = {"backend": "mock", "total_memories": 0}
    return storage


@pytest.fixture
def memory_service(mock_storage):
    """Create a MemoryService instance with mock storage."""
    return MemoryService(storage=mock_storage)


@pytest.fixture
def sample_memory():
    """Create a sample memory object for testing."""
    return Memory(
        content="Test memory content",
        content_hash="test_hash_123",
        tags=["test", "sample"],
        memory_type="note",
        metadata={"source": "test"},
        created_at=1698765432.0,
        updated_at=1698765432.0,
    )


@pytest.fixture
def sample_memories():
    """Create a list of sample memories."""
    return [
        Memory(
            content=f"Test memory {i + 1}",
            content_hash=f"hash_{i + 1}",
            tags=[f"tag{i + 1}", "test"],
            memory_type="note",
            metadata={"index": i + 1},
            created_at=1698765432.0 + i * 100,
            updated_at=1698765432.0 + i * 100,
        )
        for i in range(5)
    ]


# =============================================================================
# Pagination Logic Tests
# These test the CALCULATION logic, not mock interactions
# =============================================================================


class TestPaginationLogic:
    """Test pagination metadata calculations."""

    def test_pagination_metadata_basic(self, memory_service):
        """Test basic pagination metadata structure."""
        metadata = memory_service._build_pagination_metadata(total=100, page=1, page_size=10)

        assert metadata["total"] == 100
        assert metadata["page"] == 1
        assert metadata["page_size"] == 10
        assert metadata["has_more"] is True
        assert metadata["total_pages"] == 10

    def test_pagination_has_more_true_when_more_pages(self, memory_service):
        """Test has_more is True when not on last page."""
        metadata = memory_service._build_pagination_metadata(total=25, page=1, page_size=10)

        assert metadata["has_more"] is True

    def test_pagination_has_more_false_on_last_page(self, memory_service):
        """Test has_more is False on the last page."""
        metadata = memory_service._build_pagination_metadata(total=25, page=3, page_size=10)

        assert metadata["has_more"] is False

    def test_pagination_has_more_false_when_exact_fit(self, memory_service):
        """Test has_more is False when results exactly fill pages."""
        metadata = memory_service._build_pagination_metadata(total=20, page=2, page_size=10)

        assert metadata["has_more"] is False

    def test_pagination_total_pages_calculation(self, memory_service):
        """Test total_pages ceiling division."""
        # 25 items / 10 per page = 3 pages
        metadata = memory_service._build_pagination_metadata(total=25, page=1, page_size=10)
        assert metadata["total_pages"] == 3

        # 20 items / 10 per page = 2 pages
        metadata = memory_service._build_pagination_metadata(total=20, page=1, page_size=10)
        assert metadata["total_pages"] == 2

        # 1 item / 10 per page = 1 page
        metadata = memory_service._build_pagination_metadata(total=1, page=1, page_size=10)
        assert metadata["total_pages"] == 1

    def test_pagination_empty_results(self, memory_service):
        """Test pagination with zero results."""
        metadata = memory_service._build_pagination_metadata(total=0, page=1, page_size=10)

        assert metadata["total"] == 0
        assert metadata["has_more"] is False
        # With 0 results, total_pages formula gives 0 (ceiling division: (0 + 10 - 1) // 10 = 0)
        assert metadata["total_pages"] == 0


# =============================================================================
# Input Normalization Tests
# =============================================================================


class TestInputNormalization:
    """Test input validation and normalization."""

    @pytest.mark.asyncio
    async def test_tag_string_converted_to_list(self, memory_service, mock_storage):
        """Test that string tags are converted to list."""
        mock_storage.count_tag_search.return_value = 0
        mock_storage.search_by_tag.return_value = []

        result = await memory_service.search_by_tag(tags="single-tag")

        # Verify tags were normalized
        assert result["tags"] == ["single-tag"]

    @pytest.mark.asyncio
    async def test_none_tags_normalized_to_empty_list(self, memory_service, mock_storage):
        """Test that None tags become empty list in stored memory."""
        mock_storage.store.return_value = (True, "Success")

        await memory_service.store_memory(content="Test", tags=None)

        # Get the memory that was stored
        stored_memory = mock_storage.store.call_args.args[0]
        assert isinstance(stored_memory.tags, list)

    @pytest.mark.asyncio
    async def test_hostname_tagging_adds_source_tag(self, memory_service, mock_storage):
        """Test hostname creates source: prefixed tag."""
        mock_storage.store.return_value = (True, "Success")

        await memory_service.store_memory(content="Test", tags=["existing"], client_hostname="my-machine")

        stored_memory = mock_storage.store.call_args.args[0]
        assert "source:my-machine" in stored_memory.tags
        assert stored_memory.metadata["hostname"] == "my-machine"

    @pytest.mark.asyncio
    async def test_hostname_not_duplicated(self, memory_service, mock_storage):
        """Test hostname tag isn't added if already present."""
        mock_storage.store.return_value = (True, "Success")

        await memory_service.store_memory(content="Test", tags=["source:my-machine"], client_hostname="my-machine")

        stored_memory = mock_storage.store.call_args.args[0]
        source_tags = [t for t in stored_memory.tags if t.startswith("source:")]
        assert len(source_tags) == 1


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Test error handling and recovery."""

    @pytest.mark.asyncio
    async def test_store_validation_error_returns_structured_response(self, memory_service, mock_storage):
        """Test ValueError produces clean error response."""
        mock_storage.store.side_effect = ValueError("Invalid content")

        result = await memory_service.store_memory(content="Test")

        assert result["success"] is False
        assert "Invalid memory data" in result["error"]

    @pytest.mark.asyncio
    async def test_store_connection_error_returns_structured_response(self, memory_service, mock_storage):
        """Test ConnectionError produces appropriate response."""
        mock_storage.store.side_effect = ConnectionError("DB unavailable")

        result = await memory_service.store_memory(content="Test")

        assert result["success"] is False
        assert "Storage connection failed" in result["error"]

    @pytest.mark.asyncio
    async def test_store_unexpected_error_returns_generic_response(self, memory_service, mock_storage):
        """Test unexpected errors are caught and wrapped."""
        mock_storage.store.side_effect = RuntimeError("Unexpected")

        result = await memory_service.store_memory(content="Test")

        assert result["success"] is False
        assert "Failed to store memory" in result["error"]

    @pytest.mark.asyncio
    async def test_retrieve_error_returns_empty_results(self, memory_service, mock_storage):
        """Test retrieval errors return empty list, not crash."""
        mock_storage.retrieve.side_effect = Exception("Search failed")

        result = await memory_service.retrieve_memories(query="test")

        assert result["memories"] == []
        assert "error" in result

    @pytest.mark.asyncio
    async def test_list_memories_error_returns_empty_results(self, memory_service, mock_storage):
        """Test list errors return empty list with error message."""
        mock_storage.get_all_memories.side_effect = Exception("Database error")

        result = await memory_service.list_memories(page=1, page_size=10)

        assert result["success"] is False
        assert result["memories"] == []
        assert "error" in result

    @pytest.mark.asyncio
    async def test_delete_error_returns_failure(self, memory_service, mock_storage):
        """Test delete errors produce failure response."""
        mock_storage.delete.side_effect = Exception("Delete failed")

        result = await memory_service.delete_memory("test_hash")

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_health_check_error_returns_unhealthy(self, memory_service, mock_storage):
        """Test health check failures report unhealthy."""
        mock_storage.get_stats.side_effect = Exception("Stats failed")

        result = await memory_service.check_database_health()

        assert result["healthy"] is False
        assert "error" in result


# =============================================================================
# Response Formatting Tests
# =============================================================================


class TestResponseFormatting:
    """Test response data transformation."""

    def test_format_memory_includes_all_fields(self, memory_service, sample_memory):
        """Test formatted response has all required fields."""
        formatted = memory_service._format_memory_response(sample_memory)

        required_fields = [
            "content",
            "content_hash",
            "tags",
            "memory_type",
            "metadata",
            "created_at",
            "updated_at",
            "created_at_iso",
            "updated_at_iso",
        ]

        for field in required_fields:
            assert field in formatted, f"Missing field: {field}"

    def test_format_memory_preserves_content(self, memory_service, sample_memory):
        """Test content is not modified during formatting."""
        formatted = memory_service._format_memory_response(sample_memory)

        assert formatted["content"] == sample_memory.content
        assert formatted["content_hash"] == sample_memory.content_hash
        assert formatted["tags"] == sample_memory.tags

    def test_format_memory_includes_iso_timestamps(self, memory_service, sample_memory):
        """Test ISO timestamp conversion is included."""
        formatted = memory_service._format_memory_response(sample_memory)

        # ISO timestamps should be strings
        assert isinstance(formatted["created_at_iso"], str)
        assert isinstance(formatted["updated_at_iso"], str)


# =============================================================================
# Offset Calculation Tests
# =============================================================================


class TestOffsetCalculation:
    """Test pagination offset calculations."""

    @pytest.mark.asyncio
    async def test_page_1_has_zero_offset(self, memory_service, mock_storage):
        """Test first page has offset 0."""
        mock_storage.get_all_memories.return_value = []
        mock_storage.count_all_memories.return_value = 0

        await memory_service.list_memories(page=1, page_size=10)

        call_kwargs = mock_storage.get_all_memories.call_args.kwargs
        assert call_kwargs["offset"] == 0

    @pytest.mark.asyncio
    async def test_page_2_has_correct_offset(self, memory_service, mock_storage):
        """Test page 2 offset equals page_size."""
        mock_storage.get_all_memories.return_value = []
        mock_storage.count_all_memories.return_value = 0

        await memory_service.list_memories(page=2, page_size=10)

        call_kwargs = mock_storage.get_all_memories.call_args.kwargs
        assert call_kwargs["offset"] == 10

    @pytest.mark.asyncio
    async def test_page_3_offset_calculation(self, memory_service, mock_storage):
        """Test page 3 with page_size 10 has offset 20."""
        mock_storage.get_all_memories.return_value = []
        mock_storage.count_all_memories.return_value = 0

        await memory_service.list_memories(page=3, page_size=10)

        call_kwargs = mock_storage.get_all_memories.call_args.kwargs
        assert call_kwargs["offset"] == 20
        assert call_kwargs["limit"] == 10


# =============================================================================
# Match Type Tests
# =============================================================================


class TestMatchTypeReporting:
    """Test match type is reported correctly in tag search."""

    @pytest.mark.asyncio
    async def test_match_any_reported(self, memory_service, mock_storage):
        """Test match_all=False reports as ANY."""
        mock_storage.count_tag_search.return_value = 0
        mock_storage.search_by_tag.return_value = []

        result = await memory_service.search_by_tag(tags=["a", "b"], match_all=False)

        assert result["match_type"] == "ANY"

    @pytest.mark.asyncio
    async def test_match_all_reported(self, memory_service, mock_storage):
        """Test match_all=True reports as ALL."""
        mock_storage.count_tag_search.return_value = 0
        mock_storage.search_by_tag.return_value = []

        result = await memory_service.search_by_tag(tags=["a", "b"], match_all=True)

        assert result["match_type"] == "ALL"


# =============================================================================
# Found/Not Found Response Tests
# =============================================================================


class TestFoundNotFoundResponses:
    """Test found/not found response structures."""

    @pytest.mark.asyncio
    async def test_get_by_hash_found_response(self, memory_service, mock_storage, sample_memory):
        """Test response structure when memory is found."""
        mock_storage.get_memory_by_hash.return_value = sample_memory

        result = await memory_service.get_memory_by_hash("test_hash")

        assert result["found"] is True
        assert "memory" in result

    @pytest.mark.asyncio
    async def test_get_by_hash_not_found_response(self, memory_service, mock_storage):
        """Test response structure when memory is not found."""
        mock_storage.get_memory_by_hash.return_value = None

        result = await memory_service.get_memory_by_hash("nonexistent")

        assert result["found"] is False
        assert result["content_hash"] == "nonexistent"

    @pytest.mark.asyncio
    async def test_delete_success_response(self, memory_service, mock_storage):
        """Test delete success response structure."""
        mock_storage.delete.return_value = (True, "Deleted")

        result = await memory_service.delete_memory("test_hash")

        assert result["success"] is True
        assert result["content_hash"] == "test_hash"

    @pytest.mark.asyncio
    async def test_delete_not_found_response(self, memory_service, mock_storage):
        """Test delete not-found response structure."""
        mock_storage.delete.return_value = (False, "Not found")

        result = await memory_service.delete_memory("nonexistent")

        assert result["success"] is False


# =============================================================================
# Health Check Response Tests
# =============================================================================


class TestHealthCheckResponses:
    """Test health check response structure."""

    @pytest.mark.asyncio
    async def test_healthy_response_includes_stats(self, memory_service, mock_storage):
        """Test healthy response includes storage stats."""
        mock_storage.get_stats.return_value = {"backend": "test", "total_memories": 42}

        result = await memory_service.check_database_health()

        assert result["healthy"] is True
        assert result["total_memories"] == 42
        assert result["storage_type"] == "test"
        assert "last_updated" in result
