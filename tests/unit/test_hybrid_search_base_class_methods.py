"""
Tests for MemoryStorage base class methods required by hybrid search.

Bug: retrieve_memories() calls self.storage.count() and self.storage.search_by_tags()
but neither method exists on the MemoryStorage ABC, causing AttributeError on QdrantStorage
(and any storage backend that only implements the base class interface).

Error from production: 'QdrantStorage' object has no attribute 'count'

These tests verify:
1. MemoryStorage base class exposes count() and search_by_tags()
2. retrieve_memories() hybrid path works without AttributeError
"""

import time
from unittest.mock import AsyncMock, patch

import pytest

from mcp_memory_service.models.memory import Memory, MemoryQueryResult
from mcp_memory_service.services.memory_service import MemoryService
from mcp_memory_service.storage.base import MemoryStorage

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_memory():
    """A simple memory for testing."""
    return Memory(
        content="Rathole project architecture uses reverse proxy tunneling",
        content_hash="abc123def456",
        tags=["rathole", "architecture"],
        memory_type="note",
        metadata={},
        created_at=time.time(),
        updated_at=time.time(),
    )


@pytest.fixture
def mock_storage(sample_memory):
    """
    Create a mock storage using MemoryStorage as the spec.

    CRITICAL: Using spec=MemoryStorage means the mock will ONLY have methods
    that exist on MemoryStorage. If count() or search_by_tags() are missing
    from the ABC, accessing them on the mock raises AttributeError -- exactly
    reproducing the production bug.
    """
    storage = AsyncMock(spec=MemoryStorage)
    storage.max_content_length = None
    storage.supports_chunking = True

    # Setup returns for methods we expect to be called
    storage.get_all_tags.return_value = ["rathole", "architecture", "python"]
    storage.count_all_memories.return_value = 50
    storage.count.return_value = 50  # Used by hybrid search path
    storage.count_semantic_search.return_value = 5

    # Setup retrieve to return a MemoryQueryResult
    query_result = MemoryQueryResult(
        memory=sample_memory,
        relevance_score=0.85,
        debug_info={"score": 0.85, "backend": "test"},
    )
    storage.retrieve.return_value = [query_result]

    # search_by_tag (singular) and search_by_tags (plural) return Memory objects
    storage.search_by_tag.return_value = [sample_memory]
    storage.search_by_tags.return_value = [sample_memory]

    return storage


@pytest.fixture
def service(mock_storage):
    """Create MemoryService with spec-constrained mock storage."""
    return MemoryService(storage=mock_storage)


# =============================================================================
# Bug 1: self.storage.count() does not exist on base class
# =============================================================================


class TestCountMethodOnBaseClass:
    """Verify count() exists on MemoryStorage and delegates to count_all_memories()."""

    def test_count_method_exists_on_base_class(self):
        """MemoryStorage must have a count() method for hybrid search."""
        assert hasattr(MemoryStorage, "count"), (
            "MemoryStorage base class is missing count() method. "
            "hybrid search in memory_service.py line 332 calls self.storage.count()"
        )

    @pytest.mark.asyncio
    async def test_count_delegates_to_count_all_memories(self):
        """count() should delegate to count_all_memories() with no arguments.

        Uses a concrete subclass to test REAL delegation (not mock auto-stubs).
        """
        # Create a minimal concrete subclass with count_all_memories tracked
        from unittest.mock import AsyncMock as AM

        class FakeStorage(MemoryStorage):
            max_content_length = None
            supports_chunking = False

            # Implement all abstract methods as no-ops
            async def initialize(self):
                pass

            async def store(self, memory):
                return (True, "ok")

            async def retrieve(self, query, n_results=5, tags=None, memory_type=None, min_similarity=None, offset=0):
                return []

            async def search_by_tag(self, tags, limit=10, offset=0, match_all=False, start_timestamp=None, end_timestamp=None):
                return []

            async def get_memory_by_hash(self, content_hash):
                return None

            async def delete(self, content_hash):
                return (True, "ok")

            async def delete_by_tag(self, tag):
                return (0, "ok")

            async def delete_by_all_tags(self, tags):
                return (0, "ok")

            async def cleanup_duplicates(self):
                return (0, "ok")

            async def update_memory_metadata(self, content_hash, updates, preserve_timestamps=True):
                return (True, "ok")

            async def count_semantic_search(self, query, tags=None, memory_type=None, min_similarity=None):
                return 0

            async def count_tag_search(self, tags, match_all=False, start_timestamp=None, end_timestamp=None):
                return 0

            async def count_time_range(self, start_timestamp=None, end_timestamp=None, tags=None, memory_type=None):
                return 0

        storage = FakeStorage()
        storage.count_all_memories = AM(return_value=42)

        result = await storage.count()

        assert result == 42
        storage.count_all_memories.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_hybrid_search_does_not_raise_on_count(self, service):
        """retrieve_memories() hybrid path must not raise AttributeError on count()."""
        # Patch settings to ensure hybrid search is attempted (alpha < 1.0)
        from mcp_memory_service.config import HybridSearchSettings

        mock_config = HybridSearchSettings(
            hybrid_alpha=0.5,  # Force hybrid mode
            recency_decay=0.0,
        )

        with patch("mcp_memory_service.services.memory_service.settings") as mock_settings:
            mock_settings.hybrid_search = mock_config

            # This must NOT raise AttributeError: 'QdrantStorage' object has no attribute 'count'
            result = await service.retrieve_memories(query="rathole architecture", page=1, page_size=10)

            assert "error" not in result or "count" not in result.get("error", "")


# =============================================================================
# Bug 2: self.storage.search_by_tags() does not exist on base class
# =============================================================================


class TestSearchByTagsMethodOnBaseClass:
    """Verify search_by_tags() exists on MemoryStorage and delegates to search_by_tag()."""

    def test_search_by_tags_method_exists_on_base_class(self):
        """MemoryStorage must have a search_by_tags() method for hybrid search."""
        assert hasattr(MemoryStorage, "search_by_tags"), (
            "MemoryStorage base class is missing search_by_tags() method. "
            "hybrid search in memory_service.py line 353 calls self.storage.search_by_tags()"
        )

    @pytest.mark.asyncio
    async def test_search_by_tags_delegates_to_search_by_tag(self, sample_memory):
        """search_by_tags() should delegate to search_by_tag() with same params.

        Uses a concrete subclass to test REAL delegation (not mock auto-stubs).
        """
        from unittest.mock import AsyncMock as AM

        class FakeStorage(MemoryStorage):
            max_content_length = None
            supports_chunking = False

            async def initialize(self):
                pass

            async def store(self, memory):
                return (True, "ok")

            async def retrieve(self, query, n_results=5, tags=None, memory_type=None, min_similarity=None, offset=0):
                return []

            async def search_by_tag(self, tags, limit=10, offset=0, match_all=False, start_timestamp=None, end_timestamp=None):
                return []

            async def get_memory_by_hash(self, content_hash):
                return None

            async def delete(self, content_hash):
                return (True, "ok")

            async def delete_by_tag(self, tag):
                return (0, "ok")

            async def delete_by_all_tags(self, tags):
                return (0, "ok")

            async def cleanup_duplicates(self):
                return (0, "ok")

            async def update_memory_metadata(self, content_hash, updates, preserve_timestamps=True):
                return (True, "ok")

            async def count_semantic_search(self, query, tags=None, memory_type=None, min_similarity=None):
                return 0

            async def count_tag_search(self, tags, match_all=False, start_timestamp=None, end_timestamp=None):
                return 0

            async def count_time_range(self, start_timestamp=None, end_timestamp=None, tags=None, memory_type=None):
                return 0

        storage = FakeStorage()
        storage.search_by_tag = AM(return_value=[sample_memory])

        result = await storage.search_by_tags(
            tags=["rathole", "architecture"],
            match_all=False,
            limit=30,
        )

        assert len(result) == 1
        assert result[0].content == sample_memory.content
        storage.search_by_tag.assert_awaited_once_with(
            tags=["rathole", "architecture"],
            match_all=False,
            limit=30,
            offset=0,
        )


# =============================================================================
# End-to-end: hybrid retrieve_memories must work
# =============================================================================


class TestHybridRetrieveWithBaseClassMethods:
    """End-to-end test that hybrid search works when storage only implements base class."""

    @pytest.mark.asyncio
    async def test_retrieve_memories_hybrid_path_returns_results(self, service, mock_storage):
        """
        retrieve_memories() must return results through the hybrid path.

        This is the actual production bug: hybrid search calls count() and
        search_by_tags() which don't exist, causing an exception that gets
        caught and returns empty results.
        """
        from mcp_memory_service.config import HybridSearchSettings

        mock_config = HybridSearchSettings(
            hybrid_alpha=0.5,
            recency_decay=0.0,
        )

        with patch("mcp_memory_service.services.memory_service.settings") as mock_settings:
            mock_settings.hybrid_search = mock_config

            result = await service.retrieve_memories(query="rathole architecture", page=1, page_size=10)

            # The bug caused empty results due to caught AttributeError
            assert "memories" in result
            # If hybrid worked, we should have results (not empty due to error)
            # Check there's no error mentioning attribute issues
            if "error" in result:
                assert "attribute" not in result["error"].lower()
                assert "count" not in result["error"].lower()
                assert "search_by_tags" not in result["error"].lower()
