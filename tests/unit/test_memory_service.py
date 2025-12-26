"""
Unit tests for MemoryService business logic.

These tests verify the MemoryService class centralizes memory operations
correctly and provides consistent behavior across all interfaces.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from typing import List

from mcp_memory_service.services.memory_service import MemoryService
from mcp_memory_service.models.memory import Memory, MemoryQueryResult
from mcp_memory_service.storage.base import MemoryStorage


# Test Fixtures

@pytest.fixture
def mock_storage():
    """Create a mock storage backend."""
    storage = AsyncMock(spec=MemoryStorage)
    # Add required properties
    storage.max_content_length = 1000
    storage.supports_chunking = True
    # Setup method return values to avoid AttributeError
    storage.store.return_value = (True, "Success")
    storage.delete.return_value = (True, "Deleted")
    storage.get_stats.return_value = {
        "backend": "mock",
        "total_memories": 0
    }
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
        updated_at=1698765432.0
    )


@pytest.fixture
def sample_memories():
    """Create a list of sample memories."""
    memories = []
    for i in range(5):
        memories.append(Memory(
            content=f"Test memory {i+1}",
            content_hash=f"hash_{i+1}",
            tags=[f"tag{i+1}", "test"],
            memory_type="note",
            metadata={"index": i+1},
            created_at=1698765432.0 + i * 100,
            updated_at=1698765432.0 + i * 100
        ))
    return memories


# Test list_memories method

@pytest.mark.asyncio
async def test_list_memories_basic_pagination(memory_service, mock_storage, sample_memories):
    """Test basic pagination functionality."""
    # Setup mock
    mock_storage.get_all_memories.return_value = sample_memories[:2]
    mock_storage.count_all_memories.return_value = 5

    # Execute
    result = await memory_service.list_memories(page=1, page_size=2)

    # Verify
    assert result["page"] == 1
    assert result["page_size"] == 2
    assert result["total"] == 5
    assert result["has_more"] is True
    assert len(result["memories"]) == 2

    # Verify storage called with correct parameters
    mock_storage.get_all_memories.assert_called_once_with(
        limit=2,
        offset=0,
        memory_type=None,
        tags=None
    )


@pytest.mark.asyncio
async def test_list_memories_with_tag_filter(memory_service, mock_storage, sample_memories):
    """Test filtering by tag."""
    filtered_memories = [m for m in sample_memories if "tag1" in m.tags]
    mock_storage.get_all_memories.return_value = filtered_memories
    mock_storage.count_all_memories.return_value = len(filtered_memories)

    result = await memory_service.list_memories(page=1, page_size=10, tag="tag1")

    # Verify tag passed to storage as list
    mock_storage.get_all_memories.assert_called_once()
    call_kwargs = mock_storage.get_all_memories.call_args.kwargs
    assert call_kwargs["tags"] == ["tag1"]


@pytest.mark.asyncio
async def test_list_memories_with_type_filter(memory_service, mock_storage, sample_memories):
    """Test filtering by memory type."""
    mock_storage.get_all_memories.return_value = sample_memories
    mock_storage.count_all_memories.return_value = 5

    result = await memory_service.list_memories(page=1, page_size=10, memory_type="note")

    # Verify type passed to storage
    mock_storage.get_all_memories.assert_called_once()
    call_kwargs = mock_storage.get_all_memories.call_args.kwargs
    assert call_kwargs["memory_type"] == "note"


@pytest.mark.asyncio
async def test_list_memories_offset_calculation(memory_service, mock_storage):
    """Test correct offset calculation for different pages."""
    mock_storage.get_all_memories.return_value = []
    mock_storage.count_all_memories.return_value = 0

    # Page 3 with page_size 10 should have offset 20
    await memory_service.list_memories(page=3, page_size=10)

    call_kwargs = mock_storage.get_all_memories.call_args.kwargs
    assert call_kwargs["offset"] == 20
    assert call_kwargs["limit"] == 10


@pytest.mark.asyncio
async def test_list_memories_has_more_false(memory_service, mock_storage, sample_memories):
    """Test has_more is False when no more results."""
    mock_storage.get_all_memories.return_value = sample_memories
    mock_storage.count_all_memories.return_value = 5

    # Requesting page that includes last item
    result = await memory_service.list_memories(page=1, page_size=10)

    assert result["has_more"] is False


@pytest.mark.asyncio
async def test_list_memories_error_handling(memory_service, mock_storage):
    """Test error handling in list_memories."""
    mock_storage.get_all_memories.side_effect = Exception("Database error")

    result = await memory_service.list_memories(page=1, page_size=10)

    assert result["success"] is False
    assert "error" in result
    assert "Database error" in result["error"]
    assert result["memories"] == []


# Test store_memory method

@pytest.mark.asyncio
async def test_store_memory_basic(memory_service, mock_storage):
    """Test basic memory storage."""
    mock_storage.store.return_value = (True, "Success")

    result = await memory_service.store_memory(
        content="Test content",
        tags=["test"],
        memory_type="note"
    )

    assert result["success"] is True
    assert "memory" in result
    assert result["memory"]["content"] == "Test content"

    # Verify storage.store was called
    mock_storage.store.assert_called_once()
    stored_memory = mock_storage.store.call_args.args[0]
    assert stored_memory.content == "Test content"
    assert stored_memory.tags == ["test"]


@pytest.mark.asyncio
async def test_store_memory_with_hostname_tagging(memory_service, mock_storage):
    """Test hostname tagging is applied correctly."""
    mock_storage.store.return_value = None

    result = await memory_service.store_memory(
        content="Test content",
        tags=["test"],
        client_hostname="my-machine"
    )

    # Verify hostname tag added
    stored_memory = mock_storage.store.call_args.args[0]
    assert "source:my-machine" in stored_memory.tags
    assert stored_memory.metadata["hostname"] == "my-machine"


@pytest.mark.asyncio
async def test_store_memory_hostname_not_duplicated(memory_service, mock_storage):
    """Test hostname tag is not duplicated if already present."""
    mock_storage.store.return_value = None

    result = await memory_service.store_memory(
        content="Test content",
        tags=["test", "source:my-machine"],
        client_hostname="my-machine"
    )

    stored_memory = mock_storage.store.call_args.args[0]
    # Count occurrences of hostname tag
    hostname_tags = [t for t in stored_memory.tags if t.startswith("source:")]
    assert len(hostname_tags) == 1


@pytest.mark.asyncio
@patch('mcp_memory_service.services.memory_service.ENABLE_AUTO_SPLIT', True)
async def test_store_memory_with_chunking(memory_service, mock_storage):
    """Test content chunking when enabled and content is large."""
    mock_storage.store.return_value = (True, "Success")
    # Set max_content_length to trigger chunking
    mock_storage.max_content_length = 100

    # Create content larger than max_content_length
    long_content = "x" * 200

    with patch('mcp_memory_service.services.memory_service.split_content') as mock_split:
        mock_split.return_value = ["chunk1", "chunk2"]

        result = await memory_service.store_memory(content=long_content)

        assert result["success"] is True
        assert "memories" in result
        assert result["total_chunks"] == 2
        assert "original_hash" in result

        # Verify storage.store called twice (once per chunk)
        assert mock_storage.store.call_count == 2


@pytest.mark.asyncio
async def test_store_memory_validation_error(memory_service, mock_storage):
    """Test ValueError is caught and returned as error."""
    mock_storage.store.side_effect = ValueError("Invalid content")

    result = await memory_service.store_memory(content="Test")

    assert result["success"] is False
    assert "error" in result
    assert "Invalid memory data" in result["error"]


@pytest.mark.asyncio
async def test_store_memory_connection_error(memory_service, mock_storage):
    """Test ConnectionError is caught and handled."""
    mock_storage.store.side_effect = ConnectionError("Storage unavailable")

    result = await memory_service.store_memory(content="Test")

    assert result["success"] is False
    assert "error" in result
    assert "Storage connection failed" in result["error"]


@pytest.mark.asyncio
async def test_store_memory_unexpected_error(memory_service, mock_storage):
    """Test unexpected exceptions are caught."""
    mock_storage.store.side_effect = RuntimeError("Unexpected error")

    result = await memory_service.store_memory(content="Test")

    assert result["success"] is False
    assert "error" in result
    assert "Failed to store memory" in result["error"]


# Test retrieve_memories method

@pytest.mark.asyncio
async def test_retrieve_memories_basic(memory_service, mock_storage, sample_memories):
    """Test basic semantic search retrieval."""
    mock_storage.retrieve.return_value = [
        MemoryQueryResult(memory=mem, relevance_score=0.9 - i*0.1)
        for i, mem in enumerate(sample_memories[:3])
    ]

    result = await memory_service.retrieve_memories(query="test query", n_results=3)

    assert result["query"] == "test query"
    assert result["count"] == 3
    assert len(result["memories"]) == 3

    # After fix: storage.retrieve() only accepts query and n_results
    mock_storage.retrieve.assert_called_once_with(
        query="test query",
        n_results=3
    )


@pytest.mark.asyncio
async def test_retrieve_memories_with_filters(memory_service, mock_storage, sample_memories):
    """Test retrieval with tag and type filters."""
    # Return memories that will be filtered by MemoryService
    # Create a memory with matching tags for filtering
    from mcp_memory_service.models.memory import Memory
    import hashlib
    content_hash = hashlib.sha256("test content".encode()).hexdigest()
    matching_memory = Memory(
        content="test content",
        content_hash=content_hash,
        tags=["tag1"],
        memory_type="note",
        created_at=1234567890.0
    )
    matching_memory.metadata = {"tags": ["tag1"], "memory_type": "note"}
    mock_storage.retrieve.return_value = [
        MemoryQueryResult(memory=matching_memory, relevance_score=0.85)
    ]

    result = await memory_service.retrieve_memories(
        query="test",
        n_results=5,
        tags=["tag1"],
        memory_type="note"
    )

    # After fix: storage.retrieve() only accepts query and n_results
    # Filtering is done by MemoryService after retrieval
    mock_storage.retrieve.assert_called_once_with(
        query="test",
        n_results=5
    )


@pytest.mark.asyncio
async def test_retrieve_memories_error_handling(memory_service, mock_storage):
    """Test error handling in retrieve_memories."""
    mock_storage.retrieve.side_effect = Exception("Retrieval failed")

    result = await memory_service.retrieve_memories(query="test")

    assert "error" in result
    assert result["memories"] == []
    assert "Failed to retrieve memories" in result["error"]


# Test search_by_tag method

@pytest.mark.asyncio
async def test_search_by_tag_single_tag(memory_service, mock_storage, sample_memories):
    """Test searching by a single tag."""
    mock_storage.search_by_tag.return_value = sample_memories[:2]

    result = await memory_service.search_by_tag(tags="test")

    assert result["tags"] == ["test"]
    assert result["match_type"] == "ANY"
    assert result["count"] == 2

    mock_storage.search_by_tag.assert_called_once_with(
        tags=["test"]
    )


@pytest.mark.asyncio
async def test_search_by_tag_multiple_tags(memory_service, mock_storage, sample_memories):
    """Test searching by multiple tags."""
    mock_storage.search_by_tag.return_value = sample_memories

    result = await memory_service.search_by_tag(tags=["tag1", "tag2"])

    assert result["tags"] == ["tag1", "tag2"]
    assert result["match_type"] == "ANY"


@pytest.mark.asyncio
async def test_search_by_tag_match_all(memory_service, mock_storage, sample_memories):
    """Test searching with match_all=True."""
    mock_storage.search_by_tag.return_value = sample_memories[:1]

    result = await memory_service.search_by_tag(tags=["tag1", "tag2"], match_all=True)

    assert result["match_type"] == "ALL"
    mock_storage.search_by_tag.assert_called_once_with(
        tags=["tag1", "tag2"]
    )


@pytest.mark.asyncio
async def test_search_by_tag_error_handling(memory_service, mock_storage):
    """Test error handling in search_by_tag."""
    mock_storage.search_by_tag.side_effect = Exception("Search failed")

    result = await memory_service.search_by_tag(tags="test")

    assert "error" in result
    assert result["memories"] == []
    assert "Failed to search by tags" in result["error"]


# Test get_memory_by_hash method

@pytest.mark.asyncio
async def test_get_memory_by_hash_found(memory_service, mock_storage, sample_memory):
    """Test getting memory by hash when found."""
    mock_storage.get_by_hash.return_value = sample_memory

    result = await memory_service.get_memory_by_hash("test_hash_123")

    assert result["found"] is True
    assert "memory" in result
    assert result["memory"]["content_hash"] == "test_hash_123"
    mock_storage.get_by_hash.assert_called_once_with("test_hash_123")


@pytest.mark.asyncio
async def test_get_memory_by_hash_not_found(memory_service, mock_storage):
    """Test getting memory by hash when not found."""
    mock_storage.get_by_hash.return_value = None

    result = await memory_service.get_memory_by_hash("nonexistent_hash")

    assert result["found"] is False
    assert result["content_hash"] == "nonexistent_hash"
    mock_storage.get_by_hash.assert_called_once_with("nonexistent_hash")


@pytest.mark.asyncio
async def test_get_memory_by_hash_error(memory_service, mock_storage):
    """Test error handling in get_memory_by_hash."""
    mock_storage.get_by_hash.side_effect = Exception("Database error")

    result = await memory_service.get_memory_by_hash("test_hash")

    assert result["found"] is False
    assert "error" in result
    assert "Failed to get memory" in result["error"]


# Test delete_memory method

@pytest.mark.asyncio
async def test_delete_memory_success(memory_service, mock_storage):
    """Test successful memory deletion."""
    mock_storage.delete.return_value = (True, "Deleted successfully")

    result = await memory_service.delete_memory("test_hash")

    assert result["success"] is True
    assert result["content_hash"] == "test_hash"
    mock_storage.delete.assert_called_once_with("test_hash")


@pytest.mark.asyncio
async def test_delete_memory_not_found(memory_service, mock_storage):
    """Test deleting non-existent memory."""
    mock_storage.delete.return_value = (False, "Not found")

    result = await memory_service.delete_memory("nonexistent_hash")

    assert result["success"] is False


@pytest.mark.asyncio
async def test_delete_memory_error(memory_service, mock_storage):
    """Test error handling in delete_memory."""
    mock_storage.delete.side_effect = Exception("Delete failed")

    result = await memory_service.delete_memory("test_hash")

    assert result["success"] is False
    assert "error" in result
    assert "Failed to delete memory" in result["error"]


# Test health_check method

@pytest.mark.asyncio
async def test_health_check_success(memory_service, mock_storage):
    """Test successful health check."""
    mock_storage.get_stats.return_value = {
        "backend": "sqlite-vec",
        "total_memories": 100,
        "database_size": "5MB"
    }

    result = await memory_service.health_check()

    assert result["healthy"] is True
    assert result["storage_type"] == "sqlite-vec"
    assert result["total_memories"] == 100
    assert "last_updated" in result


@pytest.mark.asyncio
async def test_health_check_failure(memory_service, mock_storage):
    """Test health check when storage fails."""
    mock_storage.get_stats.side_effect = Exception("Health check failed")

    result = await memory_service.health_check()

    assert result["healthy"] is False
    assert "error" in result
    assert "Health check failed" in result["error"]


# Test _format_memory_response method

def test_format_memory_response(memory_service, sample_memory):
    """Test memory formatting for API responses."""
    formatted = memory_service._format_memory_response(sample_memory)

    assert formatted["content"] == sample_memory.content
    assert formatted["content_hash"] == sample_memory.content_hash
    assert formatted["tags"] == sample_memory.tags
    assert formatted["memory_type"] == sample_memory.memory_type
    assert formatted["metadata"] == sample_memory.metadata
    assert "created_at" in formatted
    assert "updated_at" in formatted
    assert "created_at_iso" in formatted
    assert "updated_at_iso" in formatted


def test_format_memory_response_preserves_all_fields(memory_service, sample_memory):
    """Test that formatting preserves all memory fields."""
    formatted = memory_service._format_memory_response(sample_memory)

    # Verify all TypedDict fields are present
    required_fields = [
        "content", "content_hash", "tags", "memory_type", "metadata",
        "created_at", "updated_at", "created_at_iso", "updated_at_iso"
    ]

    for field in required_fields:
        assert field in formatted


# Integration-style tests (still using mocks but testing workflows)

@pytest.mark.asyncio
async def test_store_and_retrieve_workflow(memory_service, mock_storage, sample_memory):
    """Test complete workflow: store then retrieve."""
    # Setup mocks
    mock_storage.store.return_value = (True, "Success")
    mock_storage.retrieve.return_value = [
        MemoryQueryResult(memory=sample_memory, relevance_score=0.92)
    ]

    # Store memory
    store_result = await memory_service.store_memory(
        content="Test workflow",
        tags=["workflow"],
        memory_type="test"
    )
    assert store_result["success"] is True

    # Retrieve memory
    retrieve_result = await memory_service.retrieve_memories(query="workflow")
    assert len(retrieve_result["memories"]) > 0


@pytest.mark.asyncio
async def test_list_memories_database_level_filtering(memory_service, mock_storage):
    """Test that list_memories uses database-level filtering (not loading all)."""
    mock_storage.get_all_memories.return_value = []
    mock_storage.count_all_memories.return_value = 1000

    # Request page 1 with 10 items from 1000 total
    result = await memory_service.list_memories(page=1, page_size=10)

    # Verify we only requested 10 items, not all 1000
    call_kwargs = mock_storage.get_all_memories.call_args.kwargs
    assert call_kwargs["limit"] == 10
    assert call_kwargs["offset"] == 0

    # This proves we're using database-level filtering, not O(n) memory loading
    mock_storage.get_all_memories.assert_called_once()


@pytest.mark.asyncio
async def test_empty_tags_list_stored_correctly(memory_service, mock_storage):
    """Test that empty or None tags are handled correctly."""
    mock_storage.store.return_value = None

    # Store with None tags
    result = await memory_service.store_memory(content="Test", tags=None)

    stored_memory = mock_storage.store.call_args.args[0]
    assert isinstance(stored_memory.tags, list)
    assert len(stored_memory.tags) == 0


@pytest.mark.asyncio
async def test_metadata_preserved_through_storage(memory_service, mock_storage):
    """Test that metadata is preserved correctly."""
    mock_storage.store.return_value = None

    custom_metadata = {"key1": "value1", "key2": 123}
    result = await memory_service.store_memory(
        content="Test",
        metadata=custom_metadata
    )

    stored_memory = mock_storage.store.call_args.args[0]
    assert "key1" in stored_memory.metadata
    assert stored_memory.metadata["key1"] == "value1"


@pytest.mark.asyncio
async def test_chunked_storage_all_chunks_fail(mock_storage):
    """
    Test that chunked storage returns success=False when ALL chunks fail.

    Regression test for issue where storing content > max_content_length
    would return {"success": True, "memories": [], "total_chunks": N}
    when all chunks failed (e.g., due to duplicates).
    """
    # Setup: max_content_length = 100, all store() calls fail
    mock_storage.max_content_length = 100
    mock_storage.store.return_value = (False, "Duplicate content detected")

    service = MemoryService(storage=mock_storage)

    # Create content > 100 chars to trigger chunking
    long_content = "A" * 250  # Will be split into ~3 chunks

    with patch('mcp_memory_service.services.memory_service.ENABLE_AUTO_SPLIT', True):
        result = await service.store_memory(
            content=long_content,
            tags=["test"],
            memory_type="test"
        )

    # Assert: should return failure, not success
    assert result["success"] is False, "Should fail when all chunks fail to store"
    assert "error" in result, "Should include error message"
    assert "Failed to store all" in result["error"], "Error should explain all chunks failed"
    assert "Duplicate content detected" in result["error"], "Error should include storage failure reason"


@pytest.mark.asyncio
async def test_chunked_storage_partial_success(mock_storage):
    """
    Test that chunked storage returns partial success when SOME chunks fail.

    This tests the scenario where content is split and some chunks succeed
    while others fail (e.g., first chunk is new, second is duplicate).
    """
    # Setup: max_content_length = 100
    mock_storage.max_content_length = 100

    # Prepare enough results for any number of chunks
    # Pattern: success, fail, success, fail, success...
    store_results = []
    for i in range(10):  # Enough for any split
        if i % 2 == 0:
            store_results.append((True, "Success"))
        else:
            store_results.append((False, "Duplicate content detected"))
    mock_storage.store.side_effect = store_results

    service = MemoryService(storage=mock_storage)

    # Create content > 100 chars to trigger chunking
    long_content = "B" * 250

    with patch('mcp_memory_service.services.memory_service.ENABLE_AUTO_SPLIT', True):
        result = await service.store_memory(
            content=long_content,
            tags=["test"],
            memory_type="test"
        )

    # Assert: should return partial success
    assert result["success"] is True, "Should succeed when some chunks store successfully"
    assert "memories" in result, "Should include stored memories"
    assert len(result["memories"]) > 0, "Should have at least one successful chunk"
    assert result.get("failed_chunks", 0) > 0, "Should report at least one failed chunk"
    total_attempted = len(result["memories"]) + result.get("failed_chunks", 0)
    assert result["total_chunks"] == total_attempted, "Total chunks should match attempted"
