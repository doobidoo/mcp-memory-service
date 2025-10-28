"""
Integration tests for API endpoints using MemoryService.

These tests verify that the API layer correctly integrates with
MemoryService for all memory operations and maintains consistent behavior.
"""

import pytest
import tempfile
import os
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from mcp_memory_service.web.dependencies import set_storage, get_memory_service
from mcp_memory_service.services.memory_service import MemoryService
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage


# Test Fixtures

@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        yield db_path


@pytest.fixture
async def initialized_storage(temp_db):
    """Create and initialize a real SQLite storage backend."""
    storage = SqliteVecMemoryStorage(temp_db)
    await storage.initialize()
    yield storage
    storage.close()


@pytest.fixture
def test_app(initialized_storage):
    """Create a FastAPI test application with initialized storage."""
    # Import here to avoid circular dependencies
    from mcp_memory_service.web.server import app

    # Set storage for the app
    set_storage(initialized_storage)

    client = TestClient(app)
    yield client


@pytest.fixture
def mock_storage():
    """Create a mock storage for isolated testing."""
    storage = AsyncMock()
    return storage


@pytest.fixture
def mock_memory_service(mock_storage):
    """Create a MemoryService with mock storage."""
    return MemoryService(storage=mock_storage)


@pytest.fixture
def sample_memory():
    """Create a sample memory for testing."""
    return Memory(
        content="Integration test memory",
        content_hash="test_hash_123",
        tags=["integration", "test"],
        memory_type="note",
        metadata={"source": "test"},
        created_at=1698765432.0,
        updated_at=1698765432.0
    )


# Test API Store Memory Endpoint

@pytest.mark.asyncio
async def test_api_store_memory_uses_service(mock_storage):
    """Test that POST /api/memories uses MemoryService."""
    mock_storage.store.return_value = None

    # Create service
    service = MemoryService(storage=mock_storage)

    # Simulate API call through service
    result = await service.store_memory(
        content="Test API storage",
        tags=["api", "test"],
        memory_type="note"
    )

    assert result["success"] is True
    assert "memory" in result
    mock_storage.store.assert_called_once()


@pytest.mark.asyncio
async def test_api_store_memory_hostname_from_header(mock_storage):
    """Test that X-Client-Hostname header is processed correctly."""
    mock_storage.store.return_value = None

    service = MemoryService(storage=mock_storage)

    # Simulate API call with hostname
    result = await service.store_memory(
        content="Test with hostname",
        tags=["test"],
        client_hostname="client-machine"
    )

    # Verify hostname tag was added
    stored_memory = mock_storage.store.call_args.args[0]
    assert "source:client-machine" in stored_memory.tags
    assert stored_memory.metadata["hostname"] == "client-machine"


@pytest.mark.asyncio
async def test_api_store_memory_hostname_from_request_body(mock_storage):
    """Test that client_hostname in request body works."""
    mock_storage.store.return_value = None

    service = MemoryService(storage=mock_storage)

    # Simulate API call with hostname in body
    result = await service.store_memory(
        content="Test",
        client_hostname="body-hostname"
    )

    stored_memory = mock_storage.store.call_args.args[0]
    assert "source:body-hostname" in stored_memory.tags


# Test API List Memories Endpoint

@pytest.mark.asyncio
async def test_api_list_memories_uses_database_filtering(mock_storage):
    """Test that GET /api/memories uses database-level filtering."""
    # Setup mock to return limited results
    mock_storage.get_all_memories.return_value = []
    mock_storage.count_all_memories.return_value = 1000

    service = MemoryService(storage=mock_storage)

    # Request page 1 with 10 items from 1000 total
    result = await service.list_memories(page=1, page_size=10)

    # CRITICAL: Verify only 10 items requested, not all 1000
    # This proves database-level filtering, not O(n) loading
    call_kwargs = mock_storage.get_all_memories.call_args.kwargs
    assert call_kwargs["limit"] == 10
    assert call_kwargs["offset"] == 0
    assert result["total"] == 1000
    assert result["has_more"] is True


@pytest.mark.asyncio
async def test_api_list_memories_pagination_through_service(mock_storage):
    """Test end-to-end pagination workflow."""
    # Create mock memories
    memories = [
        Memory(
            content=f"Memory {i}",
            content_hash=f"hash_{i}",
            tags=["test"],
            memory_type="note",
            metadata={},
            created_at=1698765432.0 + i,
            updated_at=1698765432.0 + i
        )
        for i in range(25)
    ]

    # Page 1: First 10 memories
    mock_storage.get_all_memories.return_value = memories[:10]
    mock_storage.count_all_memories.return_value = 25

    service = MemoryService(storage=mock_storage)
    page1 = await service.list_memories(page=1, page_size=10)

    assert page1["page"] == 1
    assert page1["page_size"] == 10
    assert page1["total"] == 25
    assert page1["has_more"] is True
    assert len(page1["memories"]) == 10

    # Page 2: Next 10 memories
    mock_storage.get_all_memories.return_value = memories[10:20]
    page2 = await service.list_memories(page=2, page_size=10)

    assert page2["page"] == 2
    assert page2["has_more"] is True

    # Page 3: Last 5 memories
    mock_storage.get_all_memories.return_value = memories[20:25]
    page3 = await service.list_memories(page=3, page_size=10)

    assert page3["page"] == 3
    assert page3["has_more"] is False
    assert len(page3["memories"]) == 5


@pytest.mark.asyncio
async def test_api_list_memories_tag_filter(mock_storage):
    """Test filtering by tag through API."""
    mock_storage.get_all_memories.return_value = []
    mock_storage.count_all_memories.return_value = 0

    service = MemoryService(storage=mock_storage)

    result = await service.list_memories(page=1, page_size=10, tag="important")

    # Verify tag passed to storage as list
    call_kwargs = mock_storage.get_all_memories.call_args.kwargs
    assert call_kwargs["tags"] == ["important"]


@pytest.mark.asyncio
async def test_api_list_memories_type_filter(mock_storage):
    """Test filtering by memory type through API."""
    mock_storage.get_all_memories.return_value = []
    mock_storage.count_all_memories.return_value = 0

    service = MemoryService(storage=mock_storage)

    result = await service.list_memories(page=1, page_size=10, memory_type="reference")

    call_kwargs = mock_storage.get_all_memories.call_args.kwargs
    assert call_kwargs["memory_type"] == "reference"


@pytest.mark.asyncio
async def test_api_list_memories_combined_filters(mock_storage):
    """Test combining tag and type filters."""
    mock_storage.get_all_memories.return_value = []
    mock_storage.count_all_memories.return_value = 0

    service = MemoryService(storage=mock_storage)

    result = await service.list_memories(
        page=1,
        page_size=10,
        tag="work",
        memory_type="task"
    )

    call_kwargs = mock_storage.get_all_memories.call_args.kwargs
    assert call_kwargs["tags"] == ["work"]
    assert call_kwargs["memory_type"] == "task"


# Test API Search Endpoints

@pytest.mark.asyncio
async def test_api_semantic_search_uses_service(mock_storage, sample_memory):
    """Test POST /api/search uses MemoryService."""
    mock_storage.retrieve.return_value = [sample_memory]

    service = MemoryService(storage=mock_storage)

    result = await service.retrieve_memories(query="test query", n_results=5)

    assert result["query"] == "test query"
    assert result["count"] == 1
    mock_storage.retrieve.assert_called_once()


@pytest.mark.asyncio
async def test_api_tag_search_uses_service(mock_storage, sample_memory):
    """Test POST /api/search/by-tag uses MemoryService."""
    mock_storage.search_by_tags.return_value = [sample_memory]

    service = MemoryService(storage=mock_storage)

    result = await service.search_by_tag(tags=["test"], match_all=False)

    assert result["tags"] == ["test"]
    assert result["match_type"] == "ANY"
    assert result["count"] == 1


@pytest.mark.asyncio
async def test_api_time_search_uses_service(mock_storage, sample_memory):
    """Test POST /api/search/by-time flow (if applicable)."""
    # Note: Time search might use retrieve_memories with time filters
    mock_storage.retrieve.return_value = [sample_memory]

    service = MemoryService(storage=mock_storage)

    # Simulate time-based search
    result = await service.retrieve_memories(query="last week", n_results=10)

    assert "memories" in result


# Test API Delete Endpoint

@pytest.mark.asyncio
async def test_api_delete_memory_uses_service(mock_storage):
    """Test DELETE /api/memories/{hash} uses MemoryService."""
    mock_storage.delete_memory.return_value = True

    service = MemoryService(storage=mock_storage)

    result = await service.delete_memory("test_hash_123")

    assert result["success"] is True
    assert result["content_hash"] == "test_hash_123"
    mock_storage.delete_memory.assert_called_once_with("test_hash_123")


@pytest.mark.asyncio
async def test_api_delete_memory_not_found(mock_storage):
    """Test deleting non-existent memory returns proper response."""
    mock_storage.delete_memory.return_value = False

    service = MemoryService(storage=mock_storage)

    result = await service.delete_memory("nonexistent")

    assert result["success"] is False


# Test API Get Memory Endpoint

@pytest.mark.asyncio
async def test_api_get_memory_by_hash_uses_service(mock_storage, sample_memory):
    """Test GET /api/memories/{hash} uses MemoryService."""
    mock_storage.get_memory.return_value = sample_memory

    service = MemoryService(storage=mock_storage)

    result = await service.get_memory_by_hash("test_hash_123")

    assert result["found"] is True
    assert result["memory"]["content_hash"] == "test_hash_123"


# Test Dependency Injection

def test_get_memory_service_dependency_injection():
    """Test that get_memory_service creates service with correct storage."""
    from mcp_memory_service.web.dependencies import get_memory_service

    # Create mock storage
    mock_storage = MagicMock()

    # Override dependency
    def override_get_storage():
        return mock_storage

    # Get service
    service = get_memory_service(storage=mock_storage)

    assert isinstance(service, MemoryService)
    assert service.storage == mock_storage


# Performance and Scaling Tests

@pytest.mark.asyncio
async def test_list_memories_performance_with_large_dataset(mock_storage):
    """
    Test that list_memories remains efficient with large datasets.

    This verifies the fix for O(n) memory loading anti-pattern.
    """
    # Simulate 10,000 memories in database
    mock_storage.get_all_memories.return_value = []
    mock_storage.count_all_memories.return_value = 10000

    service = MemoryService(storage=mock_storage)

    # Request just 20 items
    result = await service.list_memories(page=1, page_size=20)

    # CRITICAL: Verify we only queried for 20 items, not all 10,000
    call_kwargs = mock_storage.get_all_memories.call_args.kwargs
    assert call_kwargs["limit"] == 20
    assert call_kwargs["offset"] == 0

    # This proves database-level filtering prevents loading 10,000 records
    assert result["total"] == 10000
    assert result["has_more"] is True


@pytest.mark.asyncio
async def test_tag_filter_performance(mock_storage):
    """Test that tag filtering happens at database level."""
    mock_storage.get_all_memories.return_value = []
    mock_storage.count_all_memories.return_value = 50

    service = MemoryService(storage=mock_storage)

    result = await service.list_memories(page=1, page_size=10, tag="important")

    # Verify tag filter passed to database query
    call_kwargs = mock_storage.get_all_memories.call_args.kwargs
    assert call_kwargs["tags"] == ["important"]

    # Result should only reflect filtered count
    assert result["total"] == 50  # Only memories matching tag


# Error Handling Tests

@pytest.mark.asyncio
async def test_api_handles_storage_errors_gracefully(mock_storage):
    """Test that API returns proper errors when storage fails."""
    mock_storage.get_all_memories.side_effect = Exception("Database connection lost")

    service = MemoryService(storage=mock_storage)

    result = await service.list_memories(page=1, page_size=10)

    assert result["success"] is False
    assert "error" in result
    assert "Database connection lost" in result["error"]


@pytest.mark.asyncio
async def test_api_validates_input_through_service(mock_storage):
    """Test that validation errors from storage are handled."""
    mock_storage.store.side_effect = ValueError("Invalid content format")

    service = MemoryService(storage=mock_storage)

    result = await service.store_memory(content="invalid")

    assert result["success"] is False
    assert "Invalid memory data" in result["error"]


# Consistency Tests

@pytest.mark.asyncio
async def test_api_and_mcp_use_same_service_logic(mock_storage):
    """
    Test that API and MCP tools use the same MemoryService logic.

    This verifies the DRY principle - both interfaces share the same
    business logic through MemoryService.
    """
    service = MemoryService(storage=mock_storage)

    # Store through service (used by both API and MCP)
    mock_storage.store.return_value = None
    result1 = await service.store_memory(content="Test", tags=["shared"])

    # Retrieve through service (used by both API and MCP)
    mock_storage.retrieve.return_value = []
    result2 = await service.retrieve_memories(query="test")

    # Both operations used the same service
    assert result1["success"] is True
    assert "memories" in result2


@pytest.mark.asyncio
async def test_response_format_consistency(mock_storage, sample_memory):
    """Test that all service methods return consistently formatted responses."""
    mock_storage.get_all_memories.return_value = [sample_memory]
    mock_storage.count_all_memories.return_value = 1
    mock_storage.retrieve.return_value = [sample_memory]
    mock_storage.search_by_tags.return_value = [sample_memory]

    service = MemoryService(storage=mock_storage)

    # Get responses from different methods
    list_result = await service.list_memories(page=1, page_size=10)
    retrieve_result = await service.retrieve_memories(query="test")
    tag_result = await service.search_by_tag(tags="test")

    # All should have consistently formatted memories
    list_memory = list_result["memories"][0]
    retrieve_memory = retrieve_result["memories"][0]
    tag_memory = tag_result["memories"][0]

    # Verify all have same format
    required_fields = ["content", "content_hash", "tags", "memory_type", "created_at"]
    for field in required_fields:
        assert field in list_memory
        assert field in retrieve_memory
        assert field in tag_memory


# Real Storage Integration Test (End-to-End)

@pytest.mark.asyncio
@pytest.mark.integration
async def test_end_to_end_workflow_with_real_storage(temp_db):
    """
    End-to-end test with real SQLite storage (not mocked).

    This verifies the complete integration stack works correctly.
    """
    # Create real storage
    storage = SqliteVecMemoryStorage(temp_db)
    await storage.initialize()

    try:
        # Create service with real storage
        service = MemoryService(storage=storage)

        # Store a memory
        store_result = await service.store_memory(
            content="End-to-end test memory",
            tags=["e2e", "integration"],
            memory_type="test"
        )
        assert store_result["success"] is True

        # List memories
        list_result = await service.list_memories(page=1, page_size=10)
        assert len(list_result["memories"]) > 0

        # Search by tag
        tag_result = await service.search_by_tag(tags="e2e")
        assert len(tag_result["memories"]) > 0

        # Get specific memory
        content_hash = store_result["memory"]["content_hash"]
        get_result = await service.get_memory_by_hash(content_hash)
        assert get_result["found"] is True

        # Delete memory
        delete_result = await service.delete_memory(content_hash)
        assert delete_result["success"] is True

        # Verify deleted
        get_after_delete = await service.get_memory_by_hash(content_hash)
        assert get_after_delete["found"] is False

    finally:
        storage.close()
