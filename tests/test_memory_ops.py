"""
MCP Memory Service
Copyright (c) 2024 Heinrich Krupp
Licensed under the MIT License. See LICENSE file in the project root for full license text.
"""
"""
Test core memory operations of the MCP Memory Service.
"""
import pytest
import pytest_asyncio
import asyncio
from mcp_memory_service.server import MemoryServer

@pytest_asyncio.fixture
async def memory_server():
    """Create a test instance of the memory server."""
    server = MemoryServer()
    # MemoryServer initializes itself, no initialize() call needed
    yield server
    # No cleanup needed

@pytest.mark.asyncio
async def test_store_memory(memory_server):
    """Test storing new memory entries."""
    test_content = "The capital of France is Paris"
    test_metadata = {
        "tags": ["geography", "cities", "europe"],
        "type": "fact"
    }
    
    response = await memory_server.store_memory(
        content=test_content,
        metadata=test_metadata
    )
    
    assert response is not None
    # Add more specific assertions based on expected response format

@pytest.mark.asyncio
async def test_retrieve_memory(memory_server):
    """Test retrieving memories using semantic search."""
    # First store some test data
    test_memories = [
        "The capital of France is Paris",
        "London is the capital of England",
        "Berlin is the capital of Germany"
    ]
    
    for memory in test_memories:
        await memory_server.store_memory(content=memory)
    
    # Test retrieval
    query = "What is the capital of France?"
    results = await memory_server.retrieve_memory(
        query=query,
        n_results=1
    )
    
    assert results is not None
    assert len(results) == 1
    assert "Paris" in results[0]  # The most relevant result should mention Paris

@pytest.mark.asyncio
async def test_search_by_tag(memory_server):
    """Test retrieving memories by tags."""
    # Store memory with tags
    await memory_server.store_memory(
        content="Paris is beautiful in spring",
        metadata={"tags": ["travel", "cities", "europe"]}
    )
    
    # Search by tags
    results = await memory_server.search_by_tag(
        tags=["travel", "europe"]
    )
    
    assert results is not None
    assert len(results) > 0
    assert "Paris" in results[0]

@pytest.mark.asyncio
async def test_delete_memory(memory_server):
    """Test deleting specific memories."""
    # Store a memory and get its hash
    content = "Memory to be deleted"
    response = await memory_server.store_memory(content=content)
    content_hash = response.get("hash")
    
    # Delete the memory
    delete_response = await memory_server.delete_memory(
        content_hash=content_hash
    )
    
    assert delete_response.get("success") is True
    
    # Verify memory is deleted
    results = await memory_server.exact_match_retrieve(content=content)
    assert len(results) == 0

@pytest.mark.asyncio
async def test_memory_with_empty_content(memory_server):
    """Test handling of empty or invalid content."""
    with pytest.raises(ValueError):
        await memory_server.store_memory(content="")

@pytest.mark.asyncio
async def test_memory_with_invalid_tags(memory_server):
    """Test handling of invalid tags metadata."""
    with pytest.raises(ValueError):
        await memory_server.store_memory(
            content="Test content",
            metadata={"tags": "invalid"}  # Should be a list
        )