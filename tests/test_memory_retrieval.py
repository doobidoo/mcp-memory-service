import pytest
import asyncio
from datetime import datetime, timedelta
import json
from typing import List, Dict, Any, Optional

from memory_service.storage import ChromaMemoryStorage
from memory_service.models import Memory, MemoryQueryResult

@pytest.fixture(scope="function")
async def storage():
    """Creates a fresh ChromaMemoryStorage instance for each test."""
    storage = ChromaMemoryStorage(path="test_chroma_db")
    yield storage
    await storage.cleanup()

@pytest.fixture
async def sample_memories(storage: ChromaMemoryStorage):
    """Creates a set of test memories with various tags, types, and timestamps."""
    memories = [
        {
            "content": "Important project meeting tomorrow",
            "tags": ["meeting", "project", "important"],
            "type": "calendar",
            "timestamp": datetime.now().timestamp()
        },
        {
            "content": "Technical documentation for API v2",
            "tags": ["technical", "documentation", "api"],
            "type": "document",
            "timestamp": (datetime.now() - timedelta(days=2)).timestamp()
        },
        {
            "content": "Bug fix for memory leak issue",
            "tags": ["bug", "technical", "urgent"],
            "type": "task",
            "timestamp": (datetime.now() - timedelta(days=1)).timestamp()
        }
    ]
    
    for mem_data in memories:
        memory = Memory(
            content=mem_data["content"],
            content_hash=f"hash_{mem_data['content']}",  # Simplified for testing
            tags=mem_data["tags"],
            memory_type=mem_data["type"],
            timestamp=mem_data["timestamp"]
        )
        success, _ = await storage.store(memory)
        assert success
    
    return memories

@pytest.mark.asyncio
async def test_semantic_search(storage: ChromaMemoryStorage, sample_memories):
    """Test semantic search functionality."""
    # Test query related to meetings
    results = await storage.retrieve_memory("meeting schedule", n_results=2)
    assert len(results) > 0
    assert any("meeting" in result.memory.content.lower() for result in results)
    
    # Test query related to technical content
    results = await storage.retrieve_memory("technical documentation", n_results=2)
    assert len(results) > 0
    assert any("technical" in result.memory.content.lower() for result in results)

@pytest.mark.asyncio
async def test_tag_based_retrieval(storage: ChromaMemoryStorage, sample_memories):
    """Test retrieval by tags."""
    # Test single tag
    results = await storage.search_by_tag(["technical"])
    assert len(results) == 2
    assert all("technical" in memory.tags for memory in results)
    
    # Test multiple tags
    results = await storage.search_by_tag(["urgent", "bug"])
    assert len(results) == 1
    assert "bug" in results[0].tags
    assert "urgent" in results[0].tags

@pytest.mark.asyncio
async def test_time_based_retrieval(storage: ChromaMemoryStorage, sample_memories):
    """Test retrieval based on time ranges."""
    now = datetime.now()
    
    # Test recent memories (last 24 hours)
    results = await storage.recall(
        start_timestamp=(now - timedelta(days=1)).timestamp(),
        end_timestamp=now.timestamp()
    )
    assert len(results) == 2
    
    # Test older memories
    results = await storage.recall(
        start_timestamp=(now - timedelta(days=3)).timestamp(),
        end_timestamp=(now - timedelta(days=1)).timestamp()
    )
    assert len(results) == 1

@pytest.mark.asyncio
async def test_combined_retrieval(storage: ChromaMemoryStorage, sample_memories):
    """Test combining different retrieval methods."""
    now = datetime.now()
    
    # Test semantic search within a time range
    results = await storage.retrieve_memory(
        "technical",
        start_timestamp=(now - timedelta(days=2)).timestamp(),
        end_timestamp=now.timestamp(),
        n_results=5
    )
    assert len(results) > 0
    assert all(
        result.memory.timestamp >= (now - timedelta(days=2)).timestamp()
        for result in results
    )

@pytest.mark.asyncio
async def test_exact_match_retrieve(storage: ChromaMemoryStorage, sample_memories):
    """Test exact content match retrieval."""
    exact_content = "Technical documentation for API v2"
    results = await storage.exact_match_retrieve(exact_content)
    assert len(results) == 1
    assert results[0].memory.content == exact_content

@pytest.mark.asyncio
async def test_debug_retrieve(storage: ChromaMemoryStorage, sample_memories):
    """Test debug retrieval with similarity scores."""
    results = await storage.debug_retrieve(
        query="technical documentation",
        n_results=2,
        similarity_threshold=0.5
    )
    assert len(results) > 0
    assert all(hasattr(result, 'similarity') for result in results)
    assert all(result.similarity >= 0.5 for result in results)

if __name__ == "__main__":
    pytest.main(["-v", __file__])
