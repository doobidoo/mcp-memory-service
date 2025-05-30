"""
Test Vector Store Client for EchoVault Memory Service
Copyright (c) 2025 EchoVault
Licensed under the MIT License.
"""

import os
import json
import pytest
import asyncio
from typing import Dict, Any, List

# Import the VectorStoreClient
from src.mcp_memory_service.storage.vector_store import VectorStoreClient

# Set test environment variables
os.environ["NEON_DSN"] = os.environ.get("NEON_DSN", "postgresql://postgres:postgres@localhost:5432/postgres")
os.environ["USE_QDRANT"] = "false"  # Disable Qdrant for basic tests

@pytest.fixture
async def vector_store():
    """Create and initialize a VectorStoreClient instance for testing."""
    client = VectorStoreClient()
    await client.initialize()
    
    try:
        yield client
    finally:
        # Cleanup after tests
        await client.close()

@pytest.mark.asyncio
async def test_initialization():
    """Test basic initialization of the vector store client."""
    client = VectorStoreClient()
    await client.initialize()
    
    # Verify initialization
    assert client._is_initialized
    assert client.neon_client is not None
    
    # Cleanup
    await client.close()

@pytest.mark.asyncio
async def test_upsert(vector_store):
    """Test upserting vectors to the store."""
    # Create test data
    test_id = f"test_upsert_{asyncio.get_event_loop().time()}"
    content = "Test vector content"
    embedding = [0.1, 0.2, 0.3] * 512  # 1536 dimensions
    metadata = {
        "content_hash": test_id,
        "memory_type": "test",
        "tags": ["test", "vector", "upsert"],
        "timestamp": int(asyncio.get_event_loop().time())
    }
    
    # Upsert vector
    success = await vector_store.upsert(
        id=test_id,
        content=content,
        embedding=embedding,
        metadata=metadata
    )
    
    # Verify success
    assert success is True
    
    # Verify it was stored correctly
    async with vector_store.neon_client.pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM memories WHERE content_hash = $1", test_id)
        
        assert row is not None
        assert row["content"] == content
        
        # Clean up
        await conn.execute("DELETE FROM memories WHERE content_hash = $1", test_id)

@pytest.mark.asyncio
async def test_search(vector_store):
    """Test vector search functionality."""
    # Create and insert test vectors
    test_vectors = []
    for i in range(3):
        test_id = f"test_search_{i}_{asyncio.get_event_loop().time()}"
        content = f"Test search content {i}"
        
        # Create embeddings with varying similarity to search query
        if i == 0:
            # This one should be the most similar
            embedding = [0.9, 0.8, 0.7] * 512
        elif i == 1:
            # This one should be second most similar
            embedding = [0.7, 0.6, 0.5] * 512
        else:
            # This one should be the least similar
            embedding = [0.1, 0.2, 0.3] * 512
        
        metadata = {
            "content_hash": test_id,
            "memory_type": "test",
            "tags": ["test", "vector", "search"],
            "timestamp": int(asyncio.get_event_loop().time())
        }
        
        # Upsert vector
        await vector_store.upsert(
            id=test_id,
            content=content,
            embedding=embedding,
            metadata=metadata
        )
        
        test_vectors.append(test_id)
    
    # Define search query (most similar to first vector)
    search_embedding = [0.85, 0.75, 0.65] * 512
    
    # Search for similar vectors
    results = await vector_store.search(
        embedding=search_embedding,
        limit=3,
        similarity_threshold=0.0
    )
    
    # Verify results
    assert len(results) == 3
    
    # Results should be in order of similarity
    assert results[0]["id"] == test_vectors[0]
    assert results[1]["id"] == test_vectors[1]
    assert results[2]["id"] == test_vectors[2]
    
    # Clean up
    for test_id in test_vectors:
        await vector_store.delete(test_id)

@pytest.mark.asyncio
async def test_delete(vector_store):
    """Test deleting vectors from the store."""
    # Create test vector
    test_id = f"test_delete_{asyncio.get_event_loop().time()}"
    content = "Test delete content"
    embedding = [0.4, 0.5, 0.6] * 512  # 1536 dimensions
    metadata = {
        "content_hash": test_id,
        "memory_type": "test",
        "tags": ["test", "vector", "delete"],
        "timestamp": int(asyncio.get_event_loop().time())
    }
    
    # Upsert vector
    await vector_store.upsert(
        id=test_id,
        content=content,
        embedding=embedding,
        metadata=metadata
    )
    
    # Verify it exists
    async with vector_store.neon_client.pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM memories WHERE content_hash = $1", test_id)
        assert row is not None
    
    # Delete vector
    success = await vector_store.delete(test_id)
    assert success is True
    
    # Verify it's gone
    async with vector_store.neon_client.pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM memories WHERE content_hash = $1", test_id)
        assert row is None

@pytest.mark.asyncio
async def test_delete_by_tag(vector_store):
    """Test deleting vectors by tag."""
    # Create unique tag for this test
    unique_tag = f"unique_delete_tag_{asyncio.get_event_loop().time()}"
    test_vectors = []
    
    # Create multiple test vectors with the unique tag
    for i in range(3):
        test_id = f"test_delete_tag_{i}_{asyncio.get_event_loop().time()}"
        test_vectors.append(test_id)
        
        content = f"Test delete by tag content {i}"
        embedding = [0.3, 0.4, 0.5] * 512  # 1536 dimensions
        metadata = {
            "content_hash": test_id,
            "memory_type": "test",
            "tags": ["test", "vector", unique_tag],
            "timestamp": int(asyncio.get_event_loop().time())
        }
        
        # Upsert vector
        await vector_store.upsert(
            id=test_id,
            content=content,
            embedding=embedding,
            metadata=metadata
        )
    
    # Delete by tag
    count = await vector_store.delete_by_tag(unique_tag)
    
    # Verify correct number of vectors were deleted
    assert count == 3
    
    # Verify they're gone
    async with vector_store.neon_client.pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM memories WHERE content_hash = ANY($1::text[])", test_vectors)
        assert len(rows) == 0

@pytest.mark.asyncio
async def test_get_stats(vector_store):
    """Test getting vector store statistics."""
    # Insert some test vectors
    test_vectors = []
    for i in range(2):
        test_id = f"test_stats_{i}_{asyncio.get_event_loop().time()}"
        test_vectors.append(test_id)
        
        content = f"Test stats content {i}"
        embedding = [0.2, 0.3, 0.4] * 512  # 1536 dimensions
        metadata = {
            "content_hash": test_id,
            "memory_type": "test",
            "tags": ["test", "vector", "stats"],
            "timestamp": int(asyncio.get_event_loop().time())
        }
        
        # Upsert vector
        await vector_store.upsert(
            id=test_id,
            content=content,
            embedding=embedding,
            metadata=metadata
        )
    
    # Get stats
    stats = await vector_store.get_stats()
    
    # Verify stats
    assert "vector_count" in stats
    assert stats["vector_count"] > 0
    assert "providers" in stats
    assert len(stats["providers"]) > 0
    
    # First provider should be neon_pgvector
    assert stats["providers"][0]["name"] == "neon_pgvector"
    
    # Clean up
    for test_id in test_vectors:
        await vector_store.delete(test_id)

# Optional test for Qdrant functionality
@pytest.mark.skipif(not os.environ.get("USE_QDRANT", "").lower() == "true", 
                  reason="Qdrant is not enabled")
@pytest.mark.asyncio
async def test_qdrant_integration():
    """Test Qdrant integration when enabled."""
    # Set Qdrant environment variables for this test
    os.environ["USE_QDRANT"] = "true"
    
    # Create client
    client = VectorStoreClient()
    await client.initialize()
    
    try:
        # Verify Qdrant is enabled
        assert client.use_qdrant
        assert client.qdrant_client is not None
        
        # Test basic operations
        test_id = f"test_qdrant_{asyncio.get_event_loop().time()}"
        content = "Test Qdrant integration"
        embedding = [0.5, 0.5, 0.5] * 512  # 1536 dimensions
        metadata = {
            "content_hash": test_id,
            "memory_type": "test",
            "tags": ["test", "qdrant"],
            "timestamp": int(asyncio.get_event_loop().time())
        }
        
        # Upsert vector
        success = await client.upsert(
            id=test_id,
            content=content,
            embedding=embedding,
            metadata=metadata
        )
        
        assert success is True
        
        # Search for the vector
        results = await client.search(
            embedding=embedding,
            limit=1,
            similarity_threshold=0.9
        )
        
        assert len(results) == 1
        assert results[0]["id"] == test_id
        
        # Delete vector
        success = await client.delete(test_id)
        assert success is True
    finally:
        # Cleanup and reset environment
        await client.close()
        os.environ["USE_QDRANT"] = "false"