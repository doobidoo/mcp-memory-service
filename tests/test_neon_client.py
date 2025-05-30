"""
Test Neon PostgreSQL Client for EchoVault Memory Service
Copyright (c) 2025 EchoVault
Licensed under the MIT License.
"""

import os
import json
import pytest
import asyncio
from typing import Dict, Any, List

# Import the NeonClient
from src.mcp_memory_service.storage.neon_client import NeonClient

# Set test environment variables
os.environ["NEON_DSN"] = os.environ.get("NEON_DSN", "postgresql://postgres:postgres@localhost:5432/postgres")
os.environ["NEON_POOL_SIZE"] = "2"

@pytest.fixture
async def neon_client():
    """Create and initialize a NeonClient instance for testing."""
    client = NeonClient()
    await client.initialize()
    
    try:
        yield client
    finally:
        # Cleanup after tests
        await client.close()

@pytest.mark.asyncio
async def test_connection():
    """Test basic connection to Neon PostgreSQL."""
    client = NeonClient()
    try:
        await client.initialize()
        assert client.pool is not None
        assert client._is_initialized
    finally:
        await client.close()

@pytest.mark.asyncio
async def test_insert_event(neon_client):
    """Test inserting an event into the database."""
    # Create test data
    content = "Test memory content"
    content_hash = "test_hash_12345"
    embedding = [0.1, 0.2, 0.3] * 512  # 1536 dimensions
    tags = ["test", "memory", "postgres"]
    memory_type = "test"
    metadata = {"test_key": "test_value"}
    timestamp = int(asyncio.get_event_loop().time())
    
    # Insert event
    success = await neon_client.insert_event(
        content=content,
        content_hash=content_hash,
        embedding=embedding,
        tags=tags,
        memory_type=memory_type,
        metadata=metadata,
        timestamp=timestamp
    )
    
    # Verify success
    assert success is True
    
    # Verify event was inserted correctly
    async with neon_client.pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM memories WHERE content_hash = $1", content_hash)
        
        assert row is not None
        assert row["content"] == content
        assert row["memory_type"] == memory_type
        assert row["timestamp"] == timestamp
        
        # Clean up
        await conn.execute("DELETE FROM memories WHERE content_hash = $1", content_hash)

@pytest.mark.asyncio
async def test_search_by_vector(neon_client):
    """Test vector search functionality."""
    # Create test data
    content = "This is a vector search test"
    content_hash = "vector_search_test_hash"
    embedding = [0.2, 0.3, 0.5] * 512  # 1536 dimensions
    tags = ["vector", "search", "test"]
    memory_type = "vector_test"
    timestamp = int(asyncio.get_event_loop().time())
    
    # Insert test data
    await neon_client.insert_event(
        content=content,
        content_hash=content_hash,
        embedding=embedding,
        tags=tags,
        memory_type=memory_type,
        timestamp=timestamp
    )
    
    # Perform vector search
    results = await neon_client.search_by_vector(
        embedding=embedding,
        limit=5,
        similarity_threshold=0.5
    )
    
    # Verify results
    assert len(results) > 0
    assert any(r["content_hash"] == content_hash for r in results)
    
    # Clean up
    async with neon_client.pool.acquire() as conn:
        await conn.execute("DELETE FROM memories WHERE content_hash = $1", content_hash)

@pytest.mark.asyncio
async def test_search_by_tags(neon_client):
    """Test tag search functionality."""
    # Create test data
    content = "This is a tag search test"
    content_hash = "tag_search_test_hash"
    embedding = [0.4, 0.2, 0.6] * 512  # 1536 dimensions
    tags = ["unique_tag_123", "test"]
    memory_type = "tag_test"
    timestamp = int(asyncio.get_event_loop().time())
    
    # Insert test data
    await neon_client.insert_event(
        content=content,
        content_hash=content_hash,
        embedding=embedding,
        tags=tags,
        memory_type=memory_type,
        timestamp=timestamp
    )
    
    # Search by tag
    results = await neon_client.search_by_tags(tags=["unique_tag_123"])
    
    # Verify results
    assert len(results) > 0
    assert results[0]["content"] == content
    
    # Clean up
    async with neon_client.pool.acquire() as conn:
        await conn.execute("DELETE FROM memories WHERE content_hash = $1", content_hash)

@pytest.mark.asyncio
async def test_delete_memory(neon_client):
    """Test deleting a memory by content hash."""
    # Create test data
    content = "This is a delete test"
    content_hash = "delete_test_hash"
    embedding = [0.1, 0.5, 0.8] * 512  # 1536 dimensions
    tags = ["delete", "test"]
    memory_type = "delete_test"
    timestamp = int(asyncio.get_event_loop().time())
    
    # Insert test data
    await neon_client.insert_event(
        content=content,
        content_hash=content_hash,
        embedding=embedding,
        tags=tags,
        memory_type=memory_type,
        timestamp=timestamp
    )
    
    # Verify it exists
    async with neon_client.pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM memories WHERE content_hash = $1", content_hash)
        assert row is not None
    
    # Delete memory
    success = await neon_client.delete_memory(content_hash)
    assert success is True
    
    # Verify it's gone
    async with neon_client.pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM memories WHERE content_hash = $1", content_hash)
        assert row is None

@pytest.mark.asyncio
async def test_delete_by_tag(neon_client):
    """Test deleting memories by tag."""
    # Create multiple test memories with a unique tag
    unique_tag = "unique_delete_tag_456"
    test_hashes = []
    
    for i in range(3):
        content_hash = f"delete_tag_test_hash_{i}"
        test_hashes.append(content_hash)
        
        await neon_client.insert_event(
            content=f"Delete tag test content {i}",
            content_hash=content_hash,
            embedding=[0.1, 0.2, 0.3] * 512,
            tags=["test", unique_tag],
            memory_type="delete_tag_test",
            timestamp=int(asyncio.get_event_loop().time())
        )
    
    # Delete by tag
    count = await neon_client.delete_by_tag(unique_tag)
    
    # Verify correct number of memories were deleted
    assert count == 3
    
    # Verify they're gone
    async with neon_client.pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM memories WHERE content_hash = ANY($1::text[])", test_hashes)
        assert len(rows) == 0

@pytest.mark.asyncio
async def test_search_by_timeframe(neon_client):
    """Test searching memories by timeframe."""
    # Create memories with different timestamps
    base_time = int(asyncio.get_event_loop().time()) - 86400  # 1 day ago
    
    # Memory 1: 1 hour ago
    await neon_client.insert_event(
        content="Recent timeframe test",
        content_hash="timeframe_test_recent",
        embedding=[0.1, 0.2, 0.3] * 512,
        tags=["timeframe", "test"],
        memory_type="timeframe_test",
        timestamp=base_time + 82800  # 23 hours after base_time
    )
    
    # Memory 2: 12 hours ago
    await neon_client.insert_event(
        content="Middle timeframe test",
        content_hash="timeframe_test_middle",
        embedding=[0.1, 0.2, 0.3] * 512,
        tags=["timeframe", "test"],
        memory_type="timeframe_test",
        timestamp=base_time + 43200  # 12 hours after base_time
    )
    
    # Memory 3: 23 hours ago
    await neon_client.insert_event(
        content="Old timeframe test",
        content_hash="timeframe_test_old",
        embedding=[0.1, 0.2, 0.3] * 512,
        tags=["timeframe", "test"],
        memory_type="timeframe_test",
        timestamp=base_time + 3600  # 1 hour after base_time
    )
    
    # Search within last 6 hours
    results = await neon_client.search_by_timeframe(
        start_timestamp=base_time + 64800,  # 18 hours after base_time
        end_timestamp=base_time + 86400,    # 24 hours after base_time
        limit=10
    )
    
    # Verify only the recent memory is found
    assert len(results) == 1
    assert results[0]["content_hash"] == "timeframe_test_recent"
    
    # Search within 6-18 hours ago
    results = await neon_client.search_by_timeframe(
        start_timestamp=base_time + 25200,  # 7 hours after base_time
        end_timestamp=base_time + 64800,    # 18 hours after base_time
        limit=10
    )
    
    # Verify only the middle memory is found
    assert len(results) == 1
    assert results[0]["content_hash"] == "timeframe_test_middle"
    
    # Search for all memories in the last 24 hours
    results = await neon_client.search_by_timeframe(
        start_timestamp=base_time,
        end_timestamp=base_time + 86400,
        limit=10
    )
    
    # Verify all three memories are found
    assert len(results) == 3
    
    # Clean up
    test_hashes = ["timeframe_test_recent", "timeframe_test_middle", "timeframe_test_old"]
    async with neon_client.pool.acquire() as conn:
        await conn.execute("DELETE FROM memories WHERE content_hash = ANY($1::text[])", test_hashes)