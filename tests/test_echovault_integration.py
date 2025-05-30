"""
Integration Tests for EchoVault Memory Service
Copyright (c) 2025 EchoVault
Licensed under the MIT License.
"""

import os
import json
import pytest
import asyncio
import uuid
import time
from typing import Dict, Any, List

# Import the EchoVault storage
from src.mcp_memory_service.storage.echovault import EchoVaultStorage
from src.mcp_memory_service.models.memory import Memory
from src.mcp_memory_service.utils.hashing import generate_content_hash

# Enable EchoVault for tests
os.environ["USE_ECHOVAULT"] = "true"

@pytest.fixture
async def storage():
    """Create and initialize an EchoVaultStorage instance for testing."""
    client = EchoVaultStorage()
    await client.initialize()
    
    yield client
    
    # Cleanup
    if hasattr(client, "neon_client"):
        await client.neon_client.close()

@pytest.mark.asyncio
async def test_initialize():
    """Test initialization of EchoVault storage."""
    storage = EchoVaultStorage()
    await storage.initialize()
    
    assert storage._is_initialized
    assert storage.neon_client is not None
    assert storage.vector_store is not None
    assert storage.blob_store is not None
    
    # Clean up
    await storage.neon_client.close()

@pytest.mark.asyncio
async def test_store_memory(storage):
    """Test storing a memory."""
    # Create a test memory
    content = "This is a test memory for EchoVault integration"
    content_hash = generate_content_hash(content, {"test": True})
    
    memory = Memory(
        content=content,
        content_hash=content_hash,
        tags=["test", "integration"],
        memory_type="test",
        metadata={"test": True}
    )
    
    # Store the memory
    success, message = await storage.store(memory)
    
    # Verify success
    assert success is True
    assert "Successfully stored memory" in message
    
    # Verify stored in Neon
    async with storage.neon_client.pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM memories WHERE content_hash = $1", content_hash)
        
        assert row is not None
        assert row["content"] == content
        assert row["memory_type"] == "test"
        
        # Clean up
        await conn.execute("DELETE FROM memories WHERE content_hash = $1", content_hash)

@pytest.mark.asyncio
async def test_store_large_memory(storage):
    """Test storing a large memory that should be moved to blob storage."""
    # Set a lower threshold temporarily for testing
    original_threshold = storage.blob_store.blob_threshold
    storage.blob_store.blob_threshold = 200  # 200 bytes
    
    try:
        # Skip test if R2 is not configured
        if not storage.blob_store.is_configured():
            pytest.skip("R2 not configured, skipping large memory test")
        
        # Create a large memory
        content = "This is a large memory content for testing blob storage integration. " * 50
        content_hash = generate_content_hash(content, {"test": True})
        
        memory = Memory(
            content=content,
            content_hash=content_hash,
            tags=["test", "large", "blob"],
            memory_type="test_large",
            metadata={"test": True, "large": True}
        )
        
        # Store the memory
        success, message = await storage.store(memory)
        
        # Verify success
        assert success is True
        assert "Successfully stored memory" in message
        
        # Verify stored in Neon with a payload_url
        async with storage.neon_client.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM memories WHERE content_hash = $1", content_hash)
            
            assert row is not None
            assert row["content"] != content  # Should contain a summary, not the full content
            assert row["payload_url"] is not None
            assert "memories/" in row["payload_url"]
            
            # Remember payload_url for cleanup
            payload_url = row["payload_url"]
        
        # Retrieve the memory to verify blob storage
        results = await storage.retrieve(query=content[:100])
        
        # Verify retrieved memory
        assert len(results) > 0
        assert results[0].memory.content == content
        assert results[0].memory.content_hash == content_hash
        
        # Clean up
        async with storage.neon_client.pool.acquire() as conn:
            await conn.execute("DELETE FROM memories WHERE content_hash = $1", content_hash)
        
        # Delete blob
        if storage.blob_store.is_configured() and payload_url:
            await storage.blob_store.delete_blob(payload_url)
    finally:
        # Restore original threshold
        storage.blob_store.blob_threshold = original_threshold

@pytest.mark.asyncio
async def test_retrieve_memory(storage):
    """Test retrieving memories."""
    # Create multiple memories with varying similarity to a query
    base_content = "The quick brown fox jumps over the lazy dog"
    memories = [
        {"content": base_content, "similarity": "high"},
        {"content": "A fast auburn fox leaps above a sleepy canine", "similarity": "medium"},
        {"content": "The lazy cat sleeps on the couch", "similarity": "low"}
    ]
    
    # Store memories
    memory_hashes = []
    for i, memory_data in enumerate(memories):
        content = memory_data["content"]
        content_hash = generate_content_hash(content, {"test": True, "index": i})
        memory_hashes.append(content_hash)
        
        memory = Memory(
            content=content,
            content_hash=content_hash,
            tags=["test", "retrieve", memory_data["similarity"]],
            memory_type="test_retrieve",
            metadata={"test": True, "index": i}
        )
        
        await storage.store(memory)
    
    # Wait a moment for indexing
    await asyncio.sleep(1)
    
    # Retrieve memories
    query = "fox jumps over dog"
    results = await storage.retrieve(query=query, n_results=3)
    
    # Verify results
    assert len(results) == 3
    
    # Results should be in order of decreasing similarity
    assert "fox" in results[0].memory.content.lower()
    assert "dog" in results[0].memory.content.lower() or "canine" in results[0].memory.content.lower()
    
    # Clean up
    for content_hash in memory_hashes:
        await storage.delete(content_hash)

@pytest.mark.asyncio
async def test_search_by_tag(storage):
    """Test searching memories by tags."""
    # Create unique tag for this test
    unique_tag = f"unique_tag_{uuid.uuid4()}"
    
    # Create memories with different tags
    test_memories = []
    for i in range(3):
        content = f"Tag search test memory {i}"
        content_hash = generate_content_hash(content, {"test": True, "index": i})
        
        tags = ["test", "tag_search"]
        if i < 2:  # Add the unique tag to only the first two memories
            tags.append(unique_tag)
        
        memory = Memory(
            content=content,
            content_hash=content_hash,
            tags=tags,
            memory_type="test_tag_search",
            metadata={"test": True, "index": i}
        )
        
        await storage.store(memory)
        test_memories.append(content_hash)
    
    # Search by unique tag
    results = await storage.search_by_tag([unique_tag])
    
    # Verify results
    assert len(results) == 2  # Only 2 memories should have the unique tag
    
    # Clean up
    for content_hash in test_memories:
        await storage.delete(content_hash)

@pytest.mark.asyncio
async def test_delete_memory(storage):
    """Test deleting a memory."""
    # Create a test memory
    content = "This memory will be deleted"
    content_hash = generate_content_hash(content, {"test": True})
    
    memory = Memory(
        content=content,
        content_hash=content_hash,
        tags=["test", "delete"],
        memory_type="test_delete",
        metadata={"test": True}
    )
    
    # Store the memory
    await storage.store(memory)
    
    # Verify it exists
    async with storage.neon_client.pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM memories WHERE content_hash = $1", content_hash)
        assert row is not None
    
    # Delete the memory
    success, message = await storage.delete(content_hash)
    
    # Verify success
    assert success is True
    assert "Successfully deleted memory" in message
    
    # Verify it's gone
    async with storage.neon_client.pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM memories WHERE content_hash = $1", content_hash)
        assert row is None

@pytest.mark.asyncio
async def test_delete_by_tag(storage):
    """Test deleting memories by tag."""
    # Create unique tag for this test
    unique_tag = f"unique_delete_tag_{uuid.uuid4()}"
    
    # Create memories with the unique tag
    test_memories = []
    for i in range(3):
        content = f"Tag delete test memory {i}"
        content_hash = generate_content_hash(content, {"test": True, "index": i})
        
        memory = Memory(
            content=content,
            content_hash=content_hash,
            tags=["test", "delete_by_tag", unique_tag],
            memory_type="test_delete_by_tag",
            metadata={"test": True, "index": i}
        )
        
        await storage.store(memory)
        test_memories.append(content_hash)
    
    # Delete by unique tag
    count, message = await storage.delete_by_tag(unique_tag)
    
    # Verify success
    assert count == 3
    assert "Successfully deleted" in message
    
    # Verify all are gone
    async with storage.neon_client.pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM memories WHERE content_hash = ANY($1::text[])", test_memories)
        assert len(rows) == 0

@pytest.mark.asyncio
async def test_cleanup_duplicates(storage):
    """Test cleaning up duplicate memories."""
    # Create unique content for this test
    unique_content = f"Duplicate test content {uuid.uuid4()}"
    
    # Create two memories with the same content but different hashes
    content_hash1 = generate_content_hash(unique_content, {"test": True, "instance": 1})
    content_hash2 = generate_content_hash(unique_content, {"test": True, "instance": 2})
    
    memory1 = Memory(
        content=unique_content,
        content_hash=content_hash1,
        tags=["test", "duplicate"],
        memory_type="test_duplicate",
        metadata={"test": True, "instance": 1}
    )
    
    memory2 = Memory(
        content=unique_content,
        content_hash=content_hash2,
        tags=["test", "duplicate"],
        memory_type="test_duplicate",
        metadata={"test": True, "instance": 2}
    )
    
    # Store both memories
    await storage.store(memory1)
    await storage.store(memory2)
    
    # Verify both exist
    async with storage.neon_client.pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM memories WHERE content = $1", unique_content)
        assert len(rows) == 2
    
    # Clean up duplicates
    count, message = await storage.cleanup_duplicates()
    
    # Verify success
    assert count > 0
    assert "Successfully removed" in message
    
    # Verify only one remains
    async with storage.neon_client.pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM memories WHERE content = $1", unique_content)
        assert len(rows) == 1
        
        # Clean up remaining memory
        await conn.execute("DELETE FROM memories WHERE content = $1", unique_content)

@pytest.mark.asyncio
async def test_otel_instrumentation():
    """Test OpenTelemetry instrumentation."""
    from src.mcp_memory_service.utils import otel_prom
    
    # Initialize OpenTelemetry
    initialized = otel_prom.initialize("test-echovault")
    
    # Create a test span
    with otel_prom.create_span("test_span", {"test": True}) as span:
        # Do something inside the span
        await asyncio.sleep(0.1)
    
    # Test metrics
    otel_prom.trace_write(content_length=1000, has_payload_url=True, tags_count=3)
    otel_prom.trace_read(latency_ms=50, results_count=5)
    otel_prom.update_stats(memory_count=100, blob_count=20, connection_pool_size=5, connection_pool_available=3)