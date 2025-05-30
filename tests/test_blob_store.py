"""
Test Blob Storage Client for EchoVault Memory Service
Copyright (c) 2025 EchoVault
Licensed under the MIT License.
"""

import os
import pytest
import asyncio
import uuid
from typing import Dict, Any, List

# Import the BlobStoreClient
from src.mcp_memory_service.storage.blob_store import BlobStoreClient

# This test requires R2 credentials to be set
pytestmark = pytest.mark.skipif(
    not all([
        os.environ.get("R2_ENDPOINT"),
        os.environ.get("R2_ACCESS_KEY_ID"),
        os.environ.get("R2_SECRET_ACCESS_KEY"),
        os.environ.get("R2_BUCKET")
    ]),
    reason="R2 credentials not set"
)

@pytest.fixture
async def blob_store():
    """Create and initialize a BlobStoreClient instance for testing."""
    client = BlobStoreClient()
    await client.initialize()
    
    # Set a smaller threshold for testing
    client.blob_threshold = 100  # 100 bytes for testing
    
    yield client

@pytest.mark.asyncio
async def test_is_configured():
    """Test checking if blob store is configured."""
    client = BlobStoreClient()
    await client.initialize()
    
    # Check if configured based on environment
    if all([
        os.environ.get("R2_ENDPOINT"),
        os.environ.get("R2_ACCESS_KEY_ID"),
        os.environ.get("R2_SECRET_ACCESS_KEY"),
        os.environ.get("R2_BUCKET")
    ]):
        assert client.is_configured() is True
    else:
        assert client.is_configured() is False

@pytest.mark.asyncio
async def test_save_if_large(blob_store):
    """Test saving content to blob storage if it exceeds the threshold."""
    # Generate unique content hash
    content_hash = str(uuid.uuid4())
    
    # Small content - should not be saved to blob storage
    small_content = "Small content"
    small_result, small_url = await blob_store.save_if_large(small_content, content_hash)
    
    # Should return original content and no URL
    assert small_result == small_content
    assert small_url is None
    
    # Large content - should be saved to blob storage
    large_content = "x" * 1000  # 1000 characters, well above our test threshold
    large_result, large_url = await blob_store.save_if_large(large_content, content_hash + "_large")
    
    # Should return a preview and a URL
    assert len(large_result) < len(large_content)
    assert "..." in large_result
    assert large_url is not None
    assert "memories/" in large_url
    assert content_hash + "_large" in large_url
    
    # Clean up
    if large_url:
        await blob_store.delete_blob(large_url)

@pytest.mark.asyncio
async def test_generate_presigned_url(blob_store):
    """Test generating a presigned URL for a blob."""
    # First save a blob
    content_hash = str(uuid.uuid4())
    large_content = "x" * 1000
    _, blob_key = await blob_store.save_if_large(large_content, content_hash)
    
    assert blob_key is not None
    
    # Generate presigned URL
    url = await blob_store.generate_presigned_url(blob_key)
    
    # Verify URL
    assert url is not None
    assert "https://" in url
    assert os.environ.get("R2_BUCKET", "") in url
    
    # Clean up
    await blob_store.delete_blob(blob_key)

@pytest.mark.asyncio
async def test_retrieve_content(blob_store):
    """Test retrieving content from blob storage."""
    # First save a blob
    content_hash = str(uuid.uuid4())
    original_content = "This is a test content that will be stored in blob storage and then retrieved."
    original_content = original_content * 10  # Make it large enough
    
    _, blob_key = await blob_store.save_if_large(original_content, content_hash)
    
    assert blob_key is not None
    
    # Retrieve content
    retrieved_content = await blob_store.retrieve_content(blob_key)
    
    # Verify content
    assert retrieved_content is not None
    assert retrieved_content == original_content
    
    # Clean up
    await blob_store.delete_blob(blob_key)

@pytest.mark.asyncio
async def test_delete_blob(blob_store):
    """Test deleting a blob from storage."""
    # First save a blob
    content_hash = str(uuid.uuid4())
    content = "This is content that will be deleted." * 20
    
    _, blob_key = await blob_store.save_if_large(content, content_hash)
    
    assert blob_key is not None
    
    # Delete blob
    success = await blob_store.delete_blob(blob_key)
    
    # Verify deletion
    assert success is True
    
    # Try to retrieve it - should fail
    retrieved = await blob_store.retrieve_content(blob_key)
    assert retrieved is None

@pytest.mark.asyncio
async def test_batch_delete_blobs(blob_store):
    """Test deleting multiple blobs in a batch."""
    # Create multiple blobs
    keys = []
    for i in range(3):
        content_hash = f"{str(uuid.uuid4())}_{i}"
        content = f"Batch delete test content {i}" * 50
        
        _, blob_key = await blob_store.save_if_large(content, content_hash)
        assert blob_key is not None
        keys.append(blob_key)
    
    # Delete in batch
    success = await blob_store.batch_delete_blobs(keys)
    
    # Verify batch deletion
    assert success is True
    
    # Verify all blobs are gone
    for key in keys:
        retrieved = await blob_store.retrieve_content(key)
        assert retrieved is None

@pytest.mark.asyncio
async def test_content_threshold(blob_store):
    """Test the content threshold functionality."""
    # Adjust threshold for test
    original_threshold = blob_store.blob_threshold
    blob_store.blob_threshold = 200
    
    content_hash = str(uuid.uuid4())
    
    try:
        # Content just below threshold
        below_content = "x" * 199
        below_result, below_url = await blob_store.save_if_large(below_content, content_hash + "_below")
        
        # Should not be stored in blob storage
        assert below_result == below_content
        assert below_url is None
        
        # Content just above threshold
        above_content = "x" * 201
        above_result, above_url = await blob_store.save_if_large(above_content, content_hash + "_above")
        
        # Should be stored in blob storage
        assert len(above_result) < len(above_content)
        assert above_url is not None
        
        # Clean up
        if above_url:
            await blob_store.delete_blob(above_url)
    finally:
        # Restore original threshold
        blob_store.blob_threshold = original_threshold