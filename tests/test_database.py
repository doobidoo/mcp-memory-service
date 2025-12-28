"""
MCP Memory Service
Copyright (c) 2024 Heinrich Krupp
Licensed under the MIT License. See LICENSE file in the project root for full license text.
"""
"""
Test database operations of the MCP Memory Service.
"""
import pytest
import pytest_asyncio
import asyncio
import os
from mcp_memory_service.server import MemoryServer

@pytest_asyncio.fixture
async def memory_server():
    """Create a test instance of the memory server."""
    server = MemoryServer()
    # MemoryServer initializes itself, no initialize() call needed
    yield server
    # No cleanup needed - MemoryServer doesn't have shutdown()

@pytest.mark.asyncio
async def test_create_backup(memory_server):
    """Test database backup creation."""
    # Store some test data
    await memory_server.store_memory(
        content="Test memory for backup"
    )
    
    # Create backup
    backup_response = await memory_server.create_backup()
    
    assert backup_response.get("success") is True
    assert backup_response.get("backup_path") is not None
    assert os.path.exists(backup_response.get("backup_path"))

@pytest.mark.asyncio
async def test_database_health(memory_server):
    """Test database health check functionality."""
    health_status = await memory_server.check_database_health()
    
    assert health_status is not None
    assert "status" in health_status
    assert "memory_count" in health_status
    assert "database_size" in health_status

@pytest.mark.asyncio
async def test_optimize_database(memory_server):
    """Test database optimization."""
    # Store multiple memories to trigger optimization
    for i in range(10):
        await memory_server.store_memory(
            content=f"Test memory {i}"
        )
    
    # Run optimization
    optimize_response = await memory_server.optimize_db()
    
    assert optimize_response.get("success") is True
    assert "optimized_size" in optimize_response

@pytest.mark.asyncio
async def test_cleanup_duplicates(memory_server):
    """Test duplicate memory cleanup."""
    # Store duplicate memories
    duplicate_content = "This is a duplicate memory"
    await memory_server.store_memory(content=duplicate_content)
    await memory_server.store_memory(content=duplicate_content)
    
    # Clean up duplicates
    cleanup_response = await memory_server.cleanup_duplicates()
    
    assert cleanup_response.get("success") is True
    assert cleanup_response.get("duplicates_removed") >= 1
    
    # Verify only one copy remains
    results = await memory_server.exact_match_retrieve(
        content=duplicate_content
    )
    assert len(results) == 1

@pytest.mark.asyncio
async def test_database_persistence(memory_server):
    """Test database persistence across server restarts."""
    test_content = "Persistent memory test"

    # Store memory
    await memory_server.store_memory(content=test_content)

    # Simulate server restart
    await memory_server.shutdown()
    # Create new server instance to simulate restart
    new_server = MemoryServer()

    # Verify memory persists
    results = await new_server.exact_match_retrieve(
        content=test_content
    )
    assert len(results) == 1
    assert results[0] == test_content

    # No cleanup needed