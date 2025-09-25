"""
Test chronological ordering and pagination for the /api/memories endpoint.

Tests verify that the GitHub issue #79 has been properly resolved by ensuring:
1. Memories are returned in chronological order (newest first)
2. Pagination works correctly with chronological ordering
3. All storage backends support the new ordering
"""

import pytest
import asyncio
import time
import tempfile
from datetime import datetime, timedelta
from typing import List, Dict, Any
import os

# Import project modules
# Note: This assumes the project is installed in editable mode with `pip install -e .`
# or PYTHONPATH is configured correctly for the test environment
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage


class TestChronologicalOrdering:
    """Test chronological ordering functionality across all storage backends."""

    async def create_test_memories(self, storage) -> List[Memory]:
        """Create test memories with different timestamps."""
        memories = []
        base_time = time.time() - 3600  # Start 1 hour ago

        # Create 5 test memories with 10-minute intervals
        for i in range(5):
            timestamp = base_time + (i * 600)  # 10-minute intervals
            memory = Memory(
                content=f"Test memory {i + 1}",
                content_hash=f"hash_{i + 1}",
                tags=[f"tag{i + 1}", "test"],
                memory_type="test",
                metadata={"index": i + 1},
                created_at=timestamp,
                updated_at=timestamp
            )
            memories.append(memory)

            success, message = await storage.store(memory)
            assert success, f"Failed to store memory {i + 1}: {message}"

        return memories

    @pytest.mark.asyncio
    async def test_get_all_memories_chronological_order_sqlite(self):
        """Test that get_all_memories returns memories in chronological order (SQLite)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = SqliteVecMemoryStorage(os.path.join(tmp_dir, "test.db"))
            await storage.initialize()

            # Create test memories
            original_memories = await self.create_test_memories(storage)

            # Get all memories
            retrieved_memories = await storage.get_all_memories()

            # Verify we got all memories
            assert len(retrieved_memories) == 5, f"Expected 5 memories, got {len(retrieved_memories)}"

            # Verify chronological order (newest first)
            for i in range(len(retrieved_memories) - 1):
                current_time = retrieved_memories[i].created_at or 0
                next_time = retrieved_memories[i + 1].created_at or 0
                assert current_time >= next_time, f"Memory at index {i} is older than memory at index {i + 1}"

            # Verify the actual order matches expectations (newest first)
            expected_order = [5, 4, 3, 2, 1]  # Newest to oldest
            actual_order = [int(mem.content.split()[-1]) for mem in retrieved_memories]
            assert actual_order == expected_order, f"Expected order {expected_order}, got {actual_order}"

    @pytest.mark.asyncio
    async def test_pagination_with_chronological_order_sqlite(self):
        """Test pagination maintains chronological order (SQLite)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = SqliteVecMemoryStorage(os.path.join(tmp_dir, "test.db"))
            await storage.initialize()

            # Create test memories
            await self.create_test_memories(storage)

            # Test pagination: Get first 2 memories
            first_page = await storage.get_all_memories(limit=2, offset=0)
            assert len(first_page) == 2

            # Test pagination: Get next 2 memories
            second_page = await storage.get_all_memories(limit=2, offset=2)
            assert len(second_page) == 2

            # Test pagination: Get last memory
            third_page = await storage.get_all_memories(limit=2, offset=4)
            assert len(third_page) == 1

            # Verify chronological order across pages
            all_paginated = first_page + second_page + third_page

            # Should be in chronological order (newest first)
            for i in range(len(all_paginated) - 1):
                current_time = all_paginated[i].created_at or 0
                next_time = all_paginated[i + 1].created_at or 0
                assert current_time >= next_time, f"Pagination broke chronological order at position {i}"

            # Verify content order
            expected_content_order = ["Test memory 5", "Test memory 4", "Test memory 3", "Test memory 2", "Test memory 1"]
            actual_content_order = [mem.content for mem in all_paginated]
            assert actual_content_order == expected_content_order

    @pytest.mark.asyncio
    async def test_count_all_memories_sqlite(self):
        """Test count_all_memories method (SQLite)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = SqliteVecMemoryStorage(os.path.join(tmp_dir, "test.db"))
            await storage.initialize()

            # Initially should be empty
            initial_count = await storage.count_all_memories()
            assert initial_count == 0

            # Create test memories
            await self.create_test_memories(storage)

            # Should now have 5 memories
            final_count = await storage.count_all_memories()
            assert final_count == 5

    @pytest.mark.asyncio
    async def test_empty_storage_handling_sqlite(self):
        """Test handling of empty storage (SQLite)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = SqliteVecMemoryStorage(os.path.join(tmp_dir, "test.db"))
            await storage.initialize()

            # Test get_all_memories on empty storage
            memories = await storage.get_all_memories()
            assert memories == []

            # Test with pagination on empty storage
            paginated = await storage.get_all_memories(limit=10, offset=0)
            assert paginated == []

            # Test count on empty storage
            count = await storage.count_all_memories()
            assert count == 0

    @pytest.mark.asyncio
    async def test_offset_beyond_total_sqlite(self):
        """Test offset beyond total records (SQLite)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = SqliteVecMemoryStorage(os.path.join(tmp_dir, "test.db"))
            await storage.initialize()

            # Create test memories
            await self.create_test_memories(storage)

            # Test offset beyond total records
            memories = await storage.get_all_memories(limit=10, offset=100)
            assert memories == []

    @pytest.mark.asyncio
    async def test_large_limit_sqlite(self):
        """Test large limit parameter (SQLite)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = SqliteVecMemoryStorage(os.path.join(tmp_dir, "test.db"))
            await storage.initialize()

            # Create test memories
            await self.create_test_memories(storage)

            # Test limit larger than total records
            memories = await storage.get_all_memories(limit=100, offset=0)
            assert len(memories) == 5  # Should return all 5 memories

    @pytest.mark.asyncio
    async def test_mixed_timestamps_ordering_sqlite(self):
        """Test ordering with mixed/unsorted timestamps (SQLite)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage = SqliteVecMemoryStorage(os.path.join(tmp_dir, "test.db"))
            await storage.initialize()

            # Create memories with deliberately mixed timestamps
            base_time = time.time()
            timestamps = [base_time + 300, base_time + 100, base_time + 500, base_time + 200, base_time + 400]

            for i, timestamp in enumerate(timestamps):
                memory = Memory(
                    content=f"Mixed memory {i + 1}",
                    content_hash=f"mixed_hash_{i + 1}",
                    tags=["mixed", "test"],
                    memory_type="mixed",
                    metadata={"timestamp": timestamp},
                    created_at=timestamp,
                    updated_at=timestamp
                )

                success, message = await storage.store(memory)
                assert success, f"Failed to store mixed memory {i + 1}: {message}"

            # Retrieve all memories
            memories = await storage.get_all_memories()

            # Should be ordered by timestamp (newest first)
            expected_order = [base_time + 500, base_time + 400, base_time + 300, base_time + 200, base_time + 100]
            actual_timestamps = [mem.created_at for mem in memories]

            assert actual_timestamps == expected_order, f"Expected {expected_order}, got {actual_timestamps}"


class TestAPIChronologicalIntegration:
    """Integration tests that would test the actual API endpoints.

    These tests are structured to be easily adaptable for testing the actual
    FastAPI endpoints when a test client is available.
    """

    def test_api_endpoint_structure(self):
        """Test that the API endpoint imports and structure are correct."""
        # Import the API router to ensure it loads correctly
        from mcp_memory_service.web.api.memories import router

        # Verify the router exists and has the expected endpoints
        routes = [route.path for route in router.routes]
        assert "/memories" in routes
        assert "/memories/{content_hash}" in routes

    def test_memory_response_model(self):
        """Test that the response models include necessary fields for chronological ordering."""
        from mcp_memory_service.web.api.memories import MemoryResponse, MemoryListResponse

        # Verify MemoryResponse has timestamp fields
        response_fields = MemoryResponse.__fields__.keys()
        assert "created_at" in response_fields
        assert "created_at_iso" in response_fields
        assert "updated_at" in response_fields
        assert "updated_at_iso" in response_fields

        # Verify MemoryListResponse has pagination fields
        list_fields = MemoryListResponse.__fields__.keys()
        assert "memories" in list_fields
        assert "total" in list_fields
        assert "page" in list_fields
        assert "page_size" in list_fields
        assert "has_more" in list_fields

    def test_storage_backend_type_compatibility(self):
        """Test that the API endpoints use the correct base storage type."""
        from mcp_memory_service.web.api.memories import list_memories
        import inspect

        # Get the signature of the list_memories function
        sig = inspect.signature(list_memories)
        storage_param = sig.parameters['storage']

        # Check that it uses the base MemoryStorage type, not a specific implementation
        assert 'MemoryStorage' in str(storage_param.annotation)


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])