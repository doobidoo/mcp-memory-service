#!/usr/bin/env python3
"""
Test shared storage functionality between HTTP and MCP servers.

This test verifies that both servers can share a single storage instance,
preventing duplicate model loading and initialization issues.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from src.mcp_memory_service.shared_storage import (
    StorageManager,
    close_shared_storage,
    get_shared_storage,
    is_storage_initialized,
)


class TestSharedStorage:
    """Test shared storage management between servers."""

    @pytest.mark.asyncio
    async def test_singleton_storage_manager(self):
        """Test that StorageManager is a true singleton."""
        manager1 = StorageManager.get_instance()
        manager2 = StorageManager.get_instance()

        assert manager1 is manager2, "StorageManager should be a singleton"

    @pytest.mark.asyncio
    async def test_storage_initialization_once(self):
        """Test that storage is only initialized once despite multiple calls."""
        # Reset the singleton for testing
        StorageManager._instance = None

        with patch("src.mcp_memory_service.shared_storage.create_storage_instance") as mock_create:
            mock_storage = AsyncMock()
            mock_storage.close = AsyncMock()
            mock_create.return_value = mock_storage

            # Multiple concurrent calls to get_shared_storage
            tasks = [get_shared_storage() for _ in range(5)]
            results = await asyncio.gather(*tasks)

            # All should return the same instance
            for result in results:
                assert result is mock_storage

            # create_storage_instance should only be called once
            mock_create.assert_called_once()

            # Clean up
            await close_shared_storage()

    @pytest.mark.asyncio
    async def test_is_initialized_state(self):
        """Test that is_initialized correctly reports storage state."""
        # Reset the singleton for testing
        StorageManager._instance = None

        assert not is_storage_initialized(), "Storage should not be initialized initially"

        with patch("src.mcp_memory_service.shared_storage.create_storage_instance") as mock_create:
            mock_storage = AsyncMock()
            mock_storage.close = AsyncMock()
            mock_create.return_value = mock_storage

            # Initialize storage
            await get_shared_storage()
            assert is_storage_initialized(), "Storage should be initialized after get_shared_storage"

            # Close storage
            await close_shared_storage()
            assert not is_storage_initialized(), "Storage should not be initialized after close"

    @pytest.mark.asyncio
    async def test_thread_safe_initialization(self):
        """Test that storage initialization is thread-safe."""
        # Reset the singleton for testing
        StorageManager._instance = None

        call_count = 0

        async def mock_create_storage(*args):
            nonlocal call_count
            call_count += 1
            # Simulate slow initialization
            await asyncio.sleep(0.1)
            mock_storage = AsyncMock()
            mock_storage.close = AsyncMock()
            return mock_storage

        with patch("src.mcp_memory_service.shared_storage.create_storage_instance", mock_create_storage):
            # Launch many concurrent initialization attempts
            tasks = [get_shared_storage() for _ in range(20)]
            results = await asyncio.gather(*tasks)

            # Should only initialize once despite concurrent calls
            assert call_count == 1, f"Storage initialized {call_count} times, expected 1"

            # All results should be the same instance
            first_result = results[0]
            for result in results:
                assert result is first_result

            # Clean up
            await close_shared_storage()

    @pytest.mark.asyncio
    async def test_close_idempotent(self):
        """Test that close_shared_storage is idempotent."""
        # Reset the singleton for testing
        StorageManager._instance = None

        with patch("src.mcp_memory_service.shared_storage.create_storage_instance") as mock_create:
            mock_storage = AsyncMock()
            mock_storage.close = AsyncMock()
            mock_create.return_value = mock_storage

            # Initialize storage
            await get_shared_storage()

            # Close multiple times
            await close_shared_storage()
            await close_shared_storage()
            await close_shared_storage()

            # Storage.close should only be called once
            mock_storage.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_reinitialize_after_close(self):
        """Test that storage can be reinitialized after closing."""
        # Reset the singleton for testing
        StorageManager._instance = None

        with patch("src.mcp_memory_service.shared_storage.create_storage_instance") as mock_create:
            # First storage instance
            mock_storage1 = AsyncMock()
            mock_storage1.close = AsyncMock()

            # Second storage instance
            mock_storage2 = AsyncMock()
            mock_storage2.close = AsyncMock()

            mock_create.side_effect = [mock_storage1, mock_storage2]

            # First initialization
            storage1 = await get_shared_storage()
            assert storage1 is mock_storage1
            assert is_storage_initialized()

            # Close
            await close_shared_storage()
            assert not is_storage_initialized()

            # Second initialization
            storage2 = await get_shared_storage()
            assert storage2 is mock_storage2
            assert is_storage_initialized()

            # Clean up
            await close_shared_storage()


@pytest.mark.asyncio
async def test_unified_server_shared_storage():
    """Test that unified_server properly initializes shared storage."""
    from src.mcp_memory_service.unified_server import UnifiedServer

    # Reset singleton for testing
    StorageManager._instance = None

    with patch("src.mcp_memory_service.shared_storage.create_storage_instance") as mock_create:
        mock_storage = AsyncMock()
        mock_storage.close = AsyncMock()
        mock_create.return_value = mock_storage

        with patch.object(UnifiedServer, "run_http_server", new_callable=AsyncMock) as mock_http:
            with patch.object(UnifiedServer, "run_mcp_server", new_callable=AsyncMock) as mock_mcp:
                # Make servers complete quickly for test
                mock_http.return_value = None
                mock_mcp.return_value = None

                server = UnifiedServer()
                server.http_enabled = True
                server.mcp_enabled = True

                # Run server (will complete quickly due to mocked servers)
                await server.run()

                # Storage should have been initialized
                mock_create.assert_called_once()

                # Both servers should have been started
                mock_http.assert_called_once()
                mock_mcp.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
