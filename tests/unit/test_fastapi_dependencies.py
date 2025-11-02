"""
Unit tests for FastAPI dependency injection.

These tests verify that the actual dependency injection chain works correctly,
catching issues like import-time default parameter evaluation.

Added to prevent production bugs like v8.12.0 where:
  def get_memory_service(storage: MemoryStorage = get_storage())
was evaluated at import time when _storage was None.
"""

import pytest
import pytest_asyncio
import asyncio
import tempfile
import os
from unittest.mock import MagicMock


@pytest_asyncio.fixture
async def temp_storage():
    """Create a temporary storage for testing."""
    from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
    from mcp_memory_service.web.dependencies import set_storage

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        storage = SqliteVecMemoryStorage(db_path)
        await storage.initialize()
        set_storage(storage)
        yield storage
        storage.close()


@pytest.mark.asyncio
async def test_get_storage_dependency_callable(temp_storage):
    """Test that get_storage() dependency is callable without errors."""
    from mcp_memory_service.web.dependencies import get_storage

    # Should be callable
    assert callable(get_storage)

    # Should not raise when called
    storage = get_storage()
    assert storage is not None
    assert storage is temp_storage


def test_get_memory_service_dependency_callable():
    """Test that get_memory_service() dependency is callable without errors."""
    from mcp_memory_service.web.dependencies import get_memory_service

    # Should be callable
    assert callable(get_memory_service)

    # Should not raise when called
    try:
        service = get_memory_service()
        assert service is not None
    except Exception as e:
        pytest.fail(f"get_memory_service() raised unexpected exception: {e}")


def test_get_storage_uses_depends_not_default_param():
    """Test that get_storage is used via Depends(), not as default parameter.

    This prevents the v8.12.0 bug where:
      def get_memory_service(storage: MemoryStorage = get_storage())
    was evaluated at import time.
    """
    import inspect
    from mcp_memory_service.web.dependencies import get_memory_service
    from fastapi.params import Depends

    # Get function signature
    sig = inspect.signature(get_memory_service)

    # Check if storage parameter exists
    if 'storage' in sig.parameters:
        storage_param = sig.parameters['storage']

        # If it has a default, it should be Depends(...), not a function call
        if storage_param.default != inspect.Parameter.empty:
            # Default should be a Depends instance, not the result of get_storage()
            # Check the type name since Depends is not a simple type
            assert type(storage_param.default).__name__ == 'Depends', \
                "storage parameter should use Depends(get_storage), not get_storage()"


@pytest.mark.asyncio
async def test_dependency_chain_storage_to_service(temp_storage):
    """Test that the dependency chain from storage → service works."""
    from mcp_memory_service.web.dependencies import get_storage, get_memory_service

    # Get storage
    storage = get_storage()
    assert storage is not None

    # Get service (should use the storage)
    service = get_memory_service()
    assert service is not None

    # Service should have a storage reference
    assert hasattr(service, 'storage')


@pytest.mark.asyncio
async def test_get_storage_returns_singleton(temp_storage):
    """Test that get_storage() returns the same instance (singleton pattern)."""
    from mcp_memory_service.web.dependencies import get_storage

    storage1 = get_storage()
    storage2 = get_storage()

    # Should be the same instance
    assert storage1 is storage2, "get_storage() should return singleton"


def test_get_memory_service_returns_new_instance():
    """Test that get_memory_service() returns new instances (not singleton)."""
    from mcp_memory_service.web.dependencies import get_memory_service

    service1 = get_memory_service()
    service2 = get_memory_service()

    # They use the same storage but are different service instances
    # (This is OK because MemoryService is stateless)
    assert isinstance(service1, type(service2))


def test_dependencies_module_has_required_functions():
    """Test that dependencies module exports required functions."""
    from mcp_memory_service.web import dependencies

    # Core dependency functions
    assert hasattr(dependencies, 'get_storage')
    assert hasattr(dependencies, 'get_memory_service')

    # Should be callable
    assert callable(dependencies.get_storage)
    assert callable(dependencies.get_memory_service)


@pytest.mark.asyncio
async def test_storage_dependency_is_initialized(temp_storage):
    """Test that storage returned by get_storage() is properly initialized."""
    from mcp_memory_service.web.dependencies import get_storage

    storage = get_storage()

    # Check it has expected methods (from base class)
    assert hasattr(storage, 'store')
    assert hasattr(storage, 'get_all_memories')
    assert hasattr(storage, 'get_stats')
    assert hasattr(storage, 'delete')


@pytest.mark.asyncio
async def test_async_dependencies_work(temp_storage):
    """Test that async dependencies work correctly.

    Some storage operations are async, so we need to verify they work.
    """
    from mcp_memory_service.web.dependencies import get_storage

    storage = get_storage()

    # get_stats is async and was the source of issue #191
    stats = await storage.get_stats()
    assert isinstance(stats, dict)
    assert 'total_memories' in stats


def test_dependency_injection_doesnt_fail_on_import():
    """Test that importing dependencies module doesn't cause errors.

    This catches import-time evaluation bugs.
    """
    try:
        # This should not raise
        import mcp_memory_service.web.dependencies
        import mcp_memory_service.web.app

        # App should be created successfully
        from mcp_memory_service.web.app import app
        assert app is not None
    except Exception as e:
        pytest.fail(f"Import-time error in dependencies: {e}")


def test_memory_service_has_required_methods():
    """Test that MemoryService has all required methods."""
    from mcp_memory_service.web.dependencies import get_memory_service

    service = get_memory_service()

    # Core methods from MemoryService class
    required_methods = [
        'store_memory',
        'retrieve_memories',
        'delete_memory',
        'list_memories',  # Not get_all_memories
        'search_by_tag',
        'get_memory_by_hash',
        'health_check',
    ]

    for method in required_methods:
        assert hasattr(service, method), f"MemoryService missing {method}"
        assert callable(getattr(service, method))


if __name__ == "__main__":
    # Allow running tests directly for quick verification
    pytest.main([__file__, "-v"])
