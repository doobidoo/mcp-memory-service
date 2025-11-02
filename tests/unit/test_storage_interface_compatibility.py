"""
Unit tests for storage backend interface compatibility.

These tests verify that all storage backends implement the same interface,
catching issues like mismatched method signatures or missing methods.

Added to prevent production bugs like v8.12.0 where:
  - count_all_memories() had different signatures across backends
  - Some backends had 'tags' parameter, others didn't
  - Database-level filtering wasn't uniformly implemented
"""

import pytest
import inspect
from abc import ABC
from typing import get_type_hints


def get_all_storage_classes():
    """Get all concrete storage backend classes."""
    from mcp_memory_service.storage.base import MemoryStorage
    from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
    from mcp_memory_service.storage.cloudflare import CloudflareStorage
    from mcp_memory_service.storage.hybrid import HybridMemoryStorage

    return [
        ('SqliteVecMemoryStorage', SqliteVecMemoryStorage),
        ('CloudflareStorage', CloudflareStorage),
        ('HybridMemoryStorage', HybridMemoryStorage),
    ]


def test_base_class_is_abstract():
    """Test that MemoryStorage base class is abstract."""
    from mcp_memory_service.storage.base import MemoryStorage

    # Should be an ABC
    assert issubclass(MemoryStorage, ABC)

    # Should not be instantiable directly
    with pytest.raises(TypeError):
        MemoryStorage()


def test_all_backends_inherit_from_base():
    """Test that all storage backends inherit from MemoryStorage."""
    from mcp_memory_service.storage.base import MemoryStorage

    for name, storage_class in get_all_storage_classes():
        assert issubclass(storage_class, MemoryStorage), \
            f"{name} must inherit from MemoryStorage"


def test_all_backends_implement_required_methods():
    """Test that all backends implement required abstract methods."""
    from mcp_memory_service.storage.base import MemoryStorage

    # Get abstract methods from base class
    abstract_methods = {
        name for name, method in inspect.getmembers(MemoryStorage)
        if getattr(method, '__isabstractmethod__', False)
    }

    # Each backend must implement all abstract methods
    for name, storage_class in get_all_storage_classes():
        for method_name in abstract_methods:
            assert hasattr(storage_class, method_name), \
                f"{name} missing required method: {method_name}"


def test_store_signature_compatibility():
    """Test that store has compatible signature across backends."""
    signatures = {}

    for name, storage_class in get_all_storage_classes():
        sig = inspect.signature(storage_class.store)
        signatures[name] = sig

    # All signatures should have same parameters (ignoring 'self')
    first_name = list(signatures.keys())[0]
    first_params = list(signatures[first_name].parameters.keys())[1:]  # Skip 'self'

    for name, sig in signatures.items():
        params = list(sig.parameters.keys())[1:]  # Skip 'self'
        assert params == first_params, \
            f"{name}.store parameters {params} don't match {first_name} {first_params}"


def test_get_all_memories_signature_compatibility():
    """Test that get_all_memories has compatible signature across backends."""
    signatures = {}

    for name, storage_class in get_all_storage_classes():
        sig = inspect.signature(storage_class.get_all_memories)
        signatures[name] = sig

    # All signatures should have same parameters (ignoring 'self')
    first_name = list(signatures.keys())[0]
    first_params = list(signatures[first_name].parameters.keys())[1:]  # Skip 'self'

    for name, sig in signatures.items():
        params = list(sig.parameters.keys())[1:]  # Skip 'self'
        assert params == first_params, \
            f"{name}.get_all_memories parameters {params} don't match {first_name} {first_params}"


def test_count_all_memories_signature_compatibility():
    """Test that count_all_memories has compatible signature across backends.

    This test specifically prevents the v8.12.0 bug where count_all_memories()
    had different signatures across backends (some had 'tags', others didn't).
    """
    signatures = {}

    for name, storage_class in get_all_storage_classes():
        if hasattr(storage_class, 'count_all_memories'):
            sig = inspect.signature(storage_class.count_all_memories)
            signatures[name] = sig

    # All signatures should have same parameters (ignoring 'self')
    if len(signatures) > 1:
        first_name = list(signatures.keys())[0]
        first_params = list(signatures[first_name].parameters.keys())[1:]  # Skip 'self'

        for name, sig in signatures.items():
            params = list(sig.parameters.keys())[1:]  # Skip 'self'
            assert params == first_params, \
                f"{name}.count_all_memories parameters {params} don't match {first_name} {first_params}"


def test_retrieve_signature_compatibility():
    """Test that retrieve has compatible signature across backends."""
    signatures = {}

    for name, storage_class in get_all_storage_classes():
        sig = inspect.signature(storage_class.retrieve)
        signatures[name] = sig

    # All signatures should have same parameters (ignoring 'self')
    first_name = list(signatures.keys())[0]
    first_params = list(signatures[first_name].parameters.keys())[1:]  # Skip 'self'

    for name, sig in signatures.items():
        params = list(sig.parameters.keys())[1:]  # Skip 'self'
        assert params == first_params, \
            f"{name}.retrieve parameters {params} don't match {first_name} {first_params}"


def test_delete_signature_compatibility():
    """Test that delete has compatible signature across backends."""
    signatures = {}

    for name, storage_class in get_all_storage_classes():
        sig = inspect.signature(storage_class.delete)
        signatures[name] = sig

    # All signatures should have same parameters (ignoring 'self')
    first_name = list(signatures.keys())[0]
    first_params = list(signatures[first_name].parameters.keys())[1:]  # Skip 'self'

    for name, sig in signatures.items():
        params = list(sig.parameters.keys())[1:]  # Skip 'self'
        assert params == first_params, \
            f"{name}.delete parameters {params} don't match {first_name} {first_params}"


def test_get_stats_signature_compatibility():
    """Test that get_stats has compatible signature across backends."""
    signatures = {}

    for name, storage_class in get_all_storage_classes():
        sig = inspect.signature(storage_class.get_stats)
        signatures[name] = sig

    # All signatures should have same parameters (ignoring 'self')
    first_name = list(signatures.keys())[0]
    first_params = list(signatures[first_name].parameters.keys())[1:]  # Skip 'self'

    for name, sig in signatures.items():
        params = list(sig.parameters.keys())[1:]  # Skip 'self'
        assert params == first_params, \
            f"{name}.get_stats parameters {params} don't match {first_name} {first_params}"


def test_all_backends_have_same_public_methods():
    """Test that all backends expose the same public interface.

    This catches missing methods that should be implemented.
    """
    from mcp_memory_service.storage.base import MemoryStorage

    # Get public methods from base class (those without leading underscore)
    base_methods = {
        name for name, method in inspect.getmembers(MemoryStorage, predicate=inspect.isfunction)
        if not name.startswith('_')
    }

    for name, storage_class in get_all_storage_classes():
        backend_methods = {
            method_name for method_name, method in inspect.getmembers(storage_class, predicate=inspect.isfunction)
            if not method_name.startswith('_')
        }

        # Backend should implement all base methods
        missing = base_methods - backend_methods
        assert not missing, \
            f"{name} missing public methods: {missing}"


def test_async_method_consistency():
    """Test that async methods are consistently async across backends.

    If one backend makes a method async, all should be async.
    """
    from mcp_memory_service.storage.base import MemoryStorage

    # Get all public methods from base class
    base_methods = [
        name for name, method in inspect.getmembers(MemoryStorage, predicate=inspect.isfunction)
        if not name.startswith('_')
    ]

    # Track which methods are async in each backend
    async_status = {method: set() for method in base_methods}

    for name, storage_class in get_all_storage_classes():
        for method_name in base_methods:
            if hasattr(storage_class, method_name):
                method = getattr(storage_class, method_name)
                if inspect.iscoroutinefunction(method):
                    async_status[method_name].add(name)

    # Each method should either be async in all backends or none
    for method_name, async_backends in async_status.items():
        if async_backends:
            all_backends = {name for name, _ in get_all_storage_classes()}
            assert async_backends == all_backends, \
                f"{method_name} is async in {async_backends} but not in {all_backends - async_backends}"


def test_backends_handle_tags_parameter_consistently():
    """Test that all backends handle 'tags' parameter consistently.

    This specifically targets the v8.12.0 bug where count_all_memories()
    had 'tags' in some backends but not others.
    """
    methods_with_tags = ['get_all_memories', 'count_all_memories']

    for method_name in methods_with_tags:
        has_tags_param = {}

        for name, storage_class in get_all_storage_classes():
            if hasattr(storage_class, method_name):
                sig = inspect.signature(getattr(storage_class, method_name))
                has_tags_param[name] = 'tags' in sig.parameters

        # All backends should handle tags consistently
        if has_tags_param:
            first_name = list(has_tags_param.keys())[0]
            first_value = has_tags_param[first_name]

            for name, has_tags in has_tags_param.items():
                assert has_tags == first_value, \
                    f"{name}.{method_name} 'tags' parameter inconsistent: {name}={has_tags}, {first_name}={first_value}"


def test_return_type_consistency():
    """Test that methods return consistent types across backends.

    This helps catch issues where one backend returns dict and another returns a custom class.
    """
    from mcp_memory_service.storage.base import MemoryStorage

    # Methods to check return types
    methods_to_check = ['get_stats', 'store', 'delete']

    for method_name in methods_to_check:
        if not hasattr(MemoryStorage, method_name):
            continue

        return_types = {}

        for name, storage_class in get_all_storage_classes():
            if hasattr(storage_class, method_name):
                method = getattr(storage_class, method_name)
                try:
                    type_hints = get_type_hints(method)
                    if 'return' in type_hints:
                        return_types[name] = type_hints['return']
                except Exception:
                    # Some methods may not have type hints
                    pass

        # If we have return types, they should match
        if len(return_types) > 1:
            first_name = list(return_types.keys())[0]
            first_type = return_types[first_name]

            for name, return_type in return_types.items():
                # Allow for Coroutine wrappers in async methods
                assert return_type == first_type or str(return_type).startswith('typing.Coroutine'), \
                    f"{name}.{method_name} return type {return_type} doesn't match {first_name} {first_type}"


if __name__ == "__main__":
    # Allow running tests directly for quick verification
    pytest.main([__file__, "-v"])
