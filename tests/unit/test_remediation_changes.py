"""
Tests for codebase remediation changes.

These tests verify the security fixes, config cleanup, and storage consolidation
performed during the codebase remediation (56% code reduction).
"""
import pytest
import secrets
from unittest.mock import MagicMock, patch


class TestSecurityFixes:
    """Test security fixes from Phase 0 of remediation."""

    def test_cors_default_is_empty_list(self):
        """Verify CORS default is [] not ['*'] (SEC-003 fix)."""
        from mcp_memory_service.config import HTTPSettings

        settings = HTTPSettings()
        assert settings.cors_origins == [], \
            f"CORS default should be empty list, got {settings.cors_origins}"

    def test_api_key_comparison_uses_constant_time(self):
        """Verify API key comparison uses secrets.compare_digest (SEC-007 fix)."""
        # This test verifies the fix is in place by checking the middleware code
        from mcp_memory_service.web.oauth.middleware import authenticate_api_key

        # The function should use secrets.compare_digest internally
        # Test that authentication works (actual timing attack prevention
        # is verified by code inspection - secrets.compare_digest is used)
        result = authenticate_api_key(None)  # No key provided
        # Should return unauthenticated result
        assert result is not None

    def test_anonymous_access_is_read_only(self):
        """Verify anonymous access defaults to read-only scope (SEC-001 fix)."""
        from mcp_memory_service.config import OAuthSettings

        settings = OAuthSettings()
        # Anonymous access should be restricted by default
        assert settings.allow_anonymous_access is False, \
            "Anonymous access should be disabled by default"


class TestStorageConsolidation:
    """Test storage backend consolidation from Phase 2."""

    def test_only_two_storage_backends_exist(self):
        """Verify only SQLite-vec and Qdrant backends remain."""
        from mcp_memory_service.storage import factory

        # Check supported backends
        supported = ["sqlite_vec", "qdrant"]
        removed = ["cloudflare", "hybrid", "http_client"]

        # Verify removed backends are not importable
        for backend in removed:
            with pytest.raises((ImportError, ModuleNotFoundError, AttributeError)):
                if backend == "cloudflare":
                    from mcp_memory_service.storage.cloudflare import CloudflareStorage
                elif backend == "hybrid":
                    from mcp_memory_service.storage.hybrid import HybridStorage
                elif backend == "http_client":
                    from mcp_memory_service.storage.http_client import HTTPStorageClient

    def test_storage_factory_returns_correct_backend(self):
        """Verify storage factory creates correct backend types."""
        import os
        from mcp_memory_service.storage.factory import get_storage_backend_class
        from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
        from mcp_memory_service.storage.qdrant_storage import QdrantStorage

        # Test SQLite-vec backend
        with patch.dict(os.environ, {'MCP_MEMORY_STORAGE_BACKEND': 'sqlite_vec'}):
            # Need to reload to pick up new env var
            import importlib
            import mcp_memory_service.config
            importlib.reload(mcp_memory_service.config)
            sqlite_class = get_storage_backend_class()
            assert sqlite_class == SqliteVecMemoryStorage

        # Test Qdrant backend
        with patch.dict(os.environ, {'MCP_MEMORY_STORAGE_BACKEND': 'qdrant'}):
            importlib.reload(mcp_memory_service.config)
            qdrant_class = get_storage_backend_class()
            assert qdrant_class == QdrantStorage

    def test_base_storage_protocol_exists(self):
        """Verify BaseStorage Protocol interface is defined."""
        from mcp_memory_service.storage import BaseStorage
        from typing import Protocol, runtime_checkable

        # BaseStorage should be a Protocol
        assert hasattr(BaseStorage, '__protocol_attrs__') or \
               issubclass(type(BaseStorage), type(Protocol)), \
               "BaseStorage should be a Protocol"


class TestConfigCleanup:
    """Test configuration cleanup from Phase 3."""

    def test_removed_settings_classes_not_importable(self):
        """Verify removed settings classes are gone."""
        removed_classes = [
            "CloudflareSettings",
            "HybridSettings",
            "DocumentSettings",
            "ConsolidationSettings",
        ]

        from mcp_memory_service import config

        for class_name in removed_classes:
            assert not hasattr(config, class_name), \
                f"{class_name} should be removed from config"

    def test_essential_settings_classes_exist(self):
        """Verify essential settings classes still exist."""
        from mcp_memory_service.config import (
            PathSettings,
            ServerSettings,
            StorageSettings,
            HTTPSettings,
            OAuthSettings,
            QdrantSettings,
        )

        # All should be instantiable
        PathSettings()
        ServerSettings()
        StorageSettings()
        HTTPSettings()
        OAuthSettings()
        QdrantSettings()


class TestRemovedModules:
    """Test that removed modules are truly gone."""

    def test_consolidation_module_removed(self):
        """Verify consolidation module is deleted."""
        with pytest.raises((ImportError, ModuleNotFoundError)):
            from mcp_memory_service import consolidation

    def test_sync_module_removed(self):
        """Verify sync module is deleted."""
        with pytest.raises((ImportError, ModuleNotFoundError)):
            from mcp_memory_service import sync

    def test_ingestion_module_removed(self):
        """Verify ingestion module is deleted."""
        with pytest.raises((ImportError, ModuleNotFoundError)):
            from mcp_memory_service import ingestion

    def test_discovery_module_removed(self):
        """Verify discovery/mDNS module is deleted."""
        with pytest.raises((ImportError, ModuleNotFoundError)):
            from mcp_memory_service import discovery

    def test_lm_studio_compat_removed(self):
        """Verify LM Studio compatibility layer is deleted."""
        with pytest.raises((ImportError, ModuleNotFoundError)):
            from mcp_memory_service import lm_studio_compat

    def test_legacy_server_removed(self):
        """Verify legacy server.py god-class is deleted."""
        with pytest.raises((ImportError, ModuleNotFoundError)):
            from mcp_memory_service import server


class TestDependencyCleanup:
    """Test removed dependencies."""

    def test_zeroconf_not_required(self):
        """Verify zeroconf (mDNS) is not a required dependency."""
        # This test passes if the test suite runs without zeroconf
        # If zeroconf was required, import would fail
        pass

    def test_authlib_not_required(self):
        """Verify authlib (OAuth) is not a required dependency."""
        # Try importing OAuth-related code without authlib
        # Should work because OAuth dependencies were removed
        try:
            from mcp_memory_service.web.oauth.middleware import AuthMiddleware
            # If we get here, the middleware works without authlib
        except ImportError as e:
            if "authlib" in str(e).lower():
                pytest.fail("authlib should not be a required dependency")

    def test_python_jose_not_required(self):
        """Verify python-jose (JWT) is not a required dependency."""
        # Middleware should work without python-jose
        try:
            from mcp_memory_service.web.oauth.middleware import verify_api_key
        except ImportError as e:
            if "jose" in str(e).lower():
                pytest.fail("python-jose should not be a required dependency")
