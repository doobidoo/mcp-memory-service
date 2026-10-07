#!/usr/bin/env python3
"""
Tests for hybrid storage delete operations capability gating (Gate 2 - TDD RED).

Tests the supports_delete_operations capability property and gating behavior that will be implemented in Gate 3.
These tests MUST FAIL until the implementation is done.

Testing Requirements for delete operations gating:
- Base MemoryStorage should have supports_delete_operations property defaulting to False
- CloudflareStorage should override supports_delete_operations to True  
- RemoteHTTPStorage should inherit False (no override)
- HybridMemoryStorage._process_single_operation should gate delete_by_timeframe and delete_before_date behind supports_delete_operations
- When secondary lacks supports_delete_operations, these operations should be skipped with logger.warning
- When secondary has supports_delete_operations, these operations should be called normally
"""

import asyncio
import pytest
import pytest_asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch, call
from typing import Dict, Any
from datetime import date, datetime

# Add src to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from mcp_memory_service.storage.base import MemoryStorage
from mcp_memory_service.storage.hybrid import BackgroundSyncService, SyncOperation
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.utils.hashing import generate_content_hash


class MockMemoryStorageWithDeleteSupport(MemoryStorage):
    """Mock storage with configurable delete support capability."""
    
    def __init__(self, supports_delete=False):
        self._supports_delete_operations = supports_delete
        self.call_log = []
        self.delete_by_timeframe_calls = []
        self.delete_before_date_calls = []

    @property
    def max_content_length(self):
        return None

    @property
    def supports_chunking(self):
        return False

    @property
    def supports_delete_operations(self):
        """This property should exist and return the configured value."""
        return self._supports_delete_operations

    async def initialize(self):
        pass

    async def store(self, memory, skip_semantic_dedup=False, store="default"):
        return True, "Success"

    async def delete(self, content_hash):
        return True, "Success"

    async def delete_by_timeframe(self, start_date: date, end_date: date, tag: str = None):
        """Track calls to delete_by_timeframe."""
        self.call_log.append(f"delete_by_timeframe({start_date}, {end_date}, {tag})")
        self.delete_by_timeframe_calls.append((start_date, end_date, tag))
        return 5, "Deleted 5 memories"

    async def delete_before_date(self, before_date: date, tag: str = None):
        """Track calls to delete_before_date."""
        self.call_log.append(f"delete_before_date({before_date}, {tag})")
        self.delete_before_date_calls.append((before_date, tag))
        return 3, "Deleted 3 memories"

    # Minimal implementation to satisfy base class
    async def retrieve(self, query, n_results=5, tags=None, min_confidence=0.0, 
                      include_superseded=False, start_time=None, end_time=None, store=None):
        return []
    
    async def search_by_tag(self, tags, time_start=None):
        return []
        
    async def search_by_tags(self, tags, operation="AND", time_start=None, time_end=None):
        return []
        
    async def get_by_exact_content(self, content):
        return []
        
    async def get_by_hash(self, content_hash, store=None):
        return None
        
    async def delete_by_tag(self, tag):
        return 0, "Success"

    async def cleanup_duplicates(self):
        return 0, "Success"

    async def get_stats(self):
        return {"total_memories": 0}

    async def update_memory_metadata(self, content_hash, updates, preserve_timestamps=True):
        return True, "Success"


class TestDeleteCapabilityProperties:
    """Test that the supports_delete_operations property exists and has correct defaults."""

    def test_base_default_supports_delete_operations_false(self):
        """Base MemoryStorage should have supports_delete_operations property defaulting to False."""
        # This will FAIL because supports_delete_operations doesn't exist in base yet
        from mcp_memory_service.storage.remote_http import RemoteHTTPStorage

        storage = RemoteHTTPStorage(base_url="https://example.invalid", api_key="x")
        assert storage.supports_delete_operations is False, \
            "Base default supports_delete_operations must be False on a non-overriding backend"

    def test_cloudflare_overrides_supports_delete_operations_true(self):
        """CloudflareStorage must override supports_delete_operations to True."""
        # This will FAIL because supports_delete_operations doesn't exist in CloudflareStorage yet
        from mcp_memory_service.storage.cloudflare import CloudflareStorage

        # Test via class descriptor to assert the actual True value, not just presence
        assert CloudflareStorage.supports_delete_operations.fget(object()) is True, \
            "CloudflareStorage must override supports_delete_operations to True"

    def test_remote_http_inherits_delete_operations_false(self):
        """RemoteHTTPStorage should inherit False from base (no override)."""
        # This will FAIL because supports_delete_operations doesn't exist in base yet
        from mcp_memory_service.storage.remote_http import RemoteHTTPStorage

        storage = RemoteHTTPStorage(base_url="https://example.invalid", api_key="x")
        assert storage.supports_delete_operations is False, \
            "RemoteHTTPStorage should inherit supports_delete_operations as False"


class TestDeleteOperationGating:
    """Test that delete_by_timeframe and delete_before_date operations are gated behind supports_delete_operations."""

    @pytest_asyncio.fixture
    async def temp_sqlite_db(self, tmp_path):
        """Create a temporary SQLite database."""
        db_path = tmp_path / "test.db"
        yield str(db_path)

    @pytest_asyncio.fixture
    async def sync_service_with_delete_support(self, temp_sqlite_db):
        """Create sync service with secondary that supports delete operations."""
        primary = SqliteVecMemoryStorage(temp_sqlite_db)
        await primary.initialize()
        
        # Secondary with delete support: supports_delete_operations = True
        secondary = MockMemoryStorageWithDeleteSupport(supports_delete=True)
        await secondary.initialize()
        
        sync_service = BackgroundSyncService(primary, secondary, sync_interval=1, batch_size=3)
        
        yield primary, secondary, sync_service
        
        if sync_service.is_running:
            await sync_service.stop()
        if hasattr(primary, 'close'):
            try:
                await primary.close()
            except Exception:
                pass

    @pytest_asyncio.fixture  
    async def sync_service_without_delete_support(self, temp_sqlite_db):
        """Create sync service with secondary that does NOT support delete operations."""
        primary = SqliteVecMemoryStorage(temp_sqlite_db)
        await primary.initialize()
        
        # Secondary without delete support: supports_delete_operations = False
        secondary = MockMemoryStorageWithDeleteSupport(supports_delete=False)
        await secondary.initialize()
        
        sync_service = BackgroundSyncService(primary, secondary, sync_interval=1, batch_size=3)
        
        yield primary, secondary, sync_service
        
        if sync_service.is_running:
            await sync_service.stop()
        if hasattr(primary, 'close'):
            try:
                await primary.close()
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_capability_property_exists_and_works_correctly(self, sync_service_with_delete_support, sync_service_without_delete_support):
        """Test that the supports_delete_operations property works as expected when implemented."""
        primary1, secondary1, sync_service1 = sync_service_with_delete_support  # should have True
        primary2, secondary2, sync_service2 = sync_service_without_delete_support  # should have False
        
        # Test that our test setup is correct
        assert secondary1.supports_delete_operations is True, \
            "Mock with delete support should have supports_delete_operations=True"
        assert secondary2.supports_delete_operations is False, \
            "Mock without delete support should have supports_delete_operations=False"
        
        # Test that getattr works correctly (this is how the implementation will check)
        assert getattr(secondary1, 'supports_delete_operations', False) is True
        assert getattr(secondary2, 'supports_delete_operations', False) is False

    @pytest.mark.asyncio
    async def test_delete_by_timeframe_gating_not_implemented_yet(self, sync_service_without_delete_support):
        """CRITICAL RED test: delete_by_timeframe should be gated but currently isn't."""
        primary, secondary, sync_service = sync_service_without_delete_support
        
        # Create delete_by_timeframe operation
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 31)
        
        delete_op = SyncOperation(
            operation='delete_by_timeframe',
            start_date=start_date,
            end_date=end_date,
            tag="test"
        )
        
        # Patch sync_stats to avoid KeyError
        with patch.object(sync_service, 'sync_stats', {'operations_synced': 0, 'operations_failed': 0, 'cloudflare_available': True}):
            await sync_service._process_single_operation(delete_op)
        
        # This FAILS because gating is not implemented - method gets called despite supports_delete_operations=False
        # When Gate 3 is implemented, this assertion should PASS (0 calls due to gating)
        assert len(secondary.delete_by_timeframe_calls) == 0, \
            "GATE 3 NOT IMPLEMENTED: delete_by_timeframe should be gated behind supports_delete_operations capability. " \
            f"Expected 0 calls (skipped), got {len(secondary.delete_by_timeframe_calls)} calls. " \
            f"The implementation should check: if not getattr(self.secondary, 'supports_delete_operations', False): skip"

    @pytest.mark.asyncio
    async def test_delete_before_date_gating_not_implemented_yet(self, sync_service_without_delete_support):
        """CRITICAL RED test: delete_before_date should be gated but currently isn't."""
        primary, secondary, sync_service = sync_service_without_delete_support
        
        # Create delete_before_date operation
        before_date = date(2024, 1, 15)
        
        delete_op = SyncOperation(
            operation='delete_before_date',
            before_date=before_date,
            tag="cleanup"
        )
        
        # Patch sync_stats to avoid KeyError
        with patch.object(sync_service, 'sync_stats', {'operations_synced': 0, 'operations_failed': 0, 'cloudflare_available': True}):
            await sync_service._process_single_operation(delete_op)
        
        # This FAILS because gating is not implemented - method gets called despite supports_delete_operations=False  
        # When Gate 3 is implemented, this assertion should PASS (0 calls due to gating)
        assert len(secondary.delete_before_date_calls) == 0, \
            "GATE 3 NOT IMPLEMENTED: delete_before_date should be gated behind supports_delete_operations capability. " \
            f"Expected 0 calls (skipped), got {len(secondary.delete_before_date_calls)} calls. " \
            f"The implementation should check: if not getattr(self.secondary, 'supports_delete_operations', False): skip"

    @pytest.mark.asyncio
    async def test_regular_operations_unaffected_by_delete_gating(self, sync_service_without_delete_support):
        """Regular operations (store, delete, update) should work normally regardless of supports_delete_operations."""
        primary, secondary, sync_service = sync_service_without_delete_support
        
        # Create test memory
        content = "test content for regular operations"
        memory = Memory(
            content=content,
            content_hash=generate_content_hash(content),
            tags=["test"],
            memory_type="observation"
        )
        
        # Test store operation - should work fine
        store_op = SyncOperation(operation='store', memory=memory)
        await sync_service._process_single_operation(store_op)
        
        # Test update operation - should work fine  
        update_op = SyncOperation(
            operation='update',
            content_hash=memory.content_hash,
            updates={"metadata": {"updated": True}}
        )
        await sync_service._process_single_operation(update_op)
        
        # Test delete operation - should work fine
        delete_op = SyncOperation(operation='delete', content_hash=memory.content_hash)
        await sync_service._process_single_operation(delete_op)
        
        # All operations should complete without error
        # The supports_delete_operations capability should ONLY affect delete_by_timeframe and delete_before_date


    @pytest.mark.asyncio
    async def test_drift_scan_skipped_when_secondary_lacks_delete_support(self, sync_service_without_delete_support):
        """Drift scan must be skipped (not run get_all_memories blindly) when the
        secondary does not support bulk operations — otherwise it advances the
        last-check clock without reconciling anything (greptile P1 #3, PR #1474)."""
        primary, secondary, sync_service = sync_service_without_delete_support
        sync_service.drift_check_enabled = True

        stats = await sync_service._detect_and_sync_drift()

        # Skipped early: zero work, no exception.
        assert stats == {'checked': 0, 'drift_detected': 0, 'synced': 0, 'failed': 0}, \
            "Drift scan should return a zeroed result and skip when secondary lacks supports_delete_operations"
