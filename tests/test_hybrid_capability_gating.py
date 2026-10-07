#!/usr/bin/env python3
"""
Tests for hybrid storage capability gating (Gate 2 - TDD RED).

Tests the capability properties and gating behavior that will be implemented in Gate 3.
These tests MUST FAIL until the implementation is done.

Testing Requirements R10-R13:
- R10: BackgroundSyncService typed against MemoryStorage
- R11: drift via polymorphic contract (already done in Phase 2 - don't retest)
- R12: CF-only behind capability-check 
- R13: secondary Cloudflare BYTE-IDENTICAL (regression zero)
"""

import asyncio
import pytest
import pytest_asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch, call
from typing import Dict, Any

# Add src to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from mcp_memory_service.storage.base import MemoryStorage
from mcp_memory_service.storage.hybrid import BackgroundSyncService, SyncOperation
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.utils.hashing import generate_content_hash


class MockMemoryStorage(MemoryStorage):
    """Mock storage with configurable capabilities."""
    
    def __init__(self, supports_capacity=False, requires_metadata_norm=False):
        self._supports_capacity_monitoring = supports_capacity
        self._requires_metadata_normalization = requires_metadata_norm
        self.call_log = []
        self.store_calls = []
        self.update_calls = []
        self.delete_calls = []

    @property
    def max_content_length(self):
        return None

    @property
    def supports_chunking(self):
        return False

    @property
    def supports_capacity_monitoring(self):
        """This property should exist and return the configured value."""
        return self._supports_capacity_monitoring

    @property  
    def requires_metadata_normalization(self):
        """This property should exist and return the configured value."""
        return self._requires_metadata_normalization

    async def initialize(self):
        pass

    async def store(self, memory, skip_semantic_dedup=False, store="default"):
        self.call_log.append(f"store({memory.content_hash})")
        self.store_calls.append(memory)
        return True, "Success"

    async def delete(self, content_hash):
        self.call_log.append(f"delete({content_hash})")  
        self.delete_calls.append(content_hash)
        return True, "Success"

    async def update_memory_metadata(self, content_hash, updates, preserve_timestamps=True):
        self.call_log.append(f"update_memory_metadata({content_hash}, {updates})")
        self.update_calls.append((content_hash, updates, preserve_timestamps))
        return True, "Success"

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


class TestCapabilityProperties:
    """Test that the capability properties exist and have correct defaults.

    These assert on INSTANCE values of real backends (inherited base defaults
    vs. Cloudflare overrides), not just ``hasattr`` on the class, so a backend
    that returns the wrong capability value cannot pass (greptile P2, PR #1474).
    """

    def test_base_defaults_false_on_inheriting_backend(self):
        """A backend that does NOT override the properties inherits False from the base.

        RemoteHTTPStorage adds no override, so its instance reflects the base
        class default. Asserting on the instance (not a mock that could shadow
        the property) proves the real inherited default is False.
        """
        from mcp_memory_service.storage.remote_http import RemoteHTTPStorage

        storage = RemoteHTTPStorage(base_url="https://example.invalid", api_key="x")
        assert storage.supports_capacity_monitoring is False, \
            "Base default supports_capacity_monitoring must be False on a non-overriding backend"
        assert storage.requires_metadata_normalization is False, \
            "Base default requires_metadata_normalization must be False on a non-overriding backend"

    def test_cloudflare_overrides_capabilities_true(self):
        """CloudflareStorage must OVERRIDE both properties to True (value, not just presence)."""
        from mcp_memory_service.storage.cloudflare import CloudflareStorage

        # The properties must resolve to True at the class level without needing
        # a configured instance. If they are plain properties, read via the
        # class descriptor; this asserts the actual True value, not mere presence.
        assert CloudflareStorage.supports_capacity_monitoring.fget(object()) is True, \
            "CloudflareStorage must override supports_capacity_monitoring to True"
        assert CloudflareStorage.requires_metadata_normalization.fget(object()) is True, \
            "CloudflareStorage must override requires_metadata_normalization to True"

    def test_remote_http_keeps_defaults_false(self):
        """RemoteHTTPStorage must keep the inherited False (it is the HTTP secondary path)."""
        from mcp_memory_service.storage.remote_http import RemoteHTTPStorage

        storage = RemoteHTTPStorage(base_url="https://example.invalid", api_key="x")
        assert storage.supports_capacity_monitoring is False
        assert storage.requires_metadata_normalization is False


class TestBackgroundSyncServiceTyping:
    """Test that BackgroundSyncService accepts MemoryStorage (not just CloudflareStorage)."""

    @pytest_asyncio.fixture
    async def temp_sqlite_db(self, tmp_path):
        """Create a temporary SQLite database."""
        db_path = tmp_path / "test.db"
        yield str(db_path)

    @pytest.mark.asyncio
    async def test_sync_service_accepts_memory_storage(self, temp_sqlite_db):
        """BackgroundSyncService should accept any MemoryStorage as secondary, not just CloudflareStorage."""
        primary = SqliteVecMemoryStorage(temp_sqlite_db)
        await primary.initialize()
        
        # Create a non-CloudflareStorage secondary
        secondary = MockMemoryStorage(supports_capacity=False, requires_metadata_norm=False)
        await secondary.initialize()
        
        try:
            # This will FAIL if BackgroundSyncService still has CloudflareStorage typing
            sync_service = BackgroundSyncService(
                primary, 
                secondary,  # Should accept any MemoryStorage, not just CloudflareStorage
                sync_interval=1,
                batch_size=3
            )
            
            # Should be able to start without type errors
            await sync_service.start()
            assert sync_service.is_running
            await sync_service.stop()
            
        except TypeError as e:
            pytest.fail(f"BackgroundSyncService should accept MemoryStorage, not just CloudflareStorage: {e}")
        finally:
            if hasattr(primary, 'close'):
                try:
                    await primary.close()
                except:
                    pass

    @pytest.mark.asyncio
    async def test_sync_service_works_with_remote_http_storage(self, temp_sqlite_db):
        """BackgroundSyncService should work with RemoteHTTPStorage as secondary."""
        primary = SqliteVecMemoryStorage(temp_sqlite_db)
        await primary.initialize()
        
        try:
            from mcp_memory_service.storage.remote_http import RemoteHTTPStorage
            
            # Create RemoteHTTP instance with minimal config
            secondary = RemoteHTTPStorage("http://localhost:8080")
            
            # This will FAIL if typing is still CloudflareStorage-only
            sync_service = BackgroundSyncService(primary, secondary)
            assert sync_service.secondary == secondary
            
        except (ImportError, TypeError) as e:
            pytest.fail(f"BackgroundSyncService should accept RemoteHTTPStorage: {e}")
        finally:
            if hasattr(primary, 'close'):
                try:
                    await primary.close()
                except:
                    pass


class TestCapabilityGating:
    """Test that CF-only functions are gated behind capability checks."""

    @pytest_asyncio.fixture
    async def temp_sqlite_db(self, tmp_path):
        """Create a temporary SQLite database."""
        db_path = tmp_path / "test.db"  
        yield str(db_path)

    @pytest_asyncio.fixture
    async def sync_service_with_cf_like_secondary(self, temp_sqlite_db):
        """Create sync service with CF-like secondary (both capabilities True)."""
        primary = SqliteVecMemoryStorage(temp_sqlite_db)
        await primary.initialize()
        
        # CF-like secondary: both capabilities True
        secondary = MockMemoryStorage(supports_capacity=True, requires_metadata_norm=True)
        await secondary.initialize()
        
        sync_service = BackgroundSyncService(primary, secondary, sync_interval=1, batch_size=3)
        
        yield primary, secondary, sync_service
        
        if sync_service.is_running:
            await sync_service.stop()
        if hasattr(primary, 'close'):
            try:
                await primary.close()
            except:
                pass

    @pytest_asyncio.fixture  
    async def sync_service_with_http_like_secondary(self, temp_sqlite_db):
        """Create sync service with HTTP-like secondary (both capabilities False)."""
        primary = SqliteVecMemoryStorage(temp_sqlite_db)
        await primary.initialize()
        
        # HTTP-like secondary: both capabilities False
        secondary = MockMemoryStorage(supports_capacity=False, requires_metadata_norm=False)
        await secondary.initialize()
        
        sync_service = BackgroundSyncService(primary, secondary, sync_interval=1, batch_size=3)
        
        yield primary, secondary, sync_service
        
        if sync_service.is_running:
            await sync_service.stop()
        if hasattr(primary, 'close'):
            try:
                await primary.close()
            except:
                pass

    @pytest.mark.asyncio
    async def test_cf_only_functions_run_with_cf_like_secondary(self, sync_service_with_cf_like_secondary):
        """When secondary has CF capabilities, CF-only functions should be called."""
        primary, secondary, sync_service = sync_service_with_cf_like_secondary
        
        # Create test memory
        content = "test content"
        memory = Memory(
            content=content,
            content_hash=generate_content_hash(content),
            tags=["test"],
            memory_type="observation"
        )

        with patch.object(sync_service, 'validate_memory_for_cloudflare') as mock_validate, \
             patch('mcp_memory_service.storage.hybrid._normalize_metadata_for_cloudflare') as mock_normalize, \
             patch.object(sync_service, 'check_cloudflare_capacity') as mock_capacity:
            
            mock_validate.side_effect = AsyncMock(return_value=(True, "Valid"))
            mock_normalize.return_value = {"normalized": True}
            mock_capacity.side_effect = AsyncMock(return_value=None)

            # Process store operation
            store_op = SyncOperation(operation='store', memory=memory)
            await sync_service._process_single_operation(store_op)
            
            # Process update operation  
            update_op = SyncOperation(
                operation='update', 
                content_hash=memory.content_hash,
                updates={"metadata": {"key": "value"}}
            )
            await sync_service._process_single_operation(update_op)

            # This will FAIL because gating is not implemented yet - CF functions run always
            # With gating, these should be called because secondary.supports_capacity_monitoring = True
            mock_validate.assert_called_once()  # Should be called for store op
            mock_normalize.assert_called_once()  # Should be called for update op  
            mock_capacity.assert_not_called()  # Should NOT be called during normal ops (only error handler + periodic)

    @pytest.mark.asyncio
    async def test_cf_only_functions_skip_with_http_like_secondary(self, sync_service_with_http_like_secondary):
        """When secondary lacks CF capabilities, CF-only functions should be skipped."""
        primary, secondary, sync_service = sync_service_with_http_like_secondary
        
        # Create test memory
        content = "test content"
        memory = Memory(
            content=content,
            content_hash=generate_content_hash(content),
            tags=["test"], 
            memory_type="observation"
        )

        with patch.object(sync_service, 'validate_memory_for_cloudflare') as mock_validate, \
             patch('mcp_memory_service.storage.hybrid._normalize_metadata_for_cloudflare') as mock_normalize, \
             patch.object(sync_service, 'check_cloudflare_capacity') as mock_capacity:
            
            mock_validate.side_effect = AsyncMock(return_value=(True, "Valid"))
            mock_normalize.return_value = {"normalized": True}
            mock_capacity.side_effect = AsyncMock(return_value=None)

            # Process store operation
            store_op = SyncOperation(operation='store', memory=memory)
            await sync_service._process_single_operation(store_op)
            
            # Process update operation
            update_op = SyncOperation(
                operation='update',
                content_hash=memory.content_hash, 
                updates={"metadata": {"key": "value"}}
            )
            await sync_service._process_single_operation(update_op)

            # This will FAIL because gating is not implemented yet - CF functions still run
            # With gating, these should NOT be called because secondary capabilities are False
            mock_validate.assert_not_called()  # Should be skipped for store op
            mock_normalize.assert_not_called()  # Should be skipped for update op
            mock_capacity.assert_not_called()  # Should be skipped

    @pytest.mark.asyncio  
    async def test_update_metadata_passes_raw_when_normalization_not_required(self, sync_service_with_http_like_secondary):
        """When secondary doesn't require metadata normalization, raw updates should be passed through."""
        primary, secondary, sync_service = sync_service_with_http_like_secondary
        
        content_hash = "test_hash"
        original_updates = {"metadata": {"key": "value", "tags": ["tag1", "tag2"]}}
        
        # Process update operation  
        update_op = SyncOperation(
            operation='update',
            content_hash=content_hash,
            updates=original_updates
        )
        
        await sync_service._process_single_operation(update_op)
        
        # This will FAIL because gating is not implemented - normalization still runs
        # With gating, raw updates should be passed to secondary without normalization
        assert len(secondary.update_calls) == 1
        _, passed_updates, _ = secondary.update_calls[0]
        
        # Should be exactly the original updates, not normalized
        assert passed_updates == original_updates, \
            f"Expected raw updates {original_updates}, got {passed_updates}"


class TestByteIdenticalBehavior:
    """Test R13: secondary Cloudflare behavior should be BYTE-IDENTICAL (regression zero)."""

    @pytest_asyncio.fixture
    async def temp_sqlite_db(self, tmp_path):
        db_path = tmp_path / "test.db"
        yield str(db_path)

    @pytest.mark.asyncio
    async def test_cloudflare_like_secondary_gets_identical_calls(self, temp_sqlite_db):
        """With CF-like secondary, the call sequence and payload should be identical to current behavior."""
        primary = SqliteVecMemoryStorage(temp_sqlite_db)
        await primary.initialize()
        
        # CF-like secondary that tracks all calls  
        secondary = MockMemoryStorage(supports_capacity=True, requires_metadata_norm=True)
        await secondary.initialize()
        
        sync_service = BackgroundSyncService(primary, secondary, sync_interval=1, batch_size=3)
        
        try:
            # Test memory
            content = "test content for byte-identical check"
            memory = Memory(
                content=content,
                content_hash=generate_content_hash(content),
                tags=["test", "byte-identical"],
                memory_type="observation"
            )

            with patch('mcp_memory_service.storage.hybrid._normalize_metadata_for_cloudflare') as mock_normalize, \
                 patch.object(sync_service, 'check_cloudflare_capacity') as mock_capacity:
                # Mock normalization to return predictable result  
                normalized_metadata = {"normalized": True, "cloudflare_ready": True}
                mock_normalize.return_value = normalized_metadata
                mock_capacity.side_effect = AsyncMock(return_value=None)

                # Process store operation
                store_op = SyncOperation(operation='store', memory=memory)
                await sync_service._process_single_operation(store_op)
                
                # Process update operation
                update_op = SyncOperation(
                    operation='update',
                    content_hash=memory.content_hash,
                    updates={"metadata": {"original": "data"}}
                )  
                await sync_service._process_single_operation(update_op)

                # Process delete operation
                delete_op = SyncOperation(operation='delete', content_hash=memory.content_hash)
                await sync_service._process_single_operation(delete_op)

                # CRITICAL: R13 regression test - store should NOT trigger extra capacity check
                # Before Phase 3: capacity check called only from error handler + periodic (2 sites)
                # After Phase 3 with regression: capacity check called after every store (3rd site)
                # This assertion prevents the regression from returning
                assert mock_capacity.call_count == 0, \
                    f"check_cloudflare_capacity should NOT be called during normal store operations (R13 regression). " \
                    f"It should only be called from error handler and periodic sync (original 2 call sites). " \
                    f"Actual calls: {mock_capacity.call_count}"

            # This will FAIL until gating is implemented correctly
            # The call sequence should be:
            # 1. store(memory) - direct pass-through
            # 2. update_memory_metadata(hash, NORMALIZED_metadata) - normalized because requires_metadata_normalization=True  
            # 3. delete(hash) - direct pass-through
            
            expected_call_log = [
                f"store({memory.content_hash})",
                f"update_memory_metadata({memory.content_hash}, {normalized_metadata})",
                f"delete({memory.content_hash})"
            ]
            
            assert secondary.call_log == expected_call_log, \
                f"Expected call sequence {expected_call_log}, got {secondary.call_log}"
            
            # Verify the normalized metadata was passed to update_memory_metadata
            assert len(secondary.update_calls) == 1
            _, passed_updates, _ = secondary.update_calls[0] 
            assert passed_updates == normalized_metadata, \
                f"Expected normalized metadata {normalized_metadata}, got {passed_updates}"

        finally:
            if sync_service.is_running:
                await sync_service.stop()
            if hasattr(primary, 'close'):
                try:
                    await primary.close()
                except:
                    pass