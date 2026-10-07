#!/usr/bin/env python3
"""
Tests for Cloudflare limit handling in hybrid storage.

Tests cover:
- Pre-sync validation for oversized metadata
- Vector count limit enforcement
- Capacity monitoring and warnings
- Error handling for limit-related failures
"""

import asyncio
import pytest
import pytest_asyncio
import json
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add src to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from mcp_memory_service.storage.hybrid import HybridMemoryStorage, BackgroundSyncService, SyncOperation
from mcp_memory_service.models.memory import Memory
import hashlib


class MockCloudflareWithLimits:
    """Mock Cloudflare storage that simulates limit errors."""

    def __init__(self, **kwargs):
        self.memories = {}
        self.vector_count = 0
        self.max_vectors = 100  # Low limit for testing
        self.initialized = False
        self.fail_on_limit = True

    @property
    def supports_capacity_monitoring(self) -> bool:
        """Mock Cloudflare storage supports capacity monitoring."""
        return True

    @property
    def requires_metadata_normalization(self) -> bool:
        """Mock Cloudflare storage requires metadata normalization."""
        return True

    async def initialize(self):
        self.initialized = True

    async def store(self, memory: Memory):
        # Check vector limit
        if self.vector_count >= self.max_vectors and self.fail_on_limit:
            raise Exception("413 Request Entity Too Large: Vector limit exceeded")

        # Check metadata size (simulate 10KB limit)
        if memory.metadata:
            metadata_json = json.dumps(memory.metadata)
            if len(metadata_json) > 10240:  # 10KB
                raise Exception("Metadata too large: exceeds 10KB limit")

        self.memories[memory.content_hash] = memory
        self.vector_count += 1
        return True, "Stored"

    async def delete(self, content_hash: str):
        if content_hash in self.memories:
            del self.memories[content_hash]
            self.vector_count -= 1
        return True, "Deleted"

    async def get_stats(self):
        return {
            "total_memories": self.vector_count,
            "storage_backend": "MockCloudflareWithLimits",
            "vector_count": self.vector_count,      # Required by hybrid.py capacity checks
            "total_vectors": self.vector_count       # Alternative field name for compatibility
        }

    async def update_memory_metadata(self, content_hash: str, updates, preserve_timestamps=True):
        return True, "Updated"

    async def close(self):
        pass


@pytest_asyncio.fixture
async def temp_db():
    """Create a temporary SQLite database."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    yield db_path
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest_asyncio.fixture
async def hybrid_with_limits(temp_db):
    """Create hybrid storage with limit-aware mock Cloudflare."""
    config = {
        'api_token': 'test',
        'account_id': 'test',
        'vectorize_index': 'test',
        'd1_database_id': 'test'
    }

    with patch('mcp_memory_service.storage.hybrid.CloudflareStorage', MockCloudflareWithLimits):
        with patch('mcp_memory_service.storage.hybrid.CLOUDFLARE_VECTORIZE_MAX_VECTORS', 100):
            with patch('mcp_memory_service.storage.hybrid.CLOUDFLARE_MAX_METADATA_SIZE_KB', 10):
                storage = HybridMemoryStorage(
                    sqlite_db_path=temp_db,
                    cloudflare_config=config,
                    sync_interval=300,
                    batch_size=5
                )
                await storage.initialize()
                yield storage
                await storage.close()


class TestCloudflareMetadataLimits:
    """Test metadata size validation."""

    @pytest.mark.asyncio
    async def test_oversized_metadata_validation(self, hybrid_with_limits):
        """Test that oversized metadata is caught during validation."""
        # Create memory with large metadata (> 10KB)
        large_metadata = {"data": "x" * 11000}  # Over 10KB when serialized

        memory = Memory(
            content="Test memory with large metadata",
            content_hash=hashlib.sha256(b"test").hexdigest(),
            tags=["test"],
            metadata=large_metadata
        )

        # Validation should fail
        is_valid, error = await hybrid_with_limits.sync_service.validate_memory_for_cloudflare(memory)
        assert not is_valid
        assert "exceeds Cloudflare limit" in error

    @pytest.mark.asyncio
    async def test_normal_metadata_passes_validation(self, hybrid_with_limits):
        """Test that normal-sized metadata passes validation."""
        normal_metadata = {"key": "value", "index": 123}

        memory = Memory(
            content="Test memory with normal metadata",
            content_hash=hashlib.sha256(b"test2").hexdigest(),
            tags=["test"],
            metadata=normal_metadata
        )

        # Validation should pass
        is_valid, error = await hybrid_with_limits.sync_service.validate_memory_for_cloudflare(memory)
        assert is_valid
        assert error is None


class TestVectorCountLimits:
    """Test vector count limit handling."""

    @pytest.mark.asyncio
    async def test_vector_limit_detection(self, hybrid_with_limits):
        """Test detection when approaching vector count limit."""
        # Simulate high vector count in the mock Cloudflare backend
        hybrid_with_limits.sync_service.secondary.vector_count = 95

        # Check capacity should detect we're at 95% (critical)
        capacity = await hybrid_with_limits.sync_service.check_cloudflare_capacity()

        assert capacity['vector_usage_percent'] == 95.0
        assert capacity['approaching_limits'] is True
        assert len(capacity['warnings']) > 0
        assert "CRITICAL" in capacity['warnings'][0]

    @pytest.mark.asyncio
    async def test_vector_limit_enforcement(self, hybrid_with_limits):
        """Test that sync stops when vector limit is reached."""
        # Set vector count at limit
        hybrid_with_limits.sync_service.cloudflare_stats['vector_count'] = 100

        memory = Memory(
            content="Memory that should be rejected",
            content_hash=hashlib.sha256(b"rejected").hexdigest(),
            tags=["test"]
        )

        # Validation should fail due to limit
        is_valid, error = await hybrid_with_limits.sync_service.validate_memory_for_cloudflare(memory)
        assert not is_valid
        assert "vector limit" in error.lower()


class TestErrorHandling:
    """Test error handling for various limit scenarios."""

    @pytest.mark.asyncio
    async def test_limit_error_no_retry(self, hybrid_with_limits):
        """Test that limit errors are not retried."""
        operation = SyncOperation(
            operation='store',
            memory=Memory(content="test", content_hash="hash123", tags=[])
        )

        # Simulate a limit error
        error = Exception("413 Request Entity Too Large: Vector limit exceeded")

        await hybrid_with_limits.sync_service._handle_sync_error(error, operation)

        # Should not be added to retry queue
        assert len(hybrid_with_limits.sync_service.failed_operations) == 0
        # Should be marked as failed
        assert hybrid_with_limits.sync_service.sync_stats['operations_failed'] == 1

    @pytest.mark.asyncio
    async def test_temporary_error_retry(self, hybrid_with_limits):
        """Test that temporary errors are retried."""
        operation = SyncOperation(
            operation='store',
            memory=Memory(content="test", content_hash="hash456", tags=[]),
            retries=0
        )

        # Simulate a temporary error
        error = Exception("503 Service Temporarily Unavailable")

        await hybrid_with_limits.sync_service._handle_sync_error(error, operation)

        # Should be added to retry queue
        assert len(hybrid_with_limits.sync_service.failed_operations) == 1
        # Should not be marked as failed yet
        assert hybrid_with_limits.sync_service.sync_stats['operations_failed'] == 0

    @pytest.mark.asyncio
    async def test_max_retries_reached(self, hybrid_with_limits):
        """Test that operations fail after max retries."""
        operation = SyncOperation(
            operation='store',
            memory=Memory(content="test", content_hash="hash789", tags=[]),
            retries=2,  # Already retried twice
            max_retries=3
        )

        # Simulate another temporary error
        error = Exception("Connection timeout")

        await hybrid_with_limits.sync_service._handle_sync_error(error, operation)

        # Should not be added to retry queue (max retries reached)
        assert len(hybrid_with_limits.sync_service.failed_operations) == 0
        # Should be marked as failed
        assert hybrid_with_limits.sync_service.sync_stats['operations_failed'] == 1


class TestCapacityMonitoring:
    """Test capacity monitoring and warnings."""

    @pytest.mark.asyncio
    async def test_capacity_warning_thresholds(self, hybrid_with_limits):
        """Test warning at 80% and critical at 95% thresholds."""
        service = hybrid_with_limits.sync_service

        # Test 50% - no warning
        service.secondary.vector_count = 50
        capacity = await service.check_cloudflare_capacity()
        assert not capacity['approaching_limits']
        assert len(capacity['warnings']) == 0

        # Test 80% - warning
        service.secondary.vector_count = 80
        capacity = await service.check_cloudflare_capacity()
        assert capacity['approaching_limits']
        assert "WARNING" in capacity['warnings'][0]

        # Test 95% - critical
        service.secondary.vector_count = 95
        capacity = await service.check_cloudflare_capacity()
        assert capacity['approaching_limits']
        assert "CRITICAL" in capacity['warnings'][0]

    @pytest.mark.asyncio
    async def test_sync_status_includes_capacity(self, hybrid_with_limits):
        """Test that sync status includes capacity information."""
        status = await hybrid_with_limits.sync_service.get_sync_status()

        assert 'capacity' in status
        assert 'vector_count' in status['capacity']
        assert 'vector_limit' in status['capacity']
        assert 'approaching_limits' in status['capacity']
        assert 'warnings' in status['capacity']


class TestIntegrationScenarios:
    """Test complete scenarios with limit handling."""

    @pytest.mark.asyncio
    async def test_sync_stops_at_limit(self, hybrid_with_limits):
        """Test that sync gracefully handles reaching the limit."""
        # Add memories up to the limit
        memories = []
        for i in range(105):  # Try to exceed limit of 100
            memory = Memory(
                content=f"Memory {i}",
                content_hash=hashlib.sha256(f"hash{i}".encode()).hexdigest(),
                tags=["bulk"],
                metadata={"index": i}
            )
            memories.append(memory)

        # Process memories
        successful = 0
        failed = 0

        for memory in memories:
            operation = SyncOperation(operation='store', memory=memory)

            # Validate first
            is_valid, _ = await hybrid_with_limits.sync_service.validate_memory_for_cloudflare(memory)

            if is_valid:
                try:
                    await hybrid_with_limits.sync_service._process_single_operation(operation)
                    successful += 1
                except Exception:
                    failed += 1
            else:
                failed += 1

        # Should stop at or before the limit
        assert successful <= 100
        assert failed >= 5  # At least 5 should fail due to limit

    @pytest.mark.asyncio
    async def test_periodic_capacity_check(self, hybrid_with_limits):
        """Test that periodic sync checks capacity once its interval is due."""
        # Set up near-limit scenario in the mock backend
        hybrid_with_limits.sync_service.secondary.vector_count = 85

        # Capacity monitoring counts rows on the secondary, so it runs on its own
        # interval rather than every cycle. Startup already used up the first slot.
        hybrid_with_limits.sync_service.cloudflare_stats['last_capacity_check'] = 0

        # Run periodic sync
        await hybrid_with_limits.sync_service._periodic_sync()

        # Should have detected approaching limits
        assert hybrid_with_limits.sync_service.cloudflare_stats['approaching_limits']
        assert len(hybrid_with_limits.sync_service.cloudflare_stats['limit_warnings']) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])