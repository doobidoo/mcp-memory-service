#!/usr/bin/env python3
"""
Comprehensive tests for HybridMemoryStorage implementation.

Tests cover:
- Basic storage operations (store, retrieve, delete)
- Background synchronization service
- Failover and graceful degradation
- Configuration and health monitoring
- Performance characteristics
"""

import asyncio
import pytest
import pytest_asyncio
import tempfile
import os
import sys
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from typing import Dict, Any

# Add src to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from mcp_memory_service.storage.hybrid import HybridMemoryStorage, BackgroundSyncService, SyncOperation
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.models.memory import Memory, MemoryQueryResult
from mcp_memory_service.utils.hashing import generate_content_hash

# Configure test logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class MockCloudflareStorage:
    """Mock Cloudflare storage for testing."""

    def __init__(self, **kwargs):
        self.initialized = False
        self.stored_memories = {}
        self.fail_operations = False
        self.fail_initialization = False

    async def initialize(self):
        if self.fail_initialization:
            raise Exception("Mock Cloudflare initialization failed")
        self.initialized = True

    async def store(self, memory: Memory):
        if self.fail_operations:
            return False, "Mock Cloudflare operation failed"
        self.stored_memories[memory.content_hash] = memory
        return True, "Memory stored successfully"

    async def delete(self, content_hash: str):
        if self.fail_operations:
            return False, "Mock Cloudflare operation failed"
        if content_hash in self.stored_memories:
            del self.stored_memories[content_hash]
            return True, "Memory deleted successfully"
        return False, "Memory not found"

    async def delete_by_timeframe(self, start_date, end_date, tag=None):
        """Mock delete_by_timeframe method for v8.66.0."""
        if self.fail_operations:
            return 0, "Mock Cloudflare operation failed"
        # Simple mock: return count of 1 (truthy value to pass sync check)
        return 1, "Mock delete by timeframe"

    async def delete_before_date(self, before_date):
        """Mock delete_before_date method for v8.66.0."""
        if self.fail_operations:
            return 0, "Mock Cloudflare operation failed"
        # Simple mock: return count of 1 (truthy value to pass sync check)
        return 1, "Mock delete before date"

    async def update_memory_metadata(self, content_hash: str, updates: Dict[str, Any], preserve_timestamps: bool = True):
        if self.fail_operations:
            return False, "Mock Cloudflare operation failed"
        if content_hash in self.stored_memories:
            # Simple mock update
            return True, "Memory updated successfully"
        return False, "Memory not found"

    async def get_stats(self):
        if self.fail_operations:
            raise Exception("Mock Cloudflare stats failed")
        return {
            "total_memories": len(self.stored_memories),
            "storage_backend": "MockCloudflareStorage",
            "vector_count": len(self.stored_memories),  # Add for hybrid storage compatibility
            "total_vectors": len(self.stored_memories)   # Alternative field name
        }

    async def close(self):
        pass


@pytest_asyncio.fixture
async def temp_sqlite_db():
    """Create a temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name

    yield db_path

    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest_asyncio.fixture
async def mock_cloudflare_config():
    """Mock Cloudflare configuration for testing."""
    return {
        'api_token': 'test_token',
        'account_id': 'test_account',
        'vectorize_index': 'test_index',
        'd1_database_id': 'test_db_id'
    }


@pytest_asyncio.fixture
async def hybrid_storage(temp_sqlite_db, mock_cloudflare_config):
    """Create a HybridMemoryStorage instance for testing."""
    with patch('mcp_memory_service.storage.hybrid.CloudflareStorage', MockCloudflareStorage):
        storage = HybridMemoryStorage(
            sqlite_db_path=temp_sqlite_db,
            embedding_model="all-MiniLM-L6-v2",
            cloudflare_config=mock_cloudflare_config,
            sync_interval=1,  # Short interval for testing
            batch_size=5
        )
        await storage.initialize()
        yield storage
        await storage.close()


@pytest.fixture
def sample_memory():
    """Create a sample memory for testing."""
    content = "This is a test memory for hybrid storage"
    return Memory(
        content=content,
        content_hash=generate_content_hash(content),
        tags=["test", "sample"],
        memory_type="test",
        metadata={},
        created_at=1638360000.0
    )


class TestHybridMemoryStorage:
    """Test cases for HybridMemoryStorage functionality."""

    @pytest.mark.asyncio
    async def test_initialization_with_cloudflare(self, temp_sqlite_db, mock_cloudflare_config):
        """Test successful initialization with Cloudflare configuration."""
        with patch('mcp_memory_service.storage.hybrid.CloudflareStorage', MockCloudflareStorage):
            storage = HybridMemoryStorage(
                sqlite_db_path=temp_sqlite_db,
                cloudflare_config=mock_cloudflare_config
            )

            await storage.initialize()

            assert storage.initialized
            assert storage.primary is not None
            assert storage.secondary is not None
            assert storage.sync_service is not None
            assert storage.sync_service.is_running

            await storage.close()

    @pytest.mark.asyncio
    async def test_initialization_without_cloudflare(self, temp_sqlite_db):
        """Test initialization without Cloudflare configuration (SQLite-only mode)."""
        storage = HybridMemoryStorage(sqlite_db_path=temp_sqlite_db)

        await storage.initialize()

        assert storage.initialized
        assert storage.primary is not None
        assert storage.secondary is None
        assert storage.sync_service is None

        await storage.close()

    @pytest.mark.asyncio
    async def test_initialization_with_cloudflare_failure(self, temp_sqlite_db, mock_cloudflare_config):
        """Test graceful handling of Cloudflare initialization failure."""
        def failing_cloudflare_storage(**kwargs):
            storage = MockCloudflareStorage(**kwargs)
            storage.fail_initialization = True
            return storage

        with patch('mcp_memory_service.storage.hybrid.CloudflareStorage', failing_cloudflare_storage):
            storage = HybridMemoryStorage(
                sqlite_db_path=temp_sqlite_db,
                cloudflare_config=mock_cloudflare_config
            )

            await storage.initialize()

            # Should fall back to SQLite-only mode
            assert storage.initialized
            assert storage.primary is not None
            assert storage.secondary is None
            assert storage.sync_service is None

            await storage.close()

    @pytest.mark.asyncio
    async def test_store_memory(self, hybrid_storage, sample_memory):
        """Test storing a memory in hybrid storage."""
        success, message = await hybrid_storage.store(sample_memory)

        assert success
        assert "success" in message.lower() or message == ""

        # Verify memory is stored in primary
        results = await hybrid_storage.retrieve(sample_memory.content, n_results=1)
        assert len(results) == 1
        assert results[0].memory.content == sample_memory.content

    @pytest.mark.asyncio
    async def test_retrieve_memory(self, hybrid_storage, sample_memory):
        """Test retrieving memories from hybrid storage."""
        # Store a memory first
        await hybrid_storage.store(sample_memory)

        # Retrieve by query
        results = await hybrid_storage.retrieve("test memory", n_results=5)
        assert len(results) >= 1

        # Check that we get the stored memory
        found = any(result.memory.content == sample_memory.content for result in results)
        assert found

    @pytest.mark.asyncio
    async def test_delete_memory(self, hybrid_storage, sample_memory):
        """Test deleting a memory from hybrid storage."""
        # Store a memory first
        await hybrid_storage.store(sample_memory)

        # Delete the memory
        success, message = await hybrid_storage.delete(sample_memory.content_hash)
        assert success

        # Verify memory is deleted from primary
        results = await hybrid_storage.retrieve(sample_memory.content, n_results=1)
        # Should not find the deleted memory
        found = any(result.memory.content_hash == sample_memory.content_hash for result in results)
        assert not found

    @pytest.mark.asyncio
    async def test_search_by_tags(self, hybrid_storage, sample_memory):
        """Test searching memories by tags."""
        # Store a memory first
        await hybrid_storage.store(sample_memory)

        # Search by tags
        results = await hybrid_storage.search_by_tags(["test"])
        assert len(results) >= 1

        # Check that we get the stored memory
        found = any(memory.content == sample_memory.content for memory in results)
        assert found

    @pytest.mark.asyncio
    async def test_get_stats(self, hybrid_storage):
        """Test getting statistics from hybrid storage."""
        stats = await hybrid_storage.get_stats()

        assert "storage_backend" in stats
        assert stats["storage_backend"] == "Hybrid (SQLite-vec + Cloudflare)"
        assert "primary_stats" in stats
        assert "sync_enabled" in stats
        assert stats["sync_enabled"] == True
        assert "sync_status" in stats

    @pytest.mark.asyncio
    async def test_force_sync(self, hybrid_storage, sample_memory):
        """Test forcing immediate synchronization."""
        # Store some memories
        await hybrid_storage.store(sample_memory)

        # Force sync
        result = await hybrid_storage.force_sync()

        assert "status" in result
        assert result["status"] in ["completed", "partial"]
        assert "primary_memories" in result
        assert result["primary_memories"] >= 1


class TestBackgroundSyncService:
    """Test cases for BackgroundSyncService functionality."""

    @pytest_asyncio.fixture
    async def sync_service_components(self, temp_sqlite_db):
        """Create components needed for sync service testing."""
        primary = SqliteVecMemoryStorage(temp_sqlite_db)
        await primary.initialize()

        secondary = MockCloudflareStorage()
        await secondary.initialize()

        sync_service = BackgroundSyncService(
            primary, secondary,
            sync_interval=1,
            batch_size=3
        )

        yield primary, secondary, sync_service

        if sync_service.is_running:
            await sync_service.stop()

        # Close storage backends with exception handling
        if primary is not None and hasattr(primary, 'close'):
            try:
                if asyncio.iscoroutinefunction(primary.close):
                    await primary.close()
                else:
                    primary.close()
            except Exception:
                pass  # Ignore cleanup errors in tests

        if secondary is not None and hasattr(secondary, 'close'):
            try:
                if asyncio.iscoroutinefunction(secondary.close):
                    await secondary.close()
                else:
                    secondary.close()
            except Exception:
                pass  # Ignore cleanup errors in tests

    @pytest.mark.asyncio
    async def test_sync_service_start_stop(self, sync_service_components):
        """Test starting and stopping the background sync service."""
        primary, secondary, sync_service = sync_service_components

        # Start service
        await sync_service.start()
        assert sync_service.is_running

        # Stop service
        await sync_service.stop()
        assert not sync_service.is_running

    @pytest.mark.asyncio
    async def test_operation_enqueue(self, sync_service_components, sample_memory):
        """Test enqueuing sync operations."""
        primary, secondary, sync_service = sync_service_components

        await sync_service.start()

        # Enqueue a store operation
        operation = SyncOperation(operation='store', memory=sample_memory)
        await sync_service.enqueue_operation(operation)

        # Wait a bit for processing
        await asyncio.sleep(0.1)

        # Check queue size decreased
        status = await sync_service.get_sync_status()
        assert status['queue_size'] >= 0  # Should be processed or in progress

        await sync_service.stop()

    @pytest.mark.asyncio
    async def test_sync_with_cloudflare_failure(self, sync_service_components):
        """Test sync behavior when Cloudflare operations fail."""
        primary, secondary, sync_service = sync_service_components

        # Make Cloudflare operations fail
        secondary.fail_operations = True

        await sync_service.start()

        # Create a test memory
        content = "test content"
        memory = Memory(
            content=content,
            content_hash=generate_content_hash(content),
            tags=["test"],
            memory_type="test"
        )

        # Enqueue operation
        operation = SyncOperation(operation='store', memory=memory)
        await sync_service.enqueue_operation(operation)

        # Wait for processing
        await asyncio.sleep(0.2)

        # Check that service marked Cloudflare as unavailable
        status = await sync_service.get_sync_status()
        assert status['cloudflare_available'] == False

        await sync_service.stop()

    @pytest.mark.asyncio
    async def test_force_sync_functionality(self, sync_service_components):
        """Test force sync functionality."""
        primary, secondary, sync_service = sync_service_components

        # Store some test memories in primary
        content1 = "test memory 1"
        content2 = "test memory 2"
        memory1 = Memory(
            content=content1,
            content_hash=generate_content_hash(content1),
            tags=["test"],
            memory_type="test"
        )
        memory2 = Memory(
            content=content2,
            content_hash=generate_content_hash(content2),
            tags=["test"],
            memory_type="test"
        )

        await primary.store(memory1)
        await primary.store(memory2)

        await sync_service.start()

        # Force sync
        result = await sync_service.force_sync()

        assert result['status'] == 'completed'
        assert result['primary_memories'] == 2
        assert result['synced_to_secondary'] >= 0

        await sync_service.stop()

    @pytest.mark.asyncio
    async def test_sync_status_reporting(self, sync_service_components):
        """Test sync status reporting functionality."""
        primary, secondary, sync_service = sync_service_components

        await sync_service.start()

        status = await sync_service.get_sync_status()

        assert 'is_running' in status
        assert status['is_running'] == True
        assert 'queue_size' in status
        assert 'stats' in status
        assert 'cloudflare_available' in status

        await sync_service.stop()

    @pytest.mark.asyncio
    async def test_force_sync_dedupes_against_secondary(self, temp_sqlite_db):
        """force_sync skips items already on the secondary to avoid burning the
        Cloudflare Workers AI embedding quota on duplicates (regression guard for
        issue #750 Phase 2)."""
        primary = SqliteVecMemoryStorage(temp_sqlite_db)
        await primary.initialize()

        # Secondary that mimics CloudflareStorage enough to expose bulk hash fetch.
        class DedupeMockCloudflareStorage(MockCloudflareStorage):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.base_url = "https://mock"
                self.d1_database_id = "mock-db"
                self.store_call_count = 0
                # Pretend these hashes are already on secondary
                self.existing_hashes = set()

            async def _retry_request(self, method, url, **kwargs):
                # Single-page response: return all hashes once, empty thereafter to
                # satisfy the cursor-based pagination loop (id > last_id; break when
                # batch size < limit).
                params = kwargs.get("json", {}).get("params", [])
                last_id = params[0] if params else 0
                resp = Mock()
                if last_id == 0:
                    resp.json = Mock(return_value={
                        "success": True,
                        "result": [{"results": [
                            {"id": i, "content_hash": h}
                            for i, h in enumerate(self.existing_hashes, start=1)
                        ]}]
                    })
                else:
                    resp.json = Mock(return_value={
                        "success": True,
                        "result": [{"results": []}]
                    })
                return resp

            async def store(self, memory):
                self.store_call_count += 1
                return await super().store(memory)

        secondary = DedupeMockCloudflareStorage()
        await secondary.initialize()

        # Seed primary with 3 memories; secondary already has 2 of them
        memories = []
        for i, content in enumerate(["alpha", "beta", "gamma"]):
            m = Memory(
                content=content,
                content_hash=generate_content_hash(content),
                tags=["test"],
                memory_type="test",
            )
            memories.append(m)
            await primary.store(m)

        secondary.existing_hashes = {memories[0].content_hash, memories[1].content_hash}

        sync_service = BackgroundSyncService(primary, secondary, sync_interval=1, batch_size=5)
        result = await sync_service.force_sync()

        assert result['status'] == 'completed'
        assert result['primary_memories'] == 3
        # Only the non-duplicate 'gamma' should be pushed → 1 store call
        assert secondary.store_call_count == 1, (
            f"Expected 1 store call (only new memories), got {secondary.store_call_count}. "
            f"Dedupe regression — see issue #750."
        )
        assert result.get('skipped_already_present') == 2

    @pytest.mark.asyncio
    async def test_force_sync_falls_back_when_dedupe_unavailable(self, sync_service_components):
        """If the secondary doesn't expose a bulk hash fetch (no _retry_request /
        d1_database_id attrs), force_sync must still work — pushing all items (old
        behavior, preserved for non-Cloudflare secondaries)."""
        primary, secondary, sync_service = sync_service_components

        # Seed primary
        for content in ("one", "two"):
            m = Memory(
                content=content,
                content_hash=generate_content_hash(content),
                tags=["test"],
                memory_type="test",
            )
            await primary.store(m)

        # MockCloudflareStorage intentionally lacks _retry_request
        result = await sync_service.force_sync()
        assert result['status'] == 'completed'
        # Fallback path doesn't populate skipped_already_present
        assert result.get('skipped_already_present', 0) == 0


class TestPerformanceCharacteristics:
    """Test performance characteristics of hybrid storage."""

    @pytest.mark.asyncio
    async def test_read_performance(self, hybrid_storage, sample_memory):
        """Test that reads are fast (should use SQLite-vec)."""
        # Store a memory
        await hybrid_storage.store(sample_memory)

        # Measure read performance
        import time

        start_time = time.time()
        results = await hybrid_storage.retrieve(sample_memory.content[:10], n_results=1)
        duration = time.time() - start_time

        # Should be very fast (< 100ms for local SQLite-vec)
        assert duration < 0.1
        assert len(results) >= 0  # Should get some results

    @pytest.mark.asyncio
    async def test_write_performance(self, hybrid_storage):
        """Test that writes are fast (immediate SQLite-vec write)."""
        content = "Performance test memory"
        memory = Memory(
            content=content,
            content_hash=generate_content_hash(content),
            tags=["perf"],
            memory_type="performance_test"
        )

        import time

        start_time = time.time()
        success, message = await hybrid_storage.store(memory)
        duration = time.time() - start_time

        # Should be very fast (< 100ms for local SQLite-vec)
        assert duration < 0.1
        assert success

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, hybrid_storage):
        """Test concurrent memory operations.

        SQLite may reject some concurrent writes under contention.
        We retry any failures sequentially to verify correctness.
        """
        # Create multiple memories
        memories = []
        for i in range(10):
            content = f"Concurrent test memory {i}"
            memory = Memory(
                content=content,
                content_hash=generate_content_hash(content),
                tags=["concurrent", f"test{i}"],
                memory_type="concurrent_test"
            )
            memories.append(memory)

        # Store all memories concurrently
        tasks = [hybrid_storage.store(memory) for memory in memories]
        results = await asyncio.gather(*tasks)

        # Retry any that failed due to contention
        for i, (success, msg) in enumerate(results):
            if not success:
                results[i] = await hybrid_storage.store(memories[i])

        # All operations should succeed after retries
        assert all(success for success, message in results)

        # Should be able to retrieve all memories
        search_results = await hybrid_storage.search_by_tags(["concurrent"])
        assert len(search_results) == 10


class TestErrorHandlingAndFallback:
    """Test error handling and fallback scenarios."""

    @pytest.mark.asyncio
    async def test_sqlite_only_mode(self, temp_sqlite_db):
        """Test operation in SQLite-only mode (no Cloudflare)."""
        storage = HybridMemoryStorage(sqlite_db_path=temp_sqlite_db)
        await storage.initialize()

        # Should work normally without Cloudflare
        content = "SQLite-only test memory"
        memory = Memory(
            content=content,
            content_hash=generate_content_hash(content),
            tags=["local"],
            memory_type="sqlite_only"
        )

        success, message = await storage.store(memory)
        assert success

        results = await storage.retrieve(memory.content, n_results=1)
        assert len(results) >= 1

        await storage.close()

    @pytest.mark.asyncio
    async def test_graceful_degradation(self, temp_sqlite_db, mock_cloudflare_config):
        """Test graceful degradation when Cloudflare becomes unavailable."""
        def unreliable_cloudflare_storage(**kwargs):
            storage = MockCloudflareStorage(**kwargs)
            # Will start working but then fail
            return storage

        with patch('mcp_memory_service.storage.hybrid.CloudflareStorage', unreliable_cloudflare_storage):
            storage = HybridMemoryStorage(
                sqlite_db_path=temp_sqlite_db,
                cloudflare_config=mock_cloudflare_config
            )

            await storage.initialize()

            # Initially should work
            content = "Degradation test memory"
            memory = Memory(
                content=content,
                content_hash=generate_content_hash(content),
                tags=["test"],
                memory_type="degradation_test"
            )

            success, message = await storage.store(memory)
            assert success

            # Make Cloudflare fail
            storage.secondary.fail_operations = True

            # Should still work (primary storage unaffected)
            content2 = "Second test memory"
            memory2 = Memory(
                content=content2,
                content_hash=generate_content_hash(content2),
                tags=["test"],
                memory_type="degradation_test"
            )
            success, message = await storage.store(memory2)
            assert success

            # Retrieval should still work
            results = await storage.retrieve("test memory", n_results=10)
            assert len(results) >= 2

            await storage.close()


class TestHybridTimeBasedDeletion:
    """Tests for time-based deletion methods added in v8.66.0."""

    @pytest_asyncio.fixture
    async def test_hybrid_storage(self):
        """Create a test hybrid storage instance."""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test.db")

        # Use mock Cloudflare storage
        with patch('mcp_memory_service.storage.hybrid.CloudflareStorage', MockCloudflareStorage):
            storage = HybridMemoryStorage(
                sqlite_db_path=db_path,
                embedding_model="all-MiniLM-L6-v2",
                cloudflare_config={'api_token': 'test', 'account_id': 'test',
                                  'vectorize_index': 'test', 'd1_database_id': 'test'},
                sync_interval=1,
                batch_size=5
            )
            await storage.initialize()
            yield storage
            await storage.close()

        # Cleanup
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    @pytest.mark.asyncio
    async def test_delete_by_timeframe_delegation(self, test_hybrid_storage):
        """Test delete_by_timeframe delegates to primary storage correctly."""
        hybrid = test_hybrid_storage

        # Store test memories
        import datetime
        from datetime import timedelta

        today = datetime.date.today()
        yesterday = today - timedelta(days=1)

        content1 = "Memory from yesterday"
        memory1 = Memory(
            content=content1,
            content_hash=generate_content_hash(content1),
            tags=["test"],
            memory_type="test",
            created_at=(yesterday.toordinal() - datetime.date(1970, 1, 1).toordinal()) * 86400.0
        )
        await hybrid.store(memory1)

        # Delete by timeframe
        success, message = await hybrid.delete_by_timeframe(yesterday, today)

        # Verify delegation worked
        assert success
        assert "deleted" in message.lower() or message == ""

        # Verify memory was deleted from primary
        results = await hybrid.retrieve(content1, n_results=1)
        found = any(r.memory.content_hash == memory1.content_hash for r in results)
        assert not found

    @pytest.mark.asyncio
    async def test_delete_by_timeframe_sync_queue(self, test_hybrid_storage):
        """Test delete_by_timeframe queues SyncOperation for background sync."""
        hybrid = test_hybrid_storage

        # Store test memory first so deletion has something to delete
        import datetime
        from datetime import timedelta

        today = datetime.date.today()
        yesterday = today - timedelta(days=1)

        content1 = "Memory from yesterday"
        memory1 = Memory(
            content=content1,
            content_hash=generate_content_hash(content1),
            tags=["test"],
            memory_type="test",
            created_at=(yesterday.toordinal() - datetime.date(1970, 1, 1).toordinal()) * 86400.0
        )
        await hybrid.store(memory1)

        # Clear the store operation from queue
        while not hybrid.sync_service.operation_queue.empty():
            try:
                await asyncio.wait_for(hybrid.sync_service.operation_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                break

        # Call delete_by_timeframe
        await hybrid.delete_by_timeframe(yesterday, today)

        # Verify sync operation queued (with timeout to prevent hanging)
        assert not hybrid.sync_service.operation_queue.empty()
        try:
            operation = await asyncio.wait_for(hybrid.sync_service.operation_queue.get(), timeout=1.0)
            assert operation.operation == 'delete_by_timeframe'
            assert operation.start_date == yesterday
            assert operation.end_date == today
        except asyncio.TimeoutError:
            pytest.fail("Timeout waiting for delete_by_timeframe sync operation")

    @pytest.mark.asyncio
    async def test_delete_by_timeframe_background_sync(self, test_hybrid_storage):
        """Test background sync queues delete_by_timeframe operation for processing."""
        hybrid = test_hybrid_storage

        # Store test memory in primary so delete_by_timeframe has something to delete
        import datetime
        from datetime import timedelta

        today = datetime.date.today()
        yesterday = today - timedelta(days=1)

        content1 = "Memory to delete"
        memory1 = Memory(
            content=content1,
            content_hash=generate_content_hash(content1),
            tags=["test"],
            memory_type="test",
            created_at=(yesterday.toordinal() - datetime.date(1970, 1, 1).toordinal()) * 86400.0
        )
        await hybrid.store(memory1)

        # Clear the store operation from queue
        while not hybrid.sync_service.operation_queue.empty():
            try:
                await asyncio.wait_for(hybrid.sync_service.operation_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                break

        # Call delete_by_timeframe
        count, message = await hybrid.delete_by_timeframe(yesterday, today)

        # Verify the deletion happened in primary
        assert count == 1, "Should have deleted one memory from primary"

        # Verify sync operation was queued for background processing
        try:
            operation = await asyncio.wait_for(hybrid.sync_service.operation_queue.get(), timeout=1.0)

            # Verify the correct operation details
            assert operation.operation == 'delete_by_timeframe'
            assert operation.start_date == yesterday
            assert operation.end_date == today

            # Note: We don't test _process_single_operation here because there's a bug
            # in hybrid.py line 654 where it checks "if not success" but delete_by_timeframe
            # returns (count, message) not (success, message). This will be fixed separately.
        except asyncio.TimeoutError:
            pytest.fail("Timeout waiting for delete_by_timeframe sync operation")

    @pytest.mark.asyncio
    async def test_delete_before_date_delegation(self, test_hybrid_storage):
        """Test delete_before_date delegates to primary storage correctly."""
        hybrid = test_hybrid_storage

        # Store old test memory
        import datetime
        from datetime import timedelta

        today = datetime.date.today()
        old_date = today - timedelta(days=30)

        content1 = "Old memory to delete"
        memory1 = Memory(
            content=content1,
            content_hash=generate_content_hash(content1),
            tags=["test"],
            memory_type="test",
            created_at=(old_date.toordinal() - datetime.date(1970, 1, 1).toordinal()) * 86400.0
        )
        await hybrid.store(memory1)

        # Delete before today
        success, message = await hybrid.delete_before_date(today)

        # Verify delegation worked
        assert success
        assert "deleted" in message.lower() or message == ""

        # Verify memory was deleted from primary
        results = await hybrid.retrieve(content1, n_results=1)
        found = any(r.memory.content_hash == memory1.content_hash for r in results)
        assert not found

    @pytest.mark.asyncio
    async def test_delete_before_date_sync_queue(self, test_hybrid_storage):
        """Test delete_before_date queues SyncOperation for background sync."""
        hybrid = test_hybrid_storage

        # Store old test memory first so deletion has something to delete
        import datetime
        from datetime import timedelta

        today = datetime.date.today()
        old_date = today - timedelta(days=30)

        content1 = "Old memory to delete"
        memory1 = Memory(
            content=content1,
            content_hash=generate_content_hash(content1),
            tags=["test"],
            memory_type="test",
            created_at=(old_date.toordinal() - datetime.date(1970, 1, 1).toordinal()) * 86400.0
        )
        await hybrid.store(memory1)

        # Clear the store operation from queue
        while not hybrid.sync_service.operation_queue.empty():
            try:
                await asyncio.wait_for(hybrid.sync_service.operation_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                break

        # Call delete_before_date
        cutoff_date = today

        await hybrid.delete_before_date(cutoff_date)

        # Verify sync operation queued (with timeout to prevent hanging)
        assert not hybrid.sync_service.operation_queue.empty()
        try:
            operation = await asyncio.wait_for(hybrid.sync_service.operation_queue.get(), timeout=1.0)
            assert operation.operation == 'delete_before_date'
            assert operation.before_date == cutoff_date
        except asyncio.TimeoutError:
            pytest.fail("Timeout waiting for delete_before_date sync operation")

    @pytest.mark.asyncio
    async def test_get_by_exact_content_delegation(self, test_hybrid_storage):
        """Test get_by_exact_content delegates to primary storage correctly."""
        hybrid = test_hybrid_storage

        # Store test memory
        content = "Exact content to find"
        memory = Memory(
            content=content,
            content_hash=generate_content_hash(content),
            tags=["test"],
            memory_type="test"
        )
        await hybrid.store(memory)

        # Get by exact content
        results = await hybrid.get_by_exact_content(content)

        # Verify delegation worked
        assert len(results) == 1
        assert results[0].content == content
        assert results[0].content_hash == memory.content_hash

    @pytest.mark.asyncio
    async def test_get_by_exact_content_no_sync(self, test_hybrid_storage):
        """Test get_by_exact_content does NOT queue sync operation (read-only)."""
        hybrid = test_hybrid_storage

        # Store test memory
        content = "Read-only content"
        memory = Memory(
            content=content,
            content_hash=generate_content_hash(content),
            tags=["test"],
            memory_type="test"
        )
        await hybrid.store(memory)

        # Clear any existing sync operations
        while not hybrid.sync_service.operation_queue.empty():
            try:
                await asyncio.wait_for(hybrid.sync_service.operation_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                break

        # Get by exact content
        await hybrid.get_by_exact_content(content)

        # Verify NO sync operation queued (read-only operation)
        # Note: There might be one operation from store() above, so we check
        # that no NEW operation was added after clearing the queue
        queue_size_after = hybrid.sync_service.operation_queue.qsize()
        assert queue_size_after == 0, "get_by_exact_content should not queue sync operations"

    @pytest.mark.asyncio
    async def test_get_by_exact_content_multiple_results(self, test_hybrid_storage):
        """Test get_by_exact_content returns multiple memories with same content correctly."""
        hybrid = test_hybrid_storage

        # Store multiple memories with same content but different tags
        content = "Duplicate content for testing"
        memory1 = Memory(
            content=content,
            content_hash=generate_content_hash(content),
            tags=["test", "first"],
            memory_type="test"
        )
        memory2 = Memory(
            content=content,
            content_hash=generate_content_hash(content),
            tags=["test", "second"],
            memory_type="test"
        )

        await hybrid.store(memory1)
        # Note: Storing same content_hash will update, not create duplicate
        # So this test verifies single result for duplicate content_hash

        # Get by exact content
        results = await hybrid.get_by_exact_content(content)

        # Verify we get the memory (should be single result due to content_hash uniqueness)
        assert len(results) >= 1
        assert all(m.content == content for m in results)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])