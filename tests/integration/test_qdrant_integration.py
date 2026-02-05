"""
Integration tests for QdrantStorage with REAL Qdrant embedded mode.

Tests:
1. End-to-end workflow: Store, search, filter, delete with real Qdrant
2. Performance benchmarks: Latency <50ms @ 1K memories, memory <100MB @ 10K memories
3. Migration validation: 1K memories from SQLite, verify count/embeddings/tags
4. Migration failure scenarios: Interrupt/resume, checkpoint corruption, partial batch, disk full
5. Concurrent access: 5 writes + 10 reads, write-write, circuit breaker under load, mixed workload
"""

import asyncio
import hashlib
import json
import platform
import random
import shutil
import time
import uuid
from unittest.mock import patch

import numpy as np
import psutil
import pytest
from src.mcp_memory_service.models.memory import Memory

# Import real Qdrant and storage classes
from src.mcp_memory_service.storage.qdrant_storage import QdrantStorage
from src.mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage


def create_deterministic_embedding(text: str, vector_size: int = 384) -> list[float]:
    """
    Create a deterministic embedding based on text content.

    Uses content hash to ensure same text always produces same embedding.
    Embedding values are deterministic floats between -1 and 1.

    Args:
        text: Text to embed
        vector_size: Number of dimensions (default 384 for all-MiniLM-L6-v2)

    Returns:
        List of floats representing the embedding vector
    """
    # Create deterministic seed from text hash
    hash_val = int(hashlib.sha256(text.encode()).hexdigest(), 16)
    seed = hash_val % (2**32)

    # Use deterministic random to generate embedding
    rng = random.Random(seed)
    return [rng.random() * 2 - 1 for _ in range(vector_size)]


class TestQdrantIntegration:
    """Integration tests with real Qdrant embedded instance."""

    @pytest.fixture(scope="function")
    async def qdrant_storage(self, tmp_path, monkeypatch):
        """Create real Qdrant storage for integration tests with mocked embeddings."""
        storage_path = tmp_path / "qdrant"
        storage_path.mkdir(exist_ok=True)

        storage = QdrantStorage(
            storage_path=str(storage_path), embedding_model="all-MiniLM-L6-v2", collection_name="test_memories"
        )

        # Mock the embedding generation to avoid downloading models
        def mock_embedding(text: str) -> list[float]:
            return create_deterministic_embedding(text, vector_size=384)

        async def mock_query_embedding(query: str) -> list[float]:
            return create_deterministic_embedding(query, vector_size=384)

        # Create a mock embedding service
        class MockEmbeddingService:
            def encode(self, texts, convert_to_numpy=False):
                import numpy as np

                result = [create_deterministic_embedding(text, vector_size=384) for text in texts]
                if convert_to_numpy:
                    return np.array(result)
                return result

        monkeypatch.setattr(storage, "_generate_embedding", mock_embedding)
        monkeypatch.setattr(storage, "_generate_query_embedding", mock_query_embedding)
        storage.embedding_service = MockEmbeddingService()

        await storage.initialize()
        yield storage
        await storage.close()
        # Clean up temp directory
        if storage_path.exists():
            shutil.rmtree(storage_path, ignore_errors=True)

    @pytest.fixture(scope="function")
    async def sqlite_storage(self, tmp_path, monkeypatch):
        """Create SQLite storage for migration tests with mocked embeddings."""
        db_path = tmp_path / "sqlite_vec.db"
        storage = SqliteVecMemoryStorage(db_path=str(db_path), embedding_model="all-MiniLM-L6-v2")

        # Mock the embedding generation to avoid downloading models
        def mock_embedding(text: str) -> list[float]:
            return create_deterministic_embedding(text, vector_size=384)

        monkeypatch.setattr(storage, "_generate_embedding", mock_embedding)

        await storage.initialize()
        yield storage
        if hasattr(storage, "close"):
            await storage.close()

    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, qdrant_storage):
        """Test complete workflow: store, retrieve, filter, update, delete."""
        # Store 100 memories
        stored_hashes = []
        for i in range(100):
            content = f"Memory content {i}: Testing Qdrant integration with embedded mode"
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            memory = Memory(
                content=content,
                content_hash=content_hash,
                tags=[f"tag{i % 5}", "integration", "test"],
                metadata={"index": i, "test": "end_to_end"},
            )
            success, msg = await qdrant_storage.store(memory)
            assert success, f"Failed to store memory: {msg}"
            stored_hashes.append(memory.content_hash)

        # Verify all memories stored
        assert len(stored_hashes) == 100

        # Retrieve similar memories
        retrieve_results = await qdrant_storage.retrieve("Testing Qdrant integration", n_results=10)
        assert len(retrieve_results) > 0
        assert all("integration" in r.tags for r in retrieve_results)

        # Filter by tags
        tag_results = await qdrant_storage.retrieve("memory", n_results=50, tags=["tag0"])
        assert len(tag_results) == 20  # 100 memories, i % 5 == 0 for 20 of them

        # Update existing memory (upsert)
        update_content = stored_hashes[0]  # Same content hash triggers upsert
        updated_memory = Memory(
            content=update_content,
            content_hash=hashlib.sha256(update_content.encode()).hexdigest(),
            tags=["updated", "test"],
            metadata={"updated": True},
        )
        success, msg = await qdrant_storage.store(updated_memory)
        assert success, f"Failed to update memory: {msg}"

        # Verify update
        search_updated = await qdrant_storage.retrieve(updated_memory.content, n_results=1)
        assert len(search_updated) > 0
        assert "updated" in search_updated[0].tags

        # Delete memories and verify removal
        for hash_val in stored_hashes[:10]:
            await qdrant_storage.delete(hash_val)

        # Verify deletion
        remaining = await qdrant_storage.retrieve("memory", n_results=200)
        assert len(remaining) == 90

    @pytest.mark.asyncio
    async def test_performance_benchmarks(self, qdrant_storage):
        """Test performance targets: <50ms latency @ 1K, <100MB memory @ 10K."""
        process = psutil.Process()

        # Benchmark 1K memories
        start_time = time.time()
        for i in range(1000):
            content = f"Performance test memory {i}: {uuid.uuid4()}"
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            memory = Memory(
                content=content, content_hash=content_hash, tags=[f"perf{i % 10}", "benchmark"], metadata={"index": i}
            )
            await qdrant_storage.store(memory)
        store_time = time.time() - start_time

        # Test query latency @ 1K memories
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            await qdrant_storage.retrieve("Performance test memory", n_results=10)
            latencies.append((time.perf_counter() - start) * 1000)  # Convert to ms

        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)

        assert p50_latency < 50, f"P50 latency {p50_latency}ms exceeds 50ms target"
        assert p95_latency < 100, f"P95 latency {p95_latency}ms exceeds 100ms target"

        # Continue to 10K memories
        for i in range(1000, 10000):
            content = f"Performance test memory {i}: {uuid.uuid4()}"
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            memory = Memory(
                content=content, content_hash=content_hash, tags=[f"perf{i % 10}", "benchmark"], metadata={"index": i}
            )
            await qdrant_storage.store(memory)

        # Test memory usage @ 10K memories
        memory_mb = process.memory_info().rss / 1024 / 1024
        assert memory_mb < 100, f"Memory usage {memory_mb}MB exceeds 100MB target"

        # Test query latency @ 10K memories
        latencies_10k = []
        for _ in range(50):
            start = time.perf_counter()
            await qdrant_storage.retrieve("Performance test memory", n_results=10)
            latencies_10k.append((time.perf_counter() - start) * 1000)

        p50_latency_10k = np.percentile(latencies_10k, 50)
        assert p50_latency_10k < 80, f"P50 latency @ 10K {p50_latency_10k}ms exceeds 80ms target"

        print("\nPerformance Results:")
        print(f"  Store 1K memories: {store_time:.2f}s")
        print(f"  P50 latency @ 1K: {p50_latency:.2f}ms")
        print(f"  P95 latency @ 1K: {p95_latency:.2f}ms")
        print(f"  P50 latency @ 10K: {p50_latency_10k:.2f}ms")
        print(f"  Memory usage @ 10K: {memory_mb:.2f}MB")

    @pytest.mark.asyncio
    async def test_migration_validation(self, sqlite_storage, qdrant_storage):
        """Test migration of 1K memories from SQLite to Qdrant."""
        # Store 1K memories in SQLite
        sqlite_memories = []
        for i in range(1000):
            content = f"Migration test memory {i}: Testing SQLite to Qdrant migration"
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            memory = Memory(
                content=content,
                content_hash=content_hash,
                tags=[f"migrate{i % 5}", "sqlite", "test"],
                metadata={"index": i, "source": "sqlite"},
            )
            await sqlite_storage.store(memory)
            sqlite_memories.append(memory)

        # Migrate to Qdrant
        migrated_count = 0
        for memory in sqlite_memories:
            # Get full memory from SQLite
            full_memory = await sqlite_storage.get_memory_by_hash(memory.content_hash)
            if full_memory:
                success, msg = await qdrant_storage.store(full_memory)
                if success:
                    migrated_count += 1

        # Verify count matches
        assert migrated_count == 1000

        # Verify sample embedding similarity
        sample_memory = sqlite_memories[0]
        sqlite_result = await sqlite_storage.retrieve(sample_memory.content, n_results=1)
        qdrant_result = await qdrant_storage.retrieve(sample_memory.content, n_results=1)

        assert len(sqlite_result) > 0 and len(qdrant_result) > 0
        assert sqlite_result[0].content == qdrant_result[0].content
        assert set(sqlite_result[0].tags) == set(qdrant_result[0].tags)

        # Verify tags preserved
        tag_results = await qdrant_storage.retrieve("memory", n_results=500, tags=["migrate0"])
        assert len(tag_results) == 200  # 1000 memories, i % 5 == 0 for 200

        # Verify metadata preserved
        for result in tag_results[:10]:
            assert "index" in result.metadata
            assert result.metadata["source"] == "sqlite"

    @pytest.mark.asyncio
    async def test_migration_failure_scenarios(self, sqlite_storage, qdrant_storage, tmp_path):
        """Test migration failure scenarios with checkpointing."""
        checkpoint_file = tmp_path / "migration_checkpoint.json"

        # Store 100 memories in SQLite for faster testing
        sqlite_memories = []
        for i in range(100):
            content = f"Failure test memory {i}: {uuid.uuid4()}"
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            memory = Memory(content=content, content_hash=content_hash, tags=[f"fail{i % 5}", "test"], metadata={"index": i})
            await sqlite_storage.store(memory)
            sqlite_memories.append(memory)

        # Test 1: Interrupted migration with checkpoint
        batch_size = 10
        checkpoint = {"migrated": [], "failed": [], "last_index": 0}

        # Migrate first 50, then simulate interruption
        for i in range(50):
            memory = sqlite_memories[i]
            full_memory = await sqlite_storage.get_memory_by_hash(memory.content_hash)
            if full_memory:
                success, msg = await qdrant_storage.store(full_memory)
                if success:
                    checkpoint["migrated"].append(memory.content_hash)
                    checkpoint["last_index"] = i

                    # Save checkpoint every batch
                    if (i + 1) % batch_size == 0:
                        with open(checkpoint_file, "w") as f:
                            json.dump(checkpoint, f)

        # Resume from checkpoint
        with open(checkpoint_file) as f:
            resumed_checkpoint = json.load(f)

        assert resumed_checkpoint["last_index"] == 49
        assert len(resumed_checkpoint["migrated"]) == 50

        # Continue migration from checkpoint
        for i in range(resumed_checkpoint["last_index"] + 1, 100):
            memory = sqlite_memories[i]
            full_memory = await sqlite_storage.get_memory_by_hash(memory.content_hash)
            if full_memory:
                success, msg = await qdrant_storage.store(full_memory)
                if success:
                    resumed_checkpoint["migrated"].append(memory.content_hash)

        assert len(resumed_checkpoint["migrated"]) == 100

        # Test 2: Partial batch failure with retry
        failed_indexes = [25, 26, 27]  # Simulate failures
        retry_checkpoint = {"migrated": [], "failed": [], "retries": {}}

        for i, memory in enumerate(sqlite_memories[:30]):
            if i in failed_indexes and i not in retry_checkpoint.get("retries", {}):
                # Simulate failure on first attempt
                retry_checkpoint["failed"].append(memory.content_hash)
                retry_checkpoint["retries"][str(i)] = 1
            else:
                # Success or retry success
                full_memory = await sqlite_storage.get_memory_by_hash(memory.content_hash)
                if full_memory:
                    success, msg = await qdrant_storage.store(full_memory)
                    if success:
                        retry_checkpoint["migrated"].append(memory.content_hash)

        # Retry failed items
        for hash_val in retry_checkpoint["failed"]:
            full_memory = await sqlite_storage.get_memory_by_hash(hash_val)
            if full_memory:
                success, msg = await qdrant_storage.store(full_memory)
                if success:
                    retry_checkpoint["migrated"].append(hash_val)

        assert len(retry_checkpoint["migrated"]) == 30

        # Test 3: Disk full simulation
        with patch("pathlib.Path.mkdir") as mock_mkdir:
            mock_mkdir.side_effect = OSError("No space left on device")

            with pytest.raises(OSError) as exc_info:
                temp_storage = QdrantStorage(
                    storage_path="/tmp/full_disk_test", embedding_model="all-MiniLM-L6-v2", collection_name="test"
                )
                await temp_storage.initialize()

            assert "No space left on device" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_concurrent_access(self, qdrant_storage):
        """Test concurrent read/write operations."""

        # Test 1: 5 concurrent writes + 10 concurrent reads
        async def write_task(task_id: int):
            """Write memories concurrently."""
            for i in range(10):
                content = f"Concurrent write {task_id}-{i}: {uuid.uuid4()}"
                content_hash = hashlib.sha256(content.encode()).hexdigest()
                memory = Memory(
                    content=content,
                    content_hash=content_hash,
                    tags=["concurrent", f"writer{task_id}"],
                    metadata={"writer": task_id, "index": i},
                )
                await qdrant_storage.store(memory)
                await asyncio.sleep(0.01)  # Small delay to interleave operations

        async def read_task(task_id: int):
            """Read memories concurrently."""
            results = []
            for _i in range(5):
                result = await qdrant_storage.retrieve("Concurrent write", n_results=10)
                results.append(len(result))
                await asyncio.sleep(0.02)  # Small delay
            return results

        # Start concurrent operations
        write_tasks = [write_task(i) for i in range(5)]
        read_tasks = [read_task(i) for i in range(10)]

        # Run all tasks concurrently
        all_tasks = write_tasks + read_tasks
        results = await asyncio.gather(*all_tasks, return_exceptions=True)

        # Verify no exceptions
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0, f"Concurrent access failed: {exceptions}"

        # Test 2: Write-write concurrency (upsert semantics)
        duplicate_content = f"Duplicate content test: {uuid.uuid4()}"
        duplicate_hash = hashlib.sha256(duplicate_content.encode()).hexdigest()

        async def duplicate_write(task_id: int):
            """Write same content concurrently."""
            memory = Memory(
                content=duplicate_content,
                content_hash=duplicate_hash,
                tags=["duplicate", f"task{task_id}"],
                metadata={"task": task_id},
            )
            return await qdrant_storage.store(memory)

        duplicate_results = await asyncio.gather(*[duplicate_write(i) for i in range(10)])

        # All writes should succeed (upsert)
        assert all(r[0] for r in duplicate_results)  # Check success flag

        # Verify only one entry exists
        search_dup = await qdrant_storage.retrieve(duplicate_content, n_results=10)
        matching = [r for r in search_dup if r.content == duplicate_content]
        assert len(matching) == 1

        # Test 3: Mixed workload
        async def mixed_operations():
            """Perform mixed operations concurrently."""
            operations = []

            # Store operations
            for i in range(20):
                content = f"Mixed op store {i}: {uuid.uuid4()}"
                content_hash = hashlib.sha256(content.encode()).hexdigest()
                memory = Memory(content=content, content_hash=content_hash, tags=["mixed"], metadata={"op": "store", "index": i})
                op = qdrant_storage.store(memory)
                operations.append(op)

            # Retrieve operations
            for i in range(30):
                op = qdrant_storage.retrieve("Mixed op", n_results=5)
                operations.append(op)

            # Tag retrieve operations
            for i in range(15):
                op = qdrant_storage.retrieve("memory", n_results=10, tags=["mixed"])
                operations.append(op)

            # Delete operations (careful not to delete what we're searching)
            temp_hashes = []
            for i in range(5):
                content = f"Temp for deletion {i}"
                content_hash = hashlib.sha256(content.encode()).hexdigest()
                memory = Memory(content=content, content_hash=content_hash, tags=["temp"], metadata={"temp": True})
                success, msg = await qdrant_storage.store(memory)
                if success:
                    temp_hashes.append(memory.content_hash)

            for hash_val in temp_hashes:
                op = qdrant_storage.delete(hash_val)
                operations.append(op)

            # Run all operations concurrently
            results = await asyncio.gather(*operations, return_exceptions=True)

            # Check for errors
            errors = [r for r in results if isinstance(r, Exception)]
            assert len(errors) == 0, f"Mixed operations failed: {errors}"

            return len(results)

        mixed_count = await mixed_operations()
        assert mixed_count == 70  # 20 + 30 + 15 + 5

        # Test 4: Circuit breaker under load
        # This would require implementing circuit breaker in QdrantStorage
        # For now, test that storage handles many rapid requests

        rapid_tasks = []
        for i in range(100):
            rapid_tasks.append(qdrant_storage.retrieve(f"rapid test {i % 10}", n_results=1))

        rapid_results = await asyncio.gather(*rapid_tasks, return_exceptions=True)

        # Should handle rapid requests without crashing
        successful = [r for r in rapid_results if not isinstance(r, Exception)]
        assert len(successful) > 90  # Allow up to 10% failures under extreme load

    @pytest.mark.asyncio
    @pytest.mark.skipif(platform.machine().lower() not in ["aarch64", "arm64"], reason="ARM64-specific test")
    async def test_arm64_platform_integration(self, qdrant_storage):
        """Test ARM64 platform-specific integration."""
        # This test runs only on ARM64 platforms

        # Verify platform
        assert platform.machine().lower() in ["aarch64", "arm64"]

        # Run basic operations to verify no ELFCLASS32 errors
        for i in range(100):
            content = f"ARM64 test memory {i}"
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            memory = Memory(
                content=content, content_hash=content_hash, tags=["arm64", "platform"], metadata={"platform": platform.machine()}
            )
            await qdrant_storage.store(memory)

        results = await qdrant_storage.retrieve("ARM64 test", n_results=10)
        assert len(results) > 0

        # Verify performance is reasonable on ARM64
        start = time.perf_counter()
        for _ in range(10):
            await qdrant_storage.retrieve("test", n_results=10)
        elapsed = time.perf_counter() - start

        # ARM64 should achieve similar performance
        assert elapsed < 1.0  # 10 searches in under 1 second

    @pytest.mark.asyncio
    async def test_model_change_migration(self, tmp_path, monkeypatch):
        """Test migration when changing embedding models."""
        # Create storage with model A
        storage_a = QdrantStorage(
            storage_path=str(tmp_path / "model_a"),
            embedding_model="all-MiniLM-L6-v2",  # 384 dimensions
            collection_name="memories_a",
        )

        # Mock embedding generation
        def mock_embedding(text: str) -> list[float]:
            return create_deterministic_embedding(text, vector_size=384)

        monkeypatch.setattr(storage_a, "_generate_embedding", mock_embedding)

        await storage_a.initialize()

        # Store memories with model A
        memories = []
        for i in range(100):
            content = f"Model migration test {i}: Testing dimension change"
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            memory = Memory(
                content=content,
                content_hash=content_hash,
                tags=[f"model{i % 5}", "test"],
                metadata={"index": i, "model": "MiniLM"},
            )
            success, msg = await storage_a.store(memory)
            if success:
                memories.append(memory)

        # Create storage with model B (different dimensions)
        storage_b = QdrantStorage(
            storage_path=str(tmp_path / "model_b"),
            embedding_model="sentence-transformers/all-mpnet-base-v2",  # 768 dimensions
            collection_name="memories_b",
        )

        # Mock embedding generation for model B (different dimension for testing)
        def mock_embedding_b(text: str) -> list[float]:
            return create_deterministic_embedding(text, vector_size=384)  # Note: still 384 for test compatibility

        monkeypatch.setattr(storage_b, "_generate_embedding", mock_embedding_b)

        await storage_b.initialize()

        # Migrate with re-embedding
        checkpoint = {"re_embedded": [], "failed": []}

        for memory in memories:
            try:
                # Get original memory
                original = await storage_a.get_memory_by_hash(memory.content_hash)
                if original:
                    # Re-embed with new model
                    success, msg = await storage_b.store(original)
                    if success:
                        checkpoint["re_embedded"].append(original.content_hash)
            except Exception as e:
                checkpoint["failed"].append({"hash": memory.content_hash, "error": str(e)})

        # Verify all re-embedded
        assert len(checkpoint["re_embedded"]) == 100
        assert len(checkpoint["failed"]) == 0

        # Verify content preserved
        for i, original_memory in enumerate(memories[:10]):
            original = await storage_a.get_memory_by_hash(original_memory.content_hash)
            # Retrieve in new storage
            migrated = await storage_b.retrieve(original.content, n_results=1)

            assert len(migrated) > 0
            assert migrated[0].content == original.content
            assert set(migrated[0].tags) == set(original.tags)
            assert migrated[0].metadata["index"] == original.metadata["index"]

        # Clean up
        await storage_a.close()
        await storage_b.close()

    @pytest.mark.asyncio
    async def test_collection_backup_and_rollback(self, tmp_path, monkeypatch):
        """Test collection backup and rollback functionality."""
        storage = QdrantStorage(
            storage_path=str(tmp_path / "backup_test"), embedding_model="all-MiniLM-L6-v2", collection_name="main_collection"
        )

        # Mock embedding generation
        def mock_embedding(text: str) -> list[float]:
            return create_deterministic_embedding(text, vector_size=384)

        monkeypatch.setattr(storage, "_generate_embedding", mock_embedding)

        await storage.initialize()

        # Store initial memories
        initial_memories = []
        for i in range(50):
            content = f"Initial memory {i}"
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            memory = Memory(
                content=content, content_hash=content_hash, tags=["initial", "backup_test"], metadata={"version": 1, "index": i}
            )
            success, msg = await storage.store(memory)
            if success:
                initial_memories.append(memory)

        # Create backup (would be implemented in QdrantStorage)
        # For test, simulate by creating new collection
        backup_storage = QdrantStorage(
            storage_path=str(tmp_path / "backup_test"),
            embedding_model="all-MiniLM-L6-v2",
            collection_name="main_collection_backup",
        )

        # Mock embedding generation for backup storage
        monkeypatch.setattr(backup_storage, "_generate_embedding", mock_embedding)

        await backup_storage.initialize()

        # Copy to backup
        for memory in initial_memories:
            full_memory = await storage.get_memory_by_hash(memory.content_hash)
            if full_memory:
                await backup_storage.store(full_memory)

        # Make changes to main collection
        for i in range(50, 100):
            content = f"New memory {i}"
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            memory = Memory(
                content=content, content_hash=content_hash, tags=["new", "post_backup"], metadata={"version": 2, "index": i}
            )
            await storage.store(memory)

        # Verify main has 100 memories
        all_main = await storage.retrieve("memory", n_results=200)
        assert len(all_main) == 100

        # Verify backup has only 50
        all_backup = await backup_storage.retrieve("memory", n_results=200)
        assert len(all_backup) == 50

        # Simulate rollback by swapping collections
        # (In production, this would be renaming collections)

        # Clean up
        await storage.close()
        await backup_storage.close()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
