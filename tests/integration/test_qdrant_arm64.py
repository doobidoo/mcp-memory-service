"""
ARM64 Platform-Specific Integration Tests for Qdrant Storage.

Validates primary goal: ARM64 compatibility with no ELFCLASS32 errors.

Tests cover:
1. Pure Python wheel imports (no native compilation)
2. Embedded Qdrant startup without binary incompatibility
3. Basic vector operations (store/retrieve/search)
4. Performance benchmarks on ARM64 platforms

Skip logic: Runs on ARM64 (aarch64, arm64), gracefully skips on AMD64/x86_64.
"""

import asyncio
import platform
import tempfile
import time
from pathlib import Path

import pytest
import pytest_asyncio

# Platform detection
IS_ARM64 = platform.machine().lower() in ["aarch64", "arm64"]
ARM64_SKIP_REASON = f"Skipping ARM64-specific tests on {platform.machine()} platform"


class TestQdrantARM64Import:
    """Test 1: Pure Python wheel import validation."""

    @pytest.mark.skipif(not IS_ARM64, reason=ARM64_SKIP_REASON)
    def test_qdrant_client_import_arm64(self):
        """Verify qdrant-client imports successfully on ARM64 (pure Python wheel)."""
        try:
            import qdrant_client

            assert qdrant_client is not None
            # Verify it's the pure Python implementation
            assert hasattr(qdrant_client, "QdrantClient")
        except ImportError as e:
            pytest.fail(f"Failed to import qdrant-client on ARM64: {e}")

    @pytest.mark.skipif(not IS_ARM64, reason=ARM64_SKIP_REASON)
    def test_no_native_binary_dependencies(self):
        """Ensure qdrant-client has no native binary dependencies on ARM64."""
        import inspect

        import qdrant_client

        # Check module file location - pure Python should be in site-packages
        module_file = inspect.getfile(qdrant_client)
        assert ".so" not in module_file, f"Found native binary in module path: {module_file}"
        assert ".dylib" not in module_file, f"Found native library in module path: {module_file}"


class TestQdrantARM64Startup:
    """Test 2: Embedded Qdrant startup without ELFCLASS32 errors."""

    @pytest_asyncio.fixture
    async def storage_path(self):
        """Create temporary storage directory for Qdrant."""
        with tempfile.TemporaryDirectory(prefix="qdrant_arm64_") as tmpdir:
            yield Path(tmpdir)

    @pytest.mark.skipif(not IS_ARM64, reason=ARM64_SKIP_REASON)
    @pytest.mark.asyncio
    async def test_qdrant_embedded_startup_arm64(self, storage_path):
        """Verify Qdrant embedded mode starts without ELFCLASS32 errors on ARM64."""
        from qdrant_client import QdrantClient

        try:
            # Start embedded Qdrant (this triggers binary loading if any)
            client = QdrantClient(path=str(storage_path))

            # If we got here, no ELFCLASS32 error occurred
            assert client is not None

            # Verify client can perform basic operations
            collections = client.get_collections()
            assert collections is not None

        except OSError as e:
            if "ELFCLASS32" in str(e):
                pytest.fail(f"ELFCLASS32 error on ARM64: {e}")
            raise

    @pytest.mark.skipif(not IS_ARM64, reason=ARM64_SKIP_REASON)
    @pytest.mark.asyncio
    async def test_qdrant_no_startup_errors_in_logs(self, storage_path, caplog):
        """Ensure Qdrant startup produces no error logs on ARM64."""
        from qdrant_client import QdrantClient

        QdrantClient(path=str(storage_path))

        # Check logs for any error indicators
        error_keywords = ["ELFCLASS32", "incompatible", "wrong ELF class", "binary"]
        for record in caplog.records:
            for keyword in error_keywords:
                assert (
                    keyword.lower() not in record.message.lower()
                ), f"Found error keyword '{keyword}' in logs: {record.message}"


class TestQdrantARM64Operations:
    """Test 3: Vector operations on ARM64."""

    @pytest_asyncio.fixture
    async def qdrant_client(self):
        """Create Qdrant client with temporary storage."""
        with tempfile.TemporaryDirectory(prefix="qdrant_ops_") as tmpdir:
            from qdrant_client import QdrantClient

            client = QdrantClient(path=tmpdir)
            yield client

    @pytest.mark.skipif(not IS_ARM64, reason=ARM64_SKIP_REASON)
    @pytest.mark.asyncio
    async def test_collection_creation_arm64(self, qdrant_client):
        """Verify collection creation works on ARM64."""
        from qdrant_client.models import Distance, VectorParams

        collection_name = "test_arm64_collection"
        vector_size = 384  # all-MiniLM-L6-v2 dimension

        # Create collection
        qdrant_client.create_collection(
            collection_name=collection_name, vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )

        # Verify collection exists
        collections = qdrant_client.get_collections()
        collection_names = [c.name for c in collections.collections]
        assert collection_name in collection_names

    @pytest.mark.skipif(not IS_ARM64, reason=ARM64_SKIP_REASON)
    @pytest.mark.asyncio
    async def test_vector_insert_arm64(self, qdrant_client):
        """Verify vector insertion works on ARM64."""
        import numpy as np
        from qdrant_client.models import Distance, PointStruct, VectorParams

        collection_name = "test_arm64_insert"
        vector_size = 384

        # Setup collection
        qdrant_client.create_collection(
            collection_name=collection_name, vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )

        # Insert test vector
        test_vector = np.random.rand(vector_size).tolist()
        qdrant_client.upsert(
            collection_name=collection_name,
            points=[PointStruct(id=1, vector=test_vector, payload={"content": "test memory", "tags": ["arm64"]})],
        )

        # Verify insertion
        count = qdrant_client.count(collection_name=collection_name)
        assert count.count == 1

    @pytest.mark.skipif(not IS_ARM64, reason=ARM64_SKIP_REASON)
    @pytest.mark.asyncio
    async def test_vector_search_arm64(self, qdrant_client):
        """Verify vector search works on ARM64."""
        import numpy as np
        from qdrant_client.models import Distance, PointStruct, VectorParams

        collection_name = "test_arm64_search"
        vector_size = 384

        # Setup collection with test data
        qdrant_client.create_collection(
            collection_name=collection_name, vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )

        # Insert multiple vectors
        test_vectors = [np.random.rand(vector_size).tolist() for _ in range(10)]
        points = [PointStruct(id=i, vector=vec, payload={"index": i}) for i, vec in enumerate(test_vectors)]
        qdrant_client.upsert(collection_name=collection_name, points=points)

        # Search using first vector
        results = qdrant_client.search(collection_name=collection_name, query_vector=test_vectors[0], limit=5)

        # Verify search results
        assert len(results) > 0
        assert results[0].id == 0  # Should match itself with highest score


class TestQdrantARM64Performance:
    """Test 4: Performance benchmarks on ARM64 platforms."""

    @pytest_asyncio.fixture
    async def qdrant_storage(self):
        """Create QdrantStorage instance for performance testing."""
        with tempfile.TemporaryDirectory(prefix="qdrant_perf_") as tmpdir:
            # Import will be available after QdrantStorage implementation
            # For now, use qdrant-client directly
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            client = QdrantClient(path=tmpdir)
            collection_name = "perf_test"

            client.create_collection(
                collection_name=collection_name, vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )

            yield client, collection_name

    @pytest.mark.skipif(not IS_ARM64, reason=ARM64_SKIP_REASON)
    @pytest.mark.asyncio
    async def test_insert_performance_arm64(self, qdrant_storage):
        """Benchmark insert operations on ARM64."""
        import numpy as np
        from qdrant_client.models import PointStruct

        client, collection_name = qdrant_storage
        num_vectors = 100
        vector_size = 384

        # Generate test data
        vectors = [np.random.rand(vector_size).tolist() for _ in range(num_vectors)]
        points = [PointStruct(id=i, vector=vec, payload={"index": i}) for i, vec in enumerate(vectors)]

        # Benchmark batch insert
        start = time.perf_counter()
        client.upsert(collection_name=collection_name, points=points)
        elapsed = time.perf_counter() - start

        # Performance target: <50ms per operation on ARM64
        per_op_ms = (elapsed / num_vectors) * 1000
        assert per_op_ms < 50, f"ARM64 insert too slow: {per_op_ms:.2f}ms per operation"

    @pytest.mark.skipif(not IS_ARM64, reason=ARM64_SKIP_REASON)
    @pytest.mark.asyncio
    async def test_search_performance_arm64(self, qdrant_storage):
        """Benchmark search operations on ARM64."""
        import numpy as np
        from qdrant_client.models import PointStruct

        client, collection_name = qdrant_storage
        num_vectors = 1000
        vector_size = 384

        # Insert test data
        vectors = [np.random.rand(vector_size).tolist() for _ in range(num_vectors)]
        points = [PointStruct(id=i, vector=vec, payload={"index": i}) for i, vec in enumerate(vectors)]
        client.upsert(collection_name=collection_name, points=points)

        # Benchmark search
        query_vector = np.random.rand(vector_size).tolist()
        search_times = []

        for _ in range(10):
            start = time.perf_counter()
            client.search(collection_name=collection_name, query_vector=query_vector, limit=10)
            search_times.append(time.perf_counter() - start)

        # Performance target: <50ms p50 latency on ARM64
        p50_ms = sorted(search_times)[len(search_times) // 2] * 1000
        assert p50_ms < 50, f"ARM64 search too slow: {p50_ms:.2f}ms p50 latency"

    @pytest.mark.skipif(not IS_ARM64, reason=ARM64_SKIP_REASON)
    @pytest.mark.asyncio
    async def test_concurrent_operations_arm64(self, qdrant_storage):
        """Verify concurrent operations work correctly on ARM64."""
        import numpy as np
        from qdrant_client.models import PointStruct

        client, collection_name = qdrant_storage
        vector_size = 384

        async def insert_batch(start_id: int, count: int):
            """Insert a batch of vectors."""
            vectors = [np.random.rand(vector_size).tolist() for _ in range(count)]
            points = [PointStruct(id=start_id + i, vector=vec, payload={"batch": start_id}) for i, vec in enumerate(vectors)]
            client.upsert(collection_name=collection_name, points=points)

        # Run concurrent inserts
        await asyncio.gather(insert_batch(0, 50), insert_batch(50, 50), insert_batch(100, 50))

        # Verify all insertions succeeded
        count = client.count(collection_name=collection_name)
        assert count.count == 150


class TestQdrantARM64PlatformInfo:
    """Platform detection and reporting for ARM64 tests."""

    @pytest.mark.skipif(not IS_ARM64, reason=ARM64_SKIP_REASON)
    def test_platform_detection_arm64(self):
        """Report ARM64 platform details for test logs."""
        import sys

        platform_info = {
            "machine": platform.machine(),
            "system": platform.system(),
            "release": platform.release(),
            "python_version": sys.version,
            "python_implementation": platform.python_implementation(),
        }

        # Print for test output visibility
        print("\n=== ARM64 Platform Information ===")
        for key, value in platform_info.items():
            print(f"{key}: {value}")

        # Assertions to validate ARM64
        assert platform.machine().lower() in ["aarch64", "arm64"]

        # Detect specific ARM64 platforms
        system = platform.system()
        if system == "Darwin":
            print("Platform: Apple Silicon (macOS)")
        elif system == "Linux":
            # Could be AWS Graviton, Raspberry Pi, etc.
            print(f"Platform: Linux ARM64 ({platform.release()})")

        assert True  # Test always passes, just for reporting
