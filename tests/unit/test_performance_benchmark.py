"""
Performance benchmark tests for tag filtering at scale.

These tests verify that tag filtering remains fast (<500ms) even with
10K+ memories in the database. This validates that database-level filtering
with indexes provides O(log n) performance, not O(n).

Note: These are marked as slow tests and can be skipped in CI with:
    pytest -m "not slow"
"""

import time

import pytest

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash


class TestPerformanceBenchmark:
    """Performance tests for tag filtering at scale."""

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_tag_filtering_performance_10k_memories(self, temp_db_path):
        """
        Verify tag-filtered queries complete in <500ms with 10K memories.

        This validates that:
        1. Database-level filtering is used (not Python post-filtering)
        2. Indexes are effective at scale
        3. Query performance is acceptable for production use
        """
        storage = SqliteVecMemoryStorage(str(temp_db_path / "test.db"))
        await storage.initialize()

        print("\n=== PERFORMANCE BENCHMARK: 10K Memories ===")

        # Store 10K memories with various tags
        tags_pool = ["python", "java", "javascript", "rust", "go", "ruby", "php", "swift"]
        print("Storing 10,000 memories...")
        store_start = time.time()

        for i in range(10000):
            # Distribute tags: 1/8 of memories get each tag
            tag = tags_pool[i % len(tags_pool)]
            await storage.store(
                Memory(
                    content=f"Programming tutorial {i} about {tag}",
                    content_hash=generate_content_hash(f"Tutorial {i}"),
                    tags=[tag, "tutorial"],
                    memory_type="note",
                )
            )

            if (i + 1) % 1000 == 0:
                elapsed = time.time() - store_start
                print(f"  Stored {i + 1} memories ({elapsed:.2f}s elapsed)")

        store_duration = time.time() - store_start
        print(f"Stored 10,000 memories in {store_duration:.2f}s")

        # Test 1: Single tag filter query
        print("\nTest 1: Single tag filter ('python')")
        query_start = time.time()
        results = await storage.retrieve(query="programming tutorial", n_results=10, tags=["python"])
        query_duration = time.time() - query_start

        print(f"  Query completed in {query_duration * 1000:.2f}ms")
        print(f"  Returned {len(results)} results")

        # SUCCESS CRITERIA: <500ms
        assert query_duration < 0.5, f"Tag-filtered query took {query_duration * 1000:.2f}ms (should be <500ms)"

        # Verify results are correct
        assert len(results) > 0, "Should return results"
        for result in results:
            assert "python" in result.memory.tags, f"All results should have 'python' tag, got {result.memory.tags}"

        # Test 2: Multi-tag OR filter
        print("\nTest 2: Multi-tag OR filter ('python' OR 'java')")
        query_start = time.time()
        results = await storage.retrieve(query="programming tutorial", n_results=10, tags=["python", "java"])
        query_duration = time.time() - query_start

        print(f"  Query completed in {query_duration * 1000:.2f}ms")
        print(f"  Returned {len(results)} results")

        assert query_duration < 0.5, f"Multi-tag query took {query_duration * 1000:.2f}ms (should be <500ms)"

        # Test 3: Combined filters (memory_type + tags)
        print("\nTest 3: Combined filters (memory_type + tags)")
        query_start = time.time()
        results = await storage.retrieve(query="programming tutorial", n_results=10, tags=["python"], memory_type="note")
        query_duration = time.time() - query_start

        print(f"  Query completed in {query_duration * 1000:.2f}ms")
        print(f"  Returned {len(results)} results")

        assert query_duration < 0.5, f"Combined filter query took {query_duration * 1000:.2f}ms (should be <500ms)"

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_search_by_tag_performance_10k(self, temp_db_path):
        """
        Verify search_by_tag() performance at 10K scale.

        This method doesn't use semantic search, so it should be even faster.
        Target: <200ms
        """
        storage = SqliteVecMemoryStorage(str(temp_db_path / "test.db"))
        await storage.initialize()

        print("\n=== SEARCH_BY_TAG PERFORMANCE: 10K Memories ===")

        # Store 10K memories
        tags_pool = ["python", "java", "javascript", "rust", "go"]
        print("Storing 10,000 memories...")

        for i in range(10000):
            tag = tags_pool[i % len(tags_pool)]
            await storage.store(
                Memory(
                    content=f"Content {i}", content_hash=generate_content_hash(f"Content {i}"), tags=[tag], memory_type="note"
                )
            )

        print("Store complete")

        # Test search_by_tag performance
        print("\nTesting search_by_tag(['python'])")
        query_start = time.time()
        results = await storage.search_by_tag(["python"], limit=2000)
        query_duration = time.time() - query_start

        print(f"  Query completed in {query_duration * 1000:.2f}ms")
        print(f"  Returned {len(results)} results")

        # Target: <200ms (no semantic search overhead)
        assert query_duration < 0.2, f"search_by_tag took {query_duration * 1000:.2f}ms (should be <200ms)"

        # Verify correctness (10K memories / 5 tags = 2000 per tag)
        assert len(results) == 2000, f"Should return 2000 results (10K / 5 tags), got {len(results)}"

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_count_performance_at_scale(self, temp_db_path):
        """
        Verify count_all_memories() performance with filters.

        Counting should be even faster than retrieval since no data transfer.
        Target: <100ms
        """
        storage = SqliteVecMemoryStorage(str(temp_db_path / "test.db"))
        await storage.initialize()

        print("\n=== COUNT PERFORMANCE: 10K Memories ===")

        # Store 10K memories
        print("Storing 10,000 memories...")
        for i in range(10000):
            await storage.store(
                Memory(
                    content=f"Content {i}",
                    content_hash=generate_content_hash(f"Content {i}"),
                    tags=["python"] if i % 2 == 0 else ["java"],
                    memory_type="note",
                )
            )

        print("Store complete")

        # Test count with tag filter
        print("\nTesting count_all_memories(tags=['python'])")
        query_start = time.time()
        count = await storage.count_all_memories(tags=["python"])
        query_duration = time.time() - query_start

        print(f"  Count completed in {query_duration * 1000:.2f}ms")
        print(f"  Result: {count} memories")

        # Target: <100ms
        assert query_duration < 0.1, f"count_all_memories took {query_duration * 1000:.2f}ms (should be <100ms)"

        # Verify correctness
        assert count == 5000, f"Should count 5000 python-tagged memories, got {count}"

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_performance_comparison_old_vs_new(self, temp_db_path):
        """
        Compare performance: database-level filtering vs Python post-filtering.

        This demonstrates the performance improvement from our fix.
        """
        storage = SqliteVecMemoryStorage(str(temp_db_path / "test.db"))
        await storage.initialize()

        print("\n=== PERFORMANCE COMPARISON: DB Filter vs Python Filter ===")

        # Store 5K memories (smaller dataset for comparison test)
        print("Storing 5,000 memories...")
        for i in range(5000):
            await storage.store(
                Memory(
                    content=f"Tutorial {i}",
                    content_hash=generate_content_hash(f"Tutorial {i}"),
                    tags=["python"] if i % 2 == 0 else ["java"],
                    memory_type="note",
                )
            )

        print("Store complete")

        # NEW approach: database-level filtering
        print("\nNEW: Database-level filtering")
        start = time.time()
        results_new = await storage.retrieve(query="tutorial", n_results=10, tags=["python"])
        duration_new = time.time() - start
        print(f"  Duration: {duration_new * 1000:.2f}ms")
        print(f"  Results: {len(results_new)}")

        # OLD approach simulation: get all, filter in Python
        print("\nOLD: Python post-filtering (simulated)")
        start = time.time()
        # Get unfiltered results
        results_unfiltered = await storage.retrieve(
            query="tutorial",
            n_results=100,  # Get more to simulate old behavior
        )
        # Filter in Python
        results_old = [r for r in results_unfiltered if "python" in r.memory.tags]
        duration_old = time.time() - start
        print(f"  Duration: {duration_old * 1000:.2f}ms")
        print(f"  Results: {len(results_old)}")

        print(f"\nPerformance improvement: {duration_old / duration_new:.2f}x faster")

        # NEW approach should be significantly faster
        assert duration_new < duration_old, "Database-level filtering should be faster than Python post-filtering"

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_olog_n_scaling_verification(self, temp_db_path):
        """
        Verify O(log n) scaling by testing at multiple data sizes.

        O(log n) behavior: Doubling data size increases query time by ~40% or less
        O(n) behavior: Doubling data size doubles query time (100% increase)

        This test proves our relational tag indexes provide true O(log n) performance.
        """
        storage = SqliteVecMemoryStorage(str(temp_db_path / "test.db"))
        await storage.initialize()

        print("\n=== O(log n) SCALING VERIFICATION ===")
        print("Testing at 10K, 50K, and 100K memory scales")
        print("Expected: ~30-40% time increase per doubling (O(log n))")
        print("Failure mode: ~100% time increase per doubling (O(n))")

        # Test at multiple scales
        test_sizes = [10_000, 50_000, 100_000]
        durations = {}

        for size in test_sizes:
            print(f"\n--- Testing with {size:,} memories ---")

            # Clear and repopulate
            # (In practice, we'd use a fresh database, but for this test we'll accumulate)
            current_count = await storage.count_all_memories()
            memories_to_add = size - current_count

            if memories_to_add > 0:
                print(f"Adding {memories_to_add:,} more memories...")
                tags_pool = ["python", "java", "rust", "go", "javascript"]
                store_start = time.time()

                for i in range(current_count, size):
                    tag = tags_pool[i % len(tags_pool)]
                    await storage.store(
                        Memory(
                            content=f"Tutorial {i} about {tag}",
                            content_hash=generate_content_hash(f"Tutorial {i}"),
                            tags=[tag, "programming"],
                            memory_type="note",
                        )
                    )

                    if (i + 1) % 10_000 == 0:
                        elapsed = time.time() - store_start
                        print(f"  Stored {i + 1:,} memories ({elapsed:.2f}s elapsed)")

                store_duration = time.time() - store_start
                print(f"Added {memories_to_add:,} memories in {store_duration:.2f}s")

            # Run tag-filtered query and measure time
            print("Running tag-filtered query...")
            query_start = time.time()
            results = await storage.retrieve(query="programming tutorial", n_results=10, tags=["python"])
            query_duration = time.time() - query_start

            durations[size] = query_duration
            print(f"Query completed in {query_duration * 1000:.2f}ms")
            print(f"Returned {len(results)} results")

            # Sanity check: should return results
            assert len(results) > 0, "Should return results"

        # Analyze scaling behavior
        print("\n=== SCALING ANALYSIS ===")

        # 10K → 50K (5x increase)
        increase_10k_to_50k = (durations[50_000] / durations[10_000]) - 1
        print(f"10K → 50K (5x data): {increase_10k_to_50k * 100:.1f}% time increase")
        print(f"  10K: {durations[10_000] * 1000:.2f}ms")
        print(f"  50K: {durations[50_000] * 1000:.2f}ms")

        # 50K → 100K (2x increase)
        increase_50k_to_100k = (durations[100_000] / durations[50_000]) - 1
        print(f"\n50K → 100K (2x data): {increase_50k_to_100k * 100:.1f}% time increase")
        print(f"  50K: {durations[50_000] * 1000:.2f}ms")
        print(f"  100K: {durations[100_000] * 1000:.2f}ms")

        # O(log n) validation:
        # For a 2x data increase, O(log n) should see ~40% or less time increase
        # For O(n), we'd see ~100% time increase
        print("\n=== VERDICT ===")
        if increase_50k_to_100k < 0.6:
            print(f"✓ O(log n) performance confirmed ({increase_50k_to_100k * 100:.1f}% < 60%)")
        elif increase_50k_to_100k < 1.0:
            print(f"⚠ Sub-linear but not O(log n) ({increase_50k_to_100k * 100:.1f}%)")
        else:
            print(f"✗ Linear O(n) behavior detected ({increase_50k_to_100k * 100:.1f}% ≈ 100%)")

        # CRITICAL ASSERTION: Doubling data should NOT double query time
        assert (
            increase_50k_to_100k < 0.6
        ), f"Query time should scale O(log n): 2x data → <60% time increase, got {increase_50k_to_100k * 100:.1f}%"

        # Additional assertion: 100K query should complete in reasonable time
        assert durations[100_000] < 0.5, f"100K query should complete in <500ms, took {durations[100_000] * 1000:.2f}ms"


@pytest.fixture
def temp_db_path(tmp_path):
    """Provide temporary directory for test databases."""
    return tmp_path
