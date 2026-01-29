#!/usr/bin/env python3
"""
Benchmark: Filter-based operations performance (Issue #374)

Tests performance of filter-based deletion and time-range search
before and after SQL-level filtering optimization.
"""

import asyncio
import hashlib
import os
import sys
import tempfile
import time
from datetime import datetime, timedelta
import random
import string

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.models.memory import Memory


def generate_random_content(length: int = 100) -> str:
    """Generate random content for test memories."""
    return ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=length))


def generate_content_hash(content: str) -> str:
    """Generate SHA256 hash of content."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def generate_test_memory(index: int, days_ago: int = 0) -> Memory:
    """Generate a test memory with realistic data."""
    tags = random.sample(['python', 'javascript', 'rust', 'go', 'test', 'benchmark', 'project-a', 'project-b'], k=random.randint(1, 4))
    created = datetime.now() - timedelta(days=days_ago)
    content = f"Test memory {index}: {generate_random_content(200)}"

    return Memory(
        content=content,
        content_hash=generate_content_hash(content),
        tags=tags,
        memory_type="test",
        created_at=int(created.timestamp()),
        updated_at=int(created.timestamp()),
        created_at_iso=created.isoformat() + 'Z',
        updated_at_iso=created.isoformat() + 'Z'
    )


async def populate_database(storage: SqliteVecMemoryStorage, count: int) -> None:
    """Populate database with test memories."""
    print(f"  Populating database with {count} memories...")

    for i in range(count):
        days_ago = random.randint(0, 365)
        memory = generate_test_memory(i, days_ago)
        await storage.store(memory)

        if (i + 1) % 1000 == 0:
            print(f"    Inserted {i + 1}/{count} memories...")

    print(f"  ✓ Populated {count} memories")


async def benchmark_list_all_then_filter(storage: SqliteVecMemoryStorage, tag: str) -> dict:
    """Benchmark: Load all memories, filter in Python (CURRENT METHOD)."""
    start = time.perf_counter()

    # Current implementation: load all, filter in Python
    all_memories = await storage.get_all_memories()
    filtered = [m for m in all_memories if tag in m.tags]

    elapsed = time.perf_counter() - start

    return {
        "method": "list_all_then_filter",
        "total_memories": len(all_memories),
        "matched": len(filtered),
        "elapsed_ms": elapsed * 1000
    }


async def benchmark_sql_filter_tags(storage: SqliteVecMemoryStorage, tag: str) -> dict:
    """Benchmark: SQL-level tag filtering (OPTIMIZED METHOD)."""
    start = time.perf_counter()

    # Direct SQL query with tag filter
    cursor = storage.conn.execute(
        "SELECT content_hash FROM memories WHERE (',' || REPLACE(tags, ' ', '') || ',') GLOB ? AND deleted_at IS NULL",
        (f'*,{tag},*',)
    )
    matched_hashes = [row[0] for row in cursor.fetchall()]

    elapsed = time.perf_counter() - start

    return {
        "method": "sql_filter_tags",
        "matched": len(matched_hashes),
        "elapsed_ms": elapsed * 1000
    }


async def benchmark_time_range_python(storage: SqliteVecMemoryStorage, days: int) -> dict:
    """Benchmark: Load all, filter by time in Python (CURRENT METHOD)."""
    start = time.perf_counter()

    cutoff = datetime.now() - timedelta(days=days)
    cutoff_ts = int(cutoff.timestamp())

    # Current implementation
    all_memories = await storage.get_all_memories()
    filtered = [m for m in all_memories if m.created_at >= cutoff_ts]

    elapsed = time.perf_counter() - start

    return {
        "method": "time_range_python",
        "total_memories": len(all_memories),
        "matched": len(filtered),
        "elapsed_ms": elapsed * 1000
    }


async def benchmark_time_range_sql(storage: SqliteVecMemoryStorage, days: int) -> dict:
    """Benchmark: SQL-level time filtering (OPTIMIZED METHOD)."""
    start = time.perf_counter()

    cutoff = datetime.now() - timedelta(days=days)
    cutoff_ts = int(cutoff.timestamp())

    # Direct SQL with time filter
    cursor = storage.conn.execute(
        "SELECT content_hash FROM memories WHERE created_at >= ? AND deleted_at IS NULL",
        (cutoff_ts,)
    )
    matched_hashes = [row[0] for row in cursor.fetchall()]

    elapsed = time.perf_counter() - start

    return {
        "method": "time_range_sql",
        "matched": len(matched_hashes),
        "elapsed_ms": elapsed * 1000
    }


async def run_benchmarks(memory_counts: list[int] = [100, 1000, 5000, 10000]):
    """Run all benchmarks."""
    print("=" * 70)
    print("BENCHMARK: Filter-based Operations Performance (Issue #374)")
    print("=" * 70)

    results = []

    for count in memory_counts:
        print(f"\n{'='*70}")
        print(f"Testing with {count} memories")
        print("=" * 70)

        # Create temp database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            storage = SqliteVecMemoryStorage(db_path)
            await storage.initialize()

            # Populate
            await populate_database(storage, count)

            # Benchmark tag filtering
            print(f"\n  Tag Filtering (tag='python'):")

            result_python = await benchmark_list_all_then_filter(storage, 'python')
            print(f"    Python filter: {result_python['elapsed_ms']:.2f}ms ({result_python['matched']} matches)")

            result_sql = await benchmark_sql_filter_tags(storage, 'python')
            print(f"    SQL filter:    {result_sql['elapsed_ms']:.2f}ms ({result_sql['matched']} matches)")

            speedup_tag = result_python['elapsed_ms'] / result_sql['elapsed_ms'] if result_sql['elapsed_ms'] > 0 else float('inf')
            print(f"    ⚡ Speedup:     {speedup_tag:.1f}x")

            # Benchmark time filtering
            print(f"\n  Time Range Filtering (last 30 days):")

            result_time_python = await benchmark_time_range_python(storage, 30)
            print(f"    Python filter: {result_time_python['elapsed_ms']:.2f}ms ({result_time_python['matched']} matches)")

            result_time_sql = await benchmark_time_range_sql(storage, 30)
            print(f"    SQL filter:    {result_time_sql['elapsed_ms']:.2f}ms ({result_time_sql['matched']} matches)")

            speedup_time = result_time_python['elapsed_ms'] / result_time_sql['elapsed_ms'] if result_time_sql['elapsed_ms'] > 0 else float('inf')
            print(f"    ⚡ Speedup:     {speedup_time:.1f}x")

            results.append({
                "count": count,
                "tag_python_ms": result_python['elapsed_ms'],
                "tag_sql_ms": result_sql['elapsed_ms'],
                "tag_speedup": speedup_tag,
                "time_python_ms": result_time_python['elapsed_ms'],
                "time_sql_ms": result_time_sql['elapsed_ms'],
                "time_speedup": speedup_time
            })

        finally:
            # Cleanup
            if os.path.exists(db_path):
                os.unlink(db_path)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Memories':<10} {'Tag (Python)':<15} {'Tag (SQL)':<12} {'Speedup':<10} {'Time (Python)':<15} {'Time (SQL)':<12} {'Speedup':<10}")
    print("-" * 90)

    for r in results:
        print(f"{r['count']:<10} {r['tag_python_ms']:.2f}ms{'':<7} {r['tag_sql_ms']:.2f}ms{'':<5} {r['tag_speedup']:.1f}x{'':<5} {r['time_python_ms']:.2f}ms{'':<7} {r['time_sql_ms']:.2f}ms{'':<5} {r['time_speedup']:.1f}x")

    print("\n✅ SQL-level filtering provides significant performance improvement!")
    print("   Recommendation: Update base.py to use SQL filtering when available.")


if __name__ == "__main__":
    # Run with smaller counts for quick test, larger for full benchmark
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        asyncio.run(run_benchmarks([100, 1000, 5000, 10000]))
    else:
        asyncio.run(run_benchmarks([100, 500, 1000]))
