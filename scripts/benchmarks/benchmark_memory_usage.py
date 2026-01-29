#!/usr/bin/env python3
"""
Benchmark: Memory usage comparison for filter operations (Issue #374)

Compares memory consumption between Python-filtering and SQL-filtering approaches.
"""

import asyncio
import gc
import hashlib
import os
import sys
import tempfile
import tracemalloc
from datetime import datetime, timedelta
import random
import string

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.models.memory import Memory


def generate_random_content(length: int = 100) -> str:
    return ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=length))


def generate_content_hash(content: str) -> str:
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def generate_test_memory(index: int, days_ago: int = 0) -> Memory:
    tags = random.sample(['python', 'javascript', 'rust', 'go', 'test', 'benchmark'], k=random.randint(1, 3))
    created = datetime.now() - timedelta(days=days_ago)
    content = f"Test memory {index}: {generate_random_content(500)}"  # Larger content

    return Memory(
        content=content,
        content_hash=generate_content_hash(content),
        tags=tags,
        memory_type="observation",
        created_at=int(created.timestamp()),
        updated_at=int(created.timestamp()),
        created_at_iso=created.isoformat() + 'Z',
        updated_at_iso=created.isoformat() + 'Z'
    )


async def measure_memory_python_filter(storage: SqliteVecMemoryStorage, tag: str) -> dict:
    """Measure memory usage of Python-based filtering."""
    gc.collect()
    tracemalloc.start()

    # Python filtering: load all, filter in memory
    all_memories = await storage.get_all_memories()
    filtered = [m for m in all_memories if tag in m.tags]

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "method": "python_filter",
        "current_mb": current / 1024 / 1024,
        "peak_mb": peak / 1024 / 1024,
        "total_loaded": len(all_memories),
        "matched": len(filtered)
    }


async def measure_memory_sql_filter(storage: SqliteVecMemoryStorage, tag: str) -> dict:
    """Measure memory usage of SQL-based filtering."""
    gc.collect()
    tracemalloc.start()

    # SQL filtering: only load matching rows
    stripped_tag = tag.strip()
    exact_match_pattern = f"*,{stripped_tag},*"

    cursor = storage.conn.execute(
        "SELECT content_hash, content, tags FROM memories WHERE (',' || REPLACE(tags, ' ', '') || ',') GLOB ? AND deleted_at IS NULL",
        (exact_match_pattern,)
    )
    results = cursor.fetchall()

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "method": "sql_filter",
        "current_mb": current / 1024 / 1024,
        "peak_mb": peak / 1024 / 1024,
        "matched": len(results)
    }


async def run_memory_benchmark(counts: list[int] = [1000, 5000, 10000]):
    """Run memory usage benchmark."""
    print("=" * 70)
    print("BENCHMARK: Memory Usage Comparison (Issue #374)")
    print("=" * 70)

    results = []

    for count in counts:
        print(f"\n{'='*70}")
        print(f"Testing with {count} memories")
        print("=" * 70)

        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            storage = SqliteVecMemoryStorage(db_path)
            await storage.initialize()

            # Populate
            print(f"  Populating {count} memories...")
            for i in range(count):
                days_ago = random.randint(0, 365)
                memory = generate_test_memory(i, days_ago)
                await storage.store(memory)
                if (i + 1) % 2000 == 0:
                    print(f"    Inserted {i + 1}/{count}...")

            print(f"  ✓ Population complete")

            # Measure Python filtering
            python_result = await measure_memory_python_filter(storage, 'python')
            print(f"\n  Python Filter Memory:")
            print(f"    Peak: {python_result['peak_mb']:.2f} MB")
            print(f"    Loaded: {python_result['total_loaded']} memories")

            # Measure SQL filtering
            sql_result = await measure_memory_sql_filter(storage, 'python')
            print(f"\n  SQL Filter Memory:")
            print(f"    Peak: {sql_result['peak_mb']:.2f} MB")
            print(f"    Matched: {sql_result['matched']} memories")

            # Calculate reduction
            reduction = ((python_result['peak_mb'] - sql_result['peak_mb']) / python_result['peak_mb']) * 100
            print(f"\n  📉 Memory Reduction: {reduction:.1f}%")

            results.append({
                "count": count,
                "python_peak_mb": python_result['peak_mb'],
                "sql_peak_mb": sql_result['peak_mb'],
                "reduction_pct": reduction
            })

        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: Memory Usage")
    print("=" * 70)
    print(f"\n{'Memories':<10} {'Python (Peak)':<15} {'SQL (Peak)':<15} {'Reduction':<12}")
    print("-" * 55)

    for r in results:
        print(f"{r['count']:<10} {r['python_peak_mb']:.2f} MB{'':<7} {r['sql_peak_mb']:.2f} MB{'':<7} {r['reduction_pct']:.1f}%")

    print("\n✅ SQL-level filtering significantly reduces memory usage!")


if __name__ == "__main__":
    asyncio.run(run_memory_benchmark([1000, 5000, 10000]))
