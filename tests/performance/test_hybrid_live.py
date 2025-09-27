#!/usr/bin/env python3
"""
Live performance test of the hybrid storage backend implementation.
Demonstrates performance, functionality, and sync capabilities under load.
"""

import asyncio
import sys
import time
import tempfile
import os
from pathlib import Path

# Add src to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from mcp_memory_service.storage.hybrid import HybridMemoryStorage
from mcp_memory_service.models.memory import Memory
import hashlib

async def test_hybrid_storage():
    """Test the hybrid storage implementation with live demonstrations."""

    print("🚀 Testing Hybrid Storage Backend")
    print("=" * 50)

    # Create a temporary SQLite database for testing
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name

    try:
        # Initialize hybrid storage (without Cloudflare for this demo)
        print("📍 Step 1: Initializing Hybrid Storage")
        storage = HybridMemoryStorage(
            sqlite_db_path=db_path,
            embedding_model="all-MiniLM-L6-v2",
            cloudflare_config=None,  # Will operate in SQLite-only mode
            sync_interval=30,  # Short interval for demo
            batch_size=5
        )

        print("   Initializing storage backend...")
        await storage.initialize()
        print(f"   ✅ Storage initialized")
        print(f"   📊 Primary: {storage.primary.__class__.__name__}")
        print(f"   📊 Secondary: {storage.secondary.__class__.__name__ if storage.secondary else 'None (SQLite-only mode)'}")
        print(f"   📊 Sync Service: {'Running' if storage.sync_service and storage.sync_service.is_running else 'Disabled'}")
        print()

        # Test 1: Performance measurement
        print("📍 Step 2: Performance Test")
        memories_to_test = []

        for i in range(5):
            content = f"Performance test memory #{i+1} - testing hybrid storage speed"
            content_hash = hashlib.sha256(content.encode()).hexdigest()

            memory = Memory(
                content=content,
                content_hash=content_hash,
                tags=["hybrid", "test", f"batch_{i}"],
                memory_type="performance_test",
                metadata={"test_batch": i, "test_type": "performance"},
                created_at=time.time()
            )
            memories_to_test.append(memory)

        # Measure write performance
        print("   Testing write performance...")
        write_times = []
        for i, memory in enumerate(memories_to_test):
            start_time = time.time()
            success, message = await storage.store(memory)
            duration = time.time() - start_time
            write_times.append(duration)

            if success:
                print(f"   ✅ Write #{i+1}: {duration*1000:.1f}ms")
            else:
                print(f"   ❌ Write #{i+1} failed: {message}")

        avg_write_time = sum(write_times) / len(write_times)
        print(f"   📊 Average write time: {avg_write_time*1000:.1f}ms")
        print()

        # Test 2: Read performance
        print("📍 Step 3: Read Performance Test")
        read_times = []

        for i in range(3):
            start_time = time.time()
            results = await storage.retrieve("performance test", n_results=5)
            duration = time.time() - start_time
            read_times.append(duration)

            print(f"   ✅ Read #{i+1}: {duration*1000:.1f}ms ({len(results)} results)")

        avg_read_time = sum(read_times) / len(read_times)
        print(f"   📊 Average read time: {avg_read_time*1000:.1f}ms")
        print()

        # Test 3: Different operations
        print("📍 Step 4: Testing Various Operations")

        # Search by tags
        start_time = time.time()
        tagged_memories = await storage.search_by_tags(["hybrid"])
        tag_search_time = time.time() - start_time
        print(f"   ✅ Tag search: {tag_search_time*1000:.1f}ms ({len(tagged_memories)} results)")

        # Get stats
        start_time = time.time()
        stats = await storage.get_stats()
        stats_time = time.time() - start_time
        print(f"   ✅ Stats retrieval: {stats_time*1000:.1f}ms")
        print(f"      - Backend: {stats.get('storage_backend')}")
        print(f"      - Total memories: {stats.get('total_memories', 0)}")
        print(f"      - Sync enabled: {stats.get('sync_enabled', False)}")

        # Test delete
        if memories_to_test:
            test_memory = memories_to_test[0]
            start_time = time.time()
            success, message = await storage.delete(test_memory.content_hash)
            delete_time = time.time() - start_time
            print(f"   ✅ Delete operation: {delete_time*1000:.1f}ms ({'Success' if success else 'Failed'})")

        print()

        # Test 4: Concurrent operations
        print("📍 Step 5: Concurrent Operations Test")

        async def store_memory(content_suffix):
            content = f"Concurrent test memory {content_suffix}"
            content_hash = hashlib.sha256(content.encode()).hexdigest()

            memory = Memory(
                content=content,
                content_hash=content_hash,
                tags=["concurrent", "hybrid"],
                memory_type="concurrent_test",
                metadata={"test_id": content_suffix},
                created_at=time.time()
            )
            return await storage.store(memory)

        start_time = time.time()
        concurrent_tasks = [store_memory(i) for i in range(10)]
        results = await asyncio.gather(*concurrent_tasks)
        concurrent_time = time.time() - start_time

        successful_ops = sum(1 for success, _ in results if success)
        print(f"   ✅ Concurrent operations: {concurrent_time*1000:.1f}ms")
        print(f"      - Operations: 10 concurrent stores")
        print(f"      - Successful: {successful_ops}/10")
        print(f"      - Avg per operation: {(concurrent_time/10)*1000:.1f}ms")
        print()

        # Final stats
        print("📍 Step 6: Final Statistics")
        final_stats = await storage.get_stats()
        print(f"   📊 Total memories stored: {final_stats.get('total_memories', 0)}")
        print(f"   📊 Storage backend: {final_stats.get('storage_backend')}")

        if storage.sync_service:
            sync_status = await storage.sync_service.get_sync_status()
            print(f"   📊 Sync queue size: {sync_status.get('queue_size', 0)}")
            print(f"   📊 Operations processed: {sync_status.get('stats', {}).get('operations_processed', 0)}")

        print()
        print("🎉 Hybrid Storage Test Complete!")
        print("=" * 50)

        # Performance summary
        print("📊 PERFORMANCE SUMMARY:")
        print(f"   • Average Write: {avg_write_time*1000:.1f}ms")
        print(f"   • Average Read:  {avg_read_time*1000:.1f}ms")
        print(f"   • Tag Search:    {tag_search_time*1000:.1f}ms")
        print(f"   • Stats Query:   {stats_time*1000:.1f}ms")
        print(f"   • Delete Op:     {delete_time*1000:.1f}ms")
        print(f"   • Concurrent:    {(concurrent_time/10)*1000:.1f}ms per op")

        # Cleanup
        await storage.close()

    finally:
        # Clean up temp file
        if os.path.exists(db_path):
            os.unlink(db_path)

if __name__ == "__main__":
    print("🚀 Hybrid Storage Live Demo")
    print("Testing the new hybrid backend implementation...")
    print()

    # Run the async test
    asyncio.run(test_hybrid_storage())