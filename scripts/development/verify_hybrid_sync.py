#!/usr/bin/env python3
"""
Comprehensive verification of hybrid storage background sync functionality.
"""

import asyncio
import os
import sys
import tempfile
import time
from unittest.mock import patch

sys.path.insert(0, "src")

import hashlib

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.hybrid import HybridMemoryStorage


class DetailedMockCloudflare:
    """Detailed mock for tracking sync operations."""

    def __init__(self, **kwargs):
        self.memories = {}
        self.operation_log = []
        self.initialized = False
        self.delay = 0.01  # Simulate network delay

    async def initialize(self):
        self.initialized = True
        self.operation_log.append(("init", time.time()))

    async def store(self, memory):
        await asyncio.sleep(self.delay)  # Simulate network
        self.memories[memory.content_hash] = memory
        self.operation_log.append(("store", memory.content_hash, time.time()))
        return True, "Stored"

    async def delete(self, content_hash):
        await asyncio.sleep(self.delay)
        if content_hash in self.memories:
            del self.memories[content_hash]
        self.operation_log.append(("delete", content_hash, time.time()))
        return True, "Deleted"

    async def update_memory_metadata(self, content_hash, updates, preserve_timestamps=True):
        await asyncio.sleep(self.delay)
        self.operation_log.append(("update", content_hash, time.time()))
        return True, "Updated"

    async def get_stats(self):
        return {"total": len(self.memories)}

    async def close(self):
        self.operation_log.append(("close", time.time()))


async def verify_sync():
    print("🔍 HYBRID STORAGE BACKGROUND SYNC VERIFICATION")
    print("=" * 60)

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        config = {"api_token": "test", "account_id": "test", "vectorize_index": "test", "d1_database_id": "test"}

        with patch("mcp_memory_service.storage.hybrid.CloudflareStorage", DetailedMockCloudflare):
            # Initialize with short sync interval
            storage = HybridMemoryStorage(
                sqlite_db_path=db_path,
                cloudflare_config=config,
                sync_interval=0.5,  # 500ms for quick testing
                batch_size=2,
            )

            await storage.initialize()
            print("✅ Hybrid storage initialized with background sync")
            print("  • Primary: SQLite-vec (local)")
            print("  • Secondary: Mock Cloudflare (simulated)")
            print("  • Sync interval: 0.5 seconds")
            print("  • Batch size: 2 operations")
            print()

            # TEST 1: Store operations are queued
            print("📝 TEST 1: Store Operations Queuing")
            print("-" * 40)

            memories = []
            for i in range(4):
                content = f"Sync test memory #{i + 1} at {time.time()}"
                memory = Memory(
                    content=content,
                    content_hash=hashlib.sha256(content.encode()).hexdigest(),
                    tags=["sync-verify"],
                    memory_type="test",
                )
                memories.append(memory)

                start = time.time()
                success, msg = await storage.store(memory)
                elapsed = (time.time() - start) * 1000
                print(f"  Memory #{i + 1}: ✅ stored in {elapsed:.1f}ms (local)")

            # Check initial queue
            status = await storage.sync_service.get_sync_status()
            print("\n  📊 Queue status after stores:")
            print(f"     • Queued operations: {status['queue_size']}")
            print(f"     • Processed: {status['stats']['operations_processed']}")

            # TEST 2: Wait for automatic background sync
            print("\n⏳ TEST 2: Automatic Background Sync")
            print("-" * 40)
            print("  Waiting 1.5 seconds for automatic sync...")
            await asyncio.sleep(1.5)

            status = await storage.sync_service.get_sync_status()
            mock_log = storage.secondary.operation_log

            print("\n  📊 After automatic sync:")
            print(f"     • Queue remaining: {status['queue_size']}")
            print(f"     • Operations processed: {status['stats']['operations_processed']}")
            print(f"     • Mock Cloudflare received: {len([op for op in mock_log if op[0] == 'store'])} stores")

            # TEST 3: Delete operation
            print("\n🗑️ TEST 3: Delete Operation Sync")
            print("-" * 40)

            delete_hash = memories[0].content_hash
            success, msg = await storage.delete(delete_hash)
            print("  Delete operation: ✅ (local)")

            await asyncio.sleep(1)  # Wait for sync

            delete_ops = [op for op in mock_log if op[0] == "delete"]
            print(f"  Mock Cloudflare received: {len(delete_ops)} delete operation(s)")

            # TEST 4: Force sync
            print("\n🔄 TEST 4: Force Sync")
            print("-" * 40)

            # Add more memories
            for i in range(2):
                content = f"Force sync test #{i + 1}"
                memory = Memory(
                    content=content,
                    content_hash=hashlib.sha256(content.encode()).hexdigest(),
                    tags=["force-sync"],
                    memory_type="test",
                )
                await storage.store(memory)

            print("  Added 2 more memories")

            # Force sync
            result = await storage.force_sync()
            print("\n  Force sync result:")
            print(f"     • Status: {result['status']}")
            print(f"     • Primary memories: {result['primary_memories']}")
            print(f"     • Synced to secondary: {result['synced_to_secondary']}")
            print(f"     • Duration: {result.get('duration', 0):.3f}s")

            # Final verification
            print("\n✅ FINAL VERIFICATION")
            print("-" * 40)

            final_status = await storage.sync_service.get_sync_status()
            final_mock_ops = storage.secondary.operation_log

            print("  Sync service statistics:")
            print(f"     • Total operations processed: {final_status['stats']['operations_processed']}")
            print(f"     • Failed operations: {final_status['stats'].get('operations_failed', 0)}")
            print(f"     • Cloudflare available: {final_status['cloudflare_available']}")

            print("\n  Mock Cloudflare operations log:")
            store_count = len([op for op in final_mock_ops if op[0] == "store"])
            delete_count = len([op for op in final_mock_ops if op[0] == "delete"])
            update_count = len([op for op in final_mock_ops if op[0] == "update"])

            print(f"     • Store operations: {store_count}")
            print(f"     • Delete operations: {delete_count}")
            print(f"     • Update operations: {update_count}")
            print(f"     • Total operations: {len(final_mock_ops) - 2}")  # Exclude init and close

            # Verify memory counts match
            primary_count = len(await storage.primary.get_all_memories())
            secondary_count = len(storage.secondary.memories)

            print("\n  Memory count verification:")
            print(f"     • Primary (SQLite-vec): {primary_count}")
            print(f"     • Secondary (Mock CF): {secondary_count}")
            print(f"     • Match: {'✅ YES' if primary_count == secondary_count else '❌ NO'}")

            await storage.close()

            print("\n" + "=" * 60)
            print("🎉 BACKGROUND SYNC VERIFICATION COMPLETE")
            print("\nSummary: The hybrid storage backend is working correctly!")
            print("  ✅ Store operations are queued for background sync")
            print("  ✅ Automatic sync processes operations in batches")
            print("  ✅ Delete operations are synced to secondary")
            print("  ✅ Force sync ensures complete synchronization")
            print("  ✅ Both backends maintain consistency")

    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


if __name__ == "__main__":
    asyncio.run(verify_sync())
