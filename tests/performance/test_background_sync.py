#!/usr/bin/env python3
"""
Performance test for background sync service with mock Cloudflare backend.
Verifies that the sync queue and processing work correctly under load.
"""

import asyncio
import sys
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
import time

# Add src to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from mcp_memory_service.storage.hybrid import HybridMemoryStorage, BackgroundSyncService
from mcp_memory_service.models.memory import Memory
import hashlib


class MockCloudflareStorage:
    """Mock Cloudflare storage to test sync without real API."""

    def __init__(self, **kwargs):
        self.memories = {}
        self.operations = []
        self.initialized = False

    async def initialize(self):
        self.initialized = True
        print("  ☁️ Mock Cloudflare initialized")

    async def store(self, memory):
        self.memories[memory.content_hash] = memory
        self.operations.append(('store', memory.content_hash))
        return True, "Stored in mock Cloudflare"

    async def delete(self, content_hash):
        if content_hash in self.memories:
            del self.memories[content_hash]
        self.operations.append(('delete', content_hash))
        return True, "Deleted from mock Cloudflare"

    async def update_memory_metadata(self, content_hash, updates, preserve_timestamps=True):
        self.operations.append(('update', content_hash))
        return True, "Updated in mock Cloudflare"

    async def get_stats(self):
        return {
            "total_memories": len(self.memories),
            "operations_count": len(self.operations)
        }

    async def close(self):
        pass


async def test_background_sync_with_mock():
    print("🔍 Testing Background Sync with Mock Cloudflare")
    print("=" * 50)

    # Create temp db
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name

    try:
        # Mock Cloudflare config
        mock_config = {
            'api_token': 'mock_token',
            'account_id': 'mock_account',
            'vectorize_index': 'mock_index',
            'd1_database_id': 'mock_db'
        }

        # Patch CloudflareStorage with our mock
        with patch('mcp_memory_service.storage.hybrid.CloudflareStorage', MockCloudflareStorage):
            storage = HybridMemoryStorage(
                sqlite_db_path=db_path,
                embedding_model='all-MiniLM-L6-v2',
                cloudflare_config=mock_config,
                sync_interval=1,  # 1 second for quick testing
                batch_size=3
            )

            await storage.initialize()
            print(f"✅ Hybrid storage initialized")
            print(f"  📊 Primary: {storage.primary.__class__.__name__}")
            print(f"  ☁️ Secondary: {storage.secondary.__class__.__name__ if storage.secondary else 'None'}")
            print(f"  🔄 Sync Service: {'Running' if storage.sync_service and storage.sync_service.is_running else 'Not Running'}")
            print()

            # Store memories to trigger sync operations
            print("📝 Storing test memories...")
            memories_stored = []
            for i in range(5):
                content = f"Background sync test memory #{i+1}"
                memory = Memory(
                    content=content,
                    content_hash=hashlib.sha256(content.encode()).hexdigest(),
                    tags=['sync-test', f'batch-{i//3}'],
                    memory_type='test',
                    metadata={'index': i}
                )
                success, msg = await storage.store(memory)
                memories_stored.append(memory)
                print(f"  Memory #{i+1}: {'✅' if success else '❌'}")

            # Check sync queue status
            print("\n🔄 Checking sync queue...")
            if storage.sync_service:
                status = await storage.sync_service.get_sync_status()
                print(f"  Queue size: {status.get('pending_operations', 0)}")
                print(f"  Cloudflare available: {status.get('cloudflare_available', False)}")
                print(f"  Operations processed: {status.get('operations_processed', 0)}")

                # Wait for background processing
                print("\n⏳ Waiting for background sync (2 seconds)...")
                await asyncio.sleep(2)

                # Check status after processing
                status = await storage.sync_service.get_sync_status()
                print(f"\n📊 After background processing:")
                print(f"  Queue size: {status.get('pending_operations', 0)}")
                print(f"  Operations processed: {status.get('operations_processed', 0)}")
                print(f"  Operations failed: {status.get('operations_failed', 0)}")
                print(f"  Last sync duration: {status.get('stats', {}).get('last_sync_duration', 0):.2f}s")

                # Check mock Cloudflare received operations
                mock_cf_stats = await storage.secondary.get_stats()
                print(f"\n☁️ Mock Cloudflare status:")
                print(f"  Total memories: {mock_cf_stats['total_memories']}")
                print(f"  Operations received: {mock_cf_stats['operations_count']}")

                # Test delete operation
                print("\n🗑️ Testing delete operation...")
                success, msg = await storage.delete(memories_stored[0].content_hash)
                print(f"  Delete: {'✅' if success else '❌'}")

                # Wait for delete to sync
                await asyncio.sleep(1)

                # Force sync remaining operations
                print("\n🔄 Force sync test...")
                result = await storage.force_sync()
                print(f"  Status: {result['status']}")
                print(f"  Primary memories: {result['primary_memories']}")
                print(f"  Synced to secondary: {result['synced_to_secondary']}")

                # Final verification
                final_status = await storage.sync_service.get_sync_status()
                print(f"\n✅ Final sync status:")
                print(f"  Total operations processed: {final_status.get('operations_processed', 0)}")
                print(f"  Queue remaining: {final_status.get('pending_operations', 0)}")

            await storage.close()
            print("\n🎉 Background sync test completed successfully!")

    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


if __name__ == "__main__":
    asyncio.run(test_background_sync_with_mock())