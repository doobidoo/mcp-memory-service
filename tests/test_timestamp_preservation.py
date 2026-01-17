"""
Comprehensive tests for timestamp preservation during sync operations.

This test suite verifies that the fix for the timestamp regression bug
(where created_at was being reset during metadata sync) works correctly.
"""

import pytest
import pytest_asyncio
import time
import tempfile
import os
from datetime import datetime
from pathlib import Path

# Direct imports from submodules to avoid namespace collision with mcp.server.models
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.models.memory import Memory


@pytest_asyncio.fixture
async def storage():
    """Create a temporary SQLite storage for testing."""
    # Create temporary database file
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test.db")

    storage = SqliteVecMemoryStorage(
        db_path=db_path,
        embedding_model="all-MiniLM-L6-v2"
    )
    await storage.initialize()

    yield storage

    # Cleanup
    storage.close()  # Not async
    try:
        os.remove(db_path)
        os.rmdir(temp_dir)
    except:
        pass


@pytest.fixture
def old_memory():
    """Create a memory with an old timestamp (24 hours ago)."""
    old_time = time.time() - 86400  # 24 hours ago
    old_iso = datetime.utcfromtimestamp(old_time).isoformat() + "Z"

    return Memory(
        content="This is a test memory from yesterday",
        content_hash="test_hash_old_memory",
        tags=["test", "old"],
        memory_type="note",
        metadata={"original": True},
        created_at=old_time,
        created_at_iso=old_iso,
        updated_at=old_time,
        updated_at_iso=old_iso
    )


@pytest.mark.asyncio
class TestTimestampPreservation:
    """Test suite for timestamp preservation during metadata updates."""

    async def test_preserve_timestamps_true_only_updates_updated_at(self, storage, old_memory):
        """
        Test that preserve_timestamps=True only updates updated_at,
        leaving created_at unchanged.
        """
        # Store memory with old timestamp
        await storage.store(old_memory)

        # Update metadata with preserve_timestamps=True (default)
        updates = {
            "tags": ["test", "updated"],
            "metadata": {"updated": True}
        }

        success, _ = await storage.update_memory_metadata(
            old_memory.content_hash,
            updates,
            preserve_timestamps=True
        )

        assert success

        # Retrieve and verify timestamps
        cursor = storage.conn.execute('''
            SELECT created_at, created_at_iso, updated_at, updated_at_iso
            FROM memories WHERE content_hash = ?
        ''', (old_memory.content_hash,))

        row = cursor.fetchone()
        created_at, created_at_iso, updated_at, updated_at_iso = row

        # created_at should be preserved (within 1 second tolerance)
        assert abs(created_at - old_memory.created_at) < 1.0, \
            f"created_at changed! Expected {old_memory.created_at}, got {created_at}"
        assert created_at_iso == old_memory.created_at_iso

        # updated_at should be recent (within last 5 seconds)
        now = time.time()
        assert abs(updated_at - now) < 5.0, \
            f"updated_at not updated! Expected ~{now}, got {updated_at}"

    async def test_preserve_timestamps_false_without_source_preserves_created_at(self, storage, old_memory):
        """
        Test that preserve_timestamps=False without providing timestamps
        still preserves created_at (regression test for the bug).
        """
        # Store memory with old timestamp
        await storage.store(old_memory)

        # Update metadata with preserve_timestamps=False but NO timestamps in updates
        # This simulates the BUGGY behavior that was resetting created_at
        updates = {
            "tags": ["test", "synced"],
            "metadata": {"synced": True}
        }

        success, _ = await storage.update_memory_metadata(
            old_memory.content_hash,
            updates,
            preserve_timestamps=False
        )

        assert success

        # Retrieve and verify timestamps
        cursor = storage.conn.execute('''
            SELECT created_at, created_at_iso, updated_at, updated_at_iso
            FROM memories WHERE content_hash = ?
        ''', (old_memory.content_hash,))

        row = cursor.fetchone()
        created_at, created_at_iso, updated_at, updated_at_iso = row

        # created_at should STILL be preserved (this is the fix!)
        assert abs(created_at - old_memory.created_at) < 1.0, \
            f"BUG: created_at was reset! Expected {old_memory.created_at}, got {created_at}"
        assert created_at_iso == old_memory.created_at_iso, \
            f"BUG: created_at_iso was reset! Expected {old_memory.created_at_iso}, got {created_at_iso}"

    async def test_preserve_timestamps_false_with_source_uses_source_timestamps(self, storage, old_memory):
        """
        Test that preserve_timestamps=False WITH source timestamps
        uses the provided timestamps (for drift detection sync).
        """
        # Store memory with old timestamp
        await storage.store(old_memory)

        # Simulate a sync from Cloudflare with newer metadata but original created_at
        cloudflare_updated_at = time.time() - 3600  # 1 hour ago (newer than local)
        cloudflare_updated_iso = datetime.utcfromtimestamp(cloudflare_updated_at).isoformat() + "Z"

        updates = {
            "tags": ["test", "synced-from-cloudflare"],
            "metadata": {"source": "cloudflare"},
            "created_at": old_memory.created_at,  # Original creation time
            "created_at_iso": old_memory.created_at_iso,
            "updated_at": cloudflare_updated_at,  # Newer update time
            "updated_at_iso": cloudflare_updated_iso
        }

        success, _ = await storage.update_memory_metadata(
            old_memory.content_hash,
            updates,
            preserve_timestamps=False
        )

        assert success

        # Retrieve and verify timestamps
        cursor = storage.conn.execute('''
            SELECT created_at, created_at_iso, updated_at, updated_at_iso
            FROM memories WHERE content_hash = ?
        ''', (old_memory.content_hash,))

        row = cursor.fetchone()
        created_at, created_at_iso, updated_at, updated_at_iso = row

        # created_at should match the source (Cloudflare)
        assert abs(created_at - old_memory.created_at) < 1.0, \
            f"created_at not preserved from source! Expected {old_memory.created_at}, got {created_at}"
        assert created_at_iso == old_memory.created_at_iso

        # updated_at should match the source (Cloudflare)
        assert abs(updated_at - cloudflare_updated_at) < 1.0, \
            f"updated_at not from source! Expected {cloudflare_updated_at}, got {updated_at}"
        assert updated_at_iso == cloudflare_updated_iso

    async def test_drift_detection_scenario(self, storage, old_memory):
        """
        Test the complete drift detection scenario:
        1. Memory exists locally with old metadata
        2. Cloudflare has newer metadata
        3. Drift detection syncs metadata while preserving created_at
        """
        # Store memory with old timestamp
        await storage.store(old_memory)

        # Simulate Cloudflare memory with newer metadata
        cf_updated_at = time.time() - 1800  # 30 minutes ago
        cf_updated_iso = datetime.utcfromtimestamp(cf_updated_at).isoformat() + "Z"

        # This is what hybrid storage does during drift detection
        cf_updates = {
            'tags': ["test", "cloudflare-updated"],
            'memory_type': "reference",
            'metadata': {"updated_via": "cloudflare"},
            'created_at': old_memory.created_at,  # Preserve original
            'created_at_iso': old_memory.created_at_iso,
            'updated_at': cf_updated_at,  # Use Cloudflare's update time
            'updated_at_iso': cf_updated_iso,
        }

        success, _ = await storage.update_memory_metadata(
            old_memory.content_hash,
            cf_updates,
            preserve_timestamps=False  # Use Cloudflare timestamps
        )

        assert success

        # Verify the sync preserved created_at but updated metadata
        cursor = storage.conn.execute('''
            SELECT created_at, updated_at, tags, memory_type, metadata
            FROM memories WHERE content_hash = ?
        ''', (old_memory.content_hash,))

        row = cursor.fetchone()
        created_at, updated_at, tags, memory_type, metadata_str = row

        # Timestamps
        assert abs(created_at - old_memory.created_at) < 1.0, \
            "Drift detection reset created_at!"
        assert abs(updated_at - cf_updated_at) < 1.0, \
            "Drift detection didn't use Cloudflare updated_at!"

        # Metadata
        assert tags == "test,cloudflare-updated"
        assert memory_type == "reference"

    async def test_multiple_syncs_preserve_original_created_at(self, storage, old_memory):
        """
        Test that multiple sync operations (as would happen over time)
        never reset the original created_at timestamp.
        """
        # Store memory with old timestamp
        await storage.store(old_memory)
        original_created_at = old_memory.created_at

        # Simulate 3 sync operations over time
        for i in range(3):
            sync_time = time.time() - (3600 * (3 - i))  # 3h, 2h, 1h ago
            sync_iso = datetime.utcfromtimestamp(sync_time).isoformat() + "Z"

            updates = {
                'tags': ["test", f"sync-{i+1}"],
                'metadata': {"sync_count": i + 1},
                'created_at': original_created_at,  # Always the original
                'created_at_iso': old_memory.created_at_iso,
                'updated_at': sync_time,
                'updated_at_iso': sync_iso,
            }

            success, _ = await storage.update_memory_metadata(
                old_memory.content_hash,
                updates,
                preserve_timestamps=False
            )

            assert success

        # Verify created_at never changed
        cursor = storage.conn.execute('''
            SELECT created_at, created_at_iso
            FROM memories WHERE content_hash = ?
        ''', (old_memory.content_hash,))

        row = cursor.fetchone()
        created_at, created_at_iso = row

        assert abs(created_at - original_created_at) < 1.0, \
            f"After {3} syncs, created_at changed! Expected {original_created_at}, got {created_at}"
        assert created_at_iso == old_memory.created_at_iso

    async def test_new_memory_store_sets_timestamps_correctly(self, storage):
        """
        Test that storing a new memory without explicit timestamps
        sets them correctly (current time).
        """
        now_before = time.time()

        memory = Memory(
            content="New memory without explicit timestamps",
            content_hash="test_hash_new"
        )

        await storage.store(memory)

        now_after = time.time()

        # Retrieve and verify timestamps
        cursor = storage.conn.execute('''
            SELECT created_at, updated_at
            FROM memories WHERE content_hash = ?
        ''', (memory.content_hash,))

        row = cursor.fetchone()
        created_at, updated_at = row

        # Both should be recent (between before and after)
        assert now_before <= created_at <= now_after, \
            f"created_at not set to current time: {created_at}"
        assert now_before <= updated_at <= now_after, \
            f"updated_at not set to current time: {updated_at}"

    async def test_store_memory_with_explicit_timestamps_preserves_them(self, storage, old_memory):
        """
        Test that storing a memory WITH explicit timestamps
        (e.g., synced from Cloudflare) preserves those timestamps.
        """
        await storage.store(old_memory)

        # Retrieve and verify timestamps
        cursor = storage.conn.execute('''
            SELECT created_at, created_at_iso, updated_at, updated_at_iso
            FROM memories WHERE content_hash = ?
        ''', (old_memory.content_hash,))

        row = cursor.fetchone()
        created_at, created_at_iso, updated_at, updated_at_iso = row

        # All timestamps should match what was provided
        assert abs(created_at - old_memory.created_at) < 1.0
        assert created_at_iso == old_memory.created_at_iso
        assert abs(updated_at - old_memory.updated_at) < 1.0
        assert updated_at_iso == old_memory.updated_at_iso


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
