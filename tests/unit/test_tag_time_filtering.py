"""
Comprehensive tests for tag+time filtering functionality across all storage backends.

Tests the time_start parameter added in PR #215 to fix semantic over-filtering bug (issue #214).
"""

import pytest
import pytest_asyncio
import tempfile
import os
import shutil
import time
from datetime import datetime, timedelta
from typing import List

from src.mcp_memory_service.models.memory import Memory
from src.mcp_memory_service.utils.hashing import generate_content_hash

# Skip tests if sqlite-vec is not available
try:
    import sqlite_vec
    SQLITE_VEC_AVAILABLE = True
except ImportError:
    SQLITE_VEC_AVAILABLE = False

if SQLITE_VEC_AVAILABLE:
    from src.mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage

# Import Cloudflare storage for testing (may be skipped if not configured)
try:
    from src.mcp_memory_service.storage.cloudflare import CloudflareMemoryStorage
    CLOUDFLARE_AVAILABLE = True
except ImportError:
    CLOUDFLARE_AVAILABLE = False

# Import Hybrid storage
try:
    from src.mcp_memory_service.storage.hybrid import HybridMemoryStorage
    HYBRID_AVAILABLE = SQLITE_VEC_AVAILABLE  # Hybrid requires SQLite-vec
except ImportError:
    HYBRID_AVAILABLE = False


class TestTagTimeFilteringSqliteVec:
    """Test tag+time filtering for SQLite-vec storage backend."""

    pytestmark = pytest.mark.skipif(not SQLITE_VEC_AVAILABLE, reason="sqlite-vec not available")

    @pytest_asyncio.fixture
    async def storage(self):
        """Create a test storage instance."""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_tag_time.db")

        storage = SqliteVecMemoryStorage(db_path)
        await storage.initialize()

        yield storage

        # Cleanup
        if storage.conn:
            storage.conn.close()
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def old_memory(self):
        """Create a memory with timestamp 2 days ago."""
        content = "Old memory from 2 days ago"
        # Set timestamp to 2 days ago
        two_days_ago = time.time() - (2 * 24 * 60 * 60)
        return Memory(
            content=content,
            content_hash=generate_content_hash(content),
            tags=["test", "old"],
            memory_type="note",
            created_at=two_days_ago
        )

    @pytest.fixture
    def recent_memory(self):
        """Create a memory with current timestamp."""
        content = "Recent memory from now"
        return Memory(
            content=content,
            content_hash=generate_content_hash(content),
            tags=["test", "recent"],
            memory_type="note",
            created_at=time.time()
        )

    @pytest.mark.asyncio
    async def test_search_by_tag_with_time_filter_returns_recent(self, storage, old_memory, recent_memory):
        """Test that time_start filters out old memories."""
        # Store both memories
        await storage.store(old_memory)
        await storage.store(recent_memory)

        # Search with time_start = 1 day ago (should only return recent_memory)
        one_day_ago = time.time() - (24 * 60 * 60)
        results = await storage.search_by_tag(["test"], time_start=one_day_ago)

        # Should only return the recent memory
        assert len(results) == 1
        assert results[0].content_hash == recent_memory.content_hash
        assert "recent" in results[0].tags

    @pytest.mark.asyncio
    async def test_search_by_tag_with_time_filter_excludes_old(self, storage, old_memory, recent_memory):
        """Test that old memories are excluded when time_start is recent."""
        # Store both memories
        await storage.store(old_memory)
        await storage.store(recent_memory)

        # Search with time_start = 10 seconds ago (should not return 2-day-old memory)
        ten_seconds_ago = time.time() - 10
        results = await storage.search_by_tag(["old"], time_start=ten_seconds_ago)

        # Should return empty (old_memory is from 2 days ago)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_by_tag_without_time_filter_backward_compat(self, storage, old_memory, recent_memory):
        """Test backward compatibility - no time_start returns all matching memories."""
        # Store both memories
        await storage.store(old_memory)
        await storage.store(recent_memory)

        # Search without time_start (backward compatibility)
        results = await storage.search_by_tag(["test"])

        # Should return both memories
        assert len(results) == 2
        hashes = {r.content_hash for r in results}
        assert old_memory.content_hash in hashes
        assert recent_memory.content_hash in hashes

    @pytest.mark.asyncio
    async def test_search_by_tag_with_none_time_start(self, storage, old_memory):
        """Test that time_start=None behaves same as no time_start."""
        await storage.store(old_memory)

        # Explicit None should be same as not passing parameter
        results = await storage.search_by_tag(["test"], time_start=None)

        assert len(results) == 1
        assert results[0].content_hash == old_memory.content_hash

    @pytest.mark.asyncio
    async def test_search_by_tag_with_future_time_start(self, storage, recent_memory):
        """Test that future time_start returns empty results."""
        await storage.store(recent_memory)

        # Set time_start to 1 hour in the future
        future_time = time.time() + (60 * 60)
        results = await storage.search_by_tag(["test"], time_start=future_time)

        # Should return empty (memory is older than future time)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_by_tag_with_zero_time_start(self, storage, recent_memory):
        """Test that time_start=0 returns all memories (epoch time)."""
        await storage.store(recent_memory)

        # time_start=0 (Unix epoch) should return all memories
        results = await storage.search_by_tag(["test"], time_start=0)

        assert len(results) == 1
        assert results[0].content_hash == recent_memory.content_hash

    @pytest.mark.asyncio
    async def test_search_by_tag_multiple_tags_with_time_filter(self, storage):
        """Test multiple tags with time filtering."""
        # Create memories with different tag combinations
        memory1 = Memory(
            content="Memory with tag1 and tag2",
            content_hash=generate_content_hash("Memory with tag1 and tag2"),
            tags=["tag1", "tag2"],
            created_at=time.time()
        )
        memory2 = Memory(
            content="Old memory with tag1",
            content_hash=generate_content_hash("Old memory with tag1"),
            tags=["tag1"],
            created_at=time.time() - (2 * 24 * 60 * 60)  # 2 days ago
        )

        await storage.store(memory1)
        await storage.store(memory2)

        # Search for tag1 with time_start = 1 day ago
        one_day_ago = time.time() - (24 * 60 * 60)
        results = await storage.search_by_tag(["tag1"], time_start=one_day_ago)

        # Should only return memory1 (recent)
        assert len(results) == 1
        assert results[0].content_hash == memory1.content_hash

    @pytest.mark.asyncio
    async def test_search_by_tag_with_percent_in_tag(self, storage):
        """Tags containing % must not be treated as LIKE wildcards."""
        memory = Memory(
            content="Memory tagged with special char",
            content_hash=generate_content_hash("Memory tagged with special char"),
            tags=["100%"],
            created_at=time.time(),
        )
        unrelated = Memory(
            content="Unrelated memory",
            content_hash=generate_content_hash("Unrelated memory"),
            tags=["other"],
            created_at=time.time(),
        )
        await storage.store(memory)
        await storage.store(unrelated)

        results = await storage.search_by_tag(["100%"])
        assert len(results) == 1
        assert results[0].content_hash == memory.content_hash

    @pytest.mark.asyncio
    async def test_search_by_tag_with_underscore_in_tag(self, storage):
        """Tags containing _ must not be treated as single-char LIKE wildcards."""
        memory = Memory(
            content="Memory with underscore tag",
            content_hash=generate_content_hash("Memory with underscore tag"),
            tags=["my_tag"],
            created_at=time.time(),
        )
        decoy = Memory(
            content="Memory that should not match",
            content_hash=generate_content_hash("Memory that should not match"),
            tags=["myXtag"],
            created_at=time.time(),
        )
        await storage.store(memory)
        await storage.store(decoy)

        results = await storage.search_by_tag(["my_tag"])
        assert len(results) == 1
        assert results[0].content_hash == memory.content_hash


@pytest.mark.skipif(not CLOUDFLARE_AVAILABLE, reason="Cloudflare storage not available")
class TestTagTimeFilteringCloudflare:
    """Test tag+time filtering for Cloudflare storage backend."""

    @pytest_asyncio.fixture
    async def storage(self):
        """Create a test Cloudflare storage instance."""
        # Note: Requires CLOUDFLARE_* environment variables to be set
        storage = CloudflareMemoryStorage()
        await storage.initialize()

        yield storage

        # Cleanup: delete test memories
        # (Cloudflare doesn't have direct cleanup, so we skip)

    @pytest.fixture
    def recent_memory(self):
        """Create a recent test memory."""
        content = f"Cloudflare test memory {time.time()}"
        return Memory(
            content=content,
            content_hash=generate_content_hash(content),
            tags=["cloudflare-test", "recent"],
            memory_type="note",
            created_at=time.time()
        )

    @pytest.mark.asyncio
    async def test_search_by_tag_with_time_filter(self, storage, recent_memory):
        """Test Cloudflare backend time filtering."""
        await storage.store(recent_memory)

        # Search with time_start = 1 hour ago
        one_hour_ago = time.time() - (60 * 60)
        results = await storage.search_by_tag(["cloudflare-test"], time_start=one_hour_ago)

        # Should return the recent memory
        assert len(results) >= 1
        # Verify at least one result matches our memory
        hashes = {r.content_hash for r in results}
        assert recent_memory.content_hash in hashes

    @pytest.mark.asyncio
    async def test_search_by_tag_without_time_filter(self, storage, recent_memory):
        """Test Cloudflare backward compatibility (no time filter)."""
        await storage.store(recent_memory)

        # Search without time_start
        results = await storage.search_by_tag(["cloudflare-test"])

        # Should return memories (at least our test memory)
        assert len(results) >= 1
        hashes = {r.content_hash for r in results}
        assert recent_memory.content_hash in hashes


@pytest.mark.skipif(not HYBRID_AVAILABLE, reason="Hybrid storage not available")
class TestTagTimeFilteringHybrid:
    """Test tag+time filtering for Hybrid storage backend."""

    @pytest_asyncio.fixture
    async def storage(self):
        """Create a test Hybrid storage instance."""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_hybrid_tag_time.db")

        # Create hybrid storage (local SQLite + Cloudflare sync)
        storage = HybridMemoryStorage(db_path)
        await storage.initialize()

        yield storage

        # Cleanup
        if hasattr(storage, 'local_storage') and storage.local_storage.conn:
            storage.local_storage.conn.close()
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def test_memory(self):
        """Create a test memory for hybrid backend."""
        content = f"Hybrid test memory {time.time()}"
        return Memory(
            content=content,
            content_hash=generate_content_hash(content),
            tags=["hybrid-test", "time-filter"],
            memory_type="note",
            created_at=time.time()
        )

    @pytest.mark.asyncio
    async def test_search_by_tag_with_time_filter(self, storage, test_memory):
        """Test Hybrid backend time filtering."""
        await storage.store(test_memory)

        # Search with time_start = 1 minute ago
        one_minute_ago = time.time() - 60
        results = await storage.search_by_tag(["hybrid-test"], time_start=one_minute_ago)

        # Should return the test memory from local storage
        assert len(results) == 1
        assert results[0].content_hash == test_memory.content_hash

    @pytest.mark.asyncio
    async def test_search_by_tag_without_time_filter(self, storage, test_memory):
        """Test Hybrid backward compatibility (no time filter)."""
        await storage.store(test_memory)

        # Search without time_start
        results = await storage.search_by_tag(["hybrid-test"])

        # Should return the test memory
        assert len(results) == 1
        assert results[0].content_hash == test_memory.content_hash

    @pytest.mark.asyncio
    async def test_search_by_tag_hybrid_uses_local_storage(self, storage, test_memory):
        """Verify that Hybrid backend searches local storage for tag+time queries."""
        await storage.store(test_memory)

        # Hybrid should use local storage for fast tag+time queries
        one_hour_ago = time.time() - (60 * 60)
        results = await storage.search_by_tag(["time-filter"], time_start=one_hour_ago)

        # Should return results from local SQLite storage
        assert len(results) == 1
        assert results[0].content_hash == test_memory.content_hash
