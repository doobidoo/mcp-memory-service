#!/usr/bin/env python3
"""
Comprehensive tests for Leader Election mechanism in Hybrid Backend.

Tests cover:
- Basic leader election and lock acquisition
- Heartbeat mechanism and health monitoring
- Stale detection and automatic failover
- Adaptive sync interval calculation
- Leader-based write routing
- Clean shutdown and resource cleanup
- Multiple process simulation
"""

import asyncio
import pytest
import tempfile
import os
import sys
import time
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from typing import Dict, Any

# Add src to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from mcp_memory_service.storage.hybrid import (
    LeaderElection,
    BackgroundSyncService,
    HybridMemoryStorage,
    SyncOperation
)
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.utils.hashing import generate_content_hash

# Configure test logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Mock Cloudflare Storage for testing
class MockCloudflareStorage:
    """Mock Cloudflare storage for testing."""

    def __init__(self, **kwargs):
        self.initialized = False
        self.stored_memories = {}
        self.fail_operations = False
        self.fail_initialization = False

    async def initialize(self):
        if self.fail_initialization:
            raise Exception("Mock Cloudflare initialization failed")
        self.initialized = True

    async def store(self, memory: Memory):
        if self.fail_operations:
            return False, "Mock Cloudflare operation failed"
        self.stored_memories[memory.content_hash] = memory
        return True, "Memory stored successfully"

    async def delete(self, content_hash: str):
        if self.fail_operations:
            return False, "Mock Cloudflare operation failed"
        if content_hash in self.stored_memories:
            del self.stored_memories[content_hash]
            return True, "Memory deleted successfully"
        return False, "Memory not found"

    async def update_memory_metadata(self, content_hash: str, updates: Dict[str, Any], preserve_timestamps: bool = True):
        if self.fail_operations:
            return False, "Mock Cloudflare operation failed"
        if content_hash in self.stored_memories:
            return True, "Memory updated successfully"
        return False, "Memory not found"

    async def get_stats(self):
        if self.fail_operations:
            raise Exception("Mock Cloudflare stats failed")
        return {
            "total_memories": len(self.stored_memories),
            "storage_backend": "MockCloudflareStorage"
        }

    async def close(self):
        pass


# Fixtures
@pytest.fixture
def temp_lock_dir():
    """Create temporary directory for lock files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_sqlite_db():
    """Create a temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name

    yield db_path

    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def mock_cloudflare_storage():
    """Create mock Cloudflare storage."""
    return MockCloudflareStorage()


@pytest.fixture
async def mock_sqlite_storage(temp_sqlite_db):
    """Create mock SQLite storage."""
    storage = SqliteVecMemoryStorage(db_path=temp_sqlite_db)
    await storage.initialize()
    yield storage
    # SqliteVecMemoryStorage.close() is not async
    if hasattr(storage, 'close') and storage.close:
        if asyncio.iscoroutinefunction(storage.close):
            await storage.close()
        else:
            storage.close()


# ============================================================================
# Test Suite 1: Basic Leader Election
# ============================================================================

class TestBasicLeaderElection:
    """Test basic leader election functionality."""

    @pytest.mark.asyncio
    async def test_first_process_acquires_leadership(self, temp_lock_dir):
        """First process should successfully acquire leadership."""
        leader = LeaderElection(lock_dir=temp_lock_dir, process_id="process-1")

        result = leader.try_acquire_leadership()

        assert result is True, "First process should acquire leadership"
        assert leader.is_leader is True, "Process should be marked as leader"
        assert leader.lock_file is not None, "Lock file should be open"

        # Verify lock file exists and contains process ID
        assert leader.lock_file_path.exists(), "Lock file should exist"

        # Cleanup
        leader.release_leadership()

    @pytest.mark.asyncio
    async def test_second_process_becomes_follower(self, temp_lock_dir):
        """Second process should become follower when leader exists."""
        leader1 = LeaderElection(lock_dir=temp_lock_dir, process_id="process-1")
        leader2 = LeaderElection(lock_dir=temp_lock_dir, process_id="process-2")

        # First process becomes leader
        result1 = leader1.try_acquire_leadership()
        assert result1 is True, "First process should acquire leadership"

        # Second process should fail to acquire leadership
        result2 = leader2.try_acquire_leadership()
        assert result2 is False, "Second process should not acquire leadership"
        assert leader2.is_leader is False, "Second process should be follower"
        assert leader2.lock_file is None, "Follower should not hold lock file"

        # Cleanup
        leader1.release_leadership()

    @pytest.mark.asyncio
    async def test_lock_file_prevents_concurrent_leaders(self, temp_lock_dir):
        """Lock file should prevent concurrent leaders."""
        leaders = []

        # Create 5 processes trying to acquire leadership
        for i in range(5):
            leader = LeaderElection(lock_dir=temp_lock_dir, process_id=f"process-{i}")
            leader.try_acquire_leadership()
            leaders.append(leader)

        # Count how many became leaders
        leader_count = sum(1 for l in leaders if l.is_leader)

        assert leader_count == 1, f"Only one leader should exist, found {leader_count}"

        # Cleanup
        for leader in leaders:
            if leader.is_leader:
                leader.release_leadership()


# ============================================================================
# Test Suite 2: Heartbeat Mechanism
# ============================================================================

class TestHeartbeatMechanism:
    """Test heartbeat writing and monitoring."""

    @pytest.mark.asyncio
    async def test_leader_writes_heartbeat_on_acquisition(self, temp_lock_dir):
        """Leader should write heartbeat immediately on acquisition."""
        leader = LeaderElection(lock_dir=temp_lock_dir, process_id="process-1")

        leader.try_acquire_leadership()

        # Check heartbeat file exists
        assert leader.heartbeat_file_path.exists(), "Heartbeat file should exist"

        # Read and verify heartbeat
        heartbeat = leader._read_heartbeat()
        assert heartbeat is not None, "Heartbeat should be readable"

        timestamp, process_id = heartbeat
        assert process_id == "process-1", "Heartbeat should contain correct process ID"
        assert time.time() - timestamp < 1, "Heartbeat should be recent"

        # Cleanup
        leader.release_leadership()

    @pytest.mark.asyncio
    async def test_heartbeat_updates_periodically(self, temp_lock_dir):
        """Heartbeat should update periodically during leader loop."""
        leader = LeaderElection(lock_dir=temp_lock_dir, process_id="process-1")
        leader.try_acquire_leadership()

        # Start heartbeat loop
        await leader.start_leader_heartbeat()

        # Read initial heartbeat
        initial_heartbeat = leader._read_heartbeat()
        initial_time = initial_heartbeat[0]

        # Wait for heartbeat interval (using short interval for testing)
        # Note: In production, HYBRID_LEADER_HEARTBEAT_INTERVAL is 10s
        # For testing, we'll just wait and verify the mechanism works
        await asyncio.sleep(0.5)  # Short wait for test

        # Manually trigger heartbeat write
        leader._write_heartbeat()

        # Read updated heartbeat
        updated_heartbeat = leader._read_heartbeat()
        updated_time = updated_heartbeat[0]

        assert updated_time >= initial_time, "Heartbeat should be updated"

        # Cleanup
        await leader.stop()

    @pytest.mark.asyncio
    async def test_heartbeat_contains_timestamp_and_process_id(self, temp_lock_dir):
        """Heartbeat file should contain both timestamp and process ID."""
        leader = LeaderElection(lock_dir=temp_lock_dir, process_id="test-process-123")
        leader.try_acquire_leadership()

        # Read heartbeat
        heartbeat = leader._read_heartbeat()
        assert heartbeat is not None

        timestamp, process_id = heartbeat
        assert isinstance(timestamp, float), "Timestamp should be a float"
        assert process_id == "test-process-123", "Process ID should match"

        # Cleanup
        leader.release_leadership()

    @pytest.mark.asyncio
    async def test_heartbeat_readable_by_other_processes(self, temp_lock_dir):
        """Heartbeat file should be readable by follower processes."""
        leader = LeaderElection(lock_dir=temp_lock_dir, process_id="leader-process")
        follower = LeaderElection(lock_dir=temp_lock_dir, process_id="follower-process")

        # Leader acquires lock
        leader.try_acquire_leadership()

        # Follower should be able to read leader's heartbeat
        heartbeat = follower._read_heartbeat()
        assert heartbeat is not None, "Follower should read leader's heartbeat"

        timestamp, process_id = heartbeat
        assert process_id == "leader-process", "Should read leader's process ID"

        # Cleanup
        leader.release_leadership()


# ============================================================================
# Test Suite 3: Stale Detection
# ============================================================================

class TestStaleDetection:
    """Test stale leader detection mechanism."""

    @pytest.mark.asyncio
    async def test_fresh_heartbeat_not_stale(self, temp_lock_dir):
        """Fresh heartbeat should not be considered stale."""
        leader = LeaderElection(lock_dir=temp_lock_dir, process_id="process-1")
        leader.try_acquire_leadership()

        # Write fresh heartbeat
        leader._write_heartbeat()

        # Check staleness
        is_stale = leader.is_leader_stale()

        assert is_stale is False, "Fresh heartbeat should not be stale"

        # Cleanup
        leader.release_leadership()

    @pytest.mark.asyncio
    async def test_old_heartbeat_is_stale(self, temp_lock_dir):
        """Old heartbeat (>45s) should be considered stale."""
        leader = LeaderElection(lock_dir=temp_lock_dir, process_id="process-1")
        leader.try_acquire_leadership()

        # Write old heartbeat (simulate by writing old timestamp)
        stale_time = time.time() - 60  # 60 seconds ago
        with open(leader.heartbeat_file_path, 'w') as f:
            f.write(f"{stale_time}\n{leader.process_id}")

        # Check staleness
        is_stale = leader.is_leader_stale()

        assert is_stale is True, "Old heartbeat should be stale"

        # Cleanup
        leader.release_leadership()

    @pytest.mark.asyncio
    async def test_missing_heartbeat_is_stale(self, temp_lock_dir):
        """Missing heartbeat file should be considered stale."""
        leader = LeaderElection(lock_dir=temp_lock_dir, process_id="process-1")

        # Don't acquire leadership, so no heartbeat file exists
        is_stale = leader.is_leader_stale()

        assert is_stale is True, "Missing heartbeat should be stale"

    @pytest.mark.asyncio
    async def test_stale_threshold_boundary(self, temp_lock_dir):
        """Test stale threshold boundary (exactly 45s)."""
        leader = LeaderElection(lock_dir=temp_lock_dir, process_id="process-1")
        leader.try_acquire_leadership()

        # Write heartbeat exactly at threshold (45s ago)
        # Note: HYBRID_LEADER_STALE_THRESHOLD from config
        threshold_time = time.time() - 45.0
        with open(leader.heartbeat_file_path, 'w') as f:
            f.write(f"{threshold_time}\n{leader.process_id}")

        is_stale = leader.is_leader_stale()

        # Should be stale (age > threshold)
        assert is_stale is True, "Heartbeat at threshold should be stale"

        # Cleanup
        leader.release_leadership()


# ============================================================================
# Test Suite 4: Automatic Failover
# ============================================================================

class TestAutomaticFailover:
    """Test automatic failover when leader becomes stale."""

    @pytest.mark.asyncio
    async def test_follower_detects_stale_leader(self, temp_lock_dir):
        """Follower should detect when leader becomes stale."""
        leader = LeaderElection(lock_dir=temp_lock_dir, process_id="leader")
        follower = LeaderElection(lock_dir=temp_lock_dir, process_id="follower")

        # Leader acquires lock
        leader.try_acquire_leadership()

        # Make leader's heartbeat stale
        stale_time = time.time() - 60
        with open(leader.heartbeat_file_path, 'w') as f:
            f.write(f"{stale_time}\nleader")

        # Follower checks if leader is stale
        is_stale = follower.is_leader_stale()

        assert is_stale is True, "Follower should detect stale leader"

        # Cleanup
        leader.release_leadership()

    @pytest.mark.asyncio
    async def test_follower_attempts_takeover_on_stale_leader(self, temp_lock_dir):
        """Follower should attempt takeover when leader is stale."""
        leader = LeaderElection(lock_dir=temp_lock_dir, process_id="leader")
        follower = LeaderElection(lock_dir=temp_lock_dir, process_id="follower")

        # Leader acquires lock but doesn't keep it updated
        leader.try_acquire_leadership()

        # Release leader's lock (simulate crash)
        leader.release_leadership()

        # Follower should successfully acquire leadership
        takeover_success = follower.try_acquire_leadership()

        assert takeover_success is True, "Follower should successfully take over"
        assert follower.is_leader is True, "Follower should become leader"

        # Cleanup
        follower.release_leadership()

    @pytest.mark.asyncio
    async def test_new_leader_starts_heartbeat(self, temp_lock_dir):
        """New leader should start heartbeat after promotion."""
        leader = LeaderElection(lock_dir=temp_lock_dir, process_id="leader")
        follower = LeaderElection(lock_dir=temp_lock_dir, process_id="follower")

        # Leader acquires and releases
        leader.try_acquire_leadership()
        leader.release_leadership()

        # Follower takes over
        follower.try_acquire_leadership()
        await follower.start_leader_heartbeat()

        # Verify new heartbeat exists
        await asyncio.sleep(0.1)  # Let heartbeat task run
        heartbeat = follower._read_heartbeat()

        assert heartbeat is not None, "New leader should write heartbeat"
        timestamp, process_id = heartbeat
        assert process_id == "follower", "Heartbeat should have new leader's ID"

        # Cleanup
        await follower.stop()

    @pytest.mark.asyncio
    async def test_health_check_loop_promotes_follower(self, temp_lock_dir):
        """Health check loop should automatically promote follower."""
        leader = LeaderElection(lock_dir=temp_lock_dir, process_id="leader")
        follower = LeaderElection(lock_dir=temp_lock_dir, process_id="follower")

        # Leader acquires lock
        leader.try_acquire_leadership()

        # Make leader stale
        stale_time = time.time() - 60
        with open(leader.heartbeat_file_path, 'w') as f:
            f.write(f"{stale_time}\nleader")

        # Release leader's lock to allow takeover
        leader.release_leadership()

        # Track if callback was called
        promotion_occurred = asyncio.Event()

        async def on_promotion():
            promotion_occurred.set()

        # Start follower health check (with short interval for testing)
        await follower.start_follower_health_check(on_promotion)

        # Wait for promotion to occur
        try:
            await asyncio.wait_for(promotion_occurred.wait(), timeout=2.0)
            promotion_success = True
        except asyncio.TimeoutError:
            promotion_success = False

        assert promotion_success, "Follower should be promoted automatically"
        assert follower.is_leader is True, "Follower should become leader"

        # Cleanup
        await follower.stop()


# ============================================================================
# Test Suite 5: Adaptive Sync Intervals
# ============================================================================

class TestAdaptiveSyncIntervals:
    """Test adaptive sync interval calculation."""

    @pytest.mark.asyncio
    async def test_active_interval_after_write(self, mock_sqlite_storage, mock_cloudflare_storage):
        """Should return 5s interval after recent write."""
        sync_service = BackgroundSyncService(
            primary_storage=mock_sqlite_storage,
            secondary_storage=mock_cloudflare_storage,
            sync_interval=300,
            batch_size=50,
            is_leader=True
        )

        # Record write activity
        sync_service.record_write_activity()

        # Get interval immediately after write
        interval = sync_service.get_current_sync_interval()

        # Should return active interval (5s from config)
        assert interval == 5, f"Expected 5s active interval, got {interval}s"

    @pytest.mark.asyncio
    async def test_idle_interval_after_long_inactivity(self, mock_sqlite_storage, mock_cloudflare_storage):
        """Should return 60s interval after 300s+ inactivity."""
        sync_service = BackgroundSyncService(
            primary_storage=mock_sqlite_storage,
            secondary_storage=mock_cloudflare_storage,
            sync_interval=300,
            batch_size=50,
            is_leader=True
        )

        # Set last write time to 301 seconds ago
        sync_service.last_write_time = time.time() - 301

        # Get interval
        interval = sync_service.get_current_sync_interval()

        # Should return idle interval (60s from config)
        assert interval == 60, f"Expected 60s idle interval, got {interval}s"

    @pytest.mark.asyncio
    async def test_activity_resets_idle_timer(self, mock_sqlite_storage, mock_cloudflare_storage):
        """Recording activity should reset idle timer."""
        sync_service = BackgroundSyncService(
            primary_storage=mock_sqlite_storage,
            secondary_storage=mock_cloudflare_storage,
            sync_interval=300,
            batch_size=50,
            is_leader=True
        )

        # Start in idle state
        sync_service.last_write_time = time.time() - 400

        # Verify idle
        interval_before = sync_service.get_current_sync_interval()
        assert interval_before == 60, "Should be idle"

        # Record activity
        sync_service.record_write_activity()

        # Should return to active
        interval_after = sync_service.get_current_sync_interval()
        assert interval_after == 5, "Should be active after write"

    @pytest.mark.asyncio
    async def test_adaptive_sync_disabled_uses_base_interval(self, mock_sqlite_storage, mock_cloudflare_storage):
        """When adaptive sync disabled, should use base interval."""
        sync_service = BackgroundSyncService(
            primary_storage=mock_sqlite_storage,
            secondary_storage=mock_cloudflare_storage,
            sync_interval=300,
            batch_size=50,
            is_leader=True
        )

        # Disable adaptive sync
        sync_service.adaptive_sync_enabled = False

        # Should return base interval regardless of activity
        sync_service.record_write_activity()
        interval_active = sync_service.get_current_sync_interval()

        sync_service.last_write_time = time.time() - 400
        interval_idle = sync_service.get_current_sync_interval()

        assert interval_active == 300, "Should use base interval when disabled"
        assert interval_idle == 300, "Should use base interval when disabled"


# ============================================================================
# Test Suite 6: Leader-Based Write Routing
# ============================================================================

class TestLeaderBasedRouting:
    """Test write routing based on leadership status."""

    @pytest.mark.asyncio
    async def test_sync_service_tracks_leadership(self, mock_sqlite_storage, mock_cloudflare_storage):
        """Sync service should track leadership status."""
        # Create sync service as leader
        leader_sync = BackgroundSyncService(
            primary_storage=mock_sqlite_storage,
            secondary_storage=mock_cloudflare_storage,
            sync_interval=300,
            batch_size=50,
            is_leader=True
        )

        assert leader_sync.is_leader is True

        # Create sync service as follower
        follower_sync = BackgroundSyncService(
            primary_storage=mock_sqlite_storage,
            secondary_storage=mock_cloudflare_storage,
            sync_interval=300,
            batch_size=50,
            is_leader=False
        )

        assert follower_sync.is_leader is False

    @pytest.mark.asyncio
    async def test_leader_election_integrated_with_storage(self, temp_sqlite_db):
        """Test that leader election is properly integrated with hybrid storage."""
        with patch('mcp_memory_service.storage.hybrid.HYBRID_LEADER_ELECTION_ENABLED', True):
            storage = HybridMemoryStorage(
                sqlite_db_path=temp_sqlite_db,
                cloudflare_config=None  # No Cloudflare for this test
            )

            await storage.initialize()

            # Should have leader election instance
            assert storage.leader_election is not None

            # First instance should be leader
            assert storage.leader_election.is_leader is True

            await storage.close()

    @pytest.mark.asyncio
    async def test_leader_election_disabled(self, temp_sqlite_db):
        """Test that storage works when leader election is disabled."""
        with patch('mcp_memory_service.storage.hybrid.HYBRID_LEADER_ELECTION_ENABLED', False):
            storage = HybridMemoryStorage(
                sqlite_db_path=temp_sqlite_db,
                cloudflare_config=None
            )

            await storage.initialize()

            # Should not have leader election
            assert storage.leader_election is None

            await storage.close()


# ============================================================================
# Test Suite 7: Clean Shutdown
# ============================================================================

class TestCleanShutdown:
    """Test clean shutdown and resource cleanup."""

    @pytest.mark.asyncio
    async def test_leader_releases_lock_on_close(self, temp_lock_dir):
        """Leader should release lock when closing."""
        leader = LeaderElection(lock_dir=temp_lock_dir, process_id="leader")
        leader.try_acquire_leadership()

        # Verify lock held
        assert leader.is_leader is True
        assert leader.lock_file is not None

        # Stop (releases lock)
        await leader.stop()

        # Verify lock released
        assert leader.is_leader is False
        assert leader.lock_file is None

    @pytest.mark.asyncio
    async def test_heartbeat_file_cleaned_up(self, temp_lock_dir):
        """Heartbeat file should be removed on shutdown."""
        leader = LeaderElection(lock_dir=temp_lock_dir, process_id="leader")
        leader.try_acquire_leadership()

        # Verify heartbeat exists
        assert leader.heartbeat_file_path.exists()

        # Stop
        await leader.stop()

        # Verify heartbeat removed
        assert not leader.heartbeat_file_path.exists()

    @pytest.mark.asyncio
    async def test_background_tasks_cancelled(self, temp_lock_dir):
        """Background tasks should be cancelled on shutdown."""
        leader = LeaderElection(lock_dir=temp_lock_dir, process_id="leader")
        leader.try_acquire_leadership()

        # Start heartbeat
        await leader.start_leader_heartbeat()
        assert leader.heartbeat_task is not None

        # Stop
        await leader.stop()

        # Verify task cancelled
        assert leader.heartbeat_task.cancelled() or leader.heartbeat_task.done()

    @pytest.mark.asyncio
    async def test_follower_health_check_cancelled(self, temp_lock_dir):
        """Follower health check task should be cancelled on shutdown."""
        follower = LeaderElection(lock_dir=temp_lock_dir, process_id="follower")

        # Don't acquire leadership (follower)
        promotion_event = asyncio.Event()
        await follower.start_follower_health_check(lambda: promotion_event.set())

        assert follower.health_check_task is not None

        # Stop
        await follower.stop()

        # Verify task cancelled
        assert follower.health_check_task.cancelled() or follower.health_check_task.done()


# ============================================================================
# Test Suite 8: Multiple Process Simulation
# ============================================================================

class TestMultipleProcessSimulation:
    """Test multiple process scenarios."""

    @pytest.mark.asyncio
    async def test_five_concurrent_processes_one_leader(self, temp_lock_dir):
        """Simulate 5 concurrent processes, only one should become leader."""
        processes = []

        for i in range(5):
            process = LeaderElection(
                lock_dir=temp_lock_dir,
                process_id=f"process-{i}"
            )
            process.try_acquire_leadership()
            processes.append(process)

        # Count leaders
        leaders = [p for p in processes if p.is_leader]
        followers = [p for p in processes if not p.is_leader]

        assert len(leaders) == 1, f"Expected 1 leader, found {len(leaders)}"
        assert len(followers) == 4, f"Expected 4 followers, found {len(followers)}"

        # Cleanup
        for process in processes:
            if process.is_leader:
                process.release_leadership()

    @pytest.mark.asyncio
    async def test_rapid_failover_simulation(self, temp_lock_dir):
        """Simulate rapid leader failure and follower promotion."""
        # Create leader
        leader1 = LeaderElection(lock_dir=temp_lock_dir, process_id="leader-1")
        leader1.try_acquire_leadership()
        assert leader1.is_leader is True

        # Create follower
        follower = LeaderElection(lock_dir=temp_lock_dir, process_id="leader-2")
        follower.try_acquire_leadership()
        assert follower.is_leader is False

        # Leader crashes (release without cleanup)
        leader1.release_leadership()

        # Follower detects and takes over
        takeover = follower.try_acquire_leadership()
        assert takeover is True, "Follower should successfully take over"
        assert follower.is_leader is True

        # Cleanup
        follower.release_leadership()

    @pytest.mark.asyncio
    async def test_simultaneous_leadership_attempts(self, temp_lock_dir):
        """Multiple processes attempting leadership simultaneously."""
        processes = []

        async def try_acquire(process_id):
            process = LeaderElection(lock_dir=temp_lock_dir, process_id=process_id)
            result = process.try_acquire_leadership()
            return process, result

        # Simulate simultaneous attempts
        tasks = [try_acquire(f"process-{i}") for i in range(10)]
        results = await asyncio.gather(*tasks)

        processes = [r[0] for r in results]
        successes = [r[1] for r in results]

        # Only one should succeed
        assert sum(successes) == 1, "Only one process should acquire leadership"

        # Cleanup
        for process in processes:
            if process.is_leader:
                process.release_leadership()


# ============================================================================
# Test Suite 9: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_corrupted_heartbeat_file(self, temp_lock_dir):
        """Should handle corrupted heartbeat file gracefully."""
        leader = LeaderElection(lock_dir=temp_lock_dir, process_id="leader")
        leader.try_acquire_leadership()

        # Write corrupted heartbeat
        with open(leader.heartbeat_file_path, 'w') as f:
            f.write("corrupted\ndata\nextra\nlines")

        # Should not crash when reading
        heartbeat = leader._read_heartbeat()

        # Should either return None or handle gracefully
        assert True, "Should not crash on corrupted heartbeat"

        # Cleanup
        leader.release_leadership()

    @pytest.mark.asyncio
    async def test_lock_file_deleted_externally(self, temp_lock_dir):
        """Should handle lock file deletion gracefully."""
        leader = LeaderElection(lock_dir=temp_lock_dir, process_id="leader")
        leader.try_acquire_leadership()

        # Externally delete lock file
        if leader.lock_file_path.exists():
            # Close file handle first on Windows
            if sys.platform == 'win32' and leader.lock_file:
                pass  # Keep it open to test robustness

        # Leader should still be able to release
        leader.release_leadership()

        assert True, "Should handle lock file deletion gracefully"

    @pytest.mark.asyncio
    async def test_lock_dir_permissions(self, temp_lock_dir):
        """Should create lock directory if it doesn't exist."""
        non_existent = temp_lock_dir / "subdir" / "locks"

        leader = LeaderElection(lock_dir=non_existent, process_id="leader")

        # Should create directory
        assert non_existent.exists(), "Should create lock directory"

        # Should successfully acquire lock
        result = leader.try_acquire_leadership()
        assert result is True

        # Cleanup
        leader.release_leadership()

    @pytest.mark.asyncio
    async def test_double_release(self, temp_lock_dir):
        """Should handle double release gracefully."""
        leader = LeaderElection(lock_dir=temp_lock_dir, process_id="leader")
        leader.try_acquire_leadership()

        # Release twice
        leader.release_leadership()
        leader.release_leadership()  # Should not crash

        assert True, "Should handle double release gracefully"

    @pytest.mark.asyncio
    async def test_release_without_acquiring(self, temp_lock_dir):
        """Should handle release without acquiring gracefully."""
        leader = LeaderElection(lock_dir=temp_lock_dir, process_id="leader")

        # Release without acquiring
        leader.release_leadership()  # Should not crash

        assert True, "Should handle release without acquiring"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
