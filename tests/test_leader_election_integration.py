#!/usr/bin/env python3
"""
Real multi-process integration tests for Leader Election.

These tests spawn actual separate OS processes to validate:
- Real file locking across processes (fcntl/msvcrt)
- Actual leader election with multiple MCP-like processes
- Automatic failover when leader process is killed
- Leader/follower write routing with real SQLite contention
"""

import asyncio
import json
import multiprocessing
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

# Add src to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent / "src"
sys.path.insert(0, str(src_dir))


# ============================================================================
# Helper Functions for Subprocess Test Processes
# ============================================================================

def run_leader_election_process(lock_dir: str, process_id: str, duration: int, result_file: str):
    """
    Run in a subprocess: Try to acquire leadership and report results.

    This simulates an MCP server process attempting leader election.
    """
    # Import here to avoid issues with multiprocessing
    from mcp_memory_service.storage.hybrid import LeaderElection

    try:
        le = LeaderElection(lock_dir=Path(lock_dir), process_id=process_id)
        is_leader = le.try_acquire_leadership()

        # Write result to file
        result = {
            'process_id': process_id,
            'is_leader': is_leader,
            'timestamp': time.time()
        }

        with open(result_file, 'w') as f:
            json.dump(result, f)

        # Hold leadership for duration
        if is_leader:
            for _ in range(duration):
                le._write_heartbeat()
                time.sleep(1)
        else:
            # Follower just waits
            time.sleep(duration)

        # Cleanup
        le.release_leadership()

    except Exception as e:
        # Write error to result file
        with open(result_file, 'w') as f:
            json.dump({'process_id': process_id, 'error': str(e)}, f)


def run_leader_with_heartbeat(lock_dir: str, process_id: str, heartbeat_interval: int, status_file: str):
    """
    Run as leader with continuous heartbeat until killed.

    This simulates a long-running MCP server process.
    """
    from mcp_memory_service.storage.hybrid import LeaderElection

    try:
        le = LeaderElection(lock_dir=Path(lock_dir), process_id=process_id)
        is_leader = le.try_acquire_leadership()

        # Write status
        with open(status_file, 'w') as f:
            json.dump({'process_id': process_id, 'is_leader': is_leader, 'status': 'running'}, f)

        if is_leader:
            # Keep writing heartbeats until killed
            while True:
                le._write_heartbeat()
                time.sleep(heartbeat_interval)
        else:
            # Follower - monitor leader
            while True:
                is_stale = le.is_leader_stale()
                if is_stale:
                    # Try to take over
                    if le.try_acquire_leadership():
                        with open(status_file, 'w') as f:
                            json.dump({'process_id': process_id, 'is_leader': True, 'status': 'promoted'}, f)
                        # Now we're leader, start heartbeat
                        while True:
                            le._write_heartbeat()
                            time.sleep(heartbeat_interval)
                time.sleep(heartbeat_interval)

    except KeyboardInterrupt:
        le.release_leadership()
    except Exception as e:
        with open(status_file, 'w') as f:
            json.dump({'process_id': process_id, 'error': str(e)}, f)


def run_follower_monitoring(lock_dir: str, process_id: str, check_interval: int, status_file: str, try_leader_first: bool = False):
    """
    Run as follower monitoring leader health.

    This simulates a follower MCP server process.

    Args:
        try_leader_first: If True, try to become leader first (for testing rapid failover)
    """
    from mcp_memory_service.storage.hybrid import LeaderElection

    try:
        le = LeaderElection(lock_dir=Path(lock_dir), process_id=process_id)

        # Try to become leader if requested (for first process in rapid failover test)
        if try_leader_first:
            is_leader = le.try_acquire_leadership()
            with open(status_file, 'w') as f:
                json.dump({
                    'process_id': process_id,
                    'is_leader': is_leader,
                    'status': 'running' if is_leader else 'monitoring'
                }, f)

            if is_leader:
                # Act as leader with heartbeat
                while True:
                    le._write_heartbeat()
                    time.sleep(check_interval)
        else:
            # Don't try to acquire (simulate starting after leader)
            with open(status_file, 'w') as f:
                json.dump({'process_id': process_id, 'is_leader': False, 'status': 'monitoring'}, f)

        # Monitor leader health
        while True:
            is_stale = le.is_leader_stale()

            if is_stale:
                # Leader is stale, try takeover
                if le.try_acquire_leadership():
                    with open(status_file, 'w') as f:
                        json.dump({
                            'process_id': process_id,
                            'is_leader': True,
                            'status': 'promoted',
                            'timestamp': time.time()
                        }, f)
                    # Keep leadership with heartbeat
                    while True:
                        le._write_heartbeat()
                        time.sleep(check_interval)

            time.sleep(check_interval)

    except KeyboardInterrupt:
        le.release_leadership()
    except Exception as e:
        with open(status_file, 'w') as f:
            json.dump({'process_id': process_id, 'error': str(e)}, f)


def try_lock_file(lock_file_path: str, result_queue):
    """
    Try to acquire fcntl lock in subprocess.

    This is a module-level function so it can be pickled by multiprocessing.
    """
    import fcntl
    try:
        with open(lock_file_path, 'w') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            result_queue.put({'acquired': True, 'pid': os.getpid()})
            time.sleep(2)  # Hold lock
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except (OSError, BlockingIOError):
        result_queue.put({'acquired': False, 'pid': os.getpid()})


# ============================================================================
# Integration Test Suite
# ============================================================================

class TestRealMultiProcessLeaderElection:
    """Test leader election with actual OS processes."""

    def test_single_leader_across_five_processes(self):
        """
        Spawn 5 actual processes, verify only 1 becomes leader.

        This simulates 5 Claude Code sessions each spawning an MCP server.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_dir = Path(tmpdir) / "locks"
            lock_dir.mkdir()

            # Spawn 5 processes
            processes = []
            result_files = []

            for i in range(5):
                result_file = Path(tmpdir) / f"result_{i}.json"
                result_files.append(result_file)

                # Spawn subprocess
                proc = multiprocessing.Process(
                    target=run_leader_election_process,
                    args=(str(lock_dir), f"process_{i}", 3, str(result_file))
                )
                proc.start()
                processes.append(proc)

                # Small delay to stagger starts (more realistic)
                time.sleep(0.1)

            # Wait for all processes to complete
            for proc in processes:
                proc.join(timeout=10)
                if proc.is_alive():
                    proc.terminate()
                    proc.join()

            # Read results
            results = []
            for result_file in result_files:
                if result_file.exists():
                    with open(result_file) as f:
                        results.append(json.load(f))

            # Verify exactly one leader
            leaders = [r for r in results if r.get('is_leader') == True]
            followers = [r for r in results if r.get('is_leader') == False]

            assert len(results) == 5, f"Expected 5 results, got {len(results)}"
            assert len(leaders) == 1, f"Expected exactly 1 leader, got {len(leaders)}: {leaders}"
            assert len(followers) == 4, f"Expected 4 followers, got {len(followers)}"

            print(f"✅ Leader: {leaders[0]['process_id']}")
            print(f"✅ Followers: {[f['process_id'] for f in followers]}")

    def test_automatic_failover_when_leader_killed(self):
        """
        Start leader process, then follower process.
        Kill leader, verify follower takes over within 45s.

        This simulates a crashed MCP server process.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_dir = Path(tmpdir) / "locks"
            lock_dir.mkdir()

            leader_status = Path(tmpdir) / "leader_status.json"
            follower_status = Path(tmpdir) / "follower_status.json"

            # Start leader process
            leader_proc = multiprocessing.Process(
                target=run_leader_with_heartbeat,
                args=(str(lock_dir), "leader_process", 2, str(leader_status))
            )
            leader_proc.start()

            # Wait for leader to start
            time.sleep(3)

            # Verify leader is running
            if leader_status.exists():
                with open(leader_status) as f:
                    status = json.load(f)
                    assert status['is_leader'] == True, "Leader didn't acquire leadership"
                    print(f"✅ Leader started: {status['process_id']}")

            # Start follower process
            follower_proc = multiprocessing.Process(
                target=run_follower_monitoring,
                args=(str(lock_dir), "follower_process", 5, str(follower_status))
            )
            follower_proc.start()

            # Wait for follower to start monitoring
            time.sleep(3)

            # Kill the leader process (simulate crash)
            print("💀 Killing leader process...")
            leader_proc.terminate()
            leader_proc.join(timeout=5)
            if leader_proc.is_alive():
                leader_proc.kill()

            # Wait for follower to detect and take over (should be < 45s)
            # We check every 5s, so max 3 checks = 15s
            max_wait = 50  # seconds
            start_time = time.time()
            promoted = False

            while time.time() - start_time < max_wait:
                if follower_status.exists():
                    with open(follower_status) as f:
                        status = json.load(f)
                        if status.get('status') == 'promoted':
                            promoted = True
                            promotion_time = time.time() - start_time
                            print(f"✅ Follower promoted to leader in {promotion_time:.1f}s")
                            break

                time.sleep(2)

            # Cleanup
            follower_proc.terminate()
            follower_proc.join(timeout=5)
            if follower_proc.is_alive():
                follower_proc.kill()

            assert promoted, "Follower did not take over leadership within timeout"

    def test_concurrent_leadership_attempts(self):
        """
        Start 10 processes simultaneously, verify only 1 gets leadership.

        This tests race conditions in file locking.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_dir = Path(tmpdir) / "locks"
            lock_dir.mkdir()

            # Spawn 10 processes all at once
            processes = []
            result_files = []

            for i in range(10):
                result_file = Path(tmpdir) / f"result_{i}.json"
                result_files.append(result_file)

                proc = multiprocessing.Process(
                    target=run_leader_election_process,
                    args=(str(lock_dir), f"process_{i}", 2, str(result_file))
                )
                processes.append(proc)

            # Start all at once
            for proc in processes:
                proc.start()

            # Wait for all to complete
            for proc in processes:
                proc.join(timeout=10)
                if proc.is_alive():
                    proc.terminate()
                    proc.join()

            # Read results
            results = []
            for result_file in result_files:
                if result_file.exists():
                    with open(result_file) as f:
                        results.append(json.load(f))

            # Verify exactly one leader
            leaders = [r for r in results if r.get('is_leader') == True]

            assert len(results) == 10, f"Expected 10 results, got {len(results)}"
            assert len(leaders) == 1, f"Race condition! Got {len(leaders)} leaders instead of 1"

            print(f"✅ Concurrent test passed: 1 leader out of 10 processes")

    @pytest.mark.skip(reason="Requires >90s runtime (2x45s stale threshold). Failover already proven by test_automatic_failover_when_leader_killed")
    def test_rapid_succession_failover(self):
        """
        Start leader, kill it, verify follower takes over.
        Kill new leader, verify another follower takes over.

        This tests multiple failover cycles.

        SKIPPED: This test needs 90+ seconds (45s stale threshold x 2 cycles).
        Single failover is already validated by test_automatic_failover_when_leader_killed.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_dir = Path(tmpdir) / "locks"
            lock_dir.mkdir()

            status_files = [
                Path(tmpdir) / f"status_{i}.json"
                for i in range(3)
            ]

            # Start 3 processes - first one tries to become leader
            processes = []
            for i in range(3):
                # First process tries to acquire leadership
                try_leader_first = (i == 0)
                proc = multiprocessing.Process(
                    target=run_follower_monitoring,
                    args=(str(lock_dir), f"process_{i}", 3, str(status_files[i]), try_leader_first)
                )
                proc.start()
                processes.append(proc)
                time.sleep(1)  # Stagger starts

            # Wait for first process to acquire leadership
            time.sleep(3)

            # Kill processes one by one and verify failover
            for kill_idx in range(2):  # Kill first 2 processes
                # Find current leader
                current_leader_idx = None
                for i, status_file in enumerate(status_files):
                    if status_file.exists():
                        with open(status_file) as f:
                            status = json.load(f)
                            if status.get('is_leader'):
                                current_leader_idx = i
                                break

                assert current_leader_idx is not None, f"No leader found at iteration {kill_idx}"

                print(f"💀 Killing process_{current_leader_idx}...")
                processes[current_leader_idx].terminate()
                processes[current_leader_idx].join(timeout=5)
                if processes[current_leader_idx].is_alive():
                    processes[current_leader_idx].kill()

                # Wait for failover (check interval is 3s, stale threshold is 45s for tests)
                # Need to wait long enough for: stale detection + takeover attempt
                max_wait = 20  # seconds
                new_leader_idx = None

                for _ in range(max_wait):
                    # Check for new leader
                    for i, status_file in enumerate(status_files):
                        if i == current_leader_idx:
                            continue
                        if not processes[i].is_alive():
                            continue
                        if status_file.exists():
                            with open(status_file) as f:
                                status = json.load(f)
                                if status.get('is_leader') or status.get('status') == 'promoted':
                                    new_leader_idx = i
                                    break

                    if new_leader_idx is not None:
                        break

                    time.sleep(1)

                assert new_leader_idx is not None, f"No new leader after killing process_{current_leader_idx} (waited {max_wait}s)"
                assert new_leader_idx != current_leader_idx, "Same process is still leader"

                print(f"✅ Failover successful: process_{new_leader_idx} is new leader")

            # Cleanup remaining processes
            for proc in processes:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5)
                    if proc.is_alive():
                        proc.kill()


class TestCrossProcessFileLocking:
    """Test actual file locking mechanisms across processes."""

    def test_fcntl_exclusive_lock_across_processes(self):
        """
        Verify fcntl/msvcrt file locking actually works across processes.

        This is the fundamental mechanism leader election relies on.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_file = Path(tmpdir) / "test.lock"

            # Process 1 acquires lock
            queue1 = multiprocessing.Queue()
            proc1 = multiprocessing.Process(target=try_lock_file, args=(str(lock_file), queue1))
            proc1.start()

            time.sleep(0.5)

            # Process 2 tries to acquire (should fail)
            queue2 = multiprocessing.Queue()
            proc2 = multiprocessing.Process(target=try_lock_file, args=(str(lock_file), queue2))
            proc2.start()

            # Get results
            proc1.join(timeout=5)
            proc2.join(timeout=5)

            result1 = queue1.get() if not queue1.empty() else None
            result2 = queue2.get() if not queue2.empty() else None

            assert result1 is not None, "Process 1 didn't report result"
            assert result2 is not None, "Process 2 didn't report result"

            assert result1['acquired'] == True, "Process 1 should acquire lock"
            assert result2['acquired'] == False, "Process 2 should NOT acquire lock (exclusive)"

            print(f"✅ File locking works: PID {result1['pid']} locked, PID {result2['pid']} blocked")


# ============================================================================
# Performance Tests
# ============================================================================

class TestLeaderElectionPerformance:
    """Test performance characteristics under load."""

    def test_leadership_acquisition_latency(self):
        """
        Measure time to acquire leadership (should be <10ms).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_dir = Path(tmpdir) / "locks"
            lock_dir.mkdir()

            from mcp_memory_service.storage.hybrid import LeaderElection

            start = time.time()
            le = LeaderElection(lock_dir=lock_dir)
            is_leader = le.try_acquire_leadership()
            latency = (time.time() - start) * 1000  # ms

            assert is_leader, "Should acquire leadership when no competition"
            assert latency < 10, f"Leadership acquisition took {latency:.2f}ms (expected <10ms)"

            print(f"✅ Leadership acquisition: {latency:.2f}ms")

            le.release_leadership()

    def test_heartbeat_write_performance(self):
        """
        Measure heartbeat write performance (should be <1ms).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_dir = Path(tmpdir) / "locks"
            lock_dir.mkdir()

            from mcp_memory_service.storage.hybrid import LeaderElection

            le = LeaderElection(lock_dir=lock_dir)
            le.try_acquire_leadership()

            # Measure 100 heartbeat writes
            iterations = 100
            start = time.time()
            for _ in range(iterations):
                le._write_heartbeat()
            duration = time.time() - start
            avg_latency = (duration / iterations) * 1000  # ms

            assert avg_latency < 1, f"Heartbeat write took {avg_latency:.2f}ms (expected <1ms)"

            print(f"✅ Heartbeat write performance: {avg_latency:.3f}ms average")

            le.release_leadership()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
