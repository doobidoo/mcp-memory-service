# Copyright 2024 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Hybrid memory storage backend for MCP Memory Service.

This implementation provides the best of both worlds:
- SQLite-vec as primary storage for ultra-fast reads (~5ms)
- Cloudflare as secondary storage for cloud persistence and multi-device sync
- Background synchronization service for seamless integration
- Graceful degradation when cloud services are unavailable
"""

import asyncio
import logging
import os
import sys
import time
from typing import List, Dict, Any, Tuple, Optional
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from .base import MemoryStorage
from .sqlite_vec import SqliteVecMemoryStorage
from .cloudflare import CloudflareStorage
from ..models.memory import Memory, MemoryQueryResult

# Platform-specific file locking imports
if sys.platform == 'win32':
    import msvcrt
else:
    import fcntl

# Import config to check if limit constants are available
from .. import config as app_config

# Import config values (defaults handled by pydantic-settings)
CLOUDFLARE_D1_MAX_SIZE_GB = app_config.CLOUDFLARE_D1_MAX_SIZE_GB
CLOUDFLARE_VECTORIZE_MAX_VECTORS = app_config.CLOUDFLARE_VECTORIZE_MAX_VECTORS
CLOUDFLARE_MAX_METADATA_SIZE_KB = app_config.CLOUDFLARE_MAX_METADATA_SIZE_KB
CLOUDFLARE_WARNING_THRESHOLD_PERCENT = app_config.CLOUDFLARE_WARNING_THRESHOLD_PERCENT
CLOUDFLARE_CRITICAL_THRESHOLD_PERCENT = app_config.CLOUDFLARE_CRITICAL_THRESHOLD_PERCENT
HYBRID_SYNC_ON_STARTUP = app_config.HYBRID_SYNC_ON_STARTUP
HYBRID_MAX_CONTENT_LENGTH = app_config.HYBRID_MAX_CONTENT_LENGTH
HYBRID_MAX_EMPTY_BATCHES = app_config.HYBRID_MAX_EMPTY_BATCHES
HYBRID_MIN_CHECK_COUNT = app_config.HYBRID_MIN_CHECK_COUNT

# Leader election configuration
HYBRID_LEADER_ELECTION_ENABLED = app_config.HYBRID_LEADER_ELECTION_ENABLED
HYBRID_LEADER_HEALTH_CHECK_INTERVAL = app_config.HYBRID_LEADER_HEALTH_CHECK_INTERVAL
HYBRID_LEADER_HEARTBEAT_INTERVAL = app_config.HYBRID_LEADER_HEARTBEAT_INTERVAL
HYBRID_LEADER_STALE_THRESHOLD = app_config.HYBRID_LEADER_STALE_THRESHOLD

# Adaptive sync configuration
HYBRID_ADAPTIVE_SYNC_ENABLED = app_config.HYBRID_ADAPTIVE_SYNC_ENABLED
HYBRID_SYNC_ACTIVE_INTERVAL = app_config.HYBRID_SYNC_ACTIVE_INTERVAL
HYBRID_SYNC_IDLE_INTERVAL = app_config.HYBRID_SYNC_IDLE_INTERVAL
HYBRID_IDLE_THRESHOLD = app_config.HYBRID_IDLE_THRESHOLD

logger = logging.getLogger(__name__)

@dataclass
class SyncOperation:
    """Represents a pending sync operation."""
    operation: str  # 'store', 'delete', 'update'
    memory: Optional[Memory] = None
    content_hash: Optional[str] = None
    updates: Optional[Dict[str, Any]] = None
    timestamp: float = None
    retries: int = 0
    max_retries: int = 3

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class LeaderElection:
    """
    Cross-platform leader election mechanism for single-writer SQLite access.

    Uses file locking to ensure only one process can write to SQLite at a time.
    Implements automatic failover when leader becomes stale.
    """

    def __init__(self, lock_dir: Path, process_id: str = None):
        """
        Initialize leader election.

        Args:
            lock_dir: Directory for lock and heartbeat files
            process_id: Unique process identifier (defaults to PID)
        """
        self.lock_dir = Path(lock_dir)
        self.lock_dir.mkdir(parents=True, exist_ok=True)

        self.process_id = process_id or f"{os.getpid()}"
        self.lock_file_path = self.lock_dir / "leader.lock"
        self.heartbeat_file_path = self.lock_dir / "leader.heartbeat"

        self.lock_file = None
        self.is_leader = False
        self.heartbeat_task = None
        self.health_check_task = None

    def try_acquire_leadership(self) -> bool:
        """
        Attempt to acquire leader lock.

        Returns:
            True if lock acquired (now leader), False otherwise
        """
        try:
            # Try to open/create lock file
            self.lock_file = open(self.lock_file_path, 'w')

            if sys.platform == 'win32':
                # Windows file locking
                import msvcrt
                try:
                    msvcrt.locking(self.lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                    self.is_leader = True
                    logger.info(f"✅ Process {self.process_id} acquired leadership (Windows)")
                except OSError:
                    self.is_leader = False
                    self.lock_file.close()
                    self.lock_file = None
            else:
                # Unix file locking (fcntl)
                import fcntl
                try:
                    fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self.is_leader = True
                    logger.info(f"✅ Process {self.process_id} acquired leadership (Unix)")
                except (OSError, BlockingIOError):
                    self.is_leader = False
                    self.lock_file.close()
                    self.lock_file = None

            if self.is_leader:
                # Write process ID to lock file
                self.lock_file.write(self.process_id)
                self.lock_file.flush()
                self._write_heartbeat()

            return self.is_leader

        except Exception as e:
            logger.error(f"Error acquiring leadership: {e}")
            self.is_leader = False
            if self.lock_file:
                try:
                    self.lock_file.close()
                except:
                    pass
                self.lock_file = None
            return False

    def release_leadership(self):
        """Release leader lock."""
        if not self.is_leader:
            return

        try:
            if self.lock_file:
                if sys.platform == 'win32':
                    import msvcrt
                    try:
                        msvcrt.locking(self.lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                    except:
                        pass
                else:
                    import fcntl
                    try:
                        fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
                    except:
                        pass

                self.lock_file.close()
                self.lock_file = None

            # Clean up heartbeat file
            if self.heartbeat_file_path.exists():
                self.heartbeat_file_path.unlink()

            self.is_leader = False
            logger.info(f"Process {self.process_id} released leadership")

        except Exception as e:
            logger.error(f"Error releasing leadership: {e}")

    def _write_heartbeat(self):
        """Write current timestamp to heartbeat file."""
        try:
            with open(self.heartbeat_file_path, 'w') as f:
                f.write(f"{time.time()}\n{self.process_id}")
        except Exception as e:
            logger.error(f"Error writing heartbeat: {e}")

    def _read_heartbeat(self) -> Optional[Tuple[float, str]]:
        """
        Read heartbeat file.

        Returns:
            Tuple of (timestamp, process_id) or None if file doesn't exist
        """
        try:
            if not self.heartbeat_file_path.exists():
                return None

            with open(self.heartbeat_file_path, 'r') as f:
                lines = f.read().strip().split('\n')
                if len(lines) >= 2:
                    return float(lines[0]), lines[1]
                elif len(lines) == 1:
                    return float(lines[0]), "unknown"
        except Exception as e:
            logger.warning(f"Error reading heartbeat: {e}")
        return None

    def is_leader_stale(self) -> bool:
        """
        Check if current leader is stale (hasn't written heartbeat recently).

        Returns:
            True if leader is stale or doesn't exist
        """
        heartbeat = self._read_heartbeat()
        if not heartbeat:
            return True

        timestamp, process_id = heartbeat
        age = time.time() - timestamp

        is_stale = age > HYBRID_LEADER_STALE_THRESHOLD
        if is_stale:
            logger.warning(f"Leader {process_id} is stale (heartbeat {age:.1f}s old, threshold {HYBRID_LEADER_STALE_THRESHOLD}s)")

        return is_stale

    async def start_leader_heartbeat(self):
        """Start background task to write heartbeats (leader only)."""
        if not self.is_leader:
            return

        async def heartbeat_loop():
            while self.is_leader:
                try:
                    self._write_heartbeat()
                    await asyncio.sleep(HYBRID_LEADER_HEARTBEAT_INTERVAL)
                except Exception as e:
                    logger.error(f"Error in heartbeat loop: {e}")
                    await asyncio.sleep(1)

        self.heartbeat_task = asyncio.create_task(heartbeat_loop())
        logger.info(f"Leader heartbeat started (interval: {HYBRID_LEADER_HEARTBEAT_INTERVAL}s)")

    async def start_follower_health_check(self, on_leader_stale_callback):
        """
        Start background task to monitor leader health (follower only).

        Args:
            on_leader_stale_callback: Async function to call when leader becomes stale
        """
        if self.is_leader:
            return

        async def health_check_loop():
            while not self.is_leader:
                try:
                    if self.is_leader_stale():
                        logger.warning(f"Follower {self.process_id} detected stale leader, attempting takeover")
                        if self.try_acquire_leadership():
                            logger.info(f"🎉 Follower {self.process_id} promoted to leader!")
                            await on_leader_stale_callback()
                            break

                    await asyncio.sleep(HYBRID_LEADER_HEALTH_CHECK_INTERVAL)
                except Exception as e:
                    logger.error(f"Error in health check loop: {e}")
                    await asyncio.sleep(1)

        self.health_check_task = asyncio.create_task(health_check_loop())
        logger.info(f"Follower health check started (interval: {HYBRID_LEADER_HEALTH_CHECK_INTERVAL}s)")

    async def stop(self):
        """Stop background tasks and release leadership."""
        # Cancel background tasks
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass

        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass

        # Release leadership
        self.release_leadership()

class BackgroundSyncService:
    """
    Handles background synchronization between SQLite-vec and Cloudflare.

    Features:
    - Asynchronous operation queue
    - Retry logic with exponential backoff
    - Health monitoring and error handling
    - Configurable sync intervals and batch sizes
    - Adaptive sync intervals (5s active, 60s idle)
    - Graceful degradation when cloud is unavailable
    """

    def __init__(self,
                 primary_storage: SqliteVecMemoryStorage,
                 secondary_storage: CloudflareStorage,
                 sync_interval: int = 300,  # 5 minutes
                 batch_size: int = 50,
                 max_queue_size: int = 1000,
                 is_leader: bool = True):
        self.primary = primary_storage
        self.secondary = secondary_storage
        self.sync_interval = sync_interval
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        self.is_leader = is_leader

        # Sync queues and state
        self.operation_queue = asyncio.Queue(maxsize=max_queue_size)
        self.failed_operations = deque(maxlen=100)  # Keep track of failed operations
        self.is_running = False
        self.sync_task = None
        self.last_sync_time = 0
        self.sync_stats = {
            'operations_processed': 0,
            'operations_failed': 0,
            'last_sync_duration': 0,
            'cloudflare_available': True
        }

        # Health monitoring
        self.consecutive_failures = 0
        self.max_consecutive_failures = 5
        self.backoff_time = 60  # Start with 1 minute backoff

        # Cloudflare capacity tracking
        self.cloudflare_stats = {
            'vector_count': 0,
            'estimated_d1_size_gb': 0,
            'last_capacity_check': 0,
            'approaching_limits': False,
            'limit_warnings': []
        }

        # Adaptive sync tracking
        self.last_write_time = time.time()
        self.adaptive_sync_enabled = HYBRID_ADAPTIVE_SYNC_ENABLED
        self.sync_active_interval = HYBRID_SYNC_ACTIVE_INTERVAL
        self.sync_idle_interval = HYBRID_SYNC_IDLE_INTERVAL
        self.idle_threshold = HYBRID_IDLE_THRESHOLD

    def record_write_activity(self):
        """Record that a write operation occurred (for adaptive sync)."""
        self.last_write_time = time.time()

    def get_current_sync_interval(self) -> int:
        """
        Calculate current sync interval based on activity.

        Returns:
            Sync interval in seconds (5s if active, 60s if idle)
        """
        if not self.adaptive_sync_enabled:
            return self.sync_interval

        # Check time since last write
        time_since_write = time.time() - self.last_write_time

        if time_since_write < self.idle_threshold:
            # Active: sync frequently
            return self.sync_active_interval
        else:
            # Idle: sync less frequently
            return self.sync_idle_interval

    async def start(self):
        """Start the background sync service."""
        if self.is_running:
            logger.warning("Background sync service is already running")
            return

        self.is_running = True
        self.sync_task = asyncio.create_task(self._sync_loop())
        logger.info(f"Background sync service started with {self.sync_interval}s interval")

    async def stop(self):
        """Stop the background sync service and process remaining operations."""
        if not self.is_running:
            return

        self.is_running = False

        # Process remaining operations in queue
        remaining_operations = []
        while not self.operation_queue.empty():
            try:
                operation = self.operation_queue.get_nowait()
                remaining_operations.append(operation)
            except asyncio.QueueEmpty:
                break

        if remaining_operations:
            logger.info(f"Processing {len(remaining_operations)} remaining operations before shutdown")
            await self._process_operations_batch(remaining_operations)

        # Cancel the sync task
        if self.sync_task:
            self.sync_task.cancel()
            try:
                await self.sync_task
            except asyncio.CancelledError:
                pass

        logger.info("Background sync service stopped")

    async def enqueue_operation(self, operation: SyncOperation):
        """Enqueue a sync operation for background processing."""
        try:
            await self.operation_queue.put(operation)
            logger.debug(f"Enqueued {operation.operation} operation")
        except asyncio.QueueFull:
            # If queue is full, process immediately to avoid blocking
            logger.warning("Sync queue full, processing operation immediately")
            await self._process_single_operation(operation)

    async def force_sync(self) -> Dict[str, Any]:
        """Force an immediate full synchronization between backends."""
        logger.info("Starting forced sync between primary and secondary storage")
        sync_start_time = time.time()

        try:
            # Get all memories from primary storage
            primary_memories = await self.primary.get_all_memories()

            # Check Cloudflare availability
            try:
                await self.secondary.get_stats()  # Simple health check
                cloudflare_available = True
            except Exception as e:
                logger.warning(f"Cloudflare not available during force sync: {e}")
                cloudflare_available = False
                self.sync_stats['cloudflare_available'] = False
                return {
                    'status': 'partial',
                    'cloudflare_available': False,
                    'primary_memories': len(primary_memories),
                    'synced_to_secondary': 0,
                    'duration': time.time() - sync_start_time
                }

            # Sync from primary to secondary using concurrent operations
            async def sync_memory(memory):
                try:
                    success, message = await self.secondary.store(memory)
                    if success:
                        return True, None
                    else:
                        logger.debug(f"Failed to sync memory to secondary: {message}")
                        return False, message
                except Exception as e:
                    logger.debug(f"Exception syncing memory to secondary: {e}")
                    return False, str(e)

            # Process memories concurrently in batches
            synced_count = 0
            failed_count = 0

            # Process in batches to avoid overwhelming the system
            batch_size = min(self.batch_size, 10)  # Limit concurrent operations
            for i in range(0, len(primary_memories), batch_size):
                batch = primary_memories[i:i + batch_size]
                results = await asyncio.gather(*[sync_memory(m) for m in batch], return_exceptions=True)

                for result in results:
                    if isinstance(result, Exception):
                        failed_count += 1
                        logger.debug(f"Exception in batch sync: {result}")
                    elif isinstance(result, tuple):
                        success, _ = result
                        if success:
                            synced_count += 1
                        else:
                            failed_count += 1

            sync_duration = time.time() - sync_start_time
            self.sync_stats['last_sync_duration'] = sync_duration
            self.sync_stats['cloudflare_available'] = cloudflare_available

            logger.info(f"Force sync completed: {synced_count} synced, {failed_count} failed in {sync_duration:.2f}s")

            return {
                'status': 'completed',
                'cloudflare_available': cloudflare_available,
                'primary_memories': len(primary_memories),
                'synced_to_secondary': synced_count,
                'failed_operations': failed_count,
                'duration': sync_duration
            }

        except Exception as e:
            logger.error(f"Error during force sync: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'duration': time.time() - sync_start_time
            }

    async def get_sync_status(self) -> Dict[str, Any]:
        """Get current sync service status and statistics."""
        queue_size = self.operation_queue.qsize()

        status = {
            'is_running': self.is_running,
            'queue_size': queue_size,
            'failed_operations': len(self.failed_operations),
            'last_sync_time': self.last_sync_time,
            'consecutive_failures': self.consecutive_failures,
            'stats': self.sync_stats.copy(),
            'cloudflare_available': self.sync_stats['cloudflare_available'],
            'next_sync_in': max(0, self.sync_interval - (time.time() - self.last_sync_time)),
            'capacity': {
                'vector_count': self.cloudflare_stats['vector_count'],
                'vector_limit': CLOUDFLARE_VECTORIZE_MAX_VECTORS,
                'approaching_limits': self.cloudflare_stats['approaching_limits'],
                'warnings': self.cloudflare_stats['limit_warnings']
            }
        }

        return status

    async def validate_memory_for_cloudflare(self, memory: Memory) -> Tuple[bool, Optional[str]]:
        """
        Validate if a memory can be synced to Cloudflare.

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check metadata size
        if memory.metadata:
            import json
            metadata_json = json.dumps(memory.metadata)
            metadata_size_kb = len(metadata_json.encode('utf-8')) / 1024

            if metadata_size_kb > CLOUDFLARE_MAX_METADATA_SIZE_KB:
                return False, f"Metadata size {metadata_size_kb:.2f}KB exceeds Cloudflare limit of {CLOUDFLARE_MAX_METADATA_SIZE_KB}KB"

        # Check if we're approaching vector count limit
        if self.cloudflare_stats['vector_count'] >= CLOUDFLARE_VECTORIZE_MAX_VECTORS:
            return False, f"Cloudflare vector limit of {CLOUDFLARE_VECTORIZE_MAX_VECTORS} reached"

        return True, None

    async def check_cloudflare_capacity(self) -> Dict[str, Any]:
        """
        Check remaining Cloudflare capacity and return status.
        """
        try:
            # Get current stats from Cloudflare
            cf_stats = await self.secondary.get_stats()

            # Update our tracking
            self.cloudflare_stats['vector_count'] = cf_stats.get('total_memories', 0)
            self.cloudflare_stats['last_capacity_check'] = time.time()

            # Calculate usage percentages
            vector_usage_percent = (self.cloudflare_stats['vector_count'] / CLOUDFLARE_VECTORIZE_MAX_VECTORS) * 100

            # Clear previous warnings
            self.cloudflare_stats['limit_warnings'] = []

            # Check vector count limits
            if vector_usage_percent >= CLOUDFLARE_CRITICAL_THRESHOLD_PERCENT:
                warning = f"CRITICAL: Vector usage at {vector_usage_percent:.1f}% ({self.cloudflare_stats['vector_count']:,}/{CLOUDFLARE_VECTORIZE_MAX_VECTORS:,})"
                self.cloudflare_stats['limit_warnings'].append(warning)
                logger.error(warning)
                self.cloudflare_stats['approaching_limits'] = True
            elif vector_usage_percent >= CLOUDFLARE_WARNING_THRESHOLD_PERCENT:
                warning = f"WARNING: Vector usage at {vector_usage_percent:.1f}% ({self.cloudflare_stats['vector_count']:,}/{CLOUDFLARE_VECTORIZE_MAX_VECTORS:,})"
                self.cloudflare_stats['limit_warnings'].append(warning)
                logger.warning(warning)
                self.cloudflare_stats['approaching_limits'] = True
            else:
                self.cloudflare_stats['approaching_limits'] = False

            return {
                'vector_count': self.cloudflare_stats['vector_count'],
                'vector_limit': CLOUDFLARE_VECTORIZE_MAX_VECTORS,
                'vector_usage_percent': vector_usage_percent,
                'approaching_limits': self.cloudflare_stats['approaching_limits'],
                'warnings': self.cloudflare_stats['limit_warnings']
            }

        except Exception as e:
            logger.error(f"Failed to check Cloudflare capacity: {e}")
            return {
                'error': str(e),
                'approaching_limits': False
            }

    async def _sync_loop(self):
        """Main background sync loop with adaptive intervals."""
        logger.info("Background sync loop started")

        while self.is_running:
            try:
                # Process queued operations
                await self._process_operation_queue()

                # Calculate current sync interval based on activity
                current_sync_interval = self.get_current_sync_interval()

                # Periodic full sync if enough time has passed
                current_time = time.time()
                if current_time - self.last_sync_time >= current_sync_interval:
                    await self._periodic_sync()
                    self.last_sync_time = current_time
                    logger.debug(f"Next sync in {current_sync_interval}s (adaptive: {'active' if current_sync_interval == self.sync_active_interval else 'idle'})")

                # Sleep before next iteration
                await asyncio.sleep(5)  # Check every 5 seconds

            except Exception as e:
                logger.error(f"Error in sync loop: {e}")
                self.consecutive_failures += 1

                if self.consecutive_failures >= self.max_consecutive_failures:
                    logger.warning(f"Too many consecutive sync failures ({self.consecutive_failures}), backing off for {self.backoff_time}s")
                    await asyncio.sleep(self.backoff_time)
                    self.backoff_time = min(self.backoff_time * 2, 1800)  # Max 30 minutes
                else:
                    await asyncio.sleep(1)

    async def _process_operation_queue(self):
        """Process operations from the queue in batches."""
        operations = []

        # Collect up to batch_size operations
        for _ in range(self.batch_size):
            try:
                operation = self.operation_queue.get_nowait()
                operations.append(operation)
            except asyncio.QueueEmpty:
                break

        if operations:
            await self._process_operations_batch(operations)

    async def _process_operations_batch(self, operations: List[SyncOperation]):
        """Process a batch of sync operations."""
        logger.debug(f"Processing batch of {len(operations)} sync operations")

        for operation in operations:
            try:
                await self._process_single_operation(operation)
                self.sync_stats['operations_processed'] += 1

            except Exception as e:
                await self._handle_sync_error(e, operation)

    async def _handle_sync_error(self, error: Exception, operation: SyncOperation):
        """
        Handle sync operation errors with intelligent retry logic.

        Args:
            error: The exception that occurred
            operation: The failed operation
        """
        error_str = str(error).lower()

        # Check for specific Cloudflare limit errors
        is_limit_error = any(term in error_str for term in [
            'limit exceeded', 'quota exceeded', 'maximum', 'too large',
            '413', '507', 'insufficient storage', 'capacity'
        ])

        if is_limit_error:
            # Don't retry limit errors - they won't succeed
            logger.error(f"Cloudflare limit error for {operation.operation}: {error}")
            self.sync_stats['operations_failed'] += 1

            # Update capacity tracking
            self.cloudflare_stats['approaching_limits'] = True
            self.cloudflare_stats['limit_warnings'].append(f"Limit error: {error}")

            # Check capacity to understand the issue
            await self.check_cloudflare_capacity()
            return

        # Check for temporary/network errors
        is_temporary_error = any(term in error_str for term in [
            'timeout', 'connection', 'network', '500', '502', '503', '504',
            'temporarily unavailable', 'retry'
        ])

        if is_temporary_error or operation.retries < operation.max_retries:
            # Retry temporary errors
            logger.warning(f"Temporary error for {operation.operation} (retry {operation.retries + 1}/{operation.max_retries}): {error}")
            operation.retries += 1

            if operation.retries < operation.max_retries:
                # Add back to queue for retry with exponential backoff
                await asyncio.sleep(min(2 ** operation.retries, 60))  # Max 60 second delay
                self.failed_operations.append(operation)
            else:
                logger.error(f"Max retries reached for {operation.operation}")
                self.sync_stats['operations_failed'] += 1
        else:
            # Permanent error - don't retry
            logger.error(f"Permanent error for {operation.operation}: {error}")
            self.sync_stats['operations_failed'] += 1

    async def _process_single_operation(self, operation: SyncOperation):
        """Process a single sync operation to secondary storage."""
        try:
            # Record write activity for adaptive sync
            self.record_write_activity()

            if operation.operation == 'store' and operation.memory:
                # Validate memory before syncing
                is_valid, validation_error = await self.validate_memory_for_cloudflare(operation.memory)
                if not is_valid:
                    logger.warning(f"Memory validation failed for sync: {validation_error}")
                    # Don't retry if it's a hard limit
                    if "exceeds Cloudflare limit" in validation_error or "limit of" in validation_error:
                        self.sync_stats['operations_failed'] += 1
                        return  # Skip this memory permanently
                    raise Exception(validation_error)

                success, message = await self.secondary.store(operation.memory)
                if not success:
                    raise Exception(f"Store operation failed: {message}")

            elif operation.operation == 'delete' and operation.content_hash:
                success, message = await self.secondary.delete(operation.content_hash)
                if not success:
                    raise Exception(f"Delete operation failed: {message}")

            elif operation.operation == 'update' and operation.content_hash and operation.updates:
                success, message = await self.secondary.update_memory_metadata(
                    operation.content_hash, operation.updates
                )
                if not success:
                    raise Exception(f"Update operation failed: {message}")

            # Reset failure counters on success
            self.consecutive_failures = 0
            self.backoff_time = 60
            self.sync_stats['cloudflare_available'] = True

        except Exception as e:
            # Mark Cloudflare as potentially unavailable
            self.sync_stats['cloudflare_available'] = False
            raise

    async def _periodic_sync(self):
        """Perform periodic full synchronization."""
        logger.debug("Starting periodic sync")

        try:
            # Retry any failed operations first
            if self.failed_operations:
                retry_operations = list(self.failed_operations)
                self.failed_operations.clear()
                logger.info(f"Retrying {len(retry_operations)} failed operations")
                await self._process_operations_batch(retry_operations)

            # Perform a lightweight health check
            try:
                stats = await self.secondary.get_stats()
                logger.debug(f"Secondary storage health check passed: {stats}")
                self.sync_stats['cloudflare_available'] = True

                # Check Cloudflare capacity every periodic sync
                capacity_status = await self.check_cloudflare_capacity()
                if capacity_status.get('approaching_limits'):
                    logger.warning("Cloudflare approaching capacity limits")
                    for warning in capacity_status.get('warnings', []):
                        logger.warning(warning)

            except Exception as e:
                logger.warning(f"Secondary storage health check failed: {e}")
                self.sync_stats['cloudflare_available'] = False

        except Exception as e:
            logger.error(f"Error during periodic sync: {e}")


class HybridMemoryStorage(MemoryStorage):
    """
    Hybrid memory storage using SQLite-vec as primary and Cloudflare as secondary.

    This implementation provides:
    - Ultra-fast reads and writes (~5ms) via SQLite-vec
    - Cloud persistence and multi-device sync via Cloudflare
    - Background synchronization with retry logic
    - Graceful degradation when cloud services are unavailable
    - Full compatibility with the MemoryStorage interface
    """

    @property
    def max_content_length(self) -> Optional[int]:
        """
        Maximum content length constrained by Cloudflare secondary storage.
        Uses configured hybrid limit (defaults to Cloudflare limit).
        """
        return HYBRID_MAX_CONTENT_LENGTH

    @property
    def supports_chunking(self) -> bool:
        """Hybrid backend supports content chunking with metadata linking."""
        return True

    def __init__(self,
                 sqlite_db_path: str,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 cloudflare_config: Dict[str, Any] = None,
                 sync_interval: int = 300,
                 batch_size: int = 50):
        """
        Initialize hybrid storage with primary SQLite-vec and secondary Cloudflare.

        Args:
            sqlite_db_path: Path to SQLite-vec database file
            embedding_model: Embedding model name for SQLite-vec
            cloudflare_config: Cloudflare configuration dict
            sync_interval: Background sync interval in seconds (default: 5 minutes)
            batch_size: Batch size for sync operations (default: 50)
        """
        self.primary = SqliteVecMemoryStorage(
            db_path=sqlite_db_path,
            embedding_model=embedding_model
        )

        # Initialize Cloudflare storage if config provided
        self.secondary = None
        self.sync_service = None

        if cloudflare_config and all(key in cloudflare_config for key in
                                    ['api_token', 'account_id', 'vectorize_index', 'd1_database_id']):
            self.secondary = CloudflareStorage(**cloudflare_config)
        else:
            logger.warning("Cloudflare config incomplete, running in SQLite-only mode")

        self.sync_interval = sync_interval
        self.batch_size = batch_size
        self.initialized = False

        # Initial sync status tracking
        self.initial_sync_in_progress = False
        self.initial_sync_total = 0
        self.initial_sync_completed = 0
        self.initial_sync_finished = False

        # Leader election for single-writer SQLite access
        self.leader_election = None
        if HYBRID_LEADER_ELECTION_ENABLED:
            # Use SQLite db directory for lock files
            lock_dir = Path(sqlite_db_path).parent / ".hybrid_locks"
            self.leader_election = LeaderElection(lock_dir=lock_dir)

    async def initialize(self) -> None:
        """Initialize the hybrid storage system."""
        logger.info("Initializing hybrid memory storage...")

        # Try to acquire leadership if leader election is enabled
        is_leader = True
        if self.leader_election:
            is_leader = self.leader_election.try_acquire_leadership()
            if is_leader:
                logger.info("🎖️  This process is the LEADER - will write to both SQLite and Cloudflare")
                # Start leader heartbeat
                await self.leader_election.start_leader_heartbeat()
            else:
                logger.info("👥 This process is a FOLLOWER - will write to Cloudflare only")

        # Always initialize primary storage
        await self.primary.initialize()
        logger.info("Primary storage (SQLite-vec) initialized")

        # Initialize secondary storage and sync service if available
        if self.secondary:
            try:
                await self.secondary.initialize()
                logger.info("Secondary storage (Cloudflare) initialized")

                # Start background sync service
                self.sync_service = BackgroundSyncService(
                    self.primary,
                    self.secondary,
                    sync_interval=self.sync_interval,
                    batch_size=self.batch_size,
                    is_leader=is_leader
                )
                await self.sync_service.start()
                logger.info("Background sync service started")

                # Start follower health check if not leader
                if self.leader_election and not is_leader:
                    await self.leader_election.start_follower_health_check(
                        on_leader_stale_callback=self._on_promoted_to_leader
                    )

                # Schedule initial sync to run after server startup (non-blocking)
                # Only leader performs initial sync to avoid redundant work
                if HYBRID_SYNC_ON_STARTUP and is_leader:
                    asyncio.create_task(self._perform_initial_sync_after_startup())
                    logger.info("Initial sync scheduled to run after server startup")

            except Exception as e:
                logger.warning(f"Failed to initialize secondary storage: {e}")
                self.secondary = None

        self.initialized = True
        logger.info("Hybrid memory storage initialization completed")

    async def _on_promoted_to_leader(self):
        """
        Callback when a follower is promoted to leader.

        Starts leader heartbeat and updates sync service.
        """
        logger.info("🎖️  Promoted to LEADER - starting leader heartbeat and sync")
        if self.leader_election:
            await self.leader_election.start_leader_heartbeat()

        # Update sync service to leader mode
        if self.sync_service:
            self.sync_service.is_leader = True

            # Trigger initial sync to catch up on any missed writes
            if HYBRID_SYNC_ON_STARTUP:
                asyncio.create_task(self._perform_initial_sync_after_startup())
                logger.info("Initial sync scheduled after leader promotion")

    async def _perform_initial_sync_after_startup(self) -> None:
        """
        Wrapper for initial sync that waits for server startup to complete.
        This allows the web server to be accessible during the sync process.
        """
        # Wait a bit for server to fully start up
        await asyncio.sleep(2)
        logger.info("Starting initial sync in background (server is now accessible)")
        await self._perform_initial_sync()

    async def _perform_initial_sync(self) -> None:
        """
        Perform initial sync from Cloudflare to SQLite if enabled.

        This downloads all memories from Cloudflare that are missing in local SQLite,
        providing immediate access to existing cloud memories.
        """
        if not HYBRID_SYNC_ON_STARTUP or not self.secondary:
            return

        logger.info("Starting initial sync from Cloudflare to SQLite...")

        self.initial_sync_in_progress = True
        self.initial_sync_completed = 0
        self.initial_sync_finished = False

        try:
            # Get memory count from both storages to compare
            primary_stats = await self.primary.get_stats()
            secondary_stats = await self.secondary.get_stats()

            primary_count = primary_stats.get('total_memories', 0)
            secondary_count = secondary_stats.get('total_memories', 0)

            logger.info(f"Memory count comparison - Local SQLite: {primary_count}, Cloudflare: {secondary_count}")

            if secondary_count <= primary_count:
                logger.info("Local SQLite has same or more memories than Cloudflare, skipping initial sync")
                self.initial_sync_finished = True
                return

            # Get all memories from Cloudflare to sync missing ones
            missing_count = secondary_count - primary_count
            self.initial_sync_total = missing_count
            logger.info(f"Found {missing_count} memories in Cloudflare that need to be synced to local SQLite")

            # Get all Cloudflare memories using cursor-based pagination to avoid D1 OFFSET limitations
            synced_count = 0
            batch_size = min(100, self.batch_size * 2)  # Use larger batch for initial sync
            cursor = None  # Start from most recent (no cursor)
            processed_count = 0
            consecutive_empty_batches = 0  # Track empty batches for early break detection

            while True:
                try:
                    # Get batch of memories from Cloudflare using cursor-based pagination
                    logger.debug(f"Fetching batch from Cloudflare with cursor-based pagination: cursor={cursor}, batch_size={batch_size}")

                    # Try cursor-based pagination first, fallback to offset if not supported
                    if hasattr(self.secondary, 'get_all_memories_cursor'):
                        cloudflare_memories = await self.secondary.get_all_memories_cursor(
                            limit=batch_size,
                            cursor=cursor
                        )
                    else:
                        # Fallback for backends without cursor support
                        cloudflare_memories = await self.secondary.get_all_memories(
                            limit=batch_size,
                            offset=processed_count
                        )

                    if not cloudflare_memories:
                        logger.debug(f"No more memories returned from Cloudflare at cursor {cursor}")
                        break

                    logger.debug(f"Processing batch of {len(cloudflare_memories)} memories from Cloudflare")
                    batch_checked = 0
                    batch_missing = 0
                    batch_synced = 0

                    # Check which memories are missing in primary storage
                    for cf_memory in cloudflare_memories:
                        batch_checked += 1
                        processed_count += 1
                        try:
                            # Check if memory exists in primary storage
                            existing = await self.primary.get_by_hash(cf_memory.content_hash)
                            if not existing:
                                batch_missing += 1
                                # Memory doesn't exist locally, sync it
                                success, message = await self.primary.store(cf_memory)
                                if success:
                                    batch_synced += 1
                                    synced_count += 1
                                    self.initial_sync_completed = synced_count
                                    if synced_count % 10 == 0:  # Log progress every 10 memories
                                        logger.info(f"Initial sync progress: {synced_count}/{missing_count} memories synced")
                                else:
                                    logger.warning(f"Failed to sync memory {cf_memory.content_hash}: {message}")
                        except Exception as e:
                            logger.warning(f"Error checking/syncing memory {cf_memory.content_hash}: {e}")
                            continue

                    logger.debug(f"Batch complete: checked={batch_checked}, missing={batch_missing}, synced={batch_synced}")

                    # Track consecutive batches with no new syncs
                    if batch_synced == 0:
                        consecutive_empty_batches += 1
                        logger.debug(f"Empty batch detected: consecutive_empty_batches={consecutive_empty_batches}/{HYBRID_MAX_EMPTY_BATCHES}")
                    else:
                        consecutive_empty_batches = 0  # Reset counter when we find missing memories

                    # Log progress summary
                    if processed_count > 0 and processed_count % 100 == 0:  # Every 100 memories processed
                        logger.info(f"Sync progress: processed={processed_count}, synced={synced_count}/{missing_count}, empty_batches={consecutive_empty_batches}")

                    # Update cursor to the oldest timestamp from this batch for next iteration
                    if cloudflare_memories and hasattr(self.secondary, 'get_all_memories_cursor'):
                        # Get the oldest created_at timestamp from this batch for next cursor
                        cursor = min(memory.created_at for memory in cloudflare_memories if memory.created_at)
                        logger.debug(f"Next cursor set to: {cursor}")

                    # Configurable early break conditions (v7.5.4+)
                    # Break only if we've had many consecutive empty batches AND we've synced some memories
                    if consecutive_empty_batches >= HYBRID_MAX_EMPTY_BATCHES and synced_count > 0:
                        logger.info(f"Completed sync after {consecutive_empty_batches} consecutive empty batches (threshold: {HYBRID_MAX_EMPTY_BATCHES}) - {synced_count}/{missing_count} memories synced, {processed_count} total processed")
                        break
                    # Or if we've processed minimum threshold and found no missing memories (true no-op case)
                    elif processed_count >= HYBRID_MIN_CHECK_COUNT and synced_count == 0:
                        logger.info(f"No missing memories found after checking {processed_count} memories (threshold: {HYBRID_MIN_CHECK_COUNT}) - all Cloudflare memories already exist locally")
                        break

                    # Yield control to avoid blocking the event loop
                    await asyncio.sleep(0.01)

                except Exception as e:
                    # Handle Cloudflare D1 errors (like 400 Bad Request from OFFSET limitations)
                    if "400" in str(e) and not hasattr(self.secondary, 'get_all_memories_cursor'):
                        logger.error(f"D1 OFFSET limitation hit at processed_count={processed_count}: {e}")
                        logger.warning("Cloudflare D1 OFFSET limits reached - sync incomplete due to backend limitations")
                        break
                    else:
                        logger.error(f"Error during cursor-based sync: {e}")
                        break

            logger.info(f"Initial sync completed: {synced_count} memories downloaded from Cloudflare to local SQLite")

            # Update sync tracking to reflect actual sync completion
            if synced_count == 0:
                # All memories were already present - this is a successful "no-op" sync
                self.initial_sync_completed = self.initial_sync_total
                logger.info(f"Sync completed successfully: All {self.initial_sync_total} memories were already present locally")

            self.initial_sync_finished = True

        except Exception as e:
            logger.error(f"Initial sync failed: {e}")
            # Don't fail initialization if initial sync fails
            logger.warning("Continuing with hybrid storage despite initial sync failure")
        finally:
            self.initial_sync_in_progress = False

    def get_initial_sync_status(self) -> Dict[str, Any]:
        """Get current initial sync status for monitoring."""
        return {
            "in_progress": self.initial_sync_in_progress,
            "total": self.initial_sync_total,
            "completed": self.initial_sync_completed,
            "finished": self.initial_sync_finished,
            "progress_percentage": round((self.initial_sync_completed / max(self.initial_sync_total, 1)) * 100, 1) if self.initial_sync_total > 0 else 0
        }

    async def store(self, memory: Memory) -> Tuple[bool, str]:
        """
        Store a memory with leader-based routing.

        Leader: Writes to SQLite (fast), queues Cloudflare sync
        Follower: Writes to Cloudflare only (avoids SQLite write contention)
        """
        is_leader = self.leader_election.is_leader if self.leader_election else True

        if is_leader:
            # Leader: Write to SQLite first for immediate availability
            success, message = await self.primary.store(memory)

            if success and self.sync_service:
                # Queue for background sync to secondary
                operation = SyncOperation(operation='store', memory=memory)
                await self.sync_service.enqueue_operation(operation)

            return success, message
        else:
            # Follower: Write to Cloudflare only to avoid SQLite contention
            if self.secondary:
                success, message = await self.secondary.store(memory)
                if success:
                    logger.debug(f"Follower stored memory to Cloudflare: {memory.content_hash[:8]}")
                return success, message
            else:
                return False, "Follower mode requires Cloudflare backend"

    async def retrieve(
        self,
        query: str,
        n_results: int = 5,
        tags: Optional[List[str]] = None,
        memory_type: Optional[str] = None,
        min_similarity: Optional[float] = None,
        offset: int = 0
    ) -> List[MemoryQueryResult]:
        """Retrieve memories from primary storage (fast) with pagination."""
        return await self.primary.retrieve(query, n_results, tags, memory_type, min_similarity, offset)

    async def search(
        self,
        query: str,
        n_results: int = 5,
        tags: Optional[List[str]] = None,
        memory_type: Optional[str] = None,
        min_similarity: Optional[float] = None
    ) -> List[MemoryQueryResult]:
        """Search memories in primary storage."""
        return await self.primary.search(query, n_results, tags, memory_type, min_similarity)

    async def search_by_tag(
        self,
        tags: List[str],
        match_all: bool = False,
        limit: int = 10,
        offset: int = 0,
        start_timestamp: Optional[float] = None,
        end_timestamp: Optional[float] = None
    ) -> List[Memory]:
        """Search memories by tags in primary storage with pagination and date filtering."""
        operation = "AND" if match_all else "OR"
        return await self.primary.search_by_tags(
            tags,
            operation=operation,
            limit=limit,
            offset=offset,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp
        )

    async def search_by_tags(self, tags: List[str], match_all: bool = False, limit: int = 10, offset: int = 0) -> List[Memory]:
        """Search memories by tags (alternative method signature) with pagination."""
        operation = "AND" if match_all else "OR"
        return await self.primary.search_by_tags(tags, operation=operation, limit=limit, offset=offset)

    async def get_memory_by_hash(self, content_hash: str) -> Optional[Memory]:
        """Retrieve a specific memory by its content hash from primary storage."""
        return await self.primary.get_memory_by_hash(content_hash)

    async def delete(self, content_hash: str) -> Tuple[bool, str]:
        """
        Delete a memory with leader-based routing.

        Leader: Deletes from SQLite, queues Cloudflare sync
        Follower: Deletes from Cloudflare only
        """
        is_leader = self.leader_election.is_leader if self.leader_election else True

        if is_leader:
            # Leader: Delete from SQLite
            success, message = await self.primary.delete(content_hash)

            if success and self.sync_service:
                # Queue for background sync to secondary
                operation = SyncOperation(operation='delete', content_hash=content_hash)
                await self.sync_service.enqueue_operation(operation)

            return success, message
        else:
            # Follower: Delete from Cloudflare only
            if self.secondary:
                success, message = await self.secondary.delete(content_hash)
                if success:
                    logger.debug(f"Follower deleted memory from Cloudflare: {content_hash[:8]}")
                return success, message
            else:
                return False, "Follower mode requires Cloudflare backend"

    async def delete_by_tag(self, tag: str) -> Tuple[int, str]:
        """Delete memories by tag from primary storage and queue for secondary sync."""
        # First, get the memories with this tag to get their hashes for sync
        memories_to_delete = await self.primary.search_by_tags([tag])

        # Delete from primary
        count_deleted, message = await self.primary.delete_by_tag(tag)

        # Queue individual deletes for secondary sync
        if count_deleted > 0 and self.sync_service:
            for memory in memories_to_delete:
                operation = SyncOperation(operation='delete', content_hash=memory.content_hash)
                await self.sync_service.enqueue_operation(operation)

        return count_deleted, message

    async def delete_by_tags(self, tags: List[str]) -> Tuple[int, str]:
        """
        Delete memories matching ANY of the given tags from primary storage and queue for secondary sync.

        Optimized to use primary storage's delete_by_tags if available, otherwise falls back to
        calling delete_by_tag for each tag.
        """
        if not tags:
            return 0, "No tags provided"

        # First, get all memories with any of these tags for sync queue
        memories_to_delete = await self.primary.search_by_tags(tags, operation="OR")

        # Remove duplicates based on content_hash
        unique_memories = {m.content_hash: m for m in memories_to_delete}.values()

        # Delete from primary using optimized method if available
        count_deleted, message = await self.primary.delete_by_tags(tags)

        # Queue individual deletes for secondary sync
        if count_deleted > 0 and self.sync_service:
            for memory in unique_memories:
                operation = SyncOperation(operation='delete', content_hash=memory.content_hash)
                await self.sync_service.enqueue_operation(operation)

        return count_deleted, message

    async def delete_by_all_tags(self, tags: List[str]) -> Tuple[int, str]:
        """
        Delete memories matching ALL of the given tags from primary storage and queue for secondary sync.
        """
        if not tags:
            return 0, "No tags provided"

        # First, get all memories with ALL of these tags for sync queue
        memories_to_delete = await self.primary.search_by_tags(tags, operation="AND")

        # Delete from primary using delete_by_all_tags
        count_deleted, message = await self.primary.delete_by_all_tags(tags)

        # Queue individual deletes for secondary sync
        if count_deleted > 0 and self.sync_service:
            for memory in memories_to_delete:
                operation = SyncOperation(operation='delete', content_hash=memory.content_hash)
                await self.sync_service.enqueue_operation(operation)

        return count_deleted, message

    async def cleanup_duplicates(self) -> Tuple[int, str]:
        """Clean up duplicates in primary storage."""
        # Only cleanup primary, secondary will sync naturally
        return await self.primary.cleanup_duplicates()

    async def update_memory_metadata(self, content_hash: str, updates: Dict[str, Any], preserve_timestamps: bool = True) -> Tuple[bool, str]:
        """
        Update memory metadata with leader-based routing.

        Leader: Updates SQLite, queues Cloudflare sync
        Follower: Updates Cloudflare only
        """
        is_leader = self.leader_election.is_leader if self.leader_election else True

        if is_leader:
            # Leader: Update SQLite
            success, message = await self.primary.update_memory_metadata(content_hash, updates, preserve_timestamps)

            if success and self.sync_service:
                # Queue for background sync to secondary
                operation = SyncOperation(
                    operation='update',
                    content_hash=content_hash,
                    updates=updates
                )
                await self.sync_service.enqueue_operation(operation)

            return success, message
        else:
            # Follower: Update Cloudflare only
            if self.secondary:
                success, message = await self.secondary.update_memory_metadata(content_hash, updates, preserve_timestamps)
                if success:
                    logger.debug(f"Follower updated memory in Cloudflare: {content_hash[:8]}")
                return success, message
            else:
                return False, "Follower mode requires Cloudflare backend"

    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from both storage backends."""
        # SQLite-vec get_stats is now async
        primary_stats = await self.primary.get_stats()

        stats = {
            "storage_backend": "Hybrid (SQLite-vec + Cloudflare)",
            "primary_backend": "SQLite-vec",
            "secondary_backend": "Cloudflare" if self.secondary else "None",
            "total_memories": primary_stats.get("total_memories", 0),
            "unique_tags": primary_stats.get("unique_tags", 0),
            "memories_this_week": primary_stats.get("memories_this_week", 0),
            "database_size_bytes": primary_stats.get("database_size_bytes", 0),
            "database_size_mb": primary_stats.get("database_size_mb", 0),
            "primary_stats": primary_stats,
            "sync_enabled": self.sync_service is not None
        }

        # Add leader election status
        if self.leader_election:
            heartbeat = self.leader_election._read_heartbeat()
            stats["leader_election"] = {
                "enabled": True,
                "is_leader": self.leader_election.is_leader,
                "process_id": self.leader_election.process_id,
                "leader_process_id": heartbeat[1] if heartbeat else "unknown",
                "leader_heartbeat_age_seconds": round(time.time() - heartbeat[0], 1) if heartbeat else None,
                "leader_stale_threshold": HYBRID_LEADER_STALE_THRESHOLD
            }
        else:
            stats["leader_election"] = {"enabled": False}

        # Add sync service statistics if available
        if self.sync_service:
            sync_status = await self.sync_service.get_sync_status()
            stats["sync_status"] = sync_status

            # Add adaptive sync info
            if self.sync_service.adaptive_sync_enabled:
                current_interval = self.sync_service.get_current_sync_interval()
                time_since_write = time.time() - self.sync_service.last_write_time
                stats["adaptive_sync"] = {
                    "enabled": True,
                    "current_interval": current_interval,
                    "is_active": current_interval == self.sync_service.sync_active_interval,
                    "time_since_last_write": round(time_since_write, 1),
                    "idle_threshold": self.sync_service.idle_threshold
                }

        # Add secondary stats if available and healthy
        if self.secondary and self.sync_service and self.sync_service.sync_stats['cloudflare_available']:
            try:
                secondary_stats = await self.secondary.get_stats()
                stats["secondary_stats"] = secondary_stats
            except Exception as e:
                stats["secondary_error"] = str(e)

        return stats

    async def get_all_tags_with_counts(self) -> List[Dict[str, Any]]:
        """Get all tags with their usage counts from primary storage."""
        return await self.primary.get_all_tags_with_counts()

    async def get_all_tags(self) -> List[str]:
        """Get all unique tags from primary storage."""
        return await self.primary.get_all_tags()

    async def get_recent_memories(self, n: int = 10) -> List[Memory]:
        """Get recent memories from primary storage."""
        return await self.primary.get_recent_memories(n)

    async def get_largest_memories(self, n: int = 10) -> List[Memory]:
        """Get largest memories by content length from primary storage."""
        return await self.primary.get_largest_memories(n)

    async def recall(self, query: Optional[str] = None, n_results: int = 5, start_timestamp: Optional[float] = None, end_timestamp: Optional[float] = None, offset: int = 0) -> List[MemoryQueryResult]:
        """
        Retrieve memories with combined time filtering and optional semantic search.

        Args:
            query: Optional semantic search query. If None, only time filtering is applied.
            n_results: Maximum number of results to return.
            start_timestamp: Optional start time for filtering.
            end_timestamp: Optional end time for filtering.
            offset: Number of results to skip for pagination (default: 0).

        Returns:
            List of MemoryQueryResult objects.
        """
        return await self.primary.recall(query=query, n_results=n_results, start_timestamp=start_timestamp, end_timestamp=end_timestamp, offset=offset)

    async def recall_memory(self, query: str, n_results: int = 5) -> List[Memory]:
        """Recall memories using natural language time expressions."""
        return await self.primary.recall_memory(query, n_results)

    async def get_all_memories(self, limit: int = None, offset: int = 0, memory_type: Optional[str] = None, tags: Optional[List[str]] = None) -> List[Memory]:
        """Get all memories from primary storage."""
        return await self.primary.get_all_memories(limit=limit, offset=offset, memory_type=memory_type, tags=tags)

    async def count_all_memories(self, memory_type: Optional[str] = None, tags: Optional[List[str]] = None) -> int:
        """Get total count of memories from primary storage."""
        return await self.primary.count_all_memories(memory_type=memory_type, tags=tags)

    async def count_semantic_search(
        self,
        query: str,
        tags: Optional[List[str]] = None,
        memory_type: Optional[str] = None,
        min_similarity: Optional[float] = None
    ) -> int:
        """Count memories matching semantic search criteria in primary storage."""
        return await self.primary.count_semantic_search(query, tags, memory_type, min_similarity)

    async def count_tag_search(
        self,
        tags: List[str],
        match_all: bool = False,
        start_timestamp: Optional[float] = None,
        end_timestamp: Optional[float] = None
    ) -> int:
        """Count memories matching tag search in primary storage."""
        return await self.primary.count_tag_search(tags, match_all, start_timestamp, end_timestamp)

    async def count_time_range(
        self,
        start_timestamp: Optional[float] = None,
        end_timestamp: Optional[float] = None,
        tags: Optional[List[str]] = None,
        memory_type: Optional[str] = None
    ) -> int:
        """Count memories within time range in primary storage."""
        return await self.primary.count_time_range(start_timestamp, end_timestamp, tags, memory_type)

    async def get_memories_by_time_range(self, start_time: float, end_time: float) -> List[Memory]:
        """Get memories within time range from primary storage."""
        return await self.primary.get_memories_by_time_range(start_time, end_time)

    async def close(self):
        """Clean shutdown of hybrid storage system."""
        logger.info("Shutting down hybrid memory storage...")

        # Stop leader election first
        if self.leader_election:
            await self.leader_election.stop()

        # Stop sync service
        if self.sync_service:
            await self.sync_service.stop()

        # Close storage backends
        if hasattr(self.primary, 'close') and self.primary.close:
            if asyncio.iscoroutinefunction(self.primary.close):
                await self.primary.close()
            else:
                self.primary.close()

        if self.secondary and hasattr(self.secondary, 'close') and self.secondary.close:
            if asyncio.iscoroutinefunction(self.secondary.close):
                await self.secondary.close()
            else:
                self.secondary.close()

        logger.info("Hybrid memory storage shutdown completed")

    async def force_sync(self) -> Dict[str, Any]:
        """Force immediate synchronization with secondary storage."""
        if not self.sync_service:
            return {
                'status': 'disabled',
                'message': 'Background sync service not available'
            }

        return await self.sync_service.force_sync()

    async def get_sync_status(self) -> Dict[str, Any]:
        """Get current background sync status and statistics."""
        if not self.sync_service:
            return {
                'is_running': False,
                'pending_operations': 0,
                'operations_processed': 0,
                'operations_failed': 0,
                'last_sync_time': 0,
                'sync_interval': 0
            }

        return await self.sync_service.get_sync_status()

    def sanitized(self, tags):
        """Sanitize and normalize tags to a JSON string.

        This method provides compatibility with the storage interface.
        Delegates to primary storage for consistent tag handling.
        """
        return self.primary.sanitized(tags)
