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
Automatic backup scheduler for MCP Memory Service.

Provides scheduled database backups with configurable intervals and retention policies.
"""

import asyncio
import shutil
import sqlite3
import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

from ..config import (
    BACKUPS_PATH,
    BACKUP_ENABLED,
    BACKUP_INTERVAL,
    BACKUP_RETENTION,
    BACKUP_MAX_COUNT,
    SQLITE_VEC_PATH
)

logger = logging.getLogger(__name__)


class BackupService:
    """Service for creating and managing database backups."""

    def __init__(self, backups_dir: str = None, db_path: str = None):
        """Initialize backup service.

        Args:
            backups_dir: Directory to store backups (defaults to BACKUPS_PATH)
            db_path: Path to database file (defaults to SQLITE_VEC_PATH)
        """
        self.backups_dir = Path(backups_dir or BACKUPS_PATH)
        # Determine database path with clear fallback logic
        db_path_str = db_path or SQLITE_VEC_PATH
        self.db_path = Path(db_path_str) if db_path_str else None
        self.last_backup_time: Optional[float] = None
        self.backup_count: int = 0
        self._lock = asyncio.Lock()  # Ensure thread-safe operations

        # Ensure backup directory exists
        self.backups_dir.mkdir(parents=True, exist_ok=True)

        # Load existing backup metadata
        self._load_backup_metadata()

        logger.info(f"BackupService initialized: backups_dir={self.backups_dir}, db_path={self.db_path}")

    def _load_backup_metadata(self):
        """Load metadata about existing backups."""
        backups = self.list_backups()
        self.backup_count = len(backups)
        if backups:
            # Get most recent backup time
            latest = backups[0]
            self.last_backup_time = latest.get('created_timestamp', 0)

    def _generate_backup_filename(self) -> str:
        """Generate a timestamped backup filename."""
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        return f"memory_backup_{timestamp}.db"

    async def create_backup(self, description: str = None) -> Dict[str, Any]:
        """Create a new database backup.

        Args:
            description: Optional description for the backup

        Returns:
            Dict with backup details
        """
        if not self.db_path or not self.db_path.exists():
            return {
                'success': False,
                'error': f'Database file not found: {self.db_path}',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

        async with self._lock:  # Ensure atomic operations
            try:
                start_time = time.time()
                created_at = datetime.now(timezone.utc)

                # Generate backup filename
                backup_filename = self._generate_backup_filename()
                backup_path = self.backups_dir / backup_filename

                # Use SQLite's native backup API for safe, consistent backups
                # This handles active database connections properly
                def _do_backup():
                    source = sqlite3.connect(str(self.db_path))
                    dest = sqlite3.connect(str(backup_path))
                    try:
                        source.backup(dest)
                    finally:
                        source.close()
                        dest.close()

                await asyncio.to_thread(_do_backup)

                # Calculate backup duration (just the backup operation)
                backup_duration = time.time() - start_time

                # Get backup size
                backup_size = backup_path.stat().st_size

                # Update metadata
                self.last_backup_time = created_at.timestamp()
                self.backup_count += 1

                logger.info(f"Created backup: {backup_filename} ({backup_size} bytes) in {backup_duration:.2f}s")

                # Cleanup old backups (outside of duration calculation)
                await self.cleanup_old_backups()

                return {
                    'success': True,
                    'filename': backup_filename,
                    'path': str(backup_path),
                    'size_bytes': backup_size,
                    'description': description,
                    'created_at': created_at.isoformat(),
                    'duration_seconds': round(backup_duration, 3)
                }

            except Exception as e:
                logger.error(f"Failed to create backup: {e}")
                return {
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }

    def list_backups(self) -> List[Dict[str, Any]]:
        """List all available backups.

        Returns:
            List of backup info dicts, sorted by date (newest first)
        """
        backups = []

        try:
            for backup_file in self.backups_dir.glob('memory_backup_*.db'):
                stat = backup_file.stat()

                # Parse timestamp from filename
                try:
                    timestamp_str = backup_file.stem.replace('memory_backup_', '')
                    created_dt = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                    created_dt = created_dt.replace(tzinfo=timezone.utc)
                except ValueError:
                    created_dt = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)

                backups.append({
                    'filename': backup_file.name,
                    'path': str(backup_file),
                    'size_bytes': stat.st_size,
                    'created_at': created_dt.isoformat(),
                    'created_timestamp': created_dt.timestamp(),
                    'age_days': (datetime.now(timezone.utc) - created_dt).days
                })

            # Sort by creation time, newest first
            backups.sort(key=lambda x: x['created_timestamp'], reverse=True)

        except Exception as e:
            logger.error(f"Error listing backups: {e}")

        return backups

    async def cleanup_old_backups(self) -> Dict[str, Any]:
        """Remove old backups based on retention policy.

        Removes backups that are:
        - Older than BACKUP_RETENTION days
        - Exceed BACKUP_MAX_COUNT

        Returns:
            Dict with cleanup results
        """
        removed = []
        errors = []

        try:
            backups = self.list_backups()
            retention_cutoff = datetime.now(timezone.utc) - timedelta(days=BACKUP_RETENTION)

            for i, backup in enumerate(backups):
                should_remove = False
                reason = ""

                # Check if exceeds max count
                if i >= BACKUP_MAX_COUNT:
                    should_remove = True
                    reason = f"exceeds max count ({BACKUP_MAX_COUNT})"

                # Check if older than retention period
                try:
                    created_dt = datetime.fromisoformat(backup['created_at'].replace('Z', '+00:00'))
                    if created_dt < retention_cutoff:
                        should_remove = True
                        reason = f"older than {BACKUP_RETENTION} days"
                except (ValueError, KeyError) as e:
                    logger.warning(f"Could not parse timestamp for backup {backup.get('filename', 'unknown')}: {e}")

                if should_remove:
                    try:
                        # Use asyncio.to_thread to avoid blocking the event loop
                        await asyncio.to_thread(Path(backup['path']).unlink)
                        removed.append({
                            'filename': backup['filename'],
                            'reason': reason
                        })
                        logger.info(f"Removed old backup: {backup['filename']} ({reason})")
                    except Exception as e:
                        errors.append({
                            'filename': backup['filename'],
                            'error': str(e)
                        })
                        logger.error(f"Failed to remove backup {backup['filename']}: {e}")

            # Update count more efficiently by subtracting removed count
            self.backup_count = max(0, self.backup_count - len(removed))

        except Exception as e:
            logger.error(f"Error during backup cleanup: {e}")
            errors.append({'error': str(e)})

        return {
            'removed_count': len(removed),
            'removed': removed,
            'errors': errors
        }

    async def restore_backup(self, filename: str) -> Dict[str, Any]:
        """Restore database from a backup.

        Args:
            filename: Name of backup file to restore

        Returns:
            Dict with restore results
        """
        backup_path = self.backups_dir / filename

        if not backup_path.exists():
            return {
                'success': False,
                'error': f'Backup file not found: {filename}'
            }

        if not self.db_path:
            return {
                'success': False,
                'error': 'Database path not configured'
            }

        try:
            # Create a backup of current database first
            if self.db_path.exists():
                current_backup = self.db_path.with_suffix('.db.pre_restore')
                # Use asyncio.to_thread to avoid blocking the event loop
                await asyncio.to_thread(shutil.copy2, str(self.db_path), str(current_backup))
                logger.info(f"Created pre-restore backup: {current_backup}")

            # Restore from backup
            # Use asyncio.to_thread to avoid blocking the event loop
            await asyncio.to_thread(shutil.copy2, str(backup_path), str(self.db_path))

            logger.info(f"Restored database from backup: {filename}")

            return {
                'success': True,
                'filename': filename,
                'restored_at': datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to restore backup: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def get_status(self) -> Dict[str, Any]:
        """Get current backup service status.

        Returns:
            Dict with backup service status
        """
        backups = self.list_backups()
        total_size = sum(b['size_bytes'] for b in backups)

        # Calculate time since last backup
        time_since_last = None
        if self.last_backup_time:
            time_since_last = time.time() - self.last_backup_time

        # Calculate next scheduled backup time
        next_backup = self._calculate_next_backup_time()

        return {
            'enabled': BACKUP_ENABLED,
            'interval': BACKUP_INTERVAL,
            'retention_days': BACKUP_RETENTION,
            'max_count': BACKUP_MAX_COUNT,
            'backup_count': len(backups),
            'total_size_bytes': total_size,
            'last_backup_time': self.last_backup_time,
            'time_since_last_seconds': time_since_last,
            'next_backup_at': next_backup.isoformat() if next_backup else None,
            'backups_dir': str(self.backups_dir),
            'db_path': str(self.db_path) if self.db_path else None
        }

    def _calculate_next_backup_time(self) -> Optional[datetime]:
        """Calculate the next scheduled backup time."""
        if not BACKUP_ENABLED or not self.last_backup_time:
            return None

        last_backup_dt = datetime.fromtimestamp(self.last_backup_time, tz=timezone.utc)

        if BACKUP_INTERVAL == 'hourly':
            return last_backup_dt + timedelta(hours=1)
        elif BACKUP_INTERVAL == 'daily':
            return last_backup_dt + timedelta(days=1)
        elif BACKUP_INTERVAL == 'weekly':
            return last_backup_dt + timedelta(weeks=1)

        return None


class BackupScheduler:
    """Scheduler for automatic database backups."""

    def __init__(self, backup_service: BackupService = None):
        """Initialize backup scheduler.

        Args:
            backup_service: BackupService instance (creates one if not provided)
        """
        self.backup_service = backup_service or BackupService()
        self.is_running = False
        self._task: Optional[asyncio.Task] = None

        logger.info("BackupScheduler initialized")

    def _get_interval_seconds(self) -> int:
        """Get backup interval in seconds."""
        if BACKUP_INTERVAL == 'hourly':
            return 3600
        elif BACKUP_INTERVAL == 'daily':
            return 86400
        elif BACKUP_INTERVAL == 'weekly':
            return 604800
        return 86400  # Default to daily

    async def start(self):
        """Start the backup scheduler."""
        if self.is_running:
            logger.warning("BackupScheduler already running")
            return

        if not BACKUP_ENABLED:
            logger.info("Backups disabled, scheduler not started")
            return

        self.is_running = True
        self._task = asyncio.create_task(self._schedule_loop())
        logger.info(f"BackupScheduler started with {BACKUP_INTERVAL} interval")

    async def stop(self):
        """Stop the backup scheduler."""
        if not self.is_running:
            return

        self.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info("BackupScheduler stopped")

    async def _schedule_loop(self):
        """Main scheduling loop."""
        interval_seconds = self._get_interval_seconds()

        while self.is_running:
            try:
                # Check if it's time for a backup
                should_backup = False

                if not self.backup_service.last_backup_time:
                    # No previous backup, create one
                    should_backup = True
                else:
                    time_since_last = time.time() - self.backup_service.last_backup_time
                    if time_since_last >= interval_seconds:
                        should_backup = True

                if should_backup:
                    logger.info("Scheduled backup triggered")
                    result = await self.backup_service.create_backup(
                        description=f"Scheduled {BACKUP_INTERVAL} backup"
                    )
                    if result['success']:
                        logger.info(f"Scheduled backup completed: {result['filename']}")
                    else:
                        logger.error(f"Scheduled backup failed: {result.get('error')}")

                # Sleep for a check interval (every 5 minutes)
                await asyncio.sleep(300)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in backup scheduler loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status.

        Returns:
            Dict with scheduler status and backup service status
        """
        status = self.backup_service.get_status()
        status['scheduler_running'] = self.is_running
        return status


# Global backup service instance
_backup_service: Optional[BackupService] = None
_backup_scheduler: Optional[BackupScheduler] = None


def get_backup_service() -> BackupService:
    """Get or create the global backup service instance."""
    global _backup_service
    if _backup_service is None:
        _backup_service = BackupService()
    return _backup_service


def get_backup_scheduler() -> BackupScheduler:
    """Get or create the global backup scheduler instance."""
    global _backup_scheduler
    if _backup_scheduler is None:
        _backup_scheduler = BackupScheduler(get_backup_service())
    return _backup_scheduler
