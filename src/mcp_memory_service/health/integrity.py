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
SQLite database integrity monitoring.

Runs periodic PRAGMA integrity_check to detect corruption early,
before it causes data loss. Attempts automatic WAL checkpoint repair
for minor corruption, and exports surviving memories when repair fails.

Performance characteristics:
  - PRAGMA integrity_check: ~3.5ms on a 20-memory database
  - WAL checkpoint repair: ~5ms
  - Healthy check overhead at 30-min interval: 0.0002% of wall time
"""

import asyncio
import json
import logging
import os
import sqlite3
import time
from typing import Any, Dict, Optional

from ..config import (
    INTEGRITY_CHECK_ENABLED,
    INTEGRITY_CHECK_INTERVAL,
)

logger = logging.getLogger(__name__)


class IntegrityMonitor:
    """Periodic SQLite integrity monitor for early corruption detection.

    Runs PRAGMA integrity_check at a configurable interval. On corruption,
    attempts WAL checkpoint repair before escalating to a warning with
    memory export for manual recovery.
    """

    def __init__(self, db_path: str):
        """Initialize integrity monitor.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self.is_running = False
        self._task: Optional[asyncio.Task] = None

        # Monitoring state
        self.last_check_time: Optional[float] = None
        self.last_check_healthy: Optional[bool] = None
        self.total_checks = 0
        self.total_repairs = 0
        self.total_failures = 0

        logger.info(
            f"IntegrityMonitor initialized "
            f"(enabled={INTEGRITY_CHECK_ENABLED}, "
            f"interval={INTEGRITY_CHECK_INTERVAL}s)"
        )

    async def check_integrity(self) -> tuple[bool, str]:
        """Run PRAGMA integrity_check on the database.

        Uses a separate short-lived connection to avoid interfering with
        the service's main connection. Runs in a thread to avoid blocking
        the asyncio event loop.

        Returns:
            Tuple of (is_healthy, detail_message).
        """
        def _check():
            try:
                conn = sqlite3.connect(self.db_path, timeout=5)
                try:
                    result = conn.execute("PRAGMA integrity_check").fetchone()
                finally:
                    conn.close()

                is_healthy = result is not None and result[0] == "ok"
                detail = result[0] if result else "no result"
                return is_healthy, detail

            except Exception as e:
                return False, f"connection error: {e}"

        return await asyncio.to_thread(_check)

    async def attempt_wal_repair(self) -> tuple[bool, str]:
        """Attempt to repair corruption via WAL checkpoint.

        PRAGMA wal_checkpoint(TRUNCATE) flushes the WAL file into the
        main database and truncates it. This fixes most WAL-related
        corruption caused by interrupted writes.

        Returns:
            Tuple of (repair_succeeded, detail_message).
        """
        def _repair():
            try:
                conn = sqlite3.connect(self.db_path, timeout=10)
                try:
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                finally:
                    conn.close()
                return True, ""
            except Exception as e:
                return False, f"WAL checkpoint failed: {e}"

        repaired, detail = await asyncio.to_thread(_repair)
        if not repaired:
            return False, detail

        # Verify repair worked
        is_healthy, detail = await self.check_integrity()
        if is_healthy:
            return True, "WAL checkpoint repair successful"
        else:
            return False, f"WAL checkpoint did not fix corruption: {detail}"

    async def export_memories(self, export_path: str) -> tuple[bool, int]:
        """Export surviving memories to JSON for manual recovery.

        Args:
            export_path: File path to write the JSON export.

        Returns:
            Tuple of (success, memory_count).
        """
        def _export():
            try:
                conn = sqlite3.connect(self.db_path, timeout=10)
                try:
                    rows = conn.execute(
                        "SELECT content_hash, content, created_at, metadata, tags, type "
                        "FROM memories"
                    ).fetchall()
                finally:
                    conn.close()

                memories = []
                for r in rows:
                    memories.append({
                        "hash": r[0],
                        "content": r[1],
                        "created_at": r[2],
                        "metadata": r[3],
                        "tags": r[4] or "",
                        "type": r[5] or "note",
                    })

                with open(export_path, "w") as f:
                    json.dump(memories, f, indent=2)

                logger.info(f"Exported {len(memories)} memories to {export_path}")
                return True, len(memories)

            except Exception as e:
                logger.error(f"Memory export failed: {e}")
                return False, 0

        return await asyncio.to_thread(_export)

    async def run_check(self) -> Dict[str, Any]:
        """Run a single integrity check with repair attempt on failure.

        Returns:
            Dict with check results.
        """
        start = time.perf_counter()
        is_healthy, detail = await self.check_integrity()
        check_ms = (time.perf_counter() - start) * 1000

        self.last_check_time = time.time()
        self.last_check_healthy = is_healthy
        self.total_checks += 1

        result = {
            "healthy": is_healthy,
            "detail": detail,
            "check_ms": round(check_ms, 1),
            "repaired": False,
            "exported": False,
        }

        if is_healthy:
            logger.debug(f"Integrity check passed ({check_ms:.1f}ms)")
            return result

        # Corruption detected — attempt repair
        logger.warning(f"Database corruption detected: {detail}")

        repaired, repair_detail = await self.attempt_wal_repair()
        result["repair_detail"] = repair_detail

        if repaired:
            self.total_repairs += 1
            result["repaired"] = True
            result["healthy"] = True
            logger.info(f"Auto-repair successful: {repair_detail}")
            return result

        # Repair failed — export memories for manual recovery
        self.total_failures += 1
        export_path = os.path.join(
            os.path.dirname(self.db_path),
            f"emergency_export_{int(time.time())}.json",
        )
        success, count = await self.export_memories(export_path)
        if success:
            result["exported"] = True
            result["export_path"] = export_path
            result["export_count"] = count

        logger.error(
            f"Database corruption could not be auto-repaired. "
            f"Memories exported to {export_path} ({count} memories). "
            f"Manual intervention required."
        )
        return result

    async def startup_check(self) -> Dict[str, Any]:
        """Run integrity check at startup before accepting requests.

        Returns:
            Dict with check results.
        """
        logger.info("Running startup integrity check...")
        result = await self.run_check()

        if result["healthy"]:
            if result["repaired"]:
                logger.info("Startup check: corruption found and auto-repaired")
            else:
                logger.info(f"Startup check: database healthy ({result['check_ms']}ms)")
        else:
            logger.error(
                "Startup check: database corrupt and could not be auto-repaired. "
                "The service will start but may encounter errors."
            )

        return result

    async def start(self):
        """Start the periodic integrity monitoring loop."""
        if self.is_running:
            logger.warning("IntegrityMonitor already running")
            return

        if not INTEGRITY_CHECK_ENABLED:
            logger.info("Integrity monitoring disabled, scheduler not started")
            return

        self.is_running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info(
            f"IntegrityMonitor started "
            f"(interval={INTEGRITY_CHECK_INTERVAL}s)"
        )

    async def stop(self):
        """Stop the periodic integrity monitoring loop."""
        if not self.is_running:
            return

        self.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass  # Expected when stop() cancels the monitoring task

        logger.info("IntegrityMonitor stopped")

    async def _monitor_loop(self):
        """Main monitoring loop — runs integrity check at configured interval."""
        while self.is_running:
            try:
                await asyncio.sleep(INTEGRITY_CHECK_INTERVAL)
                await self.run_check()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in integrity monitor loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    def get_status(self) -> Dict[str, Any]:
        """Get monitor status for health reporting.

        Returns:
            Dict with monitoring statistics.
        """
        return {
            "enabled": INTEGRITY_CHECK_ENABLED,
            "running": self.is_running,
            "interval_seconds": INTEGRITY_CHECK_INTERVAL,
            "last_check_time": self.last_check_time,
            "last_check_healthy": self.last_check_healthy,
            "total_checks": self.total_checks,
            "total_auto_repairs": self.total_repairs,
            "total_unrecoverable": self.total_failures,
        }
