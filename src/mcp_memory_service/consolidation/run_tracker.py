"""Persistent run tracker for incremental consolidation.

Note: Synchronous SQLite is intentional here — operations are single-row
on a <1KB table, completing in <1ms. asyncio.to_thread overhead would exceed
the actual I/O time.
"""

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class RunTracker:
    """Track consolidation run timestamps per horizon.

    Note: Uses synchronous SQLite intentionally. The consolidation_runs table
    has at most 6 rows (one per horizon). All operations are single-row lookups
    completing in <1ms — asyncio.to_thread() overhead would exceed actual I/O time.

    Schema: consolidation_runs(horizon TEXT PK, last_run_at TEXT,
            items_processed INTEGER, status TEXT, locked INTEGER)
    """

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS consolidation_runs (
                    horizon TEXT PRIMARY KEY,
                    last_run_at TEXT NOT NULL DEFAULT '',
                    items_processed INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'success',
                    locked INTEGER NOT NULL DEFAULT 0
                )
            """)
            conn.commit()
        finally:
            conn.close()

    async def get_last_run_at(self, horizon: str) -> Optional[float]:
        """Return last_run_at as unix timestamp, or None if never run."""
        conn = sqlite3.connect(str(self._db_path))
        try:
            row = conn.execute(
                "SELECT last_run_at FROM consolidation_runs WHERE horizon = ?",
                (horizon,),
            ).fetchone()
            if row and row[0]:
                return datetime.fromisoformat(row[0]).timestamp()
            return None
        finally:
            conn.close()

    async def record_run(
        self, horizon: str, items_processed: int, status: str = "success"
    ) -> None:
        """Record a consolidation run (upsert)."""
        now_iso = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.execute(
                """
                INSERT INTO consolidation_runs (horizon, last_run_at, items_processed, status)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(horizon) DO UPDATE SET
                    last_run_at = excluded.last_run_at,
                    items_processed = excluded.items_processed,
                    status = excluded.status
                """,
                (horizon, now_iso, items_processed, status),
            )
            conn.commit()
        finally:
            conn.close()

    def try_acquire(self, horizon: str) -> bool:
        """Atomic test-and-set lock. Returns True if acquired."""
        with sqlite3.connect(str(self._db_path), timeout=5) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT locked FROM consolidation_runs WHERE horizon = ?",
                (horizon,),
            ).fetchone()
            if row and row[0]:
                conn.rollback()
                return False
            conn.execute(
                "INSERT INTO consolidation_runs (horizon, last_run_at, items_processed, status, locked) "
                "VALUES (?, '', 0, 'running', 1) "
                "ON CONFLICT(horizon) DO UPDATE SET locked = 1, status = 'running'",
                (horizon,),
            )
            conn.commit()
            return True

    def release(self, horizon: str) -> None:
        """Release the lock for a horizon."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "UPDATE consolidation_runs SET locked = 0 WHERE horizon = ?",
                (horizon,),
            )
