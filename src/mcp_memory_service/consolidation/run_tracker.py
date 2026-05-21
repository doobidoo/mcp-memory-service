"""Persistent run tracker for incremental consolidation."""

import asyncio
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class RunTracker:
    """Tracks consolidation run timestamps in a separate SQLite database.

    Schema: consolidation_runs(horizon TEXT PK, last_run_at TEXT, items_processed INTEGER, status TEXT)
    """

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._lock = asyncio.Lock()
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS consolidation_runs (
                    horizon TEXT PRIMARY KEY,
                    last_run_at TEXT NOT NULL,
                    items_processed INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'success'
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
        """Non-blocking concurrency guard. Returns True if acquired."""
        return self._lock.locked() is False
