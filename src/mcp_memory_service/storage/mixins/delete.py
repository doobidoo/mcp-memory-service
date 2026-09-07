"""DeleteMixin: delete, soft-delete, purge, cleanup operations."""

import sqlite3
import logging
import time
from datetime import datetime, date, timezone
from typing import List, Tuple, Optional


logger = logging.getLogger(__name__)


def _sanitize_log_value(value: object) -> str:
    return str(value).replace("\n", "\\n").replace("\r", "\\r").replace("\x1b", "\\x1b")


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class DeleteMixin:
    """Mixin providing memory deletion operations."""

    async def delete(self, content_hash: str) -> Tuple[bool, str]:
        """Soft-delete a memory by setting deleted_at timestamp."""
        try:
            if not self.conn:
                return False, "Database not initialized"

            def _delete_memory():
                cursor = self.conn.execute(
                    'SELECT id FROM memories WHERE content_hash = ? AND deleted_at IS NULL',
                    (content_hash,)
                )
                row = cursor.fetchone()
                if not row:
                    return None
                memory_id = row[0]
                try:
                    self.conn.execute('DELETE FROM memory_embeddings WHERE rowid = ?', (memory_id,))
                except Exception as vec_err:
                    logger.warning(
                        "Could not delete embedding for memory %s (corrupted blob?): %s — "
                        "proceeding with soft-delete; run purge_deleted() to clean orphan.",
                        content_hash, vec_err,
                    )
                self.conn.execute(
                    'DELETE FROM memory_graph WHERE source_hash = ? OR target_hash = ?',
                    (content_hash, content_hash)
                )
                cursor = self.conn.execute(
                    'UPDATE memories SET deleted_at = ? WHERE content_hash = ? AND deleted_at IS NULL',
                    (time.time(), content_hash)
                )
                self.conn.commit()
                return cursor.rowcount

            rowcount = await self._execute_with_retry(_delete_memory)
            if rowcount is None:
                return False, f"Memory with hash {content_hash} not found"
            if rowcount > 0:
                logger.info(f"Soft-deleted memory: {content_hash}")
                return True, f"Successfully deleted memory {content_hash}"
            else:
                return False, f"Memory with hash {content_hash} not found"

        except Exception as e:
            try:
                self.conn.rollback()
            except sqlite3.OperationalError:
                pass
            error_msg = f"Failed to delete memory: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    # ── Consolidation Protocol Proxy Methods ──────────────────────────
    # These methods are required by the DreamInspiredConsolidator's
    # StorageProtocol but were missing from this mixin, causing
    # consolidation forgetting/archival to fail silently.
    # See the project issue tracker for related consolidation bug details.

    async def delete_memory(self, content_hash: str) -> bool:
        """Delete a memory by content hash (consolidation protocol).

        Delegates to delete() to avoid duplicating sync logic.
        """
        success, _ = await self.delete(content_hash)
        return success

    async def is_deleted(self, content_hash: str) -> bool:
        """Check if a memory has been soft-deleted (tombstone exists)."""
        try:
            if not self.conn:
                return False

            def _check_deleted():
                cursor = self.conn.execute(
                    'SELECT deleted_at FROM memories WHERE content_hash = ? AND deleted_at IS NOT NULL',
                    (content_hash,)
                )
                return cursor.fetchone() is not None
            return await self._execute_with_retry(_check_deleted)

        except Exception as e:
            logger.error(f"Failed to check if memory is deleted: {str(e)}")
            return False

    async def purge_deleted(self, older_than_days: int = 30) -> int:
        """Permanently delete tombstones older than specified days."""
        try:
            if not self.conn:
                return 0

            cutoff = time.time() - (older_than_days * 86400)

            def _purge():
                # Drop embeddings for the rows about to be hard-deleted. Skipping this
                # leaves orphan embeddings whose rowids SQLite later reuses, causing
                # "UNIQUE constraint failed: memory_embeddings" on the next store.
                rows = self.conn.execute(
                    'SELECT id FROM memories WHERE deleted_at IS NOT NULL AND deleted_at < ?',
                    (cutoff,)
                ).fetchall()
                ids = [r[0] for r in rows]
                if ids:
                    placeholders = ','.join('?' for _ in ids)
                    try:
                        self.conn.execute(
                            f'DELETE FROM memory_embeddings WHERE rowid IN ({placeholders})', ids
                        )
                    except Exception as vec_err:
                        logger.warning(
                            "Batch embedding purge failed (%s) — retrying per-row.", vec_err
                        )
                        for rid in ids:
                            try:
                                self.conn.execute(
                                    'DELETE FROM memory_embeddings WHERE rowid = ?', (rid,)
                                )
                            except Exception as row_err:
                                logger.warning(
                                    "Could not delete embedding rowid=%s during purge: %s",
                                    rid, row_err,
                                )
                cursor = self.conn.execute(
                    'DELETE FROM memories WHERE deleted_at IS NOT NULL AND deleted_at < ?',
                    (cutoff,)
                )
                self.conn.commit()
                return cursor.rowcount

            count = await self._execute_with_retry(_purge)
            if count > 0:
                logger.info(f"Purged {count} tombstones older than {older_than_days} days")
            return count

        except Exception as e:
            logger.error(f"Failed to purge deleted memories: {str(e)}")
            return 0

    async def delete_by_tag(self, tag: str) -> Tuple[int, str]:
        """Soft-delete memories by tag (exact match only)."""
        try:
            if not self.conn:
                return 0, "Database not initialized"

            stripped_tag = tag.strip()
            exact_match_pattern = f"%,{_escape_like(stripped_tag)},%"

            def _delete_by_tag():
                cursor = self.conn.execute(
                    "SELECT id, content_hash FROM memories WHERE (',' || REPLACE(tags, ' ', '') || ',') LIKE ? ESCAPE '\\' AND deleted_at IS NULL",
                    (exact_match_pattern,)
                )
                rows = cursor.fetchall()
                memory_ids = [row[0] for row in rows]
                content_hashes = [row[1] for row in rows]

                for memory_id in memory_ids:
                    try:
                        self.conn.execute('DELETE FROM memory_embeddings WHERE rowid = ?', (memory_id,))
                    except Exception as vec_err:
                        logger.warning(
                            "Could not delete embedding rowid=%s (corrupted blob?): %s — "
                            "proceeding with soft-delete.",
                            memory_id, vec_err,
                        )

                for ch in content_hashes:
                    self.conn.execute(
                        'DELETE FROM memory_graph WHERE source_hash = ? OR target_hash = ?',
                        (ch, ch)
                    )

                cursor = self.conn.execute(
                    "UPDATE memories SET deleted_at = ? WHERE (',' || REPLACE(tags, ' ', '') || ',') LIKE ? ESCAPE '\\' AND deleted_at IS NULL",
                    (time.time(), exact_match_pattern)
                )
                self.conn.commit()
                return cursor.rowcount

            count = await self._execute_with_retry(_delete_by_tag)
            logger.info(f"Soft-deleted {count} memories with tag: {_sanitize_log_value(tag)}")

            if count > 0:
                return count, f"Successfully deleted {count} memories with tag '{tag}'"
            else:
                return 0, f"No memories found with tag '{tag}'"

        except Exception as e:
            error_msg = f"Failed to delete by tag: {str(e)}"
            logger.error(error_msg)
            return 0, error_msg

    async def delete_by_tags(self, tags: List[str]) -> Tuple[int, str, List[str]]:
        """Soft-delete memories matching ANY of the given tags."""
        try:
            if not self.conn:
                return 0, "Database not initialized", []

            if not tags:
                return 0, "No tags provided", []

            stripped_tags = [tag.strip() for tag in tags]
            conditions = " OR ".join(["(',' || REPLACE(tags, ' ', '') || ',') LIKE ? ESCAPE '\\'" for _ in stripped_tags])
            params = [f"%,{_escape_like(tag)},%" for tag in stripped_tags]

            select_query = f'SELECT id, content_hash FROM memories WHERE ({conditions}) AND deleted_at IS NULL'
            update_query = f'UPDATE memories SET deleted_at = ? WHERE ({conditions}) AND deleted_at IS NULL'

            def _delete_by_tags():
                cursor = self.conn.execute(select_query, params)
                rows = cursor.fetchall()
                memory_ids = [row[0] for row in rows]
                hashes = [row[1] for row in rows]

                if memory_ids:
                    placeholders = ','.join('?' for _ in memory_ids)
                    try:
                        self.conn.execute(f'DELETE FROM memory_embeddings WHERE rowid IN ({placeholders})', memory_ids)
                    except Exception as vec_err:
                        logger.warning(
                            "Batch embedding delete failed (%s) — retrying per-row to skip corrupted blobs.",
                            vec_err,
                        )
                        for mid in memory_ids:
                            try:
                                self.conn.execute('DELETE FROM memory_embeddings WHERE rowid = ?', (mid,))
                            except Exception as row_err:
                                logger.warning("Could not delete embedding rowid=%s (corrupted blob?): %s", mid, row_err)

                for ch in hashes:
                    self.conn.execute(
                        'DELETE FROM memory_graph WHERE source_hash = ? OR target_hash = ?',
                        (ch, ch)
                    )

                cursor = self.conn.execute(update_query, [time.time()] + params)
                self.conn.commit()
                return cursor.rowcount, hashes

            count, deleted_hashes = await self._execute_with_retry(_delete_by_tags)
            logger.info(f"Soft-deleted {count} memories matching tags: {tags}")

            if count > 0:
                return count, f"Successfully deleted {count} memories matching {len(tags)} tag(s)", deleted_hashes
            else:
                return 0, f"No memories found matching any of the {len(tags)} tags", []

        except Exception as e:
            error_msg = f"Failed to delete by tags: {str(e)}"
            logger.error(error_msg)
            return 0, error_msg, []

    async def delete_by_timeframe(self, start_date: date, end_date: date, tag: Optional[str] = None) -> Tuple[int, str]:
        """Delete memories within a specific date range."""
        try:
            if not self.conn:
                return 0, "Database not initialized"

            start_ts = datetime.combine(start_date, datetime.min.time()).timestamp()
            end_ts = datetime.combine(end_date, datetime.max.time()).timestamp()

            def _select_timeframe():
                if tag:
                    stripped_tag = tag.strip()
                    cursor = self.conn.execute(
                        """
                        SELECT content_hash FROM memories
                        WHERE created_at >= ? AND created_at <= ?
                        AND (',' || REPLACE(tags, ' ', '') || ',') LIKE ? ESCAPE '\\'
                        AND deleted_at IS NULL
                    """,
                        (start_ts, end_ts, f"%,{_escape_like(stripped_tag)},%"),
                    )
                else:
                    cursor = self.conn.execute('''
                        SELECT content_hash FROM memories
                        WHERE created_at >= ? AND created_at <= ?
                        AND deleted_at IS NULL
                    ''', (start_ts, end_ts))
                return cursor.fetchall()

            hashes = [row[0] for row in await self._execute_with_retry(_select_timeframe)]

            deleted_count = 0
            for content_hash in hashes:
                success, _ = await self.delete(content_hash)
                if success:
                    deleted_count += 1

            return deleted_count, f"Deleted {deleted_count} memories from {start_date} to {end_date}" + (f" with tag '{tag}'" if tag else "")

        except Exception as e:
            logger.error(f"Error deleting by timeframe: {str(e)}")
            return 0, f"Error: {str(e)}"

    async def delete_before_date(self, before_date: date, tag: Optional[str] = None) -> Tuple[int, str]:
        """Delete memories created before a specific date."""
        try:
            if not self.conn:
                return 0, "Database not initialized"

            before_ts = datetime.combine(before_date, datetime.min.time()).timestamp()

            def _select_before_date():
                if tag:
                    stripped_tag = tag.strip()
                    cursor = self.conn.execute(
                        """
                        SELECT content_hash FROM memories
                        WHERE created_at < ?
                        AND (',' || REPLACE(tags, ' ', '') || ',') LIKE ? ESCAPE '\\'
                        AND deleted_at IS NULL
                    """,
                        (before_ts, f"%,{_escape_like(stripped_tag)},%"),
                    )
                else:
                    cursor = self.conn.execute('''
                        SELECT content_hash FROM memories
                        WHERE created_at < ?
                        AND deleted_at IS NULL
                    ''', (before_ts,))
                return cursor.fetchall()

            hashes = [row[0] for row in await self._execute_with_retry(_select_before_date)]

            deleted_count = 0
            for content_hash in hashes:
                success, _ = await self.delete(content_hash)
                if success:
                    deleted_count += 1

            return deleted_count, f"Deleted {deleted_count} memories before {before_date}" + (f" with tag '{tag}'" if tag else "")

        except Exception as e:
            logger.error(f"Error deleting before date: {str(e)}")
            return 0, f"Error: {str(e)}"

    async def cleanup_duplicates(self) -> Tuple[int, str]:
        """Soft-delete duplicate memories based on content hash."""
        try:
            if not self.conn:
                return 0, "Database not initialized"

            def _cleanup_dups():
                cursor = self.conn.execute('''
                    UPDATE memories
                    SET deleted_at = ?
                    WHERE rowid NOT IN (
                        SELECT MIN(rowid)
                        FROM memories
                        WHERE deleted_at IS NULL
                        GROUP BY content_hash
                    )
                    AND deleted_at IS NULL
                ''', (time.time(),))
                self.conn.commit()
                return cursor.rowcount

            count = await self._execute_with_retry(_cleanup_dups)
            logger.info(f"Soft-deleted {count} duplicate memories")

            if count > 0:
                return count, f"Successfully soft-deleted {count} duplicate memories"
            else:
                return 0, "No duplicate memories found"

        except Exception as e:
            error_msg = f"Failed to cleanup duplicates: {str(e)}"
            logger.error(error_msg)
            return 0, error_msg
