"""DeleteMixin: delete, soft-delete, purge, cleanup operations."""

import sqlite3
import logging
import math
import time
from datetime import datetime, date
from typing import List, Tuple, Optional


logger = logging.getLogger(__name__)


def _sanitize_log_value(value: object) -> str:
    return str(value).replace("\n", "\\n").replace("\r", "\\r").replace("\x1b", "\\x1b")


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class DeleteMixin:
    """Mixin providing memory deletion operations."""

    async def delete_if_not_updated_after(
        self, content_hash: str, deleted_at: float
    ) -> str:
        """Apply a remote tombstone unless the local memory has a newer edit.

        Returns ``applied``, ``already_deleted``, ``missing``, ``conflict``, or
        ``failed``. The timestamp predicate is repeated in the UPDATE so a write
        racing this reconciliation cannot be deleted based on a stale read.
        """
        try:
            remote_deleted_at = float(deleted_at)
            if not math.isfinite(remote_deleted_at):
                return "failed"
            if not self.conn:
                return "failed"

            def _apply_remote_tombstone():
                row = self.conn.execute(
                    'SELECT id, updated_at, deleted_at FROM memories '
                    'WHERE content_hash = ?',
                    (content_hash,),
                ).fetchone()
                if row is None:
                    return "missing"
                memory_id, updated_at, existing_deleted_at = row
                if existing_deleted_at is not None:
                    return "already_deleted"
                if updated_at is not None and float(updated_at) > remote_deleted_at:
                    return "conflict"

                cursor = self.conn.execute(
                    'UPDATE memories SET deleted_at = ? '
                    'WHERE id = ? AND deleted_at IS NULL '
                    'AND (updated_at IS NULL OR updated_at <= ?)',
                    (remote_deleted_at, memory_id, remote_deleted_at),
                )
                if cursor.rowcount == 0:
                    # Re-read to distinguish a racing edit from a disappeared row.
                    current = self.conn.execute(
                        'SELECT updated_at, deleted_at FROM memories WHERE id = ?',
                        (memory_id,),
                    ).fetchone()
                    if current is None:
                        self.conn.commit()
                        return "missing"
                    if current[1] is not None:
                        self.conn.commit()
                        return "already_deleted"
                    self.conn.commit()
                    return "conflict"

                try:
                    self.conn.execute(
                        'DELETE FROM memory_embeddings WHERE rowid = ?', (memory_id,)
                    )
                except Exception as vec_err:
                    logger.warning(
                        "Could not delete embedding for remotely deleted memory %s: %s",
                        _sanitize_log_value(content_hash), _sanitize_log_value(vec_err),
                    )
                self.conn.execute(
                    'DELETE FROM memory_graph WHERE source_hash = ? OR target_hash = ?',
                    (content_hash, content_hash),
                )
                if hasattr(self, '_append_sync_event'):
                    self._append_sync_event(
                        self.conn,
                        'delete',
                        content_hash,
                        {'content_hash': content_hash, 'deleted_at': remote_deleted_at},
                    )
                self.conn.commit()
                return "applied"

            return await self._execute_with_retry(_apply_remote_tombstone)
        except Exception as e:
            try:
                await self._run_in_thread(self.conn.rollback)
            except Exception:
                pass
            logger.error("Failed to apply remote tombstone: %s", _sanitize_log_value(e))
            return "failed"

    async def delete(self, content_hash: str, store: Optional[str] = None) -> Tuple[bool, str]:
        """Soft-delete a memory by setting deleted_at timestamp. Scoped to store partition when given (#1465)."""
        try:
            if not self.conn:
                return False, "Database not initialized"

            def _delete_memory():
                select_sql = 'SELECT id FROM memories WHERE content_hash = ? AND deleted_at IS NULL'
                select_params = [content_hash]
                if store is not None:
                    select_sql += ' AND store = ?'
                    select_params.append(store)
                cursor = self.conn.execute(select_sql, tuple(select_params))
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
                        _sanitize_log_value(content_hash), _sanitize_log_value(vec_err),
                    )
                self.conn.execute(
                    'DELETE FROM memory_graph WHERE source_hash = ? OR target_hash = ?',
                    (content_hash, content_hash)
                )
                deleted_at = time.time()
                update_sql = 'UPDATE memories SET deleted_at = ? WHERE content_hash = ? AND deleted_at IS NULL'
                update_params = [deleted_at, content_hash]
                if store is not None:
                    update_sql += ' AND store = ?'
                    update_params.append(store)
                cursor = self.conn.execute(update_sql, tuple(update_params))

                # Append sync event between UPDATE and commit (ADR-0008)
                if hasattr(self, '_append_sync_event') and cursor.rowcount > 0:
                    payload = {
                        'content_hash': content_hash,
                        'deleted_at': deleted_at
                    }
                    self._append_sync_event(self.conn, 'delete', content_hash, payload)
                
                self.conn.commit()
                return cursor.rowcount

            rowcount = await self._execute_with_retry(_delete_memory)
            if rowcount is None:
                return False, f"Memory with hash {content_hash} not found"
            if rowcount > 0:
                logger.info("Soft-deleted memory: %s", _sanitize_log_value(content_hash))
                return True, f"Successfully deleted memory {content_hash}"
            else:
                return False, f"Memory with hash {content_hash} not found"

        except Exception as e:
            try:
                await self._run_in_thread(self.conn.rollback)
            except sqlite3.OperationalError:
                pass
            error_msg = f"Failed to delete memory: {str(e)}"
            logger.error("%s", _sanitize_log_value(error_msg))
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
            logger.error("Failed to check if memory is deleted: %s", _sanitize_log_value(e))
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
                            "Batch embedding purge failed (%s) — retrying per-row.", _sanitize_log_value(vec_err)
                        )
                        for rid in ids:
                            try:
                                self.conn.execute(
                                    'DELETE FROM memory_embeddings WHERE rowid = ?', (rid,)
                                )
                            except Exception as row_err:
                                logger.warning(
                                    "Could not delete embedding rowid=%s during purge: %s",
                                    rid, _sanitize_log_value(row_err),
                                )
                cursor = self.conn.execute(
                    'DELETE FROM memories WHERE deleted_at IS NOT NULL AND deleted_at < ?',
                    (cutoff,)
                )
                self.conn.commit()
                return cursor.rowcount

            count = await self._execute_with_retry(_purge)
            if count > 0:
                logger.info("Purged %s tombstones older than %s days", count, older_than_days)
            return count

        except Exception as e:
            logger.error("Failed to purge deleted memories: %s", _sanitize_log_value(e))
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
                    "SELECT id, content_hash FROM memories WHERE (',' || tags || ',') LIKE ? ESCAPE '\\' AND deleted_at IS NULL",
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
                            memory_id, _sanitize_log_value(vec_err),
                        )

                for ch in content_hashes:
                    self.conn.execute(
                        'DELETE FROM memory_graph WHERE source_hash = ? OR target_hash = ?',
                        (ch, ch)
                    )

                cursor = self.conn.execute(
                    "UPDATE memories SET deleted_at = ? WHERE (',' || tags || ',') LIKE ? ESCAPE '\\' AND deleted_at IS NULL",
                    (time.time(), exact_match_pattern)
                )
                self.conn.commit()
                return cursor.rowcount

            count = await self._execute_with_retry(_delete_by_tag)
            logger.info("Soft-deleted %s memories with tag: %s", count, _sanitize_log_value(tag))

            if count > 0:
                return count, f"Successfully deleted {count} memories with tag '{tag}'"
            else:
                return 0, f"No memories found with tag '{tag}'"

        except Exception as e:
            error_msg = f"Failed to delete by tag: {str(e)}"
            logger.error("%s", _sanitize_log_value(error_msg))
            return 0, error_msg

    async def delete_by_tags(self, tags: List[str], store: Optional[str] = None) -> Tuple[int, str, List[str]]:
        """Soft-delete memories matching ANY of the given tags. Scoped to store partition when given (#1465)."""
        try:
            if not self.conn:
                return 0, "Database not initialized", []

            if not tags:
                return 0, "No tags provided", []

            stripped_tags = [tag.strip() for tag in tags]
            conditions = " OR ".join(["(',' || tags || ',') LIKE ? ESCAPE '\\'" for _ in stripped_tags])
            params = [f"%,{_escape_like(tag)},%" for tag in stripped_tags]

            store_clause = ""
            store_param = []
            if store is not None:
                store_clause = " AND store = ?"
                store_param = [store]

            select_query = f'SELECT id, content_hash FROM memories WHERE ({conditions}){store_clause} AND deleted_at IS NULL'
            update_query = f'UPDATE memories SET deleted_at = ? WHERE ({conditions}){store_clause} AND deleted_at IS NULL'

            def _delete_by_tags():
                cursor = self.conn.execute(select_query, params + store_param)
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
                            _sanitize_log_value(vec_err),
                        )
                        for mid in memory_ids:
                            try:
                                self.conn.execute('DELETE FROM memory_embeddings WHERE rowid = ?', (mid,))
                            except Exception as row_err:
                                logger.warning("Could not delete embedding rowid=%s (corrupted blob?): %s", mid, _sanitize_log_value(row_err))

                for ch in hashes:
                    self.conn.execute(
                        'DELETE FROM memory_graph WHERE source_hash = ? OR target_hash = ?',
                        (ch, ch)
                    )

                cursor = self.conn.execute(update_query, [time.time()] + params + store_param)
                self.conn.commit()
                return cursor.rowcount, hashes

            count, deleted_hashes = await self._execute_with_retry(_delete_by_tags)
            logger.info("Soft-deleted %s memories matching tags: %s", count, _sanitize_log_value(tags))

            if count > 0:
                return count, f"Successfully deleted {count} memories matching {len(tags)} tag(s)", deleted_hashes
            else:
                return 0, f"No memories found matching any of the {len(tags)} tags", []

        except Exception as e:
            error_msg = f"Failed to delete by tags: {str(e)}"
            logger.error("%s", _sanitize_log_value(error_msg))
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
                        AND (',' || tags || ',') LIKE ? ESCAPE '\\'
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
            logger.error("Error deleting by timeframe: %s", _sanitize_log_value(e))
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
                        AND (',' || tags || ',') LIKE ? ESCAPE '\\'
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
            logger.error("Error deleting before date: %s", _sanitize_log_value(e))
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
            logger.info("Soft-deleted %s duplicate memories", count)

            if count > 0:
                return count, f"Successfully soft-deleted {count} duplicate memories"
            else:
                return 0, "No duplicate memories found"

        except Exception as e:
            error_msg = f"Failed to cleanup duplicates: {str(e)}"
            logger.error("%s", _sanitize_log_value(error_msg))
            return 0, error_msg
