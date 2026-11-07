"""DeleteMixin: delete, soft-delete, purge, cleanup operations."""

import sqlite3
import logging
import math
import time
from datetime import datetime, date
from typing import List, Tuple, Optional

from ..base import UntaggedCountChanged
from ..source_retraction import retract_sources_for_deleted_hashes


logger = logging.getLogger(__name__)


from ...compat import _sanitize_log_value

def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


# A memory counts as untagged when its tag column is NULL, empty, or nothing
# but whitespace. Shared by the count and the delete so the confirmation count
# the caller is held to and the rows that go are decided by one predicate.
_UNTAGGED_PREDICATE = (
    "(tags IS NULL OR tags = '' OR length(trim(tags)) = 0) AND deleted_at IS NULL"
)


class DeleteMixin:
    """Mixin providing memory deletion operations."""

    async def _rollback_quietly(self) -> None:
        """Drop an open delete transaction after a failure.

        The belief/summary retraction runs before commit. A failure there has
        to undo the tombstone too, or the next statement inherits it.
        """
        try:
            await self._run_in_thread(self.conn.rollback)
        except sqlite3.OperationalError as exc:
            # Best-effort. "no transaction is active" means the failure
            # happened before any statement opened one, so there is nothing
            # to undo. A connection closed in between raises the same error.
            logger.debug("Delete rollback skipped: %s", _sanitize_log_value(exc))

    def _abort_open_delete(self) -> None:
        """Roll back a delete before the worker releases the connection lock.

        Citation cleanup runs after the tombstone update and before commit.
        Rolling back only after ``_execute_with_retry`` returns lets another
        waiter take ``_conn_lock`` and commit the shared connection first.
        The caller then sees a failed delete whose tombstone is already saved.
        """
        try:
            if self.conn is None:
                return
            # A connection double without in_transaction still gets rollback.
            if getattr(self.conn, "in_transaction", True):
                self.conn.rollback()
        except sqlite3.Error as exc:
            logger.error("Could not roll back delete: %s", _sanitize_log_value(exc))

    def _guard_delete(self, operation):
        """Run a delete body and undo it before the connection lock drops."""
        try:
            return operation()
        except Exception:
            self._abort_open_delete()
            raise

    async def get_sync_state_for_hash(
        self, content_hash: str
    ) -> tuple[float | None, float | None] | None:
        """Read the current local update/deletion timestamps for queued sync guards.

        Unlike ``get_by_hash``, this includes tombstoned rows. Database errors are
        deliberately propagated so a caller can fail closed instead of uploading
        a queued snapshot without knowing whether it was deleted locally.
        """
        if not self.conn:
            raise RuntimeError("Database not initialized")

        def _get_sync_state():
            row = self.conn.execute(
                'SELECT updated_at, deleted_at FROM memories WHERE content_hash = ?',
                (content_hash,),
            ).fetchone()
            return tuple(row) if row is not None else None

        return await self._execute_with_retry(_get_sync_state)

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
                savepoint = "remote_tombstone_reconcile"
                savepoint_open = False
                self.conn.execute(f'SAVEPOINT {savepoint}')
                savepoint_open = True
                try:
                    row = self.conn.execute(
                        'SELECT id, updated_at, deleted_at FROM memories '
                        'WHERE content_hash = ?',
                        (content_hash,),
                    ).fetchone()
                    commit_after = False
                    if row is None:
                        outcome = "missing"
                    else:
                        memory_id, updated_at, existing_deleted_at = row
                        if existing_deleted_at is not None:
                            outcome = "already_deleted"
                        elif updated_at is not None and float(updated_at) > remote_deleted_at:
                            outcome = "conflict"
                        else:
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
                                    outcome = "missing"
                                elif current[1] is not None:
                                    outcome = "already_deleted"
                                else:
                                    outcome = "conflict"
                                commit_after = True
                            else:
                                try:
                                    self.conn.execute(
                                        'DELETE FROM memory_embeddings WHERE rowid = ?',
                                        (memory_id,),
                                    )
                                except Exception as vec_err:
                                    logger.warning(
                                        "Could not delete embedding for remotely deleted memory %s: %s",
                                        _sanitize_log_value(content_hash),
                                        _sanitize_log_value(vec_err),
                                    )
                                self.conn.execute(
                                    'DELETE FROM memory_graph WHERE source_hash = ? OR target_hash = ?',
                                    (content_hash, content_hash),
                                )
                                retract_sources_for_deleted_hashes(self.conn, [content_hash], sync=self)
                                if hasattr(self, '_append_sync_event'):
                                    self._append_sync_event(
                                        self.conn,
                                        "delete",
                                        content_hash,
                                        {
                                            "content_hash": content_hash,
                                            "deleted_at": remote_deleted_at,
                                        },
                                        hlc_override=(int(remote_deleted_at * 1000), 0),
                                    )
                                outcome = "applied"
                                commit_after = True

                    if commit_after:
                        # Keep commit under the same connection lock as the update and
                        # cleanups. A concurrent store can no longer be folded into a
                        # transaction that a later error handler rolls back.
                        self.conn.commit()
                    else:
                        self.conn.execute(f'RELEASE SAVEPOINT {savepoint}')
                    return outcome
                except Exception:
                    if savepoint_open:
                        try:
                            self.conn.execute(f'ROLLBACK TO SAVEPOINT {savepoint}')
                            self.conn.execute(f'RELEASE SAVEPOINT {savepoint}')
                        except sqlite3.Error as cleanup_error:
                            logger.error(
                                "Could not roll back remote tombstone transaction for %s: %s",
                                _sanitize_log_value(content_hash),
                                _sanitize_log_value(cleanup_error),
                            )
                            try:
                                # Still under _conn_lock: reset a broken transaction
                                # before another SQLite operation can enter it.
                                self.conn.rollback()
                            except sqlite3.Error as fallback_error:
                                logger.error(
                                    "Could not reset SQLite after tombstone rollback failure for %s: %s",
                                    _sanitize_log_value(content_hash),
                                    _sanitize_log_value(fallback_error),
                                )
                    raise

            # StoreMixin spans its INSERT/commit sequence with this same lock. Keep
            # the savepoint rollback and commit ordered with stores on the shared
            # SQLite connection as well as holding _conn_lock for each SQL closure.
            async with self._savepoint_lock:
                return await self._execute_with_retry(_apply_remote_tombstone)
        except Exception as e:  # noqa: BLE001 - return a fail-closed reconciliation result
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
                # Issue #1840: read the conflict partners before the edges go.
                # Guarded like _append_sync_event: the helpers live on MetadataMixin.
                cleans_conflict_tags = hasattr(self, '_clear_stale_conflict_tags')
                conflict_partners = (
                    self._contradicts_partners(content_hash) if cleans_conflict_tags else []
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
                if cursor.rowcount > 0:
                    retract_sources_for_deleted_hashes(self.conn, [content_hash], sync=self)

                # Append sync event between UPDATE and commit (ADR-0008)
                if hasattr(self, '_append_sync_event') and cursor.rowcount > 0:
                    payload = {
                        'content_hash': content_hash,
                        'deleted_at': deleted_at
                    }
                    self._append_sync_event(self.conn, 'delete', content_hash, payload)

                if conflict_partners and cursor.rowcount > 0:
                    self._clear_stale_conflict_tags(conflict_partners)

                self.conn.commit()
                return cursor.rowcount

            def _guarded_delete():
                return self._guard_delete(_delete_memory)

            rowcount = await self._execute_with_retry(_guarded_delete)
            if rowcount is None:
                return False, f"Memory with hash {content_hash} not found"
            if rowcount > 0:
                logger.info("Soft-deleted memory: %s", _sanitize_log_value(content_hash))
                return True, f"Successfully deleted memory {content_hash}"
            else:
                return False, f"Memory with hash {content_hash} not found"

        except Exception as e:
            await self._rollback_quietly()
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
        """Permanently delete tombstones older than specified days.

        Note: `StoreMixin._purge_tombstone` also drops a tombstone on re-store of the
        same content, regardless of `older_than_days`, so a tombstone may not survive
        until this retention window ends. The old row's tags, metadata, created_at and
        deleted_at are lost at that point.

        For sync this is safe: the durable tombstone consulted by conflict resolution
        is the `sync_events` row with `op='delete'` (ADR-0009), not the `memories` row.
        """
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
                if cursor.rowcount > 0:
                    retract_sources_for_deleted_hashes(self.conn, content_hashes, sync=self)
                self.conn.commit()
                return cursor.rowcount

            def _guarded_delete_by_tag():
                return self._guard_delete(_delete_by_tag)

            count = await self._execute_with_retry(_guarded_delete_by_tag)
            logger.info("Soft-deleted %s memories with tag: %s", count, _sanitize_log_value(tag))

            if count > 0:
                return count, f"Successfully deleted {count} memories with tag '{tag}'"
            else:
                return 0, f"No memories found with tag '{tag}'"

        except Exception as e:
            await self._rollback_quietly()
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
                if cursor.rowcount > 0:
                    retract_sources_for_deleted_hashes(self.conn, hashes, sync=self)
                self.conn.commit()
                return cursor.rowcount, hashes

            def _guarded_delete_by_tags():
                return self._guard_delete(_delete_by_tags)

            count, deleted_hashes = await self._execute_with_retry(_guarded_delete_by_tags)
            logger.info("Soft-deleted %s memories matching tags: %s", count, _sanitize_log_value(tags))

            if count > 0:
                return count, f"Successfully deleted {count} memories matching {len(tags)} tag(s)", deleted_hashes
            else:
                return 0, f"No memories found matching any of the {len(tags)} tags", []

        except Exception as e:
            await self._rollback_quietly()
            error_msg = f"Failed to delete by tags: {str(e)}"
            logger.error("%s", _sanitize_log_value(error_msg))
            return 0, error_msg, []

    async def count_untagged_memories(self) -> int:
        """Count live memories that carry no tags."""
        if not self.conn:
            return 0

        def _count_untagged():
            cursor = self.conn.execute(
                f'SELECT COUNT(*) FROM memories WHERE {_UNTAGGED_PREDICATE}'
            )
            return cursor.fetchone()[0]

        return await self._execute_with_retry(_count_untagged)

    async def delete_untagged_memories(
        self, expected_count: Optional[int] = None
    ) -> Tuple[int, str]:
        """Soft-delete every memory that carries no tags.

        The tombstone is only part of a delete on this backend: graph edges
        have to go with it, beliefs and summaries built on these hashes have
        to be retracted (#1865), delta-sync peers need a delete event each,
        and all of it belongs in one transaction under the connection lock —
        the same shape as :meth:`delete_by_tags`.

        ``expected_count`` is the size the caller was held to. It is checked
        here, against the rows that are about to go, inside the connection
        lock and before the first write: a caller that counted in one request
        and deleted in the next cannot otherwise tell whether a concurrent
        write stripped the last tag off another memory in between and widened
        the delete. A mismatch raises :class:`UntaggedCountChanged` with
        nothing written.
        """
        try:
            if not self.conn:
                return 0, "Database not initialized"

            def _delete_untagged():
                cursor = self.conn.execute(
                    f'SELECT id, content_hash FROM memories WHERE {_UNTAGGED_PREDICATE}'
                )
                rows = cursor.fetchall()
                memory_ids = [row[0] for row in rows]
                hashes = [row[1] for row in rows]

                # Before any write, and under the lock that serialises them.
                if expected_count is not None and len(rows) != expected_count:
                    raise UntaggedCountChanged(expected_count, len(rows))

                if not memory_ids:
                    return 0

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

                deleted_at = time.time()
                cursor = self.conn.execute(
                    f'UPDATE memories SET deleted_at = ? WHERE {_UNTAGGED_PREDICATE}',
                    (deleted_at,)
                )
                if cursor.rowcount > 0:
                    retract_sources_for_deleted_hashes(self.conn, hashes, sync=self)

                    # One event per hash, between UPDATE and commit (ADR-0008),
                    # so a delta-sync peer sees these deletes like any other.
                    if hasattr(self, '_append_sync_event'):
                        for ch in hashes:
                            self._append_sync_event(
                                self.conn, 'delete', ch,
                                {'content_hash': ch, 'deleted_at': deleted_at},
                            )

                self.conn.commit()
                return cursor.rowcount

            def _guarded_delete_untagged():
                return self._guard_delete(_delete_untagged)

            count = await self._execute_with_retry(_guarded_delete_untagged)
            logger.info("Soft-deleted %s memories without tags", count)

            if count > 0:
                return count, (
                    f"Successfully soft-deleted {count} memories without tags "
                    f"(tombstones created for multi-PC sync)"
                )
            return 0, "No untagged memories found"

        except UntaggedCountChanged:
            # The caller confirmed a count that no longer holds; it decides what
            # to do, and nothing was written for _rollback_quietly to undo.
            raise
        except Exception as e:
            await self._rollback_quietly()
            error_msg = f"Failed to delete untagged memories: {str(e)}"
            logger.error("%s", _sanitize_log_value(error_msg))
            return 0, error_msg

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

    async def count_untagged_memories(self) -> int:
        """Count memories with no tags (NULL, empty string, or whitespace only) excluding soft-deleted."""
        try:
            if not self.conn:
                return 0

            def _count():
                cursor = self.conn.execute(
                    "SELECT COUNT(*) FROM memories WHERE (tags IS NULL OR tags = '' OR length(trim(tags)) = 0) AND deleted_at IS NULL"
                )
                return cursor.fetchone()[0]

            return await self._execute_with_retry(_count)
        except Exception as e:
            logger.error("Failed to count untagged memories: %s", _sanitize_log_value(e))
            return 0

    async def delete_untagged_memories(self) -> Tuple[int, str]:
        """Soft-delete memories with no tags, recording tombstones and sync events."""
        try:
            if not self.conn:
                return 0, "Database not initialized"

            predicate = "(tags IS NULL OR tags = '' OR length(trim(tags)) = 0) AND deleted_at IS NULL"
            deleted_at = time.time()

            def _delete():
                try:
                    rows = self.conn.execute(f"SELECT id, content_hash FROM memories WHERE {predicate}").fetchall()
                    ids = [r[0] for r in rows]
                    hashes = [r[1] for r in rows]

                    for memory_id in ids:
                        try:
                            self.conn.execute('DELETE FROM memory_embeddings WHERE rowid = ?', (memory_id,))
                        except Exception as vec_err:
                            logger.warning(
                                "Could not delete embedding rowid=%s (corrupted blob?): %s — proceeding with soft-delete.",
                                memory_id, _sanitize_log_value(vec_err),
                            )

                    for ch in hashes:
                        self.conn.execute(
                            'DELETE FROM memory_graph WHERE source_hash = ? OR target_hash = ?',
                            (ch, ch)
                        )

                    cursor = self.conn.execute(
                        f"UPDATE memories SET deleted_at = ? WHERE {predicate}",
                        (deleted_at,)
                    )

                    if hasattr(self, '_append_sync_event') and cursor.rowcount > 0:
                        for ch in hashes:
                            self._append_sync_event(self.conn, 'delete', ch, {'content_hash': ch, 'deleted_at': deleted_at})

                    self.conn.commit()
                    return cursor.rowcount, hashes
                except Exception:
                    try:
                        self.conn.rollback()
                    except Exception:
                        # Ignore rollback error if connection was closed or broken
                        pass
                    raise

            count, hashes = await self._execute_with_retry(_delete)
            logger.info("Deleted %d untagged memories", count)
            return count, f"Successfully soft-deleted {count} memories without tags (tombstones created for multi-PC sync)", hashes

        except Exception as e:
            error_msg = f"Failed to delete untagged memories: {str(e)}"
            logger.error("%s", _sanitize_log_value(error_msg))
            return 0, error_msg, []
