"""StoreMixin: store, store_batch, semantic dedup, tombstone purge."""

import sqlite3
import json
import logging
import os
import time
import traceback
import asyncio
from typing import List, Tuple, Optional

try:
    from sqlite_vec import serialize_float32
except ImportError:
    pass

from ...compat import _sanitize_log_value
from ...models.memory import Memory

logger = logging.getLogger(__name__)


class StoreMixin:
    """Mixin providing memory storage operations."""

    def _purge_tombstone(self, content_hash: str) -> None:
        """Remove a soft-deleted tombstone so the UNIQUE constraint allows re-insert.

        Also drops any embedding still attached to the tombstone's rowid. Hard-deleting
        the memories row without this leaves an orphan embedding; since memories has no
        AUTOINCREMENT, SQLite reuses that rowid later and the next store collides with a
        "UNIQUE constraint failed: memory_embeddings".
        """
        rows = self.conn.execute(
            'SELECT id FROM memories WHERE content_hash = ? AND deleted_at IS NOT NULL',
            (content_hash,)
        ).fetchall()
        for (rowid,) in rows:
            try:
                self.conn.execute('DELETE FROM memory_embeddings WHERE rowid = ?', (rowid,))
            except Exception as vec_err:
                logger.warning(
                    "Could not delete embedding rowid=%s during tombstone purge: %s", rowid, _sanitize_log_value(vec_err)
                )
        self.conn.execute(
            'DELETE FROM memories WHERE content_hash = ? AND deleted_at IS NOT NULL',
            (content_hash,)
        )

    async def _check_semantic_duplicate(
        self,
        content: str,
        time_window_hours: int = 24,
        similarity_threshold: float = 0.85
    ) -> Tuple[bool, Optional[str]]:
        """Check if a semantically similar memory was stored within time window."""
        if not self.conn:
            return False, None

        cutoff_timestamp = time.time() - (time_window_hours * 3600)

        embedding = self._generate_embedding(content)
        embedding_blob = serialize_float32(embedding)

        def _search_semantic_dup():
            cursor = self.conn.execute('''
                SELECT m.content_hash, m.content,
                       vec_distance_cosine(me.content_embedding, ?) as similarity
                FROM memories m
                JOIN memory_embeddings me ON m.rowid = me.rowid
                WHERE m.created_at > ?
                  AND m.deleted_at IS NULL
                ORDER BY similarity ASC
                LIMIT 1
            ''', (embedding_blob, cutoff_timestamp))
            return cursor.fetchone()

        result = await self._execute_with_retry(_search_semantic_dup)
        if result and result[2] <= (1.0 - similarity_threshold):
            return True, result[0]

        return False, None

    async def store(self, memory: Memory, skip_semantic_dedup: bool = False, store: str = 'default') -> Tuple[bool, str]:
        """Store a memory in the SQLite-vec database."""
        try:
            if not self.conn:
                return False, "Database not initialized"

            def _check_exact_dup():
                cursor = self.conn.execute(
                    'SELECT content_hash FROM memories WHERE content_hash = ? AND deleted_at IS NULL',
                    (memory.content_hash,)
                )
                return cursor.fetchone()
            if await self._execute_with_retry(_check_exact_dup):
                return False, "Duplicate content detected (exact match)"

            if self.semantic_dedup_enabled and not skip_semantic_dedup:
                is_duplicate, existing_hash = await self._check_semantic_duplicate(
                    memory.content,
                    time_window_hours=self.semantic_dedup_time_window,
                    similarity_threshold=self.semantic_dedup_threshold
                )
                if is_duplicate:
                    return False, f"Duplicate content detected (semantically similar to {existing_hash})"

            try:
                embedding = self._generate_embedding(memory.content)
            except Exception as e:
                logger.error(
                    "Failed to generate embedding for memory %s: %s",
                    _sanitize_log_value(memory.content_hash),
                    _sanitize_log_value(e),
                )
                return False, f"Failed to generate embedding: {str(e)}"

            tags_str = ",".join(memory.tags) if memory.tags else ""
            metadata_str = json.dumps(memory.metadata) if memory.metadata else "{}"

            _sp_name = f"store_{os.urandom(4).hex()}"
            def insert_memory_and_embedding():
                self.conn.execute(f'SAVEPOINT {_sp_name}')
                try:
                    self._purge_tombstone(memory.content_hash)
                    cursor = self.conn.execute('''
                        INSERT INTO memories (
                            content_hash, content, tags, memory_type,
                            metadata, created_at, updated_at, created_at_iso, updated_at_iso, store
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        memory.content_hash,
                        memory.content,
                        tags_str,
                        memory.memory_type,
                        metadata_str,
                        memory.created_at,
                        memory.updated_at,
                        memory.created_at_iso,
                        memory.updated_at_iso,
                        store
                    ))
                    memory_rowid = cursor.lastrowid

                    # Defensive: clear any orphan embedding already sitting at this rowid.
                    # memories has no AUTOINCREMENT, so a rowid freed by a hard-delete can be
                    # reused; a leftover embedding there would raise UNIQUE constraint failed.
                    self.conn.execute(
                        'DELETE FROM memory_embeddings WHERE rowid = ?', (memory_rowid,)
                    )
                    self.conn.execute('''
                        INSERT INTO memory_embeddings (rowid, content_embedding, store)
                        VALUES (?, ?, ?)
                    ''', (
                        memory_rowid,
                        serialize_float32(embedding),
                        store
                    ))
                    
                    # Append sync event within same transaction (ADR-0008)
                    # Called after INSERT memory_embeddings, before RELEASE SAVEPOINT
                    if hasattr(self, '_append_sync_event'):
                        payload = {
                            'content_hash': memory.content_hash,
                            'content': memory.content,
                            'memory_type': memory.memory_type,
                            'tags': memory.tags or [],
                            'created_at': memory.created_at,
                            'updated_at': memory.updated_at,
                            'metadata': memory.metadata or {},
                            'store': store,
                        }
                        self._append_sync_event(self.conn, 'create', memory.content_hash, payload)
                    
                    self.conn.execute(f'RELEASE SAVEPOINT {_sp_name}')
                except Exception:
                    self.conn.execute(f'ROLLBACK TO SAVEPOINT {_sp_name}')
                    self.conn.execute(f'RELEASE SAVEPOINT {_sp_name}')
                    raise

            async with self._savepoint_lock:
                await self._execute_with_retry(insert_memory_and_embedding)
                await self._execute_with_retry(self.conn.commit)

            # Conflict detection (P3) — runs after commit, outside the lock
            try:
                conflict_infos = await self._run_in_thread(
                    self._detect_conflicts, memory.content_hash, memory.content, embedding
                )
                if conflict_infos:
                    await self._record_conflicts(memory.content_hash, conflict_infos)
                    conflict_msg = f" {len(conflict_infos)} conflict(s) detected."
                else:
                    conflict_msg = ""
            except Exception as e:
                logger.warning("Conflict detection failed (non-fatal): %s", _sanitize_log_value(e))
                conflict_msg = ""

            logger.info("Successfully stored memory: %s", _sanitize_log_value(memory.content_hash))
            return True, f"Memory stored successfully{conflict_msg}"

        except Exception as e:
            error_msg = f"Failed to store memory: {str(e)}"
            logger.error("%s", _sanitize_log_value(error_msg))
            logger.error("%s", _sanitize_log_value(traceback.format_exc()))
            return False, error_msg

    async def store_batch(self, memories: List[Memory], store: str = 'default') -> List[Tuple[bool, str]]:
        """Store multiple memories in a single transaction with batched embedding generation."""
        if not memories:
            return []

        if not self.conn:
            return [(False, "Database not initialized")] * len(memories)

        contents = [m.content for m in memories]
        try:
            if not self.embedding_model:
                raise RuntimeError("No embedding model available")
            raw_embeddings = self.embedding_model.encode(contents, convert_to_numpy=True)
        except Exception as e:
            error_msg = f"Batch embedding generation failed: {e}"
            logger.error("%s", _sanitize_log_value(error_msg))
            return [(False, error_msg)] * len(memories)

        def batch_insert():
            local_results: List[Tuple[bool, str]] = [None] * len(memories)
            # Wrap the whole batch in an explicit transaction so the per-item SAVEPOINTs are
            # nested inside ONE transaction. Without this, RELEASE of an outermost savepoint
            # can commit, making the batch not all-or-nothing (greptile P1). The commit at
            # the call site closes this transaction; the error handler rolls it all back.
            self.conn.execute('BEGIN')
            for j, memory in enumerate(memories):
                cursor = self.conn.execute(
                    'SELECT content_hash FROM memories WHERE content_hash = ? AND deleted_at IS NULL',
                    (memory.content_hash,)
                )
                if cursor.fetchone():
                    local_results[j] = (False, "Duplicate content detected (exact match)")
                    continue

                embedding = raw_embeddings[j]
                if hasattr(embedding, "tolist"):
                    embedding_list = embedding.tolist()
                else:
                    embedding_list = list(embedding)

                tags_str = ",".join(memory.tags) if memory.tags else ""
                metadata_str = json.dumps(memory.metadata) if memory.metadata else "{}"

                sp = "batch_item"
                try:
                    self.conn.execute(f'SAVEPOINT {sp}')

                    self._purge_tombstone(memory.content_hash)

                    cur = self.conn.execute('''
                        INSERT INTO memories (
                            content_hash, content, tags, memory_type,
                            metadata, created_at, updated_at, created_at_iso, updated_at_iso, store
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        memory.content_hash, memory.content, tags_str,
                        memory.memory_type, metadata_str,
                        memory.created_at, memory.updated_at,
                        memory.created_at_iso, memory.updated_at_iso, store
                    ))
                    rowid = cur.lastrowid

                    # Defensive: clear any orphan embedding at a reused rowid (see store()).
                    self.conn.execute(
                        'DELETE FROM memory_embeddings WHERE rowid = ?', (rowid,)
                    )
                    self.conn.execute('''
                        INSERT INTO memory_embeddings (rowid, content_embedding, store)
                        VALUES (?, ?, ?)
                    ''', (rowid, serialize_float32(embedding_list), store))

                    # Append sync event for each item within SAVEPOINT batch_item (ADR-0008)
                    if hasattr(self, '_append_sync_event'):
                        payload = {
                            'content_hash': memory.content_hash,
                            'content': memory.content,
                            'memory_type': memory.memory_type,
                            'tags': memory.tags or [],
                            'created_at': memory.created_at,
                            'updated_at': memory.updated_at,
                            'metadata': memory.metadata or {},
                            'store': store,
                        }
                        self._append_sync_event(self.conn, 'create', memory.content_hash, payload)

                    self.conn.execute(f'RELEASE SAVEPOINT {sp}')
                    local_results[j] = (True, "Memory stored successfully")
                except sqlite3.IntegrityError:
                    self.conn.execute(f'ROLLBACK TO SAVEPOINT {sp}')
                    self.conn.execute(f'RELEASE SAVEPOINT {sp}')
                    local_results[j] = (False, "Duplicate content detected (race condition)")
                except sqlite3.Error as db_err:
                    self.conn.execute(f'ROLLBACK TO SAVEPOINT {sp}')
                    self.conn.execute(f'RELEASE SAVEPOINT {sp}')
                    local_results[j] = (False, f"Insert failed: {db_err}")
                except Exception:
                    # Non-sqlite failure (e.g. sync-event append raising, ADR-0008 fail-closed):
                    # roll the item's savepoint back and propagate so the batch aborts rather
                    # than committing a memory without its event.
                    self.conn.execute(f'ROLLBACK TO SAVEPOINT {sp}')
                    self.conn.execute(f'RELEASE SAVEPOINT {sp}')
                    raise
            return local_results

        if not hasattr(self, '_savepoint_lock'):
            self._savepoint_lock = asyncio.Lock()

        results: List[Tuple[bool, str]] = [None] * len(memories)
        try:
            async with self._savepoint_lock:
                results = await self._execute_with_retry(batch_insert)
                await self._execute_with_retry(self.conn.commit)

            stored = sum(1 for r in results if r and r[0])
            logger.info("Batch stored %s/%s memories in single transaction", stored, len(memories))
        except Exception as e:
            error_msg = f"Batch transaction failed: {e}"
            logger.error("%s", _sanitize_log_value(error_msg))
            logger.error("%s", _sanitize_log_value(traceback.format_exc()))
            # Roll the whole batch transaction back under the lock. batch_insert re-raised
            # after rolling back only the failing item's SAVEPOINT; earlier items were
            # RELEASE'd but never committed, so they linger in the open transaction and
            # would otherwise leak into the next unrelated commit (greptile P1). Rolling
            # back here under the same lock discards them atomically.
            try:
                async with self._savepoint_lock:
                    await self._execute_with_retry(self.conn.rollback)
            except Exception as rb_err:
                logger.error("Batch rollback failed: %s", _sanitize_log_value(rb_err))
            # The whole transaction was rolled back → NOTHING persisted. Mark every result
            # as failed, not just the None ones: items that reported (True, ...) before the
            # commit failure were not actually committed (greptile P1).
            results = [(False, error_msg) for _ in range(len(memories))]

        return [(r if r is not None else (False, "Skipped")) for r in results]
