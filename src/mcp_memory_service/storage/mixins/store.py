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
                    "Could not delete embedding rowid=%s during tombstone purge: %s", rowid, vec_err
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
                logger.error(f"Failed to generate embedding for memory {memory.content_hash}: {str(e)}")
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
                logger.warning(f"Conflict detection failed (non-fatal): {e}")
                conflict_msg = ""

            logger.info(f"Successfully stored memory: {memory.content_hash}")
            return True, f"Memory stored successfully{conflict_msg}"

        except Exception as e:
            error_msg = f"Failed to store memory: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
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
            logger.error(error_msg)
            return [(False, error_msg)] * len(memories)

        def batch_insert():
            local_results: List[Tuple[bool, str]] = [None] * len(memories)
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
            return local_results

        if not hasattr(self, '_savepoint_lock'):
            self._savepoint_lock = asyncio.Lock()

        results: List[Tuple[bool, str]] = [None] * len(memories)
        try:
            async with self._savepoint_lock:
                results = await self._execute_with_retry(batch_insert)
                await self._execute_with_retry(self.conn.commit)

            stored = sum(1 for r in results if r and r[0])
            logger.info(f"Batch stored {stored}/{len(memories)} memories in single transaction")
        except Exception as e:
            error_msg = f"Batch transaction failed: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            for j in range(len(memories)):
                if results[j] is None:
                    results[j] = (False, error_msg)

        return [(r if r is not None else (False, "Skipped")) for r in results]
