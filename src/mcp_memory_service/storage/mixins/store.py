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

    @staticmethod
    def _exact_duplicate_message(
        existing_store: str | None, requested_store: str
    ) -> str:
        """Describe a global exact-hash conflict and name its other store."""
        message = "Duplicate content detected (exact match)"
        if existing_store is not None and existing_store != requested_store:
            return f"{message}; already exists in store {existing_store!r}"
        return message

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

    def _recreate_of_for(self, content_hash: str) -> Optional[Tuple[str, list]]:
        """#1522 #2 (ADR-0012 v2): the `recreate_of` signal for a re-create of this hash.

        Returns `(event_id, [hlc_physical, hlc_logical, agent_id])` of the dominant local
        delete for `content_hash` ONLY when the hash currently carries a durable tombstone
        (a soft-deleted row). The producer stamps the `event_id` on the re-create's payload
        (retrocompat name-match) AND the HLC+agent triple as `recreate_of_hlc` — a STABLE
        identity anchor that survives event_id rewrites (P1-B, 2026-10-16): the bootstrap
        baseline rewrites the delete's event_id to `bl-...` but PRESERVES its HLC
        (bootstrap.py:132 `hlc_by_hash`), so apply can still match the named delete by the
        anchor when the raw event_id no longer matches. Call BEFORE `_purge_tombstone`,
        which hard-deletes the tombstone row but does NOT touch sync_events, so the delete
        event is still the latest for the hash.

        Returns None when the hash has no tombstone (a fresh create carries no recreate_of),
        when no delete event is recorded, or when the lookup is unavailable.

        P2 tied-deletes (Greptile): the dominant delete is selected by the resolver's FULL
        total order (`_winner_key`: HLC -> delete-vs-update -> agent_id -> event_id), NOT by
        `(hlc_physical, hlc_logical)` alone. With two deletes at the SAME HLC, a raw
        `ORDER BY hlc DESC LIMIT 1` is non-deterministic across peers; `_winner_key` breaks the
        tie the same way apply's `recompute_liveness` elects Dtop, so producer and consumer agree.
        """
        # Local import keeps the resolver's canonical order the single source of truth and
        # avoids a module-load cycle (store mixin <-> sync package).
        from ..sync.resolver import EventView, _winner_key
        try:
            tomb = self.conn.execute(
                "SELECT 1 FROM memories WHERE content_hash = ? AND deleted_at IS NOT NULL LIMIT 1",
                (content_hash,),
            ).fetchone()
            if tomb is None:
                return None
            rows = self.conn.execute(
                "SELECT event_id, hlc_physical, hlc_logical, agent_id FROM sync_events "
                "WHERE content_hash = ? AND op = 'delete'",
                (content_hash,),
            ).fetchall()
        except Exception as e:  # noqa: BLE001 - never break store() on this lookup
            logger.debug(
                "recreate_of lookup failed for %s: %s",
                _sanitize_log_value(content_hash), _sanitize_log_value(e),
            )
            return None
        if not rows:
            return None
        # Elect the dominant delete (smallest _winner_key == resolver winner among deletes).
        dtop = min(
            rows,
            key=lambda r: _winner_key(EventView(
                hlc_physical=r[1] or 0, hlc_logical=r[2] or 0, agent_id=r[3],
                event_id=r[0] or "", op="delete", content_hash=content_hash,
            )),
        )
        # event_id (name-match, retrocompat) + the STABLE HLC+agent anchor (P1-B): the anchor
        # survives the bootstrap's `bl-...` event_id rewrite, so apply can still match this
        # delete when the raw event_id has been rewritten. Delete events always carry a
        # stamped HLC, so the physical/logical are never NULL here; apply's _load_hash_events
        # coerces NULL→0 on the read side symmetrically.
        return dtop[0], [dtop[1], dtop[2], dtop[3]]

    async def _check_semantic_duplicate(
        self,
        content: str,
        time_window_hours: int = 24,
        similarity_threshold: float = 0.85,
        store: str = 'default',
    ) -> Tuple[bool, Optional[str]]:
        """Check for a similar memory in ``store`` within the configured time window."""
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
                  AND m.store = ?
                ORDER BY similarity ASC
                LIMIT 1
            ''', (embedding_blob, cutoff_timestamp, store))
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
                    'SELECT store FROM memories '
                    'WHERE content_hash = ? AND deleted_at IS NULL',
                    (memory.content_hash,)
                )
                return cursor.fetchone()
            existing = await self._execute_with_retry(_check_exact_dup)
            if existing:
                return False, self._exact_duplicate_message(existing[0], store)

            if self.semantic_dedup_enabled and not skip_semantic_dedup:
                is_duplicate, existing_hash = await self._check_semantic_duplicate(
                    memory.content,
                    time_window_hours=self.semantic_dedup_time_window,
                    similarity_threshold=self.semantic_dedup_threshold,
                    store=store,
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
                    # #1522 #2: capture the delete this create supersedes BEFORE purging the
                    # tombstone (purge does not touch sync_events). Only set when a tombstone
                    # actually exists for this hash; a fresh create carries no recreate_of.
                    _recreate_info = self._recreate_of_for(memory.content_hash)
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
                        # #1522 #2 (Henry, 3-state): v2 stamps recreate_of on EVERY create —
                        # the delete's event_id when re-creating over a tombstone, else the
                        # explicit empty sentinel "". Absence of the key now means "v1 peer"
                        # only; a v2 producer that never saw the delete emits "" (not absent),
                        # which apply treats as "delete wins". Sentinel is "" (not null): null
                        # round-trips through json as None and collapses with absence via
                        # .get(); "" survives distinct (seven G0, 2026-10-16).
                        # P1-B (2026-10-16): when NAMED, also stamp recreate_of_hlc = the
                        # delete's [hlc_physical, hlc_logical, agent_id] — a STABLE anchor that
                        # survives the bootstrap's `bl-...` event_id rewrite so apply can match
                        # the delete by HLC when the raw event_id no longer matches.
                        if _recreate_info is not None:
                            payload['recreate_of'] = _recreate_info[0]
                            payload['recreate_of_hlc'] = _recreate_info[1]
                        else:
                            payload['recreate_of'] = ""
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

        except sqlite3.IntegrityError as e:
            # A second writer can insert the same globally unique hash after the
            # pre-check. Report the same store-aware conflict as the non-racing path.
            try:
                existing = await self._execute_with_retry(_check_exact_dup)
            except sqlite3.Error:
                existing = None
            if existing:
                return False, self._exact_duplicate_message(existing[0], store)

            error_msg = f"Failed to store memory: {e!s}"
            logger.error("%s", _sanitize_log_value(error_msg))
            logger.error("%s", _sanitize_log_value(traceback.format_exc()))
            return False, error_msg
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
            # Retry-safe: if a previous attempt left a transaction open (e.g. a mid-batch
            # "database is locked"), clear it before starting, so a re-run does not hit
            # "cannot start a transaction within a transaction" (greptile P1). Then open the
            # explicit transaction that spans the whole batch. getattr guards test doubles
            # that don't expose `in_transaction`.
            if getattr(self.conn, 'in_transaction', False):
                self.conn.rollback()
            self.conn.execute('BEGIN')
            for j, memory in enumerate(memories):
                cursor = self.conn.execute(
                    'SELECT store FROM memories '
                    'WHERE content_hash = ? AND deleted_at IS NULL',
                    (memory.content_hash,)
                )
                existing = cursor.fetchone()
                if existing:
                    local_results[j] = (
                        False,
                        self._exact_duplicate_message(existing[0], store),
                    )
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

                    # #1522 #2: capture the superseded delete BEFORE purge (same as store()).
                    _recreate_info = self._recreate_of_for(memory.content_hash)
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
                        # #1522 #2 (Henry, 3-state): stamp recreate_of on EVERY create —
                        # delete's event_id over a tombstone, else "" (explicit empty
                        # sentinel). Same contract as store(); "" (not null) so absence vs
                        # empty survive the json round-trip distinctly. P1-B: when NAMED, also
                        # stamp recreate_of_hlc (the delete's [hlc_physical, hlc_logical,
                        # agent_id]) — a stable anchor robust to the bootstrap `bl-...` rewrite.
                        if _recreate_info is not None:
                            payload['recreate_of'] = _recreate_info[0]
                            payload['recreate_of_hlc'] = _recreate_info[1]
                        else:
                            payload['recreate_of'] = ""
                        self._append_sync_event(self.conn, 'create', memory.content_hash, payload)

                    self.conn.execute(f'RELEASE SAVEPOINT {sp}')
                    local_results[j] = (True, "Memory stored successfully")
                except sqlite3.IntegrityError:
                    self.conn.execute(f'ROLLBACK TO SAVEPOINT {sp}')
                    self.conn.execute(f'RELEASE SAVEPOINT {sp}')
                    existing = self.conn.execute(
                        'SELECT store FROM memories '
                        'WHERE content_hash = ? AND deleted_at IS NULL',
                        (memory.content_hash,),
                    ).fetchone()
                    if existing:
                        message = self._exact_duplicate_message(existing[0], store)
                    else:
                        message = "Duplicate content detected (race condition)"
                    local_results[j] = (False, message)
                except sqlite3.OperationalError as op_err:
                    # A transient 'database is locked'/'busy' is NOT a permanent item failure:
                    # on this deployment several agents write the same sqlite_vec.db (single
                    # writer), so a competing writer makes the item's INSERT fail mid-batch.
                    # Roll the item's savepoint back and RE-RAISE so _execute_with_retry
                    # reruns the whole (atomic, retry-safe) batch after a backoff, instead of
                    # silently dropping the memory. Non-transient OperationalErrors still
                    # degrade the item like any other sqlite error (greptile/ducanhnguyen223).
                    self.conn.execute(f'ROLLBACK TO SAVEPOINT {sp}')
                    self.conn.execute(f'RELEASE SAVEPOINT {sp}')
                    msg = str(op_err).lower()
                    if "locked" in msg or "busy" in msg:
                        raise
                    local_results[j] = (False, f"Insert failed: {op_err}")
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
            # Commit INSIDE the retried unit, under the same lock, so BEGIN+inserts+COMMIT
            # are one atomic operation. A separate commit call after the lock was released
            # let a concurrent op commit the batch early (greptile P1).
            self.conn.commit()
            return local_results

        if not hasattr(self, '_savepoint_lock'):
            self._savepoint_lock = asyncio.Lock()

        results: List[Tuple[bool, str]] = [None] * len(memories)
        try:
            async with self._savepoint_lock:
                results = await self._execute_with_retry(batch_insert)

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
