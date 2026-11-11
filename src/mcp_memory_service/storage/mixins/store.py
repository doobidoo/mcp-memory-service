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
from ..base import RecreateRefused
from ..semantic_dedup import (
    NEAR_IDENTICAL_DEFAULT,
    SUPERSEDE_REASON,
    SemanticMatch,
    annotate_store_message,
    deduplicated_result,
    verdict_for_match,
)
from ..shared import _versioned_conflict_exclusion

logger = logging.getLogger(__name__)


class StoreMixin:
    #: sqlite-vec keeps tombstones and the sync event log: it can refuse a re-create (#1560).
    supports_recreate_refusal = True

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

        This purge runs on every re-store, regardless of the tombstone's age or the
        `purge_deleted(older_than_days=...)` retention window. A new store of the same
        content supersedes the old memory's metadata (tags, created_at, deleted_at);
        those values are not preserved.

        For sync this is safe: the durable tombstone consulted by conflict resolution
        is the `sync_events` row with `op='delete'` (ADR-0009), which lives in a
        separate table and survives this purge.
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
        delete for `content_hash` whenever a delete event is recorded in sync_events —
        INDEPENDENTLY of whether a soft-deleted tombstone row still exists (#1564). The
        producer stamps the `event_id` on the re-create's payload
        (retrocompat name-match) AND the HLC+agent triple as `recreate_of_hlc` — a STABLE
        identity anchor that survives event_id rewrites (P1-B, 2026-10-16): the bootstrap
        baseline rewrites the delete's event_id to `bl-...` but PRESERVES its HLC
        (bootstrap.py:132 `hlc_by_hash`), so apply can still match the named delete by the
        anchor when the raw event_id no longer matches. Call BEFORE `_purge_tombstone`,
        which hard-deletes the tombstone row but does NOT touch sync_events, so the delete
        event is still the latest for the hash.

        Returns None when no delete event is recorded for the hash (a fresh create carries no
        recreate_of) or when the lookup is unavailable.

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
            # #1564: decide SOLELY from the recorded delete in sync_events, NOT from the
            # presence of a tombstone row. purge_deleted/_purge_tombstone hard-delete the
            # soft-deleted memories row but never touch sync_events, so the delete event
            # survives the purge. The old tombstone guard (`if tomb is None: return None`)
            # was redundant with `if not rows` below — sync_events only carries op='delete'
            # when a live row was actually deleted (delete.py emits it on rowcount>0), so a
            # fresh create (never deleted) has zero delete rows and still returns None here.
            # Requiring the tombstone row made a re-create AFTER a purge emit recreate_of=""
            # (EMPTY), which apply treats as delete-wins → origin live, peer dead forever.
            rows = list(self.conn.execute(
                "SELECT event_id, hlc_physical, hlc_logical, agent_id FROM sync_events "
                "WHERE content_hash = ? AND op = 'delete'",
                (content_hash,),
            ).fetchall())
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
        current_only: bool = False,
    ) -> Tuple[bool, Optional[str]]:
        """Check for a similar memory in ``store`` within the configured time window.

        ``current_only`` ignores memories that already have ``superseded_by`` set,
        so supersede mode replaces the live fact rather than a historical one.
        The default query is the reject-mode query. sqlite-vec's
        ``vec_distance_cosine`` is a distance (``1 - cosine``); the returned
        ``SemanticMatch.similarity`` is the cosine.
        """
        if not self.conn:
            return SemanticMatch(False, None)

        cutoff_timestamp = time.time() - (time_window_hours * 3600)
        embedding = self._generate_embedding(content)
        embedding_blob = serialize_float32(embedding)
        current_sql = ""
        if current_only:
            current_sql = " AND (m.superseded_by IS NULL OR m.superseded_by = '')"

        def _search_semantic_dup():
            cursor = self.conn.execute(
                "SELECT m.content_hash, m.content, "
                "vec_distance_cosine(me.content_embedding, ?) as similarity "
                "FROM memories m "
                "JOIN memory_embeddings me ON m.rowid = me.rowid "
                "WHERE m.created_at > ? "
                "AND m.deleted_at IS NULL "
                "AND m.store = ?"
                + current_sql
                + " ORDER BY similarity IS NULL, similarity ASC LIMIT 1",
                (embedding_blob, cutoff_timestamp, store),
            )
            return cursor.fetchone()

        result = await self._execute_with_retry(_search_semantic_dup)
        # A zero-length embedding (for example from empty content stored before
        # #1791 was fixed) makes vec_distance_cosine() return NULL. Treat that as
        # "no duplicate" instead of comparing None to a float.
        if result and result[2] is not None and result[2] <= (1.0 - similarity_threshold):
            distance = float(result[2])
            return SemanticMatch(True, result[0], 1.0 - distance, result[1])

        return SemanticMatch(False, None)

    async def _dedup_match_content(self, content_hash: str) -> Optional[str]:
        """Text of a live memory, for the normalised-identical check.

        The SQL path already returns the text. This is the fallback for a
        ``_check_semantic_duplicate`` double that only returns the hash.
        """
        def _read():
            if not self.conn:
                return None
            row = self.conn.execute(
                "SELECT content FROM memories WHERE content_hash = ? AND deleted_at IS NULL",
                (content_hash,),
            ).fetchone()
            return row[0] if row else None

        try:
            return await self._execute_with_retry(_read)
        except Exception as exc:  # noqa: BLE001 — missing text rejects, it does not supersede
            logger.debug(
                "semantic dedup could not read %s: %s",
                _sanitize_log_value(content_hash[:8]),
                _sanitize_log_value(exc),
            )
            return None

    async def _semantic_dedup_verdict(self, memory: Memory, store: str):
        """Reject, supersede, or proceed for one sqlite-vec ``store()``."""
        mode = getattr(self, "semantic_dedup_mode", "reject")
        cutoff = getattr(self, "semantic_dedup_near_identical_threshold", NEAR_IDENTICAL_DEFAULT)
        result = await self._check_semantic_duplicate(
            memory.content,
            time_window_hours=self.semantic_dedup_time_window,
            similarity_threshold=self.semantic_dedup_threshold,
            store=store,
            current_only=(mode == "supersede"),
        )
        is_duplicate, existing_hash = result[0], result[1]
        matched = getattr(result, "matched_content", None)
        if is_duplicate and existing_hash and matched is None:
            matched = await self._dedup_match_content(existing_hash)
        return verdict_for_match(
            mode=mode,
            near_identical_threshold=cutoff,
            is_duplicate=bool(is_duplicate),
            existing_hash=existing_hash,
            similarity=getattr(result, "similarity", None),
            matched_content=matched,
            new_content=memory.content,
        )

    def _write_supersede_lineage(
        self,
        old_hash: str,
        new_hash: str,
        store: str,
        new_version: int,
        old_meta: dict,
    ) -> None:
        """Columns and sync event ``update_memory_versioned`` writes for one link."""
        self.conn.execute(
            "UPDATE memories SET parent_id = ?, version = ? WHERE content_hash = ?",
            (old_hash, new_version, new_hash),
        )
        cur = self.conn.execute(
            "UPDATE memories SET superseded_by = ?, superseded_at = ?, metadata = ? "
            "WHERE content_hash = ? AND deleted_at IS NULL "
            "AND (superseded_by IS NULL OR superseded_by = '') AND store = ?",
            (new_hash, time.time(), json.dumps(old_meta), old_hash, store),
        )
        if cur.rowcount == 0:
            raise RuntimeError("old version is no longer current")
        self._append_supersede_event(old_hash)
        self._clear_stale_conflict_tags(
            [new_hash, *self._contradicts_partners(old_hash)]
        )

    def _link_semantic_supersede_row(self, old_hash: str, new_hash: str, store: str) -> bool:
        """Link one supersede inside the caller's transaction, or report it cannot."""
        row = self.conn.execute(
            "SELECT version, metadata FROM memories "
            "WHERE content_hash = ? AND deleted_at IS NULL "
            "AND (superseded_by IS NULL OR superseded_by = '') AND store = ?",
            (old_hash, store),
        ).fetchone()
        if row is None:
            return False
        old_version, old_metadata_str = row
        new_version = (old_version or 1) + 1
        old_meta = (
            self._safe_json_loads(old_metadata_str, "semantic_supersede")
            if old_metadata_str else {}
        )
        old_meta["superseded_by"] = new_hash
        old_meta["evolution_reason"] = SUPERSEDE_REASON
        sp = f"sem_supersede_{os.urandom(4).hex()}"
        self.conn.execute(f"SAVEPOINT {sp}")
        try:
            self._write_supersede_lineage(old_hash, new_hash, store, new_version, old_meta)
            self.conn.execute(f"RELEASE SAVEPOINT {sp}")
            return True
        except Exception:
            self.conn.execute(f"ROLLBACK TO SAVEPOINT {sp}")
            self.conn.execute(f"RELEASE SAVEPOINT {sp}")
            raise

    def _finish_supersede_in_savepoint(
        self, savepoint: str, old_hash: Optional[str], new_hash: str, store_name: str
    ) -> bool:
        """Link inside the open insert savepoint, or roll that savepoint back.

        The insert, its create event, and the lineage update commit together.
        A failed link undoes the insert, so the caller is left with no new row,
        no create event, and no delete event. ``delete()`` is not used: that
        soft-deletes and records a delete, and the next store with
        ``recreate=False`` then refuses the retry.
        """
        if not old_hash:
            return True
        try:
            linked = self._link_semantic_supersede_row(old_hash, new_hash, store_name)
        except Exception as exc:  # noqa: BLE001 — the insert savepoint is the rollback
            logger.error(
                "semantic supersede link failed for %s: %s",
                _sanitize_log_value(old_hash[:8]),
                _sanitize_log_value(exc),
            )
            linked = False
        if linked:
            return True
        self.conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
        self.conn.execute(f"RELEASE SAVEPOINT {savepoint}")
        return False

    async def _semantic_dedup_gate(self, memory: Memory, store: str, skip_semantic_dedup: bool):
        """Return ``(reject_message, supersede_hash)``. Either value may be None."""
        if not self.semantic_dedup_enabled or skip_semantic_dedup:
            return None, None
        verdict = await self._semantic_dedup_verdict(memory, store)
        if verdict.reject_message is not None:
            return verdict.reject_message, None
        return None, verdict.supersede_hash

    async def store(self, memory: Memory, skip_semantic_dedup: bool = False, store: str = 'default',
                    recreate: Optional[bool] = None) -> Tuple[bool, str]:
        """Store a memory in the SQLite-vec database.

        ``recreate`` (#1560, tri-state — see ``MemoryStorage.store``): ``None`` keeps
        today's behaviour (purge the tombstone, stamp ``recreate_of``); ``False`` refuses
        the write when a delete for this hash is on record in ``sync_events`` — the
        tombstone stays, no create event is appended, and the message is a
        ``RecreateRefused`` naming the delete; ``True`` revives explicitly.

        Semantic dedup (#1938): ``MCP_SEMANTIC_DEDUP_MODE=reject`` (default) still
        returns a ``SemanticDedupResult`` whose text is the old duplicate sentence.
        ``supersede`` stores the new row and points the matched row at it, unless
        the match is near-identical. The success message is then a
        ``SemanticDedupResult`` with ``outcome="superseded"``.
        """
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

            rejection, supersede_hash = await self._semantic_dedup_gate(
                memory, store, skip_semantic_dedup
            )
            if rejection is not None:
                return False, rejection

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
            _refused: List[Tuple[str, list]] = []
            _supersede_failed: List[str] = []
            def insert_memory_and_embedding():
                self.conn.execute(f'SAVEPOINT {_sp_name}')
                try:
                    # #1522 #2: capture the delete this create supersedes BEFORE purging the
                    # tombstone (purge does not touch sync_events). Only set when a tombstone
                    # actually exists for this hash; a fresh create carries no recreate_of.
                    _recreate_info = self._recreate_of_for(memory.content_hash)
                    if recreate is False and _recreate_info is not None:
                        # #1560: the caller did not ask to bring deleted content back. A
                        # stale client re-sending what this node deleted must not have the
                        # node's own delete stamped as a deliberate revive: leave the
                        # tombstone, append no create event, and name the delete instead.
                        _refused.append(_recreate_info)
                        self.conn.execute(f'RELEASE SAVEPOINT {_sp_name}')
                        return
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

                    if not self._finish_supersede_in_savepoint(
                        _sp_name, supersede_hash, memory.content_hash, store
                    ):
                        if supersede_hash:
                            _supersede_failed.append(supersede_hash)
                        return
                    self.conn.execute(f'RELEASE SAVEPOINT {_sp_name}')
                except Exception:
                    self.conn.execute(f'ROLLBACK TO SAVEPOINT {_sp_name}')
                    self.conn.execute(f'RELEASE SAVEPOINT {_sp_name}')
                    raise

            async with self._savepoint_lock:
                await self._execute_with_retry(insert_memory_and_embedding)
                await self._execute_with_retry(self.conn.commit)
                if _refused:
                    return False, RecreateRefused(_refused[0][0], _refused[0][1])
                if _supersede_failed:
                    return False, deduplicated_result(_supersede_failed[0])

            # Conflict detection (P3) — runs after commit, outside the lock
            try:
                conflict_infos = await self._run_in_thread(
                    self._detect_conflicts,
                    memory.content_hash,
                    memory.content,
                    embedding,
                    _versioned_conflict_exclusion.get(),
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
            stored = f"Memory stored successfully{conflict_msg}"
            return True, annotate_store_message(stored, supersede_hash)

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

    async def store_batch(self, memories: List[Memory], store: str = 'default',
                          recreate: Optional[bool] = None) -> List[Tuple[bool, str]]:
        """Store multiple memories in a single transaction with batched embedding generation.

        ``recreate`` has the same tri-state as ``store()`` (#1560); with ``False`` an item
        whose hash has a delete on record is refused on its own (``RecreateRefused``
        message) while the rest of the batch proceeds.
        """
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
                    if recreate is False and _recreate_info is not None:
                        # #1560: refuse this item only (see store()); tombstone and
                        # sync_events untouched.
                        self.conn.execute(f'RELEASE SAVEPOINT {sp}')
                        local_results[j] = (
                            False, RecreateRefused(_recreate_info[0], _recreate_info[1])
                        )
                        continue
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
