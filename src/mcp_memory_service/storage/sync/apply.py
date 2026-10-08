"""
Delta-sync Phase 4a apply functionality.

Implements apply_remote_event and advance_sync_cursor functions for pulling and applying
events from remote peers. Integrates with the resolver from Phase 2 for conflict resolution.

ADR-0021: Apply materializes directly to SQLite (not via POST /api/memories)
ADR-0022: Apply preserves original authorship (agent_id, event_id, HLC)
"""

import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Dict, Any

from ..base import MemoryStorage
from ..mixins.base import _sanitize_log_value
from .resolver import EventView, reduce_events

logger = logging.getLogger(__name__)


def _sync_lock(storage: MemoryStorage) -> threading.Lock:
    """Return the storage connection lock used to serialize writes on the shared conn.

    Sync apply/cursor writes run from the async request/scheduler path on the SAME SQLite
    connection as normal local writes, which run in a worker thread under ``_conn_lock`` and
    open savepoints. Committing from sync without holding that lock can commit (and drop the
    savepoint of) an in-flight local write, so the worker's later release/rollback fails
    (Greptile P1). Acquire the SAME lock here so sync and local writes are mutually exclusive.
    Mirror base.py's lazy init so the lock exists even if the backend never created one.
    """
    s = _sqlite(storage)
    if not hasattr(s, "_conn_lock") or s._conn_lock is None:
        s._conn_lock = threading.Lock()
    return s._conn_lock


def _sqlite(storage: MemoryStorage) -> MemoryStorage:
    """Resolve the concrete SQLite-backed storage.

    HybridMemoryStorage exposes its SQLite backend through ``.primary`` and has no
    ``.conn`` of its own. Accessing ``storage.conn`` directly on a hybrid instance raises
    AttributeError, breaking the feed and every scheduled sync cycle (Greptile P1). Resolve
    the concrete backend here so both direct access and SQLite-only methods work under hybrid.
    """
    if hasattr(storage, "conn"):
        return storage
    primary = getattr(storage, "primary", None)
    if primary is not None and hasattr(primary, "conn"):
        return primary
    return storage


@dataclass
class ApplyResult:
    """Result of applying a remote event."""
    applied: bool
    materialized: bool
    reason: str


def _event_identity_payload(storage: MemoryStorage, agent_id: str, event_id: str):
    """Return the stored payload (parsed) for an existing (agent_id, event_id), or None."""
    cur = _sqlite(storage).conn.execute(
        "SELECT payload FROM sync_events WHERE agent_id = ? AND event_id = ?",
        (agent_id, event_id),
    )
    row = cur.fetchone()
    if row is None:
        return None
    try:
        return json.loads(row[0]) if row[0] else {}
    except (TypeError, ValueError):
        return {}


def apply_remote_event(storage: MemoryStorage, event: Dict[str, Any]) -> ApplyResult:
    """
    Apply a remote sync event to local storage.

    Three-step process:
    1. Insert event into sync_events (idempotent via UNIQUE (agent_id, event_id))
    2. Resolve conflicts using the Phase 2 resolver
    3. Materialize if the remote event wins

    Replay safety (Greptile P1, security): an identity (agent_id, event_id) is immutable.
    If the identity already exists, we treat the event as a duplicate and never rewrite the
    memory from a replayed-but-altered payload. Only a genuinely new identity can materialize.

    Concurrency (Greptile P1): the whole insert→resolve→materialize→commit runs while holding
    the storage connection lock, so it never interleaves with an in-flight local write's
    savepoint on the shared SQLite connection.
    """
    with _sync_lock(storage):
        return _apply_remote_event_locked(storage, event)


def _apply_remote_event_locked(storage: MemoryStorage, event: Dict[str, Any]) -> ApplyResult:
    """Body of apply_remote_event; MUST run under _sync_lock (see caller)."""
    try:
        content_hash = event["content_hash"]
        agent_id = event["agent_id"]
        event_id = event["event_id"]
        op = event["op"]
        hlc_physical = event["hlc_physical"]
        hlc_logical = event["hlc_logical"]
        embedding_model = event.get("embedding_model")
        embedding_dim = event.get("embedding_dim")
        payload = event.get("payload", {})
        s = _sqlite(storage)

        # Step 1: record the event. The identity (agent_id, event_id) is immutable — an
        # already-present identity means this is a replay. INSERT OR IGNORE keeps the
        # original row; we must NOT let a replayed payload with the same identity rewrite
        # the materialized memory (Greptile P1, security).
        try:
            pre_existing = _event_identity_payload(storage, agent_id, event_id)
            if pre_existing is not None:
                # Known identity (agent_id, event_id) — the original event row is
                # authoritative and immutable. Compare the incoming payload to it:
                #   - identical payload  → benign duplicate (idempotent re-pull/resume);
                #     count as applied, do NOT re-materialize.
                #   - different payload  → replay with altered data; reject and never let
                #     it rewrite the materialized memory (Greptile P1, security).
                if pre_existing == payload:
                    s.conn.commit()
                    return ApplyResult(applied=True, materialized=False, reason="Duplicate event (idempotent, no rewrite)")
                s.conn.commit()
                return ApplyResult(applied=False, materialized=False, reason="Replay with altered payload rejected")

            s.conn.execute("""
                INSERT OR IGNORE INTO sync_events
                (agent_id, event_id, op, content_hash, hlc_physical, hlc_logical,
                 embedding_model, embedding_dim, payload, created_at, created_at_iso)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                agent_id, event_id, op, content_hash, hlc_physical, hlc_logical,
                embedding_model, embedding_dim, json.dumps(payload),
                time.time(), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            ))

            cursor = s.conn.execute(
                "SELECT 1 FROM sync_events WHERE agent_id = ? AND event_id = ?",
                (agent_id, event_id)
            )
            if cursor.fetchone() is None:
                return ApplyResult(applied=False, materialized=False, reason="Failed to insert event")

        except Exception as e:
            logger.error("Error inserting sync event: %s", _sanitize_log_value(e))
            return ApplyResult(applied=False, materialized=False, reason=f"Insert failed: {e}")

        # Step 2: resolve conflicts using the Phase 2 resolver
        try:
            remote_event_view = EventView(
                hlc_physical=hlc_physical,
                hlc_logical=hlc_logical,
                agent_id=agent_id,
                event_id=event_id,
                op=op,
                content_hash=content_hash,
            )

            cursor = s.conn.execute("""
                SELECT hlc_physical, hlc_logical, agent_id, event_id, op, content_hash
                FROM sync_events
                WHERE content_hash = ?
                ORDER BY hlc_physical DESC, hlc_logical DESC
            """, (content_hash,))

            competing_events = [
                EventView(
                    hlc_physical=row[0], hlc_logical=row[1], agent_id=row[2],
                    event_id=row[3], op=row[4], content_hash=row[5],
                )
                for row in cursor.fetchall()
            ]

            if not competing_events:
                winner = remote_event_view
            else:
                all_events = competing_events
                if remote_event_view not in competing_events:
                    all_events.append(remote_event_view)
                winner = reduce_events(all_events)

            # Keep the local HLC clock monotonic: accepting a remote event whose clock is
            # ahead must advance our saved last_hlc, otherwise a later LOCAL edit reads a
            # stale clock from metadata and can lose to the very event it meant to update
            # (Greptile P1).
            _advance_local_hlc(storage, hlc_physical, hlc_logical)

            # Step 3: materialize.
            # - create/delete: only when THIS event wins the whole-event resolution.
            # - update_metadata: ALWAYS re-materialize — the rebuild is a deterministic,
            #   order-independent fold over every update_metadata event for the hash
            #   (field-level LWW by HLC), so applying it on each event converges all peers
            #   even when this event is not the overall winner (Greptile P1 convergence).
            is_winner = bool(winner and winner.event_id == event_id and winner.agent_id == agent_id)
            if is_winner or op == "update_metadata":
                try:
                    materialized = _materialize_event(storage, event)
                    s.conn.commit()  # durability: persist event + materialization (ADR-0019/§8.5)
                    if materialized:
                        reason = ("Remote event won and materialized" if is_winner
                                  else "update_metadata merged (field-level convergence)")
                        return ApplyResult(applied=True, materialized=True, reason=reason)
                    # A win that fails to materialize is NOT applied — return failure so the
                    # sender keeps retrying and the puller does not advance past it (Greptile P1).
                    return ApplyResult(applied=False, materialized=False, reason="Materialization failed")
                except Exception as e:
                    logger.error("Materialization failed: %s", _sanitize_log_value(e))
                    return ApplyResult(applied=False, materialized=False, reason=f"Materialization error: {e}")
            else:
                # Remote event lost or tied — recorded but not materialized. This is a
                # successful apply (a legitimate conflict loser), distinct from a failure.
                s.conn.commit()
                return ApplyResult(applied=True, materialized=False, reason="Remote event lost conflict resolution")

        except Exception as e:
            logger.error("Error in conflict resolution: %s", _sanitize_log_value(e))
            return ApplyResult(applied=False, materialized=False, reason=f"Resolver failed: {e}")

    except Exception as e:
        logger.error("Error applying remote event: %s", _sanitize_log_value(e))
        return ApplyResult(applied=False, materialized=False, reason=f"Apply failed: {e}")


def _advance_local_hlc(storage: MemoryStorage, hlc_physical: int, hlc_logical: int) -> None:
    """Advance the saved last_hlc (metadata) to >= the accepted remote clock.

    Local writes read sync_hlc_physical/sync_hlc_logical from metadata to build their HLC.
    If a remote event's clock is ahead and we don't bump the saved clock, the next local
    edit gets an older clock and can lose resolution against the event it meant to supersede
    (Greptile P1). Monotonic max; same transaction as the apply.
    """
    try:
        s = _sqlite(storage)
        cur = s.conn.execute(
            "SELECT key, value FROM metadata WHERE key IN ('sync_hlc_physical', 'sync_hlc_logical')"
        )
        saved = {k: int(v) for k, v in cur.fetchall()} if cur else {}
        last_physical = saved.get('sync_hlc_physical', 0)
        last_logical = saved.get('sync_hlc_logical', 0)
        if (hlc_physical, hlc_logical) > (last_physical, last_logical):
            s.conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES ('sync_hlc_physical', ?)",
                (str(hlc_physical),),
            )
            s.conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES ('sync_hlc_logical', ?)",
                (str(hlc_logical),),
            )
    except Exception as e:
        logger.warning("Could not advance local HLC after remote event: %s", _sanitize_log_value(e))


def _materialize_event(storage: MemoryStorage, event: Dict[str, Any]) -> bool:
    """
    Materialize an event into the memories table.

    Handles create / update_metadata / delete per their distinct payload contracts
    (ADR-0021). create carries the full memory; update_metadata carries only `updates`
    to merge into an existing row; delete carries a soft-delete timestamp.
    """
    try:
        s = _sqlite(storage)
        op = event["op"]
        content_hash = event["content_hash"]
        payload = event.get("payload", {})
        embedding_model = event.get("embedding_model")

        if op == "delete":
            s.conn.execute(
                "UPDATE memories SET deleted_at = ? WHERE content_hash = ?",
                (payload.get("deleted_at", time.time()), content_hash),
            )
            return True

        if op == "update_metadata":
            # update_metadata carries ONLY {content_hash, updates, updated_at} — NOT a full
            # memory. Treating it like create would erase the memory (Greptile P1).
            #
            # Convergence (Greptile P1): update_metadata events carry PARTIAL fields, and two
            # spokes editing DIFFERENT fields must converge regardless of arrival order. We do
            # NOT apply only the winning event's updates (that discards the other field when
            # events arrive newest-first). Instead we rebuild each field from ALL
            # update_metadata events for this hash: for every field, the value comes from the
            # highest-HLC event that set it (field-level last-writer-wins by HLC). This is a
            # pure function of the event set, so every peer reaches the same row (ADR-0012/0013).
            row = s.conn.execute(
                "SELECT 1 FROM memories WHERE content_hash = ? AND deleted_at IS NULL",
                (content_hash,),
            ).fetchone()
            if row is None:
                # No local row to update (ordering/gap) — nothing to merge; not a failure.
                logger.debug("update_metadata for unknown/absent hash %s — skipped", _sanitize_log_value(content_hash))
                return True

            allowed = {"tags", "memory_type", "metadata"}
            # Winning value + its HLC per field, folded over every update_metadata event.
            best: Dict[str, Any] = {}          # field -> value
            best_hlc: Dict[str, tuple] = {}    # field -> (hlc_physical, hlc_logical, agent_id, event_id)
            latest_updated_at = None
            latest_updated_hlc = None
            ev_cursor = s.conn.execute(
                """
                SELECT hlc_physical, hlc_logical, agent_id, event_id, payload
                FROM sync_events
                WHERE content_hash = ? AND op = 'update_metadata'
                """,
                (content_hash,),
            )
            for ev_phys, ev_log, ev_agent, ev_eid, ev_payload_json in ev_cursor.fetchall():
                try:
                    ev_payload = json.loads(ev_payload_json) if ev_payload_json else {}
                except (ValueError, TypeError):
                    continue
                ev_updates = ev_payload.get("updates", {}) or {}
                # Total order mirrors the resolver: (hlc, agent_id, event_id); larger wins.
                ev_key = (ev_phys or 0, ev_log or 0, ev_agent or "", ev_eid or "")
                for key, value in ev_updates.items():
                    if key not in allowed:
                        continue
                    if ev_key > best_hlc.get(key, (-1, -1, "", "")):
                        best_hlc[key] = ev_key
                        best[key] = value
                ev_upd_at = ev_payload.get("updated_at")
                if ev_upd_at is not None and (latest_updated_hlc is None or ev_key > latest_updated_hlc):
                    latest_updated_hlc = ev_key
                    latest_updated_at = ev_upd_at

            set_clauses, params = [], []
            for key, value in best.items():
                if key == "tags":
                    # Same CSV format store() uses; the read path splits on commas.
                    value = ",".join(value) if isinstance(value, (list, tuple)) else str(value)
                elif key == "metadata":
                    value = json.dumps(value) if value else "{}"
                set_clauses.append(f"{key} = ?")
                params.append(value)

            upd_at = latest_updated_at if latest_updated_at is not None else payload.get("updated_at", time.time())
            set_clauses.append("updated_at = ?")
            params.append(upd_at)
            set_clauses.append("updated_at_iso = ?")
            params.append(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(upd_at if isinstance(upd_at, (int, float)) else time.time())))
            params.append(content_hash)
            s.conn.execute(
                f"UPDATE memories SET {', '.join(set_clauses)} WHERE content_hash = ?",
                tuple(params),
            )
            return True

        if op in ("create", "update"):
            content = payload.get("content", "")
            tags = payload.get("tags", [])
            metadata = payload.get("metadata", {})
            memory_type = payload.get("memory_type", "general")
            # Preserve the payload's store and timestamps — a named-store memory must stay in
            # that store, and an old memory must not look newly created after sync (Greptile P1).
            store = payload.get("store", "default") or "default"
            created_at = payload.get("created_at", time.time())
            updated_at = payload.get("updated_at", created_at)

            if not content and op == "create":
                logger.warning("Create event missing content for hash %s", _sanitize_log_value(content_hash))
                return False

            model_mismatch = bool(embedding_model) and embedding_model != s.embedding_model_name
            can_embed = bool(content) and not model_mismatch
            embedding_pending = 0 if can_embed else 1

            # Capture any pre-existing row id so we can clean up its stale embedding BEFORE
            # a replace changes the row id and orphans the old vector (Greptile P2).
            old = s.conn.execute(
                "SELECT id FROM memories WHERE content_hash = ?", (content_hash,)
            ).fetchone()
            old_id = old[0] if old else None

            s.conn.execute("""
                INSERT OR REPLACE INTO memories
                (content_hash, content, tags, memory_type, metadata,
                 created_at, created_at_iso, updated_at, updated_at_iso,
                 deleted_at, embedding_pending, store)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
            """, (
                content_hash,
                content,
                ",".join(tags) if tags else "",
                memory_type,
                json.dumps(metadata) if metadata else "{}",
                created_at,
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(created_at if isinstance(created_at, (int, float)) else time.time())),
                updated_at,
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(updated_at if isinstance(updated_at, (int, float)) else time.time())),
                embedding_pending,
                store,
            ))

            # Remove the old embedding if the replace changed the row id (Greptile P2);
            # otherwise a dangling vector at the old id accumulates on every round-trip.
            new_id = s.conn.execute(
                "SELECT id FROM memories WHERE content_hash = ?", (content_hash,)
            ).fetchone()[0]
            if old_id is not None and old_id != new_id:
                s.conn.execute("DELETE FROM memory_embeddings WHERE rowid = ?", (old_id,))

            if can_embed:
                try:
                    from sqlite_vec import serialize_float32
                    embedding = s._generate_embedding(content)
                    s.conn.execute("DELETE FROM memory_embeddings WHERE rowid = ?", (new_id,))
                    s.conn.execute(
                        "INSERT INTO memory_embeddings (rowid, content_embedding, store) VALUES (?, ?, ?)",
                        (new_id, serialize_float32(embedding), store),
                    )
                except Exception as emb_err:
                    logger.warning("apply: embedding generation failed for %s: %s",
                                   _sanitize_log_value(content_hash), _sanitize_log_value(emb_err))
                    s.conn.execute(
                        "UPDATE memories SET embedding_pending = 1 WHERE content_hash = ?",
                        (content_hash,),
                    )
            return True

        logger.warning("Unknown operation type: %s", _sanitize_log_value(op))
        return False

    except Exception as e:
        logger.error("Error materializing event: %s", _sanitize_log_value(e))
        return False


def advance_sync_cursor(storage: MemoryStorage, peer_id: str, last_seq: int) -> None:
    """
    Advance the sync cursor for a peer to the given sequence number.

    Args:
        storage: The local storage instance
        peer_id: Identifier of the peer
        last_seq: Last sequence number successfully processed
    """
    try:
        s = _sqlite(storage)
        # Hold the connection lock: the cursor commit shares the SQLite connection with
        # local writes and must not interleave with an in-flight savepoint (Greptile P1).
        with _sync_lock(storage):
            s.conn.execute("""
                INSERT OR REPLACE INTO sync_cursor
                (peer_id, last_seq_seen, updated_at)
                VALUES (?, ?, ?)
            """, (peer_id, last_seq, time.time()))
            # Durability (§8.5 / ADR-0019): the cursor and the applied events of the batch
            # must survive a crash. Commit here closes the batch transaction atomically.
            s.conn.commit()
        logger.debug("Advanced sync cursor for peer %s to seq %s",
                     _sanitize_log_value(peer_id), _sanitize_log_value(last_seq))
    except Exception as e:
        logger.error("Error advancing sync cursor: %s", _sanitize_log_value(e))
        raise
