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
from .resolver import EventView, reduce_events, _winner_key

logger = logging.getLogger(__name__)

# Highest sync-event envelope version this peer understands (RFC §9.3 rule 6, Phase 5).
# An incoming event with a greater schema_version is rejected/quarantined — never applied
# best-effort — and the cursor must NOT advance past it (R7). Bump this only together with
# a migration/negotiation that can actually read the newer envelope.
KNOWN_SCHEMA_VERSION = 1


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
    # When applied is False, skippable=True means the event is permanently
    # non-materializable (e.g. a create whose content is irrecoverably empty at
    # the source) and the orchestrator should advance past it instead of
    # fail-stopping the whole feed. Default False preserves the original
    # fail-stop behaviour for transient failures.
    skippable: bool = False


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

        # Version negotiation (RFC §9.3 rule 6, Phase 5 R7): reject an event whose envelope
        # version this peer cannot read. Returning applied=False keeps the puller's fail-stop
        # from advancing the cursor past it (the sender/newer peer must not have its event
        # silently dropped). Never best-effort apply an unknown-version event.
        event_schema_version = event.get("schema_version", 1)
        if event_schema_version > KNOWN_SCHEMA_VERSION:
            logger.warning(
                "Rejecting sync event %s: envelope schema_version %s > known %s (incompatible)",
                _sanitize_log_value(event.get("event_id")),
                _sanitize_log_value(event_schema_version),
                KNOWN_SCHEMA_VERSION,
            )
            return ApplyResult(applied=False, materialized=False,
                               reason=f"unknown envelope schema_version {event_schema_version} > {KNOWN_SCHEMA_VERSION}")

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
                # Content-repair upgrade case (Greptile #1499): a receiver that stalled on
                # an empty-content create (before the feed fix) has the empty payload
                # recorded; the fixed feed resends the SAME identity with the real content.
                # This is a repair, not tampering, BUT it must be strictly bounded and must
                # NOT bypass the normal conflict/materialization path:
                #   - both events must be creates (no op switch);
                #   - the ONLY accepted change is empty content -> non-empty content for the
                #     SAME content_hash; every other field (metadata, tags, memory_type,
                #     store) is preserved from the ORIGINAL stored event, so a replay cannot
                #     smuggle changed fields in under the repair (Greptile P1 apply:168);
                #   - we only rewrite the stored content, then fall through to Step 2 so the
                #     resolver decides whether this create still wins (an older create cannot
                #     resurrect a newer delete / overwrite newer state — Greptile P1 apply:174)
                #     and materialization runs under the normal commit/rollback path
                #     (a failed write is not acked as applied — Greptile P1 apply:181).
                pre_content = (pre_existing or {}).get("content") or ""
                new_content = payload.get("content") or ""
                same_hash = (pre_existing or {}).get("content_hash") == payload.get("content_hash") == content_hash
                is_repair = (
                    (not pre_content) and new_content and same_hash
                    and op == "create" and (pre_existing or {}).get("op", "create") == "create"
                )
                if not is_repair:
                    s.conn.commit()
                    return ApplyResult(applied=False, materialized=False, reason="Replay with altered payload rejected")
                # Rebuild payload = original stored payload with ONLY content filled in.
                repaired_payload = dict(pre_existing)
                repaired_payload["content"] = new_content
                s.conn.execute(
                    "UPDATE sync_events SET payload = ? WHERE agent_id = ? AND event_id = ?",
                    (json.dumps(repaired_payload), agent_id, event_id),
                )
                # Use the repaired payload for the rest of this apply and fall through to the
                # normal resolver + materialization (NOT a hand-rolled materialize). Mark so
                # the final result reason is explicit.
                payload = repaired_payload
                event = {**event, "payload": repaired_payload}
                _is_content_repair = True
            else:
                _is_content_repair = False

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
                        if locals().get("_is_content_repair"):
                            reason = "Empty-content create repaired with real content (via resolver)"
                        return ApplyResult(applied=True, materialized=True, reason=reason)
                    # A win that fails to materialize is NOT applied. A create whose content
                    # is empty in BOTH the payload and the source table is permanently
                    # non-materializable — mark it skippable so the orchestrator advances past
                    # it instead of fail-stopping the whole feed (a few irrecoverable events
                    # must not block thousands of good ones). Any other materialization miss
                    # stays a transient failure the sender keeps retrying (Greptile P1).
                    if op == "create" and not (payload.get("content") or ""):
                        return ApplyResult(applied=False, materialized=False,
                                           reason="Create event has no recoverable content — skipped",
                                           skippable=True)
                    return ApplyResult(applied=False, materialized=False, reason="Materialization failed")
                except Exception as e:
                    # Roll back any pending write (e.g. a content-repair UPDATE to
                    # sync_events) so a transient materialization failure does not leave
                    # the stored payload mutated — otherwise the next retry would match
                    # it as a duplicate and never re-attempt the memory write (Greptile
                    # #1499 apply:181, exception path).
                    try:
                        s.conn.rollback()
                    except Exception:
                        # Best-effort rollback: if the connection cannot roll back (e.g.
                        # no open transaction) there is nothing to undo; the original error
                        # below is what matters.
                        pass
                    logger.error("Materialization failed: %s", _sanitize_log_value(e))
                    return ApplyResult(applied=False, materialized=False, reason=f"Materialization error: {e}")
            else:
                # Remote event lost or tied — recorded but not materialized. This is a
                # successful apply (a legitimate conflict loser), distinct from a failure.
                s.conn.commit()
                return ApplyResult(applied=True, materialized=False, reason="Remote event lost conflict resolution")

        except Exception as e:
            try:
                s.conn.rollback()
            except Exception:
                # Best-effort rollback (see above): nothing to undo if no open transaction.
                pass
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


_MD_PROTECTED = {"tags", "memory_type", "metadata", "content", "content_hash",
                 "embedding", "created_at", "created_at_iso", "updated_at",
                 "updated_at_iso", "superseded_by", "store"}


def _ev_winner_key(content_hash, phys, log, agent, eid, ev_op):
    """_winner_key for a raw sync_events row (smaller wins, same order as the resolver)."""
    return _winner_key(EventView(
        hlc_physical=phys or 0, hlc_logical=log or 0,
        agent_id=agent, event_id=eid or "", op=ev_op, content_hash=content_hash,
    ))


def _yield_create_update_metadata(md):
    """Yield metadata items from a create/update payload's metadata dict."""
    if isinstance(md, dict):
        for k, v in md.items():
            yield k, v


def _yield_update_metadata_custom(upd):
    """Yield custom top-level keys (not protected) from an update_metadata payload."""
    for kk, vv in upd.items():
        if kk not in _MD_PROTECTED:
            yield kk, vv


def _metadata_items(ev_op, ev_payload):
    """Yield (inner_key, value) metadata contributions from one event's payload.
    Preserves None (explicit clear) for convergence with the emitter."""
    if ev_op in ("create", "update"):
        yield from _yield_create_update_metadata(ev_payload.get("metadata"))
        return
    upd = ev_payload.get("updates", {}) or {}
    yield from _yield_create_update_metadata(upd.get("metadata"))
    yield from _yield_update_metadata_custom(upd)


def _parse_event_row(content_hash: str, raw_row):
    """Parse one sync_events row and return (parsed_tuple, create_key_candidate).

    parsed_tuple: (hlc_physical, hlc_logical, agent_id, event_id, op, payload) or None if parse fails.
    create_key_candidate: _winner_key if row is create/update, else None.
    """
    phys, log, agent, eid, ev_op, pj = raw_row
    try:
        pl = json.loads(pj) if pj else {}
    except (ValueError, TypeError):
        return None, None
    parsed = (phys or 0, log or 0, agent, eid, ev_op, pl)
    if ev_op in ("create", "update"):
        k = _ev_winner_key(content_hash, phys, log, agent, eid, ev_op)
        return parsed, k
    return parsed, None


def _load_live_events(s, content_hash: str):
    """Return (parsed_events, winning_create_key).

    parsed_events: list of (hlc_physical, hlc_logical, agent_id, event_id, op, payload).
    winning_create_key: the _winner_key of the create/update that defines the LIVE incarnation
    (the single smallest-key create). None if there is no create. Using _winner_key (not raw
    HLC) elects ONE winner even when two creates share an HLC, so the recreation boundary is
    unambiguous and matches the resolver's order (arch ressalva A).
    """
    rows = s.conn.execute(
        """
        SELECT hlc_physical, hlc_logical, agent_id, event_id, op, payload
        FROM sync_events
        WHERE content_hash = ? AND op IN ('create', 'update', 'update_metadata')
        """,
        (content_hash,),
    ).fetchall()
    parsed, create_key = [], None
    for raw_row in rows:
        p, k = _parse_event_row(content_hash, raw_row)
        if p is None:
            continue
        parsed.append(p)
        if k is not None and (create_key is None or k < create_key):
            create_key = k
    return parsed, create_key


def _is_live_event(k, create_key, is_create):
    """Check if an event is part of the live incarnation.

    A create is live iff it is THE winning create (k == create_key).
    An edit (update_metadata) is live iff there's no create or it's strictly before the create
    (k < create_key) — an edit that's not strictly anterior is dead.
    """
    if is_create:
        return k == create_key
    # edit: live if no create exists, or if strictly before the winning create
    return create_key is None or k < create_key


def _top_level_fields(ev_op, pl):
    """Extract top-level fields (tags, memory_type) from an event payload, filtering None.

    create/update: read from pl.get()
    update_metadata: read from pl['updates']
    """
    if ev_op in ("create", "update"):
        return {kk: vv for kk, vv in
                {"tags": pl.get("tags"), "memory_type": pl.get("memory_type")}.items()
                if vv is not None}
    # update_metadata
    upd = (pl.get("updates", {}) or {})
    return {kk: vv for kk, vv in upd.items()
            if kk in ("tags", "memory_type") and vv is not None}


def _take_if_winner(best, best_key, k, key, value):
    """Apply LWW: update best[key] = value if k wins (smaller _winner_key)."""
    if key not in best_key or k < best_key[key]:
        best_key[key] = k
        best[key] = value


def _track_updated_at(pl, k, latest_updated_key, latest_updated_at):
    """Track the latest updated_at timestamp using LWW. Returns (new_key, new_value)."""
    ev_upd_at = pl.get("updated_at")
    if ev_upd_at is not None and (latest_updated_key is None or k < latest_updated_key):
        return k, ev_upd_at
    return latest_updated_key, latest_updated_at


def _resolve_md_fields(parsed, create_key, content_hash):
    """Field-level LWW over the live incarnation. Returns (best, meta_best, create_md_keys,
    latest_updated_at). The live incarnation = the winning create plus events STRICTLY AFTER it
    (smaller _winner_key). Everything from a dead incarnation is skipped (recreation boundary)."""
    best, best_key = {}, {}
    meta_best, meta_best_key = {}, {}
    create_md_keys = set()
    latest_updated_at, latest_updated_key = None, None

    for phys, log, agent, eid, ev_op, pl in parsed:
        k = _ev_winner_key(content_hash, phys, log, agent, eid, ev_op)
        is_create = ev_op in ("create", "update")

        if not _is_live_event(k, create_key, is_create):
            continue

        contributed = _top_level_fields(ev_op, pl)
        for key, value in contributed.items():
            _take_if_winner(best, best_key, k, key, value)

        for mk, mv in _metadata_items(ev_op, pl):
            if is_create:
                create_md_keys.add(mk)
            _take_if_winner(meta_best, meta_best_key, k, mk, mv)

        latest_updated_key, latest_updated_at = _track_updated_at(pl, k, latest_updated_key, latest_updated_at)

    return best, meta_best, create_md_keys, latest_updated_at


def _parse_current_metadata(row):
    """Parse the metadata JSON from a row, returning {} on failure."""
    try:
        current_md = json.loads(row[0]) if row[0] else {}
        if not isinstance(current_md, dict):
            current_md = {}
    except (ValueError, TypeError):
        current_md = {}
    return current_md


def _serialize_field_value(key, value):
    """Serialize a field value for SQL UPDATE.

    tags: join list/tuple with comma, or str()
    metadata: json.dumps, or "{}" if empty
    others: unchanged
    """
    if key == "tags":
        return ",".join(value) if isinstance(value, (list, tuple)) else str(value)
    elif key == "metadata":
        return json.dumps(value) if value else "{}"
    return value


def _write_merged_metadata(s, content_hash, best, meta_best, create_md_keys, latest_updated_at, payload):
    """Merge resolved fields onto the current row and UPDATE. Returns True (no-op if row gone)."""
    row = s.conn.execute(
        "SELECT metadata FROM memories WHERE content_hash = ? AND deleted_at IS NULL",
        (content_hash,),
    ).fetchone()
    if row is None:
        logger.debug("update_metadata for unknown/absent hash %s — skipped", _sanitize_log_value(content_hash))
        return True

    # metadata = resolved event keys ∪ purely-local keys the current row has that no event
    # owns (local-only, never serialized into events — arch ressalva B confirmed), minus keys
    # the dead incarnation's create owned (so they don't resurface).
    current_md = _parse_current_metadata(row)
    event_owned = set(meta_best) | create_md_keys
    merged_md = {kk: vv for kk, vv in current_md.items() if kk not in event_owned}
    merged_md.update(meta_best)
    best = dict(best)
    best["metadata"] = merged_md

    set_clauses, params = [], []
    for key, value in best.items():
        set_clauses.append(f"{key} = ?")
        params.append(_serialize_field_value(key, value))

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


def _materialize_update_metadata(s, content_hash: str, payload: Dict[str, Any]) -> bool:
    """Materialize an update_metadata event: order-independent, recreation-safe convergence.

    Thin orchestration of three steps (kept small for the complexity budget):
      1. _load_live_events: parse events + elect the winning create (recreation boundary),
         Greptile apply:315/368.
      2. _resolve_md_fields: field-level LWW by _winner_key over the live incarnation,
         Greptile apply:328 (same order as the resolver).
      3. _write_merged_metadata: per-key metadata merge preserving purely-local keys,
         Greptile apply:334.
    """
    parsed, create_key = _load_live_events(s, content_hash)
    best, meta_best, create_md_keys, latest_updated_at = _resolve_md_fields(parsed, create_key, content_hash)
    return _write_merged_metadata(s, content_hash, best, meta_best, create_md_keys, latest_updated_at, payload)


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
            return _materialize_update_metadata(s, content_hash, payload)

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
                 deleted_at, embedding_pending, store, superseded_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)
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
                # Greptile P1-4: carry superseded_by so a replaced memory stays hidden from
                # search on the target (retrieve filters on this column, not metadata).
                payload.get("superseded_by"),
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

    MONOTONIC (Phase 5, Tuvok P1): the cursor is "how far we have seen from this peer" and
    MUST NOT move backwards. An ON CONFLICT keeps MAX(existing, incoming), so a bootstrap
    install (or a stale/duplicate call) can never rewind a cursor that live sync already
    advanced past the baseline watermark — which would re-pull/re-process events (R6).

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
                INSERT INTO sync_cursor (peer_id, last_seq_seen, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(peer_id) DO UPDATE SET
                    last_seq_seen = MAX(sync_cursor.last_seq_seen, excluded.last_seq_seen),
                    updated_at = excluded.updated_at
            """, (peer_id, last_seq, time.time()))
            # Durability (§8.5 / ADR-0019): the cursor and the applied events of the batch
            # must survive a crash. Commit here closes the batch transaction atomically.
            s.conn.commit()
        logger.debug("Advanced sync cursor for peer %s to seq %s",
                     _sanitize_log_value(peer_id), _sanitize_log_value(last_seq))
    except Exception as e:
        logger.error("Error advancing sync cursor: %s", _sanitize_log_value(e))
        raise
