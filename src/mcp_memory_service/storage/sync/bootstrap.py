"""
Delta-sync Phase 5: bootstrap a new/reinstalled spoke from a peer's current state.

A new peer has no event history for the corpus that predates its join. generate_baseline()
reads the SOURCE's current memories (and tombstones) ON DEMAND and emits a deterministic,
idempotent set of baseline events plus a durable watermark (the event-log position at the
snapshot). install_baseline() applies those events on the TARGET via the existing Phase 4
apply path and parks the per-peer cursor at the watermark, so live pull/push continues from
there with no reprocessing.

Reuses, never reinvents (G1): events are applied through apply_remote_event (idempotent by
immutable (agent_id, event_id)); the cursor is set via advance_sync_cursor.

RFC §9.3 (bootstrap). SPEC: spec-delta-sync-fase5.md.
"""

import hashlib
import json
import logging
from typing import Any, Dict, List, Tuple

from ..base import MemoryStorage
from ..mixins.base import _sanitize_log_value
from .apply import apply_remote_event, advance_sync_cursor, _sqlite, _sync_lock, KNOWN_SCHEMA_VERSION

logger = logging.getLogger(__name__)


def _deterministic_event_id(content_hash: str, updated_at, op: str) -> str:
    """Stable event id for a baseline row (R2): same state → same id, never a random UUID.

    Keyed on (content_hash, updated_at, op) so re-running bootstrap on an unchanged memory
    produces the identical identity, making install idempotent through the apply path's
    (agent_id, event_id) uniqueness.
    """
    raw = f"baseline:{op}:{content_hash}:{updated_at}".encode("utf-8")
    return "bl-" + hashlib.sha256(raw).hexdigest()[:32]


def _read_snapshot(storage: MemoryStorage):
    """Read the atomic snapshot: (memory rows, watermark, hlc_by_hash). Extracted to keep
    generate_baseline under the repo complexity limit (Greptile P2).

    R3 (Tuvok P1-2): the threading lock alone does NOT make the cut atomic — a writer
    (store/delete) releases the shared connection lock BETWEEN its savepoint and its commit.
    BEGIN IMMEDIATE takes the SQLite RESERVED lock so this read serializes against any
    writer's commit: watermark and the memory snapshot come from one committed point.
    """
    s = _sqlite(storage)
    with _sync_lock(storage):
        s.conn.execute("BEGIN IMMEDIATE")
        try:
            row = s.conn.execute("SELECT COALESCE(MAX(seq), 0) FROM sync_events").fetchone()
            watermark = int(row[0]) if row and row[0] is not None else 0

            rows = s.conn.execute(
                """
                SELECT content_hash, content, tags, memory_type, metadata,
                       created_at, updated_at, deleted_at, store, superseded_by
                FROM memories
                ORDER BY id
                """
            ).fetchall()

            # Greptile P1-2/rodada2: carry the source's REAL conflict clock, not a floor of 0.
            # Take the (hlc_physical, hlc_logical) PAIR from the SAME latest event per hash —
            # NOT MAX of each column independently, which could invent a clock newer than any
            # real event ((1000,100)+(2000,0) -> (2000,100)) and make a later real edit lose.
            # "Latest" = the event with no other event of a strictly greater HLC pair.
            hlc_rows = s.conn.execute(
                """
                SELECT e.content_hash, e.hlc_physical, e.hlc_logical
                FROM sync_events e
                WHERE NOT EXISTS (
                    SELECT 1 FROM sync_events e2
                    WHERE e2.content_hash = e.content_hash
                      AND (e2.hlc_physical, e2.hlc_logical) > (e.hlc_physical, e.hlc_logical)
                )
                """
            ).fetchall()
            hlc_by_hash = {r[0]: (r[1] or 0, r[2] or 0) for r in hlc_rows}
        finally:
            # Read-only transaction: end it without writing (COMMIT releases the lock).
            s.conn.commit()
    return rows, watermark, hlc_by_hash


def generate_baseline(storage: MemoryStorage) -> Tuple[List[Dict[str, Any]], int]:
    """Generate baseline events from the current state, plus the watermark. ON DEMAND (G1).

    Returns (events, watermark). The watermark is MAX(seq) of sync_events read in the SAME
    transaction as the memory snapshot (R3 atomic cut): any memory mutated concurrently is
    either captured here or produces an event with seq > watermark — never lost.
    """
    rows, watermark, hlc_by_hash = _read_snapshot(storage)

    events: List[Dict[str, Any]] = []
    for r in rows:
        ev = _row_to_baseline_event(r, hlc_by_hash)
        if ev is not None:
            events.append(ev)

    logger.info("Generated %s baseline events at watermark %s",
                _sanitize_log_value(len(events)), _sanitize_log_value(watermark))
    return events, watermark


def _row_to_baseline_event(r, hlc_by_hash) -> Dict[str, Any]:
    """Convert one memories row into a baseline event (helper keeps generate_baseline under
    the repo's complexity limit — Greptile P2-2). Access by position: upstream rows are plain
    tuples (DictRow is fork-only). SELECT order: content_hash, content, tags, memory_type,
    metadata, created_at, updated_at, deleted_at, store, superseded_by.
    """
    (content_hash, content, tags_str, memory_type, metadata_str,
     created_at, updated_at, deleted_at, store, superseded_by) = (
        r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9],
    )
    try:
        metadata = json.loads(metadata_str) if metadata_str else {}
    except (TypeError, ValueError):
        metadata = {}

    # R4: preserve metadata.agent_id; no verifiable authorship → 'unattributed' (clean
    # single token, no slash, so it stays a valid agent_id in filters). Never invent an agent.
    agent_id = metadata.get("agent_id") or "unattributed"

    # Greptile P1-2: use the source's real HLC for this hash (floor 0 would lose to history).
    hlc_physical, hlc_logical = hlc_by_hash.get(
        content_hash,
        (int((created_at or 0) * 1000), 0),
    )

    if deleted_at is not None:
        op = "delete"
        payload = {"content_hash": content_hash, "deleted_at": deleted_at}
    else:
        op = "create"
        tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []
        payload = {
            "content_hash": content_hash,
            "content": content,
            "memory_type": memory_type,
            "tags": tags,
            "created_at": created_at,
            "updated_at": updated_at,
            "metadata": metadata,
            "store": store or "default",
            # Greptile P1-4: carry superseded_by so a replaced memory is NOT re-exposed as a
            # searchable create on the target (retrieve filters on this column).
            "superseded_by": superseded_by,
        }

    return {
        "schema_version": KNOWN_SCHEMA_VERSION,
        "agent_id": agent_id,
        "event_id": _deterministic_event_id(content_hash, updated_at, op),
        "op": op,
        "content_hash": content_hash,
        "hlc_physical": hlc_physical,
        "hlc_logical": hlc_logical,
        "embedding_model": None,
        "embedding_dim": None,
        "payload": payload,
    }


def install_baseline(storage: MemoryStorage, events: List[Dict[str, Any]],
                     peer_id: str, watermark: int) -> Dict[str, int]:
    """Install a baseline on a (fresh) peer and park the cursor at the watermark (R6).

    Applies each baseline event through the Phase 4 apply path (idempotent), then advances
    the per-peer sync cursor to the watermark so subsequent live pulls start after it, with
    no reprocessing. Re-installing the same baseline is a no-op (idempotency comes from the
    deterministic event ids + apply's (agent_id, event_id) uniqueness).
    """
    applied = failed = 0
    for ev in events:
        res = apply_remote_event(storage, ev)
        # Classify by the structured ApplyResult.applied flag, not by free-text `reason`
        # (Tuvok P2-1). For baseline install, applied=True covers both a fresh materialize
        # and a benign duplicate/idempotent re-install (the peer already holds the memory);
        # applied=False is the only genuine failure (rejected/unwritable event).
        if res.applied:
            applied += 1
        else:
            failed += 1
            logger.warning("Baseline event %s not applied: %s",
                           _sanitize_log_value(ev.get("event_id")), _sanitize_log_value(res.reason))

    # R6 / Greptile P1-1: only park the cursor at the watermark if EVERY baseline event
    # applied. If any failed (e.g. a rejected newer schema_version), advancing the cursor
    # would make later pulls start after the watermark and skip the history needed to
    # recover that memory/deletion. Leave the cursor unchanged and report incomplete.
    complete = failed == 0
    if complete:
        advance_sync_cursor(storage, peer_id, watermark)
    else:
        logger.warning("Bootstrap from peer %s INCOMPLETE: %s failed — cursor NOT advanced (watermark %s withheld)",
                       _sanitize_log_value(peer_id), failed, watermark)
    logger.info("Installed baseline from peer %s: %s applied, %s failed, complete=%s, cursor=%s",
                _sanitize_log_value(peer_id), applied, failed, complete,
                watermark if complete else "unchanged")
    return {"applied": applied, "failed": failed, "watermark": watermark, "complete": complete}


async def bootstrap_from_peer(local_storage: MemoryStorage, peer, peer_id: str) -> Dict[str, int]:
    """Bootstrap this (new) spoke from a peer over the transport (delta-sync Phase 5).

    Fetches the peer's baseline (GET /api/sync/baseline), installs it, and parks the cursor
    at the peer's watermark — after which the normal pull loop (sync_from_peer) continues
    from that watermark with no reprocessing. `peer` is a RemoteHTTPStorage-like adapter
    exposing `get_baseline()`.
    """
    logger.info("Bootstrapping from peer %s", _sanitize_log_value(peer_id))
    events, watermark = await peer.get_baseline()
    result = install_baseline(local_storage, events, peer_id=peer_id, watermark=watermark)
    logger.info("Bootstrap from peer %s complete: %s", _sanitize_log_value(peer_id), result)
    return result
