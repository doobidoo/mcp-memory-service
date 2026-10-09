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
import time
from typing import Any, Dict, List, Tuple

from ..base import MemoryStorage
from ..mixins.base import _sanitize_log_value
from .apply import apply_remote_event, advance_sync_cursor, _sqlite, _sync_lock, KNOWN_SCHEMA_VERSION

logger = logging.getLogger(__name__)

# Baseline events are HLC-ordered before any real post-watermark event. They seed the clock
# at the floor; a real local edit afterwards gets a higher HLC and wins over the baseline.
_BASELINE_HLC_PHYSICAL = 0


def _deterministic_event_id(content_hash: str, updated_at, op: str) -> str:
    """Stable event id for a baseline row (R2): same state → same id, never a random UUID.

    Keyed on (content_hash, updated_at, op) so re-running bootstrap on an unchanged memory
    produces the identical identity, making install idempotent through the apply path's
    (agent_id, event_id) uniqueness.
    """
    raw = f"baseline:{op}:{content_hash}:{updated_at}".encode("utf-8")
    return "bl-" + hashlib.sha256(raw).hexdigest()[:32]


def generate_baseline(storage: MemoryStorage) -> Tuple[List[Dict[str, Any]], int]:
    """Generate baseline events from the current state, plus the watermark. ON DEMAND (G1).

    Returns (events, watermark). The watermark is MAX(seq) of sync_events read in the SAME
    transaction as the memory snapshot (R3 atomic cut): any memory mutated concurrently is
    either captured here or produces an event with seq > watermark — never lost.
    """
    s = _sqlite(storage)
    with _sync_lock(storage):
        # R3 (Tuvok P1-2): the threading lock alone does NOT make the cut atomic — a writer
        # (store/delete) releases the shared connection lock BETWEEN its savepoint and its
        # commit, so reading MAX(seq) under the threading lock could still straddle an
        # in-flight transaction. Take the SQLite RESERVED lock with BEGIN IMMEDIATE so this
        # read serializes against any writer's commit: MAX(seq) and the memory snapshot are
        # read from a single committed point. A concurrent write is then wholly before
        # (in the snapshot + watermark) or wholly after (seq > watermark) — never split.
        s.conn.execute("BEGIN IMMEDIATE")
        try:
            row = s.conn.execute("SELECT COALESCE(MAX(seq), 0) FROM sync_events").fetchone()
            watermark = int(row[0]) if row and row[0] is not None else 0

            cur = s.conn.execute(
                """
                SELECT content_hash, content, tags, memory_type, metadata,
                       created_at, updated_at, deleted_at, store
                FROM memories
                ORDER BY id
                """
            )
            rows = cur.fetchall()
        finally:
            # Read-only transaction: end it without writing (COMMIT releases the lock).
            s.conn.commit()

    events: List[Dict[str, Any]] = []
    for r in rows:
        # Access by position, not by name: the upstream row_factory yields plain tuples
        # (the name-keyed DictRow is fork-only). SELECT order below must match these indices.
        content_hash, content, tags_str, memory_type, metadata_str, created_at, updated_at, deleted_at, store = (
            r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8],
        )
        try:
            metadata = json.loads(metadata_str) if metadata_str else {}
        except (TypeError, ValueError):
            metadata = {}

        # R4: preserve metadata.agent_id; a row with no verifiable authorship is marked
        # 'unattributed' (RFC §9.3 rule 3 "legacy/unattributed") — never invent an agent.
        # Use a clean single-token sentinel (no slash) so it stays a valid agent_id value in
        # sync_events and in any downstream agent filter.
        agent_id = metadata.get("agent_id") or "unattributed"

        if deleted_at is not None:
            # R5: a soft-deleted memory bootstraps as a tombstone, preserving the moment.
            op = "delete"
            event_id = _deterministic_event_id(content_hash, updated_at, op)
            payload = {"content_hash": content_hash, "deleted_at": deleted_at}
        else:
            op = "create"
            event_id = _deterministic_event_id(content_hash, updated_at, op)
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
            }

        events.append({
            "schema_version": KNOWN_SCHEMA_VERSION,
            "agent_id": agent_id,
            "event_id": event_id,
            "op": op,
            "content_hash": content_hash,
            "hlc_physical": _BASELINE_HLC_PHYSICAL,
            "hlc_logical": len(events),  # stable per-run order; floor clock
            "embedding_model": None,
            "embedding_dim": None,
            "payload": payload,
        })

    logger.info("Generated %s baseline events at watermark %s",
                _sanitize_log_value(len(events)), _sanitize_log_value(watermark))
    return events, watermark


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

    # R6: park the cursor at the watermark so live sync continues after the baseline.
    advance_sync_cursor(storage, peer_id, watermark)
    logger.info("Installed baseline from peer %s: %s applied, %s failed, cursor=%s",
                _sanitize_log_value(peer_id), applied, failed, watermark)
    return {"applied": applied, "failed": failed, "watermark": watermark}


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
