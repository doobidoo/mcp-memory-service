"""
Delta-sync Phase 4b: pull orchestration (sync_from_peer).

Orchestrates the complete pull flow: cursor → get_events_since → apply → advance cursor.
Operates over a local MemoryStorage and a peer adapter (RemoteHTTPStorage or test adapter).

ADR-0023: Separate from hybrid sync loop (no fusion with state-based sync)
ADR-0026: Manual invocation first (no scheduler)
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, Tuple

from ..base import MemoryStorage
from ..mixins.base import _sanitize_log_value
from .apply import apply_remote_event, advance_sync_cursor, _sqlite, _sync_lock
from .bootstrap import bootstrap_from_peer

logger = logging.getLogger(__name__)


@dataclass
class SyncResult:
    """Result of sync_from_peer operation."""
    events_applied: int
    pages_processed: int
    final_cursor: int
    events_failed: int = 0
    events_received: int = 0


class PeerAdapter(Protocol):
    """Protocol for peer adapters (RemoteHTTPStorage or test adapter)."""

    async def get_events_since(self, since_seq: int, limit: int) -> Tuple[List[Dict[str, Any]], int, bool]:
        """
        Get events from peer since given sequence number.

        Returns: (events, next_seq, has_more)
        """
        pass


async def sync_from_peer(
    local_storage: MemoryStorage,
    peer: PeerAdapter,
    peer_id: str,
    limit: int = 100
) -> SyncResult:
    """
    Orchestrate pulling and applying events from a peer.

    Implements the complete pull flow:
    1. Read current sync cursor (default 0 if not exists)
    2. Loop: get_events_since → apply_remote_event → advance_sync_cursor
    3. Continue until has_more is false
    4. Return summary result

    Args:
        local_storage: Local memory storage instance
        peer: Peer adapter implementing get_events_since
        peer_id: Unique identifier for the peer
        limit: Maximum events per page (default 100)

    Returns:
        SyncResult with events_applied, pages_processed, and final_cursor
    """
    logger.info("Starting sync from peer %s", _sanitize_log_value(peer_id))

    # Step 1: Read current cursor (default to 0 if not exists)
    cursor_result = _sqlite(local_storage).conn.execute(
        "SELECT last_seq_seen FROM sync_cursor WHERE peer_id = ?",
        (peer_id,)
    ).fetchone()

    # First sync with this peer (no cursor row): install the peer's baseline before the
    # pull loop so a fresh spoke gets the pre-existing corpus, not only events since seq 0
    # (#1522 finding #3 — delta-sync Phase 5 bootstrap was unreachable). Only when the peer
    # can serve a baseline; otherwise fall back to the legacy pull-from-0 (back-compat).
    if cursor_result is None and hasattr(peer, "get_baseline"):
        logger.info("First sync with peer %s: bootstrapping from baseline", _sanitize_log_value(peer_id))
        try:
            await bootstrap_from_peer(local_storage, peer, peer_id)
        except Exception as e:
            logger.warning("Bootstrap from peer %s failed, falling back to pull from 0: %s",
                           _sanitize_log_value(peer_id), _sanitize_log_value(e))
        # Re-read the cursor: bootstrap parks it at the peer's watermark.
        cursor_result = _sqlite(local_storage).conn.execute(
            "SELECT last_seq_seen FROM sync_cursor WHERE peer_id = ?",
            (peer_id,)
        ).fetchone()

    cursor = cursor_result[0] if cursor_result else 0
    initial_cursor = cursor

    logger.debug("Starting sync from cursor %s", _sanitize_log_value(cursor))

    # Initialize result counters
    events_applied = 0
    pages_processed = 0
    events_received = 0
    events_failed = 0

    # Step 2: Loop through pages until no more events
    while True:
        logger.debug("Fetching events since seq %s with limit %s",
                     _sanitize_log_value(cursor), limit)

        # Get events from peer
        try:
            events, next_seq, has_more = await peer.get_events_since(cursor, limit)
        except Exception as e:
            logger.error("Failed to get events from peer %s: %s",
                         _sanitize_log_value(peer_id), _sanitize_log_value(e))
            raise

        # If no events returned, we're done
        if not events:
            logger.debug("No more events to process")
            break

        pages_processed += 1
        events_received += len(events)
        logger.debug("Processing page %s with %s events", pages_processed, len(events))

        # Step 3: Apply each event in the page. FAIL-STOP (§8.5): on the first event
        # that fails to apply, stop and advance the cursor ONLY to the last event that
        # applied successfully — never past an unapplied event, or it would be lost
        # permanently (re-sync starts after the cursor).
        last_good_seq = cursor
        page_failed = False
        for event in events:
            event_seq = event.get("seq")
            try:
                result = apply_remote_event(local_storage, event)
            except Exception as e:
                logger.error("Error applying event %s (seq=%s): %s",
                             _sanitize_log_value(event.get('event_id', 'unknown')),
                             _sanitize_log_value(event_seq), _sanitize_log_value(e))
                events_failed += 1
                page_failed = True
                break
            if result.applied:
                events_applied += 1
                if event_seq is not None:
                    last_good_seq = event_seq
                logger.debug("Applied event %s: %s",
                             _sanitize_log_value(event.get('event_id', 'unknown')), result.reason)
            elif getattr(result, "skippable", False):
                # Permanently non-materializable event (e.g. create with irrecoverable
                # empty content). Advance past it so a handful of bad events cannot block
                # the whole feed. It stays recorded in the local event log; it is simply
                # not materialized into a memory row.
                if event_seq is not None:
                    last_good_seq = event_seq
                logger.warning(
                    "Skipping event %s (seq=%s): %s",
                    _sanitize_log_value(event.get("event_id", "unknown")),
                    _sanitize_log_value(event_seq),
                    _sanitize_log_value(result.reason),
                )
            else:
                # applied=False means the event was not accepted (not a benign dup).
                logger.warning("Event %s (seq=%s) not applied: %s",
                             _sanitize_log_value(event.get('event_id', 'unknown')),
                             _sanitize_log_value(event_seq), result.reason)
                events_failed += 1
                page_failed = True
                break

        # Step 4: Advance cursor to the last successfully applied seq (durably). On a
        # partial-failure page this is < next_seq, so the failed event is re-pulled next
        # time; on a clean page last_good_seq == next_seq.
        advance_target = last_good_seq if page_failed else next_seq
        try:
            advance_sync_cursor(local_storage, peer_id, advance_target)
            cursor = advance_target
            logger.debug("Advanced cursor to %s", _sanitize_log_value(cursor))
        except Exception as e:
            logger.error("Failed to advance cursor for peer %s: %s",
                         _sanitize_log_value(peer_id), _sanitize_log_value(e))
            raise

        # Step 5: Stop on partial failure (do not silently skip the failed event) or when
        # the feed is exhausted.
        if page_failed:
            logger.warning("Stopping sync from %s at seq %s due to apply failure",
                         _sanitize_log_value(peer_id), _sanitize_log_value(cursor))
            break
        if not has_more:
            logger.debug("No more pages available")
            break

    final_cursor = cursor
    logger.info(
        "Completed sync from peer %s: %s events applied, %s pages processed, "
        "cursor: %s → %s",
        _sanitize_log_value(peer_id), events_applied, pages_processed,
        _sanitize_log_value(initial_cursor), _sanitize_log_value(final_cursor),
    )

    return SyncResult(
        events_applied=events_applied,
        pages_processed=pages_processed,
        final_cursor=final_cursor,
        events_failed=events_failed,
        events_received=events_received,
    )


# --- Phase 4c: outbound push (spoke → hub), ADR-0027 ---

import time
import json


@dataclass
class PushResult:
    """Result of push_to_peer operation."""
    events_pushed: int
    events_failed: int
    final_seq: int


class PushPeerAdapter(Protocol):
    """Protocol for push targets (RemoteHTTPStorage or test adapter)."""

    async def push_events(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """POST events to the peer; returns {results, applied, skipped, failed}."""
        pass


def get_push_cursor(storage: MemoryStorage, peer_id: str) -> int:
    """Read last_seq_pushed for a peer (0 if none). Dedicated push_cursor table (ADR-0027)."""
    row = _sqlite(storage).conn.execute(
        "SELECT last_seq_pushed FROM push_cursor WHERE peer_id = ?", (peer_id,)
    ).fetchone()
    return row[0] if row else 0


def advance_push_cursor(storage: MemoryStorage, peer_id: str, last_seq: int) -> None:
    """Advance push_cursor.last_seq_pushed for a peer, committing the batch (ADR-0027)."""
    try:
        s = _sqlite(storage)
        # Hold the connection lock: shares the SQLite connection with local writes
        # and must not interleave with an in-flight savepoint (Greptile P1).
        with _sync_lock(storage):
            s.conn.execute(
                """
                INSERT OR REPLACE INTO push_cursor (peer_id, last_seq_pushed, updated_at)
                VALUES (?, ?, ?)
                """,
                (peer_id, last_seq, time.time()),
            )
            s.conn.commit()
        logger.debug("Advanced push cursor for peer %s to seq %s",
                     _sanitize_log_value(peer_id), _sanitize_log_value(last_seq))
    except Exception as e:
        logger.error("Error advancing push cursor: %s", _sanitize_log_value(e))


def _read_local_events_since(storage: MemoryStorage, since_seq: int, limit: int) -> List[Dict[str, Any]]:
    """Read local sync_events with seq > since_seq, enriching create events with content (R8)."""
    cur = _sqlite(storage).conn.execute(
        """
        SELECT seq, schema_version, event_id, op, content_hash, agent_id, hlc_physical, hlc_logical,
               embedding_model, embedding_dim, payload
        FROM sync_events
        WHERE seq > ?
        ORDER BY seq
        LIMIT ?
        """,
        (since_seq, limit),
    )
    events = []
    for row in cur.fetchall():
        seq, schema_version, event_id, op, content_hash, agent_id, hlc_p, hlc_l, emb_model, emb_dim, payload = row
        payload_dict = json.loads(payload) if payload else {}
        if op == "create":
            mrow = _sqlite(storage).conn.execute(
                "SELECT content FROM memories WHERE content_hash = ? AND deleted_at IS NULL",
                (content_hash,),
            ).fetchone()
            if mrow and mrow[0]:
                # Only override with table content when it is non-empty; an empty/zeroed
                # table row must not clobber the real content carried in the stored create
                # payload, or the push would send a recoverable create as empty content
                # (same bug the pull feed had — Greptile #1499 sync_events:240).
                payload_dict["content"] = mrow[0]
        events.append({
            "seq": seq, "schema_version": schema_version if schema_version is not None else 1,
            "event_id": event_id, "op": op, "content_hash": content_hash,
            "agent_id": agent_id, "hlc_physical": hlc_p, "hlc_logical": hlc_l,
            "embedding_model": emb_model, "embedding_dim": emb_dim, "payload": payload_dict,
        })
    return events


async def push_to_peer(
    local_storage: MemoryStorage,
    peer: PushPeerAdapter,
    peer_id: str,
    limit: int = 100,
) -> PushResult:
    """
    Push local events to a peer (delta-sync Phase 4c, ADR-0027).

    Reads local sync_events with seq > push_cursor, posts them in seq order (paginated),
    and advances push_cursor ONLY after the peer acks a batch (R6, resumable). Stops at the
    first batch that reports failures without advancing past the last fully-acked seq.
    """
    logger.info("Starting push to peer %s", _sanitize_log_value(peer_id))
    initial = get_push_cursor(local_storage, peer_id)
    cursor = initial
    pushed = failed = 0

    while True:
        events = _read_local_events_since(local_storage, cursor, limit)
        if not events:
            break
        resp = await peer.push_events(events)
        batch_failed = resp.get("failed", 0)
        if batch_failed:
            logger.warning("Push to %s stopped: %s failed in batch",
                     _sanitize_log_value(peer_id), _sanitize_log_value(batch_failed))
            # R6 (resumable): advance only to the last seq that succeeded CONTIGUOUSLY
            # before the FIRST failure. Events after the first failure must be re-pushed
            # (the hub applies in order; we cannot skip a failed event).
            status_by_id = {r["event_id"]: r.get("status") for r in resp.get("results", [])}
            last_ok_seq = cursor  # nothing new acked yet
            ok_count = 0
            for e in events:  # events are in seq order
                st = status_by_id.get(e["event_id"], "failed")
                if st == "failed":
                    break
                last_ok_seq = e["seq"]
                ok_count += 1
            failed += batch_failed
            if last_ok_seq > cursor:
                cursor = last_ok_seq
                advance_push_cursor(local_storage, peer_id, cursor)
                pushed += ok_count
            break
        # whole batch acked (applied or skipped_duplicate)
        cursor = events[-1]["seq"]
        advance_push_cursor(local_storage, peer_id, cursor)
        pushed += len(events)
        if len(events) < limit:
            break

    logger.info("Completed push to %s: %s pushed, %s failed, cursor %s → %s",
                 _sanitize_log_value(peer_id), pushed, failed,
                 _sanitize_log_value(initial), _sanitize_log_value(cursor))
    return PushResult(events_pushed=pushed, events_failed=failed, final_seq=cursor)
