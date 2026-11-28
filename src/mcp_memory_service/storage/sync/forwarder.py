"""Forward delta-sync events to a hybrid hub's secondary backend (#1506).

``apply_remote_event`` writes through the primary's SQLite connection, so a spoke's
create, edit or delete never reaches the hybrid queue that feeds Cloudflare. This
module tails the hub's own ``sync_events`` log with a persisted cursor per secondary
(the same ``push_cursor`` table and ``push_to_peer`` loop a spoke uses toward a hub,
ADR-0027) and replays each event to the secondary.

Why not enqueue from the apply path: ``BackgroundSyncService.operation_queue`` is an
in-memory ``asyncio.Queue``. A restart, or a full queue, between "applied on the hub"
and "pushed to D1" would lose the change with nothing left to notice it. The event
log is durable and the cursor only moves after the secondary acknowledged.

Passport
    in:      a ``HybridMemoryStorage`` (``.primary``, ``.secondary``, ``.sync_service``)
    out:     ``PushResult`` from ``push_to_peer``; one ``secondary_forwarded`` row per
             forwarded event (the identity a later pull can use to recognise its echo)
    callers: ``consolidation/scheduler.py`` ``_run_sync_cycle``
    breaks:  forwarding stops, Cloudflare keeps only the hub's own writes (#1506)
"""

from __future__ import annotations

import logging
import sqlite3
import time
from typing import Any

from ..mixins.base import _sanitize_log_value
from .apply import _sqlite, _sync_lock
from .orchestrator import PushResult, advance_push_cursor, push_to_peer

logger = logging.getLogger(__name__)

# Phrases a secondary answers a delete with when the row is already gone: the delete
# has nothing left to do, so the event counts as delivered.
_ALREADY_GONE = ("not found", "already deleted", "does not exist")


def secondary_id_for(secondary: Any) -> str:
    """Stable cursor key for a secondary, e.g. ``secondary:cloudflarestorage``."""
    return "secondary:" + type(secondary).__name__.lower()


def _row_store(conn, content_hash: str) -> str | None:
    row = conn.execute(
        "SELECT store FROM memories WHERE content_hash = ? AND deleted_at IS NULL", (content_hash,)
    ).fetchone()
    return row[0] if row else None


_ECHO_OPS = {"delete": ("delete",), "edit": ("create", "update", "update_metadata")}
_ECHO_CHUNK = 500  # hashes per lookup, under SQLite's bound-parameter limit


def forwarded_echoes(primary: Any, secondary: Any, stamps: dict[str, float], kind: str) -> set[str]:
    """Hashes whose secondary state is only the echo of the hub's own forward.

    The pull sees a memory row or a tombstone, not an event, so it cannot tell a spoke
    edit the hub forwarded from an edit an outside machine made on D1. D1 stamps the
    forwarded write (``updated_at`` on a tag change, ``deleted_at`` on a delete) with
    the hub's clock, inside the window ``SecondaryForwarder`` records around the call:
    ``started_at`` before the secondary is called, ``forwarded_at`` after it
    acknowledged. A stamp inside that window was written by the forward. A forward
    still pending (``acked = 0``: in flight, or the hub died between the secondary's
    ack and its own record) has no upper bound yet.

    ``kind`` is ``"delete"`` (tombstones) or ``"edit"`` (rows newer than the hub's).
    Returns an empty set when the hub predates the forward table (no forwarder ran);
    any other database error is raised, so a busy database cannot switch the guard off.
    """
    conn = getattr(primary, "conn", None)
    if conn is None or not stamps:
        return set()
    ops = _ECHO_OPS[kind]
    sid = secondary_id_for(secondary)
    hashes = list(stamps)
    echoes: set[str] = set()
    try:
        for i in range(0, len(hashes), _ECHO_CHUNK):
            chunk = hashes[i:i + _ECHO_CHUNK]
            rows = conn.execute(
                "SELECT content_hash, started_at, forwarded_at, acked FROM secondary_forwarded "
                f"WHERE secondary_id = ? AND op IN ({','.join('?' * len(ops))}) "
                f"AND content_hash IN ({','.join('?' * len(chunk))}) AND started_at IS NOT NULL",
                (sid, *ops, *chunk),
            ).fetchall()
            for content_hash, started_at, forwarded_at, acked in rows:
                stamp = stamps[content_hash]
                if stamp >= started_at and (not acked or stamp <= forwarded_at):
                    echoes.add(content_hash)
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc) or "no such column" in str(exc):
            return set()  # migrations 023/024 not applied: no forwarder ever ran here
        raise
    return echoes


class SecondaryForwarder:
    """``PushPeerAdapter`` whose peer is the hybrid hub's own secondary backend."""

    ontology_sync = False  # memory events only: the secondary has no ontology overlay

    def __init__(self, hybrid: Any):
        self.hybrid = hybrid
        self.secondary_id = secondary_id_for(hybrid.secondary)

    async def push_events(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        failed = 0
        for event in events:
            self._begin(event)
            try:
                status = await self._forward_one(event)
            except Exception as exc:  # noqa: BLE001  the cursor must stop at the first failure
                logger.warning(
                    "Forward to %s failed for event %s: %s: %s",
                    _sanitize_log_value(self.secondary_id),
                    _sanitize_log_value(event.get("event_id")),
                    _sanitize_log_value(type(exc).__name__),
                    _sanitize_log_value(exc),
                )
                results.append({"event_id": event["event_id"], "status": "failed"})
                failed += 1
                break
            results.append({"event_id": event["event_id"], "status": status})
            self._record(event)
        return {"results": results, "failed": failed}

    async def _forward_one(self, event: dict[str, Any]) -> str:
        op = event["op"]
        content_hash = event["content_hash"]
        if op == "delete":
            return await self._forward_delete(content_hash)

        # create / update_metadata: replay the CURRENT primary row, not the event body.
        # The hub already resolved conflicts when it applied the event, so the row is
        # the winner; a later event for the same hash will replay it again, harmlessly.
        primary = self.hybrid.primary
        state = await primary.get_sync_state_for_hash(content_hash)
        if state is None or state[1] is not None:
            return "skipped_gone"  # tombstoned since: its delete event follows in the log
        memory = await primary.get_by_hash(content_hash)
        if memory is None:
            raise RuntimeError("primary row unreadable")
        if op == "update_metadata":
            return await self._forward_edit(memory)
        return await self._forward_create(memory)

    async def _forward_delete(self, content_hash: str) -> str:
        state = await self.hybrid.primary.get_sync_state_for_hash(content_hash)
        if state is not None and state[1] is None:
            # Re-created on the hub after this delete: the delete is stale, and the
            # create event later in the log carries the live row to the secondary.
            return "skipped_recreated"
        ok, message = await self.hybrid.secondary.delete(content_hash)
        if not ok and not any(p in (message or "").lower() for p in _ALREADY_GONE):
            raise RuntimeError(f"delete refused: {message}")
        return "applied"

    async def _apply_update(self, memory: Any) -> None:
        from ..hybrid import SyncOperation

        updates = {"tags": memory.tags, "memory_type": memory.memory_type,
                   "metadata": memory.metadata}
        await self.hybrid.sync_service._process_single_operation(
            SyncOperation(operation="update", content_hash=memory.content_hash,
                          updates=updates, preserve_timestamps=True)
        )

    async def _forward_edit(self, memory: Any) -> str:
        try:
            await self._apply_update(memory)
            return "applied"
        except Exception as exc:  # noqa: BLE001  maybe the secondary never got the row
            logger.debug("Update not applied on the secondary, creating: %s",
                         _sanitize_log_value(exc))
        # The row may predate the forwarder: create it. A duplicate refusal here means the
        # row exists and the update itself failed, so it is NOT a delivered edit: raise,
        # the cursor stays put and the next cycle retries the update.
        return await self._forward_create(memory, duplicate_is_delivered=False)

    async def _forward_create(self, memory: Any, duplicate_is_delivered: bool = True) -> str:
        from ..hybrid import SyncOperation

        store = _row_store(_sqlite(self.hybrid).conn, memory.content_hash)
        try:
            await self.hybrid.sync_service._process_single_operation(
                SyncOperation(operation="store", memory=memory, store=store)
            )
        except RuntimeError as exc:
            # For a create, a duplicate means a hub-own write already travelled through
            # the hybrid queue.
            if not duplicate_is_delivered or "duplicate content" not in str(exc).lower():
                raise
            # A delete + re-create on a spoke can land before the delete is forwarded
            # (the delete is skipped as stale), so the secondary still holds the OLD
            # row: refresh its tags and metadata from the primary row before acking.
            await self._apply_update(memory)
        return "applied"

    def _begin(self, event: dict[str, Any]) -> None:
        """Mark the forward as in flight BEFORE the secondary is called.

        The pull's echo guard needs the window D1 may stamp in. Writing it first means a
        pull that runs mid-forward, or after a crash between the secondary's ack and
        ``_record``, still sees the forward. A retry keeps the earliest ``started_at``,
        since a crashed attempt may already have stamped D1.
        """
        s = _sqlite(self.hybrid)
        now = time.time()
        with _sync_lock(self.hybrid):
            s.conn.execute(
                """
                INSERT INTO secondary_forwarded
                    (secondary_id, agent_id, event_id, content_hash, op, forwarded_at,
                     started_at, acked)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT (secondary_id, agent_id, event_id) DO UPDATE SET
                    started_at = MIN(COALESCE(started_at, excluded.started_at),
                                     excluded.started_at),
                    acked = 0
                """,
                (self.secondary_id, event.get("agent_id") or "unknown", event["event_id"],
                 event["content_hash"], event["op"], now, now),
            )
            s.conn.commit()

    def _record(self, event: dict[str, Any]) -> None:
        """Remember which event put which (hash, op) on the secondary, and close its window."""
        s = _sqlite(self.hybrid)
        now = time.time()
        with _sync_lock(self.hybrid):
            s.conn.execute(
                """
                INSERT INTO secondary_forwarded
                    (secondary_id, agent_id, event_id, content_hash, op, forwarded_at,
                     started_at, acked)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1)
                ON CONFLICT (secondary_id, agent_id, event_id) DO UPDATE SET
                    forwarded_at = excluded.forwarded_at, acked = 1
                """,
                (self.secondary_id, event.get("agent_id") or "unknown", event["event_id"],
                 event["content_hash"], event["op"], now, now),
            )
            s.conn.commit()


async def forward_to_secondary(hybrid: Any, limit: int = 100) -> PushResult | None:
    """Forward new ``sync_events`` to the hybrid secondary; ``None`` when not applicable.

    First run on a hub that already holds history starts at the CURRENT end of the log:
    the existing drift scan and ``force_sync`` own the backlog, and replaying the whole
    log to D1 on upgrade would be a surprise bill.
    """
    if getattr(hybrid, "secondary", None) is None or getattr(hybrid, "sync_service", None) is None:
        return None
    if getattr(hybrid, "_sync_paused", False):
        return None
    s = _sqlite(hybrid)
    if not hasattr(s, "conn"):
        return None
    adapter = SecondaryForwarder(hybrid)
    peer_id = adapter.secondary_id
    has_cursor = s.conn.execute(
        "SELECT 1 FROM push_cursor WHERE peer_id = ?", (peer_id,)
    ).fetchone()
    if not has_cursor:
        with _sync_lock(hybrid):  # same connection as local writes: no half-committed seq
            end = s.conn.execute("SELECT COALESCE(MAX(seq), 0) FROM sync_events").fetchone()[0]
        advance_push_cursor(hybrid, peer_id, end)
        logger.info("Secondary forwarder %s starts at seq %s", _sanitize_log_value(peer_id), end)
    return await push_to_peer(hybrid, adapter, peer_id, limit=limit)
