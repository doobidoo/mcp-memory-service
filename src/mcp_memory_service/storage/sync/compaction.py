"""
Delta-sync sync_events LOG COMPACTION (Gate G4 implementation).

Collapses the idempotent op='update_metadata' chain per content_hash into a single
consolidated event that carries the LWW-resolved state, giving the sync_events log a bound
without breaking cross-peer convergence.

Design (reuses, never reinvents — G1):
  - Field-level LWW is NOT re-implemented here: the winning fields are read back through the
    SAME resolver the apply path uses (``_load_live_events`` + ``_resolve_md_fields`` elect the
    live incarnation and the per-field winners via ``_winner_key``). The consolidated event is
    built so that re-resolving the (now shorter) log yields a byte-identical verdict (T1/T5/T7).
  - The consolidated event carries the WINNING update_metadata event's HLC PAIR (the single
    smallest ``_winner_key``), never MAX-per-column, so bootstrap._read_snapshot elects the same
    (hlc_physical, hlc_logical) pair and _seed_last_hlc_on_first_boot cannot regress (T2).
  - Only op='update_metadata' rows are ever removed; op='create'/op='delete' are untouched (T3).
  - Watermark frontier: no event with seq > MIN(last_seq_pushed) across push_cursor may be
    removed/rewritten. When ``watermark`` is None it is computed as that MIN; when there is NO
    push_cursor row at all, no peer depends on the log tail, so the frontier is MAX(seq) (the
    whole live chain is collapsible). sync_cursor and push_cursor are never touched (T4).
  - The consolidated event_id is DETERMINISTIC (content_hash + winning HLC pair), so a second
    run recognises the already-consolidated chain and is a no-op (T6a/T6b).
  - Fail-closed: all removals + the insert run inside ONE SAVEPOINT; any error rolls the whole
    thing back, leaving the log byte-identical (T6c).

SPEC: spec-sync-eventlog-compaction.md.
"""

import hashlib
import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

from ..base import MemoryStorage
from ..mixins.base import _sanitize_log_value
from .apply import (
    _sqlite,
    _sync_lock,
    _load_live_events,
    _ev_winner_key,
    _metadata_items,
    _top_level_fields,
)

logger = logging.getLogger(__name__)

# The op of the consolidated event. The protection of free-form vs structured keys is NOT
# done here — it comes from reusing apply._top_level_fields / apply._metadata_items, which
# already classify the payload fields. This constant only names the op string.
_CONSOLIDATED_OP = "update_metadata"


def _compaction_agent_id() -> str:
    """Stable agent id for consolidated events. A dedicated, single-token id (no slash) keeps
    the row a valid, filterable event and makes the consolidated event visibly synthetic without
    impersonating a real author."""
    return "compaction"


def _deterministic_consolidated_event_id(content_hash: str, hlc_physical: int, hlc_logical: int,
                                         updates: Dict) -> str:
    """Stable id for the consolidated update_metadata event (T6b).

    Keyed on (content_hash, winning HLC pair, op, AND a canonical hash of the consolidated
    `updates`). Same inputs → same id, so a second run on an unchanged chain produces the
    identical identity and is recognised as already-consolidated. Including the payload hash
    (Greptile P1#2) ensures two peers that compact DIVERGENT chains sharing the same winning
    HLC pair mint DIFFERENT ids — otherwise identical ids with different payloads would be
    rejected by apply_remote_event as an altered replay, silently dropping the event and
    stalling the pull. Mirrors bootstrap._deterministic_event_id's shape ('prefix-' + sha256[:32]).
    """
    payload_digest = hashlib.sha256(
        json.dumps(updates, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    raw = f"compact:{_CONSOLIDATED_OP}:{content_hash}:{hlc_physical}:{hlc_logical}:{payload_digest}".encode("utf-8")
    return "cmp-" + hashlib.sha256(raw).hexdigest()[:32]


def _frontier(s, watermark: Optional[int]) -> int:
    """Compute the compaction frontier (max seq that may be collapsed).

    - explicit watermark → use it verbatim (caller caps how far compaction may reach);
    - watermark is None  → MIN(last_seq_pushed) across push_cursor; if there is NO push_cursor
      row, no peer depends on the log tail, so the frontier is MAX(seq) (collapse everything).
    """
    if watermark is not None:
        return int(watermark)
    row = s.conn.execute("SELECT COUNT(*), MIN(last_seq_pushed) FROM push_cursor").fetchone()
    has_peers = row is not None and row[0] and row[0] > 0
    if has_peers and row[1] is not None:
        return int(row[1])
    max_row = s.conn.execute("SELECT COALESCE(MAX(seq), 0) FROM sync_events").fetchone()
    return int(max_row[0]) if max_row and max_row[0] is not None else 0


def _live_hashes_with_update_metadata(s) -> List[str]:
    """content_hashes that have at least one op='update_metadata' event (compaction candidates)."""
    rows = s.conn.execute(
        "SELECT DISTINCT content_hash FROM sync_events WHERE op = 'update_metadata'"
    ).fetchall()
    return [r[0] for r in rows]


def _collapsible_update_rows(s, content_hash: str, create_key, frontier: int):
    """Return the live update_metadata rows with seq <= frontier, as
    (seq, winner_key, agent_id, event_id, hlc_physical, hlc_logical, payload_dict).

    Only events belonging to the LIVE incarnation (via create_key, same rule as
    _load_live_events) are returned; dead-incarnation edits are excluded (T5).
    """
    rows = s.conn.execute(
        """
        SELECT seq, hlc_physical, hlc_logical, agent_id, event_id, payload
        FROM sync_events
        WHERE content_hash = ? AND op = 'update_metadata' AND seq <= ?
        """,
        (content_hash, frontier),
    ).fetchall()
    out = []
    for seq, phys, log, agent, eid, pj in rows:
        try:
            pl = json.loads(pj) if pj else {}
        except (ValueError, TypeError):
            continue
        k = _ev_winner_key(content_hash, phys, log, agent, eid, "update_metadata")
        # live-incarnation boundary: an edit is live iff there is no create or it is strictly
        # before the winning create key (matches apply._is_live_event for edits).
        if create_key is not None and not (k < create_key):
            continue
        out.append((seq, k, agent, eid, phys or 0, log or 0, pl))
    return out


def _has_live_update_above_frontier(s, content_hash: str, create_key, frontier: int) -> bool:
    """True if the LIVE incarnation of `content_hash` has any update_metadata event with
    seq > frontier. Variant B (all-or-nothing): when such an event exists, the chain must NOT
    be compacted this pass — a field whose winning edit sits above the watermark would otherwise
    lose to a collapsed field carrying the inflated general-winner clock (Greptile P1#1). The
    hash is compacted on a later pass, once the watermark advances past that edit.
    Dead-incarnation edits above the frontier do NOT block (they are not part of the live state).
    """
    rows = s.conn.execute(
        """
        SELECT hlc_physical, hlc_logical, agent_id, event_id
        FROM sync_events
        WHERE content_hash = ? AND op = 'update_metadata' AND seq > ?
        """,
        (content_hash, frontier),
    ).fetchall()
    for phys, log, agent, eid in rows:
        k = _ev_winner_key(content_hash, phys, log, agent, eid, "update_metadata")
        live = create_key is None or k < create_key
        if live:
            return True
    return False


def _winning_row(collapsible):
    """Pick the winning update_metadata row (smallest _winner_key) — same order as the resolver."""
    return min(collapsible, key=lambda r: r[1])


def _lww_accumulate(items, k, best, best_key):
    """Field-level LWW step: for each (key, value) in items, keep the value whose event has the
    smaller _winner_key. Mutates best/best_key in place. Extracted to keep _consolidated_updates
    within the repo's complexity budget (CLAUDE.md: ≤8)."""
    for key, value in items:
        if key not in best_key or k < best_key[key]:
            best_key[key] = k
            best[key] = value


def _consolidated_updates(content_hash: str, collapsible) -> Dict:
    """Build the consolidated `updates` dict = the per-field / per-metadata-key LWW winners over
    the collapsed update_metadata events ONLY (the create stays untouched and keeps the fields it
    won). Uses _winner_key for the per-field LWW, identical to _resolve_md_fields' order, so
    re-resolving the shortened log reproduces the same verdict (T1)."""
    top_best: Dict[str, object] = {}
    top_key: Dict[str, tuple] = {}
    meta_best: Dict[str, object] = {}
    meta_key: Dict[str, tuple] = {}

    for _seq, k, _agent, _eid, _phys, _log, pl in collapsible:
        _lww_accumulate(_top_level_fields("update_metadata", pl).items(), k, top_best, top_key)
        _lww_accumulate(_metadata_items("update_metadata", pl), k, meta_best, meta_key)

    updates: Dict[str, object] = dict(top_best)
    if meta_best:
        updates["metadata"] = meta_best
    return updates


def _latest_updated_at(collapsible) -> Optional[float]:
    """updated_at of the winning (smallest-key) event that carries one — LWW, matches
    apply._track_updated_at."""
    best_key = None
    best_val = None
    for _seq, k, _agent, _eid, _phys, _log, pl in collapsible:
        uat = pl.get("updated_at")
        if uat is not None and (best_key is None or k < best_key):
            best_key = k
            best_val = uat
    return best_val


def _plan_for_hash(s, content_hash: str, frontier: int) -> Optional[Tuple[List[int], Dict]]:
    """Return (seqs_to_remove, consolidated_event) for one hash, or None if nothing to do.

    None when: no collapsible live edit; the live chain has an update_metadata event ABOVE the
    frontier (variant B all-or-nothing — see _has_live_update_above_frontier, Greptile P1#1); or
    the chain is ALREADY consolidated (a single collapsible edit whose event_id equals the
    deterministic consolidated id — T6a idempotence).
    """
    _parsed, create_key = _load_live_events(s, content_hash)
    collapsible = _collapsible_update_rows(s, content_hash, create_key, frontier)
    if not collapsible:
        return None

    # Variant B: if any LIVE update_metadata edit sits above the frontier, skip this hash — a
    # field's winning edit could be above the watermark and a collapsed field would otherwise
    # carry the inflated general-winner clock and win incorrectly (Greptile P1#1). Compacted on
    # a later pass once the watermark advances past it.
    if _has_live_update_above_frontier(s, content_hash, create_key, frontier):
        return None

    updates = _consolidated_updates(content_hash, collapsible)
    payload = {"content_hash": content_hash, "updates": updates}
    uat = _latest_updated_at(collapsible)
    if uat is not None:
        payload["updated_at"] = uat

    win = _winning_row(collapsible)
    _seq, _k, _win_agent, _win_eid, win_phys, win_log, _pl = win
    consolidated_id = _deterministic_consolidated_event_id(
        content_hash, win_phys, win_log, updates)

    # Already consolidated: exactly one collapsible edit and it is the consolidated event itself.
    if len(collapsible) == 1 and collapsible[0][3] == consolidated_id:
        return None

    seqs_to_remove = [r[0] for r in collapsible]
    consolidated_event = {
        # Reuse a seq WITHIN the collapsed range (the max collapsed seq, which is <= frontier)
        # so the consolidated row never appears ABOVE the watermark frontier — events above the
        # frontier must stay byte-identical and the consolidated row must not pollute that band
        # (T4). AUTOINCREMENT would otherwise assign a brand-new high seq.
        "seq": max(seqs_to_remove),
        "schema_version": 1,
        "agent_id": _compaction_agent_id(),
        "event_id": consolidated_id,
        "op": _CONSOLIDATED_OP,
        "content_hash": content_hash,
        "hlc_physical": win_phys,
        "hlc_logical": win_log,
        "embedding_model": None,
        "embedding_dim": None,
        "payload": payload,
    }
    return seqs_to_remove, consolidated_event


_DELETE_CHUNK = 500  # well under SQLite's default SQLITE_MAX_VARIABLE_NUMBER (999 on old builds)


def _delete_seqs_chunked(s, seqs: List[int]) -> None:
    """DELETE the given seqs in bounded chunks (Greptile P2). A single memory's update_metadata
    chain can exceed SQLite's host-variable limit; one oversized IN (...) would raise and roll
    back the whole compaction batch, so a job meant to drain an unbounded backlog must bound its
    own query. All chunks run in the caller's SAVEPOINT (still fail-closed as one unit)."""
    for i in range(0, len(seqs), _DELETE_CHUNK):
        chunk = seqs[i:i + _DELETE_CHUNK]
        placeholders = ",".join("?" for _ in chunk)
        s.conn.execute(
            f"DELETE FROM sync_events WHERE seq IN ({placeholders})",
            tuple(chunk),
        )


def _apply_plan(s, plans: List[Tuple[List[int], Dict]]) -> int:
    """Delete collapsed rows and insert the consolidated events for every planned hash, inside a
    SINGLE savepoint. Fail-closed: any error rolls the whole batch back (T6c). Returns the number
    of sync_events rows removed (net of the consolidated inserts)."""
    removed = 0
    sp = f"compact_{os.urandom(4).hex()}"
    s.conn.execute(f"SAVEPOINT {sp}")
    try:
        for seqs_to_remove, event in plans:
            _delete_seqs_chunked(s, seqs_to_remove)
            removed += len(seqs_to_remove)
            # INSERT (not OR IGNORE): a collision on the deterministic (agent_id,event_id) or
            # the reused seq PK must NOT be silenced. OR IGNORE after the DELETE would drop the
            # consolidated row while the chain is already gone → silent data loss (Tuvok G5 P2).
            # A raw INSERT raises on collision; the single SAVEPOINT then rolls the whole batch
            # back (fail-closed), turning a silent loss into a detectable, reverted error.
            s.conn.execute(
                """
                INSERT INTO sync_events
                (seq, schema_version, agent_id, event_id, op, content_hash,
                 hlc_physical, hlc_logical, embedding_model, embedding_dim,
                 payload, created_at, created_at_iso)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event["seq"], event["schema_version"], event["agent_id"], event["event_id"],
                    event["op"], event["content_hash"], event["hlc_physical"],
                    event["hlc_logical"], event["embedding_model"], event["embedding_dim"],
                    json.dumps(event["payload"]),
                    time.time(), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                ),
            )
            removed -= 1  # the consolidated insert replaces one of the removed rows
        s.conn.execute(f"RELEASE SAVEPOINT {sp}")
        s.conn.commit()
    except Exception:
        # Fail-closed (ADR-0008): discard every delete + insert together, log byte-identical.
        try:
            s.conn.execute(f"ROLLBACK TO SAVEPOINT {sp}")
            s.conn.execute(f"RELEASE SAVEPOINT {sp}")
        except Exception:
            # Rollback itself failed (e.g. connection already dropped). Nothing left to undo
            # here; re-raise the ORIGINAL error below so the caller sees the real failure.
            pass
        raise
    return removed


def compact_update_metadata(storage: MemoryStorage, watermark: Optional[int] = None) -> Dict:
    """Collapse the idempotent update_metadata chains per content_hash into the LWW-resolved
    consolidated state, WITHOUT breaking sync convergence.

    Returns {"events_removed": int, "bytes_reclaimed": int | None}. ``bytes_reclaimed`` is None
    (sqlite_vec does not VACUUM here; the row count is the meaningful bound). See module docstring
    for the full contract (T1–T7).
    """
    s = _sqlite(storage)
    with _sync_lock(storage):
        frontier = _frontier(s, watermark)
        plans: List[Tuple[List[int], Dict]] = []
        for content_hash in _live_hashes_with_update_metadata(s):
            plan = _plan_for_hash(s, content_hash, frontier)
            if plan is not None and len(plan[0]) > 0:
                plans.append(plan)

        if not plans:
            return {"events_removed": 0, "bytes_reclaimed": None}

        removed = _apply_plan(s, plans)
        logger.info(
            "Compacted update_metadata chains: %s events removed across %s hashes (frontier seq %s)",
            _sanitize_log_value(removed), _sanitize_log_value(len(plans)),
            _sanitize_log_value(frontier),
        )
        return {"events_removed": removed, "bytes_reclaimed": None}
