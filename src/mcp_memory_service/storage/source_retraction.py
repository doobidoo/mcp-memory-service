# Copyright 2024 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Drop deleted memory hashes from beliefs and flag consolidation summaries.

Beliefs (``derived_from``, ``contradicted_by``, ``metadata.merged_from_hashes``)
and compression/association memories (``metadata.source_memory_hashes``) record
the content hashes they were built from. Deleting a memory has to act on those
receipts in the same transaction as the ``memory_graph`` cleanup. No model call.

``ungrounded`` is the status of a belief with no sources left. ``get_beliefs``
reads ``status = 'active'`` by default, so the belief drops out of search.
``superseded`` stays what dedup uses for a loser absorbed into a canonical.
Confidence is left untouched for the next ``challenge_belief``.

The liveness check is a pure function. Callers pass the stored source list and
the set of hashes that still have a live ``memories`` row. Deleted, invalidated
and re-ingested hashes are the same answer. These columns hold memory content
hashes only; an id that cannot be resolved is treated as gone when the
``memories`` table can be asked, and left alone when it cannot (a belief-only
database is not evidence that the hash was deleted).
"""

import json
import logging
import sqlite3
import time
from datetime import datetime, timezone

from mcp_memory_service.compat import _sanitize_log_value

logger = logging.getLogger(__name__)

# Not a dedup outcome. A belief with this status has no remaining evidence.
UNGROUNDED_STATUS = "ungrounded"

# Statuses get_beliefs / derivation still treat as current knowledge.
_GROUNDABLE_STATUSES = frozenset({"active", "candidate", "disputed"})

_HASH_TABLE = "mcp_retract_hashes"
_SQL_IN_CHUNK = 400

_BELIEF_CITE_SQL = """
SELECT belief_hash, derived_from, contradicted_by, metadata, status
FROM beliefs
WHERE EXISTS (
    SELECT 1 FROM json_each(
        CASE WHEN json_valid(beliefs.derived_from) THEN beliefs.derived_from ELSE NULL END
    ) AS je
    WHERE je.value IN (SELECT hash FROM mcp_retract_hashes)
)
OR EXISTS (
    SELECT 1 FROM json_each(
        CASE WHEN json_valid(beliefs.contradicted_by) THEN beliefs.contradicted_by ELSE NULL END
    ) AS je
    WHERE je.value IN (SELECT hash FROM mcp_retract_hashes)
)
OR EXISTS (
    SELECT 1 FROM json_each(
        CASE WHEN json_valid(beliefs.metadata)
             THEN json_extract(beliefs.metadata, '$.merged_from_hashes')
             ELSE NULL END
    ) AS je
    WHERE je.value IN (SELECT hash FROM mcp_retract_hashes)
)
"""

_SUMMARY_CITE_SQL = """
SELECT content_hash, metadata, tags
FROM memories
WHERE deleted_at IS NULL
AND json_valid(metadata)
AND EXISTS (
    SELECT 1 FROM json_each(json_extract(metadata, '$.source_memory_hashes')) AS je
    WHERE je.value IN (SELECT hash FROM mcp_retract_hashes)
)
"""


def departed_sources(stored_sources, live_ids) -> list:
    """Return stored hashes that are not in ``live_ids``.

    Order-preserving, duplicates collapsed. Non-strings are ignored.
    """
    gone = []
    seen = set()
    for source in stored_sources or []:
        if (
            not isinstance(source, str)
            or not source
            or source in seen
            or source in live_ids
        ):
            continue
        seen.add(source)
        gone.append(source)
    return gone


def retract_belief_record(
    *,
    derived_from,
    contradicted_by,
    metadata,
    status,
    live_ids,
    updated_at,
) -> dict | None:
    """Return the fields to write, or None when no provenance hash departed.

    Confidence is not part of the result. A belief whose ``derived_from`` and
    ``contradicted_by`` are both empty moves to ``ungrounded`` unless it is
    already ``superseded`` (or some other non-groundable status).
    """
    derived = list(derived_from or [])
    contradicted = list(contradicted_by or [])
    meta = dict(metadata or {})
    merged = meta.get("merged_from_hashes")
    if not isinstance(merged, list):
        merged = []
    removed = departed_sources(derived + contradicted + merged, live_ids)
    if not removed:
        return None
    removed_set = set(removed)
    new_derived = _without(derived, removed_set)
    new_contradicted = _without(contradicted, removed_set)
    new_merged = _without(merged, removed_set)
    if new_merged:
        meta["merged_from_hashes"] = new_merged
    else:
        meta.pop("merged_from_hashes", None)
    meta["stale_sources"] = {
        "count": _prior_stale_count(metadata) + len(removed),
        "updated_at": updated_at,
    }
    new_status = status
    if (
        status in _GROUNDABLE_STATUSES
        and not _has_hash(new_derived)
        and not _has_hash(new_contradicted)
    ):
        new_status = UNGROUNDED_STATUS
    return {
        "derived_from": new_derived,
        "contradicted_by": new_contradicted,
        "metadata": meta,
        "status": new_status,
    }


def flag_summary_sources(metadata, *, live_ids, updated_at) -> dict:
    """Mark a compression or association memory whose sources have departed.

    ``source_memory_hashes`` is kept. The summary text is not this function's
    concern. ``stale_sources.count`` is how many of those hashes are not live.
    Repeating the call for the same count does not increment it and does not
    refresh ``updated_at``, so a later reconcile of an already-flagged summary
    does not look like a new edit.
    """
    meta = dict(metadata or {})
    sources = meta.get("source_memory_hashes")
    if not isinstance(sources, list):
        return meta
    departed = departed_sources(sources, live_ids)
    if not departed:
        return meta
    count = len(departed)
    prior = meta.get("stale_sources")
    if isinstance(prior, dict) and prior.get("count") == count:
        return meta
    meta["stale_sources"] = {"count": count, "updated_at": updated_at}
    return meta


def union_live_merged_sources(conn, supporting_hashes, merged_from):
    """Union ``merged_from`` into ``supporting_hashes``, skipping departed hashes.

    Returns ``(derived_from, kept_merged_from, changed)``. ``changed`` is False
    when liveness cannot be asked, so a belief-only database keeps every hash.
    """
    ordered, seen = _dedupe(supporting_hashes)
    string_hashes = [h for h in merged_from if isinstance(h, str) and h]
    live = live_memory_hashes(conn, string_hashes)
    gone = set() if live is None else set(departed_sources(string_hashes, live))
    kept = [h for h in merged_from if not (isinstance(h, str) and h in gone)]
    for source in kept:
        if source not in seen:
            seen.add(source)
            ordered.append(source)
    return ordered, kept, kept != list(merged_from)


def live_memory_hashes(conn, hashes):
    """Hashes that still have a non-deleted ``memories`` row.

    ``None`` means the table cannot be asked. An empty input with a present
    table returns an empty set.
    """
    if not _table_exists(conn, "memories"):
        return None
    unique = _unique_strings(hashes)
    if not unique:
        return set()
    live = set()
    for start in range(0, len(unique), _SQL_IN_CHUNK):
        chunk = unique[start : start + _SQL_IN_CHUNK]
        placeholders = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"SELECT content_hash FROM memories "
            f"WHERE deleted_at IS NULL AND content_hash IN ({placeholders})",
            chunk,
        ).fetchall()
        live.update(row[0] for row in rows)
    return live


def retract_sources_for_deleted_hashes(
    conn, content_hashes, *, updated_at=None, sync=None
) -> None:
    """Update beliefs and consolidation summaries that cite these hashes.

    The caller owns the transaction and has already set ``deleted_at`` on this
    connection, so the liveness read sees the tombstone. A missing ``beliefs``
    table is skipped. A SQLite build without JSON functions falls back to a
    scan. Per-row JSON that cannot be parsed is skipped and logged. A failure
    while writing propagates so the caller's transaction can roll the delete
    back with the retraction.

    ``sync`` is the storage that owns ``conn`` when the caller can record
    events. A summary flag then goes out through ``_append_sync_event`` in
    this same transaction. The memories ``updated_at`` column moves either way,
    which is what hybrid sync compares.
    """
    deleted = _unique_strings(content_hashes)
    if not deleted:
        return
    when = updated_at or datetime.now(timezone.utc).isoformat()
    _stage_hashes(conn, deleted)
    _retract_beliefs(conn, set(deleted), when)
    _flag_summaries(conn, set(deleted), when, sync)


def _without(items, removed: set) -> list:
    return [item for item in items if not (isinstance(item, str) and item in removed)]


def _has_hash(items) -> bool:
    return any(isinstance(item, str) and item for item in items)


def _prior_stale_count(metadata) -> int:
    raw = (metadata or {}).get("stale_sources") if isinstance(metadata, dict) else None
    if not isinstance(raw, dict):
        return 0
    try:
        return max(0, int(raw.get("count") or 0))
    except (TypeError, ValueError):
        return 0


def _dedupe(items):
    ordered = []
    seen = set()
    for item in items or []:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered, seen


def _unique_strings(hashes) -> list:
    unique = []
    seen = set()
    for source in hashes or []:
        if not isinstance(source, str) or not source or source in seen:
            continue
        seen.add(source)
        unique.append(source)
    return unique


def _table_exists(conn, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def _stage_hashes(conn, hashes: list) -> None:
    conn.execute(
        f"CREATE TEMP TABLE IF NOT EXISTS {_HASH_TABLE} (hash TEXT PRIMARY KEY)"
    )
    conn.execute(f"DELETE FROM {_HASH_TABLE}")
    conn.executemany(
        f"INSERT OR IGNORE INTO {_HASH_TABLE} (hash) VALUES (?)",
        [(source,) for source in hashes],
    )


def _retract_beliefs(conn, deleted: set, updated_at: str) -> None:
    if not _table_exists(conn, "beliefs"):
        return
    for row in _citing_beliefs(conn, deleted):
        _retract_one_belief(conn, row, updated_at)


def _citing_beliefs(conn, deleted: set):
    try:
        return conn.execute(_BELIEF_CITE_SQL).fetchall()
    except sqlite3.Error as exc:
        logger.warning(
            "Belief citation lookup fell back to a scan: %s",
            _sanitize_log_value(exc),
        )
        return _scan_beliefs(conn, deleted)


def _scan_beliefs(conn, deleted: set):
    try:
        rows = conn.execute(
            "SELECT belief_hash, derived_from, contradicted_by, metadata, status FROM beliefs"
        ).fetchall()
    except sqlite3.Error as exc:
        logger.warning("Belief retraction skipped: %s", _sanitize_log_value(exc))
        return []
    return [row for row in rows if _row_cites(row[1:4], deleted)]


def _retract_one_belief(conn, row, updated_at: str) -> None:
    belief_hash, derived_raw, contradicted_raw, metadata_raw, status = row
    parsed = _parsed_belief(belief_hash, derived_raw, contradicted_raw, metadata_raw)
    if parsed is None:
        return
    derived, contradicted, metadata = parsed
    merged = metadata.get("merged_from_hashes")
    if not isinstance(merged, list):
        merged = []
    mentioned = _unique_strings(list(derived) + list(contradicted) + list(merged))
    live = live_memory_hashes(conn, mentioned)
    if live is None:
        return
    update = retract_belief_record(
        derived_from=derived,
        contradicted_by=contradicted,
        metadata=metadata,
        status=status,
        live_ids=live,
        updated_at=updated_at,
    )
    if update is None:
        return
    conn.execute(
        "UPDATE beliefs SET derived_from = ?, contradicted_by = ?, metadata = ?, "
        "status = ?, updated_at = ? WHERE belief_hash = ?",
        (
            json.dumps(update["derived_from"]),
            json.dumps(update["contradicted_by"]),
            json.dumps(update["metadata"]),
            update["status"],
            updated_at,
            belief_hash,
        ),
    )


def _parsed_belief(belief_hash, derived_raw, contradicted_raw, metadata_raw):
    try:
        return (
            _parse_list(derived_raw),
            _parse_list(contradicted_raw),
            _parse_object(metadata_raw),
        )
    except (TypeError, ValueError) as exc:
        logger.warning(
            "Skipping belief %s with unreadable provenance: %s",
            _sanitize_log_value(belief_hash),
            _sanitize_log_value(exc),
        )
        return None


def _flag_summaries(conn, deleted: set, updated_at: str, sync=None) -> None:
    for row in _citing_summaries(conn, deleted):
        _flag_one_summary(conn, row, updated_at, sync)


def _citing_summaries(conn, deleted: set):
    try:
        return conn.execute(_SUMMARY_CITE_SQL).fetchall()
    except sqlite3.Error as exc:
        logger.warning(
            "Summary citation lookup fell back to a scan: %s",
            _sanitize_log_value(exc),
        )
        return _scan_summaries(conn, deleted)


def _scan_summaries(conn, deleted: set):
    try:
        rows = conn.execute(
            "SELECT content_hash, metadata, tags FROM memories WHERE deleted_at IS NULL"
        ).fetchall()
    except sqlite3.Error as exc:
        logger.warning("Summary flag skipped: %s", _sanitize_log_value(exc))
        return []
    return [row for row in rows if _row_cites((row[1],), deleted)]


def _flag_one_summary(conn, row, updated_at: str, sync=None) -> None:
    content_hash, metadata_raw, tags = row
    if _summary_kind(tags) is None:
        return
    try:
        metadata = _parse_object(metadata_raw)
    except (TypeError, ValueError) as exc:
        logger.warning(
            "Skipping summary %s with unreadable metadata: %s",
            _sanitize_log_value(content_hash),
            _sanitize_log_value(exc),
        )
        return
    sources = metadata.get("source_memory_hashes")
    if not isinstance(sources, list):
        return
    live = live_memory_hashes(conn, sources)
    if live is None:
        return
    flagged = flag_summary_sources(metadata, live_ids=live, updated_at=updated_at)
    if flagged == metadata:
        return
    _write_summary_flag(conn, content_hash, flagged, sync)


def _write_summary_flag(conn, content_hash: str, flagged: dict, sync) -> None:
    """Persist the flag and, when event logging is on, record it for peers."""
    epoch = time.time()
    iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))
    conn.execute(
        "UPDATE memories SET metadata = ?, updated_at = ?, updated_at_iso = ? "
        "WHERE content_hash = ? AND deleted_at IS NULL",
        (json.dumps(flagged), epoch, iso, content_hash),
    )
    _record_summary_flag(sync, conn, content_hash, flagged, epoch)


def _record_summary_flag(
    sync, conn, content_hash: str, flagged: dict, epoch: float
) -> None:
    """Same payload shape as ``update_memory_metadata``: partial metadata, one key.

    Delta sync merges ``updates.metadata`` per key, and the payload's
    ``updated_at`` is the instant peers write. Hybrid sync never sees the
    event; it compares the column this write just advanced.
    """
    if sync is None or not hasattr(sync, "_append_sync_event"):
        return
    warning = flagged.get("stale_sources")
    if not isinstance(warning, dict):
        return
    sync._append_sync_event(
        conn,
        "update_metadata",
        content_hash,
        {
            "content_hash": content_hash,
            "updates": {"metadata": {"stale_sources": warning}},
            "updated_at": epoch,
        },
    )


def _summary_kind(tags) -> str | None:
    """Compression output carries both tags; association memories carry one.

    ``compressed`` alone is also written by forgetting for a shortened user
    memory, which is not a cluster summary and is left alone.
    """
    parts = {part.strip() for part in (tags or "").split(",") if part.strip()}
    if "association" in parts:
        return "association"
    if "cluster" in parts and "compressed" in parts:
        return "compression"
    return None


def _row_cites(parts, deleted: set) -> bool:
    blob = " ".join(part for part in parts if isinstance(part, str))
    return any(source in blob for source in deleted)


def _parse_list(raw):
    if isinstance(raw, list):
        return raw
    if not raw:
        return []
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise ValueError("provenance is not a list")
    return parsed


def _parse_object(raw):
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("metadata is not an object")
    return parsed
