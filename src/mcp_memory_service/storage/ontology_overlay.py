"""Ontology overlay stored in the memory DB (#1897, phases 1 to 4).

The overlay adds memory types on top of the built-in ontology and
``MCP_CUSTOM_MEMORY_TYPES``, relationship types and inference type patterns
on top of ``RELATIONSHIPS`` and ``TYPE_PATTERNS``, and retention periods for
types that have none in ``CONSOLIDATION_CONFIG``. It lives in the ``ontology_entries`` table, and
every write bumps ``metadata.ontology_version``. ``models.ontology`` polls that
counter and reloads the overlay when it moved.

Only the sqlite_vec database carries the table. Cloudflare/D1 does not, so on
the cloudflare backend ``MCP_CUSTOM_MEMORY_TYPES`` is still the way to share
custom types. Between sqlite_vec nodes the rows travel through the hub as
``ontology_upsert`` / ``ontology_delete`` sync events (phase 2), keyed
``ontology:<kind>:<key>`` and resolved per key by the event HLC.
"""

import json
import sqlite3
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..config.consolidation import CONSOLIDATION_CONFIG
from ..models import ontology

# Same bounds as the MCP_RETENTION_* variables.
RETENTION_MIN_DAYS = 1
RETENTION_MAX_DAYS = 3650

# Kinds that models.ontology reads through read_payload_overlay(kind).
PAYLOAD_KINDS = ("relationship", "type_pattern")

# Sync (phase 2). Every overlay write records one of these events next to the row.
ONTOLOGY_KINDS = ("memory_type", "retention", "relationship", "type_pattern")
ONTOLOGY_UPSERT = "ontology_upsert"
ONTOLOGY_DELETE = "ontology_delete"
ONTOLOGY_OPS = (ONTOLOGY_UPSERT, ONTOLOGY_DELETE)
ONTOLOGY_HASH_PREFIX = "ontology:"

# emit(conn, op, content_hash, event_payload): records the sync event inside the
# write's transaction. The storage passes one; direct callers (tests) pass none.
Emit = Optional[Callable[[sqlite3.Connection, str, str, dict], None]]


# Which events a feed, push or baseline carries. "memory" is what a peer without
# ontology sync gets: it cannot apply an ontology op and would fail-stop on it.
EVENT_FILTERS = {
    "memory": " AND op NOT IN ('ontology_upsert', 'ontology_delete')",
    "ontology": " AND op IN ('ontology_upsert', 'ontology_delete')",
    "all": "",
}


def parse_row_payload(payload_json) -> Optional[dict]:
    """A row's payload_json as a dict, or None when it is empty or not a JSON object."""
    try:
        value = json.loads(payload_json) if payload_json else None
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def ontology_event_hash(kind: str, key: str) -> str:
    """The sync_events content_hash of an overlay row; never a 64-hex memory hash."""
    return f"{ONTOLOGY_HASH_PREFIX}{kind}:{key}"


def read_ontology_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT value FROM metadata WHERE key = 'ontology_version'").fetchone()
    return int(row[0]) if row else 0


def read_memory_type_overlay(conn: sqlite3.Connection) -> List[Tuple[str, Optional[str], bool]]:
    """All memory type rows as (key, parent, live), tombstoned ones included.

    Only live rows extend the ontology. Tombstoned rows still say which base
    type a removed subtype belonged to, which retention needs for the stored
    memories that keep that type.
    """
    try:
        return [
            (key, parent, bool(live))
            for key, parent, live in conn.execute(
                "SELECT key, parent, deleted_at IS NULL FROM ontology_entries "
                "WHERE kind = 'memory_type'"
            )
        ]
    except sqlite3.OperationalError:  # migration 022 not applied (yet)
        return []


def _valid_retention_days(days) -> bool:
    return (
        isinstance(days, int)
        and not isinstance(days, bool)
        and RETENTION_MIN_DAYS <= days <= RETENTION_MAX_DAYS
    )


def read_payload_overlay(conn: sqlite3.Connection, kind: str) -> Dict[str, dict]:
    """Live rows of ``kind`` as key -> parsed payload. Rows whose payload is not a JSON object are skipped."""
    try:
        rows = conn.execute(
            "SELECT key, payload_json FROM ontology_entries WHERE kind = ? AND deleted_at IS NULL",
            (kind,),
        ).fetchall()
    except sqlite3.OperationalError:  # migration 022 not applied (yet)
        return {}
    payloads = {}
    for key, payload in rows:
        try:
            value = json.loads(payload)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            payloads[key] = value
    return payloads


def read_retention_overlay(conn: sqlite3.Connection) -> Dict[str, int]:
    """Retention days per memory type. Rows with a malformed payload are skipped."""
    return {
        key: payload["days"]
        for key, payload in read_payload_overlay(conn, "retention").items()
        if _valid_retention_days(payload.get("days"))
    }


def _bump_version(conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT INTO metadata (key, value) VALUES ('ontology_version', '1') "
        "ON CONFLICT(key) DO UPDATE SET value = CAST(value AS INTEGER) + 1"
    )


def validate_memory_type_entry(name: str, parent: Optional[str]) -> Tuple[str, Optional[str]]:
    """Canonicalize a new memory type and check it against the merged ontology.

    Raises ValueError for a bad name, a name that already exists, or a parent
    that is not a base type.
    """
    name = ontology.canonicalize_memory_type(name)
    parent = ontology.canonicalize_memory_type(parent) if parent is not None else None
    for value in (name, parent):
        if value is not None and not ontology._is_valid_custom_type_name(value):
            raise ValueError(
                "Memory type names may contain lowercase letters, digits, underscore "
                "and hyphen, and must start with a letter."
            )
    if ontology.validate_memory_type(name):
        raise ValueError(f"'{name}' is already a memory type")
    if parent is not None and parent not in ontology.get_base_types():
        raise ValueError(f"'{parent}' is not a base memory type")
    return name, parent


def _emit_row(conn: sqlite3.Connection, emit: Emit, kind: str, key: str) -> None:
    """Record the row's state as a sync event, inside the caller's transaction."""
    if emit is None:
        return
    parent, payload_json, deleted_at = conn.execute(
        "SELECT parent, payload_json, deleted_at FROM ontology_entries WHERE kind = ? AND key = ?",
        (kind, key),
    ).fetchone()
    op = ONTOLOGY_DELETE if deleted_at is not None else ONTOLOGY_UPSERT
    emit(conn, op, ontology_event_hash(kind, key), {
        "kind": kind,
        "key": key,
        "parent": parent,
        "payload": parse_row_payload(payload_json),
        "deleted_at": deleted_at,
    })


def upsert_memory_type(
    conn: sqlite3.Connection, name: str, parent: Optional[str], emit: Emit = None
) -> None:
    """Write (or revive) an overlay memory type and bump the version, in one transaction.

    The caller's duplicate check reads this process's cached ontology, which can
    be stale: another process, or a concurrent call, may have written the same
    name since. The conflict update therefore only revives a tombstoned row, and
    a live row with that name makes the write fail instead of re-parenting it.
    """
    _insert_or_revive(conn, "memory_type", name, parent=parent, error="a memory type", emit=emit)


def _insert_or_revive(
    conn: sqlite3.Connection,
    kind: str,
    key: str,
    *,
    parent: Optional[str] = None,
    payload: Optional[dict] = None,
    error: str,
    emit: Emit = None,
) -> None:
    """Insert a row, or revive a tombstoned one, and bump the version in one transaction.

    A live row with the same key makes the write fail instead of replacing it.
    """
    payload_json = json.dumps(payload) if payload is not None else None
    with conn:  # commits both statements, or rolls both back
        cursor = conn.execute(
            "INSERT INTO ontology_entries (kind, key, parent, payload_json, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(kind, key) DO UPDATE SET "
            "parent = excluded.parent, payload_json = excluded.payload_json, "
            "updated_at = excluded.updated_at, deleted_at = NULL "
            "WHERE ontology_entries.deleted_at IS NOT NULL",
            (kind, key, parent, payload_json, time.time()),
        )
        if cursor.rowcount == 0:
            raise ValueError(f"'{key}' is already {error}")
        _bump_version(conn)
        _emit_row(conn, emit, kind, key)


def _upsert_payload(
    conn: sqlite3.Connection, kind: str, key: str, payload: dict, emit: Emit = None
) -> None:
    """Insert or replace a value row (retention, type_pattern) and bump the version."""
    with conn:
        conn.execute(
            "INSERT INTO ontology_entries (kind, key, payload_json, updated_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(kind, key) DO UPDATE SET "
            "payload_json = excluded.payload_json, updated_at = excluded.updated_at, "
            "deleted_at = NULL",
            (kind, key, json.dumps(payload), time.time()),
        )
        _bump_version(conn)
        _emit_row(conn, emit, kind, key)


def _tombstone(conn: sqlite3.Connection, kind: str, key: str, emit: Emit = None) -> bool:
    now = time.time()
    with conn:
        cursor = conn.execute(
            "UPDATE ontology_entries SET deleted_at = ?, updated_at = ? "
            "WHERE kind = ? AND key = ? AND deleted_at IS NULL",
            (now, now, kind, key),
        )
        if cursor.rowcount:
            _bump_version(conn)
            _emit_row(conn, emit, kind, key)
    return bool(cursor.rowcount)


def tombstone_memory_type(conn: sqlite3.Connection, name: str, emit: Emit = None) -> bool:
    """Tombstone an overlay memory type. Built-in and env types are not in the table."""
    return _tombstone(conn, "memory_type", name, emit)


def validate_retention_entry(memory_type: str, days) -> Tuple[str, int]:
    """Canonicalize a retention entry and check it against the ontology and config.

    Raises ValueError for an unknown type, for days outside 1..3650, or for a
    type that already has a period in ``CONSOLIDATION_CONFIG`` (built-in
    default or ``MCP_RETENTION_*``): the overlay only fills gaps, so such a
    row would never take effect.
    """
    memory_type = ontology.canonicalize_memory_type(memory_type)
    if not ontology.validate_memory_type(memory_type):
        raise ValueError(f"'{memory_type}' is not a memory type")
    if memory_type in CONSOLIDATION_CONFIG["retention_periods"]:
        raise ValueError(
            f"'{memory_type}' already has a retention period in the server "
            "configuration (MCP_RETENTION_*)"
        )
    if not _valid_retention_days(days):
        raise ValueError(
            f"Retention must be a whole number of days from {RETENTION_MIN_DAYS} "
            f"to {RETENTION_MAX_DAYS}."
        )
    return memory_type, days


def upsert_retention(
    conn: sqlite3.Connection, memory_type: str, days: int, emit: Emit = None
) -> None:
    """Set (or replace) the retention period of a memory type and bump the version."""
    _upsert_payload(conn, "retention", memory_type, {"days": days}, emit)


def tombstone_retention(conn: sqlite3.Connection, memory_type: str, emit: Emit = None) -> bool:
    """Tombstone a retention entry; the type falls back to its parent's period."""
    return _tombstone(conn, "retention", memory_type, emit)


def _canonical_pattern(pattern, allow_any: bool) -> Tuple[str, str]:
    """Parse ``"source → target"`` and check both ends are base types (or ``any``)."""
    pair = ontology.parse_pattern(pattern)
    if pair is None:
        raise ValueError(f"'{pattern}' is not a 'source → target' pattern")
    allowed = ontology.get_base_types() | ({"any"} if allow_any else set())
    for end in pair:
        if end not in allowed:
            raise ValueError(f"'{end}' is not a base memory type")
    return pair


def validate_relationship_entry(
    name: str, valid_patterns: Sequence[str], description: str = "", symmetric: bool = False
) -> Tuple[str, dict]:
    """Canonicalize a new relationship type and check it against the merged ontology.

    Raises ValueError for a bad or existing name, a missing or malformed
    pattern, a pattern end that is not a base type or ``any``, or a
    non-string description / non-bool ``symmetric``.
    """
    name = ontology.canonicalize_memory_type(name)
    if not ontology._is_valid_custom_type_name(name):
        raise ValueError(
            "Relationship names may contain lowercase letters, digits, underscore "
            "and hyphen, and must start with a letter."
        )
    if ontology.validate_relationship(name):
        raise ValueError(f"'{name}' is already a relationship type")
    if isinstance(valid_patterns, str) or not valid_patterns:
        raise ValueError("valid_patterns must be a non-empty list of 'source → target' strings")
    if not isinstance(description, str) or not isinstance(symmetric, bool):
        raise ValueError("description must be a string and symmetric a bool")
    patterns = [
        "{} → {}".format(*_canonical_pattern(p, allow_any=True)) for p in valid_patterns
    ]
    return name, {"description": description, "valid_patterns": patterns, "symmetric": symmetric}


def upsert_relationship(
    conn: sqlite3.Connection, name: str, spec: dict, emit: Emit = None
) -> None:
    """Write (or revive) an overlay relationship type. A live row with that name makes it fail."""
    _insert_or_revive(
        conn, "relationship", name, payload=spec, error="a relationship type", emit=emit
    )


def tombstone_relationship(conn: sqlite3.Connection, name: str, emit: Emit = None) -> bool:
    """Tombstone an overlay relationship type. Built-in relationships are not in the table."""
    return _tombstone(conn, "relationship", name, emit)


def validate_type_pattern_entry(
    source: str, target: str, relationship: str, confidence
) -> Tuple[str, dict]:
    """Check an inference type pattern and return its row key and payload.

    Raises ValueError when an end is not a base type, the pair is a built-in
    ``TYPE_PATTERNS`` pair, the relationship is unknown, one inference must not
    emit (``has_entity``), or its valid_patterns do not allow the pair (inference
    would drop it), or confidence is not in (0, 1].
    """
    pair = _canonical_pattern(f"{source} → {target}", allow_any=False)
    if pair in ontology.TYPE_PATTERNS:
        raise ValueError("{} → {} is a built-in type pattern".format(*pair))
    relationship = ontology.canonicalize_memory_type(relationship)
    if not ontology.validate_relationship(relationship):
        raise ValueError(f"'{relationship}' is not a relationship type")
    if relationship in ontology.NON_INFERABLE_RELATIONSHIPS:
        raise ValueError(f"'{relationship}' links a memory to an entity; inference cannot emit it")
    if not ontology.is_allowed_pair(relationship, *pair):
        raise ValueError(
            "the valid_patterns of '{}' do not allow {} → {}".format(relationship, *pair)
        )
    if not ontology._valid_confidence(confidence):
        raise ValueError("confidence must be a number in (0, 1]")
    return "{} → {}".format(*pair), {"relationship": relationship, "confidence": float(confidence)}


def upsert_type_pattern(
    conn: sqlite3.Connection, key: str, payload: dict, emit: Emit = None
) -> None:
    """Set (or replace) an inference type pattern and bump the version."""
    _upsert_payload(conn, "type_pattern", key, payload, emit)


def tombstone_type_pattern(conn: sqlite3.Connection, key: str, emit: Emit = None) -> bool:
    """Tombstone an inference type pattern."""
    return _tombstone(conn, "type_pattern", key, emit)


def ontology_event_error(op, content_hash, payload) -> Optional[str]:
    """Why a remote ontology event is malformed, or None. Checked before it is recorded.

    The key in the payload must be the one in ``content_hash``, so an event cannot
    write one row while being resolved against another.
    """
    if op not in ONTOLOGY_OPS:
        return f"op {op!r} is not an ontology op"
    if not isinstance(payload, dict):
        return "payload is not an object"
    kind, key = payload.get("kind"), payload.get("key")
    if kind not in ONTOLOGY_KINDS or not isinstance(key, str) or not key:
        return "unknown kind or empty key"
    if content_hash != ontology_event_hash(kind, key):
        return "content_hash does not match kind and key"
    if not _fields_typed(payload):
        return "parent or payload has the wrong type"
    return None


def _fields_typed(payload: dict) -> bool:
    parent, row_payload = payload.get("parent"), payload.get("payload")
    return (parent is None or isinstance(parent, str)) and (
        row_payload is None or isinstance(row_payload, dict)
    )


def apply_remote_row(conn: sqlite3.Connection, op: str, payload: dict) -> None:
    """Write the state a winning remote event carries. No commit: the sync apply owns it.

    Unlike a local write this replaces a live row: the event won the HLC
    resolution, so its state is what every node converges to.
    """
    deleted_at = payload.get("deleted_at") if op == ONTOLOGY_DELETE else None
    if op == ONTOLOGY_DELETE and not isinstance(deleted_at, (int, float)):
        deleted_at = time.time()
    row_payload = payload.get("payload")
    conn.execute(
        "INSERT INTO ontology_entries (kind, key, parent, payload_json, updated_at, deleted_at) "
        "VALUES (?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(kind, key) DO UPDATE SET "
        "parent = excluded.parent, payload_json = excluded.payload_json, "
        "updated_at = excluded.updated_at, deleted_at = excluded.deleted_at",
        (
            payload["kind"],
            payload["key"],
            payload.get("parent"),
            json.dumps(row_payload) if row_payload is not None else None,
            time.time(),
            deleted_at,
        ),
    )
    _bump_version(conn)


class OverlaySource:
    """What ``models.ontology`` polls.

    For a database file it opens its own short-lived connection per read, so
    it is safe from any thread and sees writes by other processes. An
    in-memory database exists only on the storage's own connection, so there
    the storage calls ``capture()`` after each write and reads come from that
    snapshot. No other process can write to it, so nothing is missed.
    """

    def __init__(self, db_path: str):
        self._snapshot: Tuple[
            int, List[Tuple[str, Optional[str], bool]], Dict[str, int], Dict[str, Dict[str, dict]]
        ] = (0, [], {}, {})
        if db_path in ("", ":memory:"):
            self.uri = None
        else:
            # Read-only: a source left behind by a storage that was never closed
            # must not create an empty database where the file used to be.
            self.uri = Path(db_path).resolve().as_uri() + "?mode=ro"

    def capture(self, conn: sqlite3.Connection) -> None:
        """Snapshot the overlay from the storage's connection (in-memory databases only)."""
        if self.uri is None:
            self._snapshot = (
                read_ontology_version(conn),
                read_memory_type_overlay(conn),
                read_retention_overlay(conn),
                {kind: read_payload_overlay(conn, kind) for kind in PAYLOAD_KINDS},
            )

    def _read(self, query, snapshot_value):
        if self.uri is None:
            return snapshot_value
        conn = sqlite3.connect(self.uri, uri=True, timeout=5)
        try:
            return query(conn)
        finally:
            conn.close()

    def read_ontology_version(self) -> int:
        return self._read(read_ontology_version, self._snapshot[0])

    def read_memory_type_overlay(self) -> List[Tuple[str, Optional[str], bool]]:
        return self._read(read_memory_type_overlay, self._snapshot[1])

    def read_retention_overlay(self) -> Dict[str, int]:
        return self._read(read_retention_overlay, self._snapshot[2])

    def read_payload_overlay(self, kind: str) -> Dict[str, dict]:
        return self._read(
            lambda conn: read_payload_overlay(conn, kind), self._snapshot[3].get(kind, {})
        )
