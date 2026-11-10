"""Ontology overlay stored in the memory DB (#1897, phases 1 and 4).

The overlay adds memory types on top of the built-in ontology and
``MCP_CUSTOM_MEMORY_TYPES``, and retention periods for types that have none
in ``CONSOLIDATION_CONFIG``. It lives in the ``ontology_entries`` table, and
every write bumps ``metadata.ontology_version``. ``models.ontology`` polls that
counter and reloads the overlay when it moved.

Only the sqlite_vec database carries the table. Cloudflare/D1 does not, so on
the cloudflare backend, and between hybrid nodes, ``MCP_CUSTOM_MEMORY_TYPES``
is still the way to share custom types.
"""

import json
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..config.consolidation import CONSOLIDATION_CONFIG
from ..models import ontology

# Same bounds as the MCP_RETENTION_* variables.
RETENTION_MIN_DAYS = 1
RETENTION_MAX_DAYS = 3650


def read_ontology_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT value FROM metadata WHERE key = 'ontology_version'").fetchone()
    return int(row[0]) if row else 0


def read_memory_type_overlay(conn: sqlite3.Connection) -> List[Tuple[str, Optional[str]]]:
    try:
        return conn.execute(
            "SELECT key, parent FROM ontology_entries "
            "WHERE kind = 'memory_type' AND deleted_at IS NULL"
        ).fetchall()
    except sqlite3.OperationalError:  # migration 022 not applied (yet)
        return []


def _valid_retention_days(days) -> bool:
    return (
        isinstance(days, int)
        and not isinstance(days, bool)
        and RETENTION_MIN_DAYS <= days <= RETENTION_MAX_DAYS
    )


def read_retention_overlay(conn: sqlite3.Connection) -> Dict[str, int]:
    """Retention days per memory type. Rows with a malformed payload are skipped."""
    try:
        rows = conn.execute(
            "SELECT key, payload_json FROM ontology_entries "
            "WHERE kind = 'retention' AND deleted_at IS NULL"
        ).fetchall()
    except sqlite3.OperationalError:  # migration 022 not applied (yet)
        return {}
    periods = {}
    for key, payload in rows:
        try:
            days = json.loads(payload)["days"]
        except (TypeError, ValueError, KeyError):
            continue
        if _valid_retention_days(days):
            periods[key] = days
    return periods


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


def upsert_memory_type(conn: sqlite3.Connection, name: str, parent: Optional[str]) -> None:
    """Write (or revive) an overlay memory type and bump the version, in one transaction.

    The caller's duplicate check reads this process's cached ontology, which can
    be stale: another process, or a concurrent call, may have written the same
    name since. The conflict update therefore only revives a tombstoned row, and
    a live row with that name makes the write fail instead of re-parenting it.
    """
    with conn:  # commits both statements, or rolls both back
        cursor = conn.execute(
            "INSERT INTO ontology_entries (kind, key, parent, updated_at) "
            "VALUES ('memory_type', ?, ?, ?) "
            "ON CONFLICT(kind, key) DO UPDATE SET "
            "parent = excluded.parent, updated_at = excluded.updated_at, deleted_at = NULL "
            "WHERE ontology_entries.deleted_at IS NOT NULL",
            (name, parent, time.time()),
        )
        if cursor.rowcount == 0:
            raise ValueError(f"'{name}' is already a memory type")
        _bump_version(conn)


def _tombstone(conn: sqlite3.Connection, kind: str, key: str) -> bool:
    now = time.time()
    with conn:
        cursor = conn.execute(
            "UPDATE ontology_entries SET deleted_at = ?, updated_at = ? "
            "WHERE kind = ? AND key = ? AND deleted_at IS NULL",
            (now, now, kind, key),
        )
        if cursor.rowcount:
            _bump_version(conn)
    return bool(cursor.rowcount)


def tombstone_memory_type(conn: sqlite3.Connection, name: str) -> bool:
    """Tombstone an overlay memory type. Built-in and env types are not in the table."""
    return _tombstone(conn, "memory_type", name)


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


def upsert_retention(conn: sqlite3.Connection, memory_type: str, days: int) -> None:
    """Set (or replace) the retention period of a memory type and bump the version."""
    with conn:
        conn.execute(
            "INSERT INTO ontology_entries (kind, key, payload_json, updated_at) "
            "VALUES ('retention', ?, ?, ?) "
            "ON CONFLICT(kind, key) DO UPDATE SET "
            "payload_json = excluded.payload_json, updated_at = excluded.updated_at, "
            "deleted_at = NULL",
            (memory_type, json.dumps({"days": days}), time.time()),
        )
        _bump_version(conn)


def tombstone_retention(conn: sqlite3.Connection, memory_type: str) -> bool:
    """Tombstone a retention entry; the type falls back to its parent's period."""
    return _tombstone(conn, "retention", memory_type)


class OverlaySource:
    """What ``models.ontology`` polls.

    For a database file it opens its own short-lived connection per read, so
    it is safe from any thread and sees writes by other processes. An
    in-memory database exists only on the storage's own connection, so there
    the storage calls ``capture()`` after each write and reads come from that
    snapshot. No other process can write to it, so nothing is missed.
    """

    def __init__(self, db_path: str):
        self._snapshot: Tuple[int, List[Tuple[str, Optional[str]]], Dict[str, int]] = (0, [], {})
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

    def read_memory_type_overlay(self) -> List[Tuple[str, Optional[str]]]:
        return self._read(read_memory_type_overlay, self._snapshot[1])

    def read_retention_overlay(self) -> Dict[str, int]:
        return self._read(read_retention_overlay, self._snapshot[2])
