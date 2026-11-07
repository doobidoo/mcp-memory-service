"""Ontology overlay stored in the memory DB (#1897, phase 1).

The overlay adds memory types on top of the built-in ontology and
``MCP_CUSTOM_MEMORY_TYPES``. It lives in the ``ontology_entries`` table, and
every write bumps ``metadata.ontology_version``. ``models.ontology`` polls that
counter and reloads the overlay when it moved.

Only the sqlite_vec database carries the table. Cloudflare/D1 does not, so on
the cloudflare backend, and between hybrid nodes, ``MCP_CUSTOM_MEMORY_TYPES``
is still the way to share custom types.
"""

import sqlite3
import time
from pathlib import Path
from typing import List, Optional, Tuple

from ..models import ontology


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
    """Write (or revive) an overlay memory type and bump the version, in one transaction."""
    with conn:  # commits both statements, or rolls both back
        conn.execute(
            "INSERT INTO ontology_entries (kind, key, parent, updated_at) "
            "VALUES ('memory_type', ?, ?, ?) "
            "ON CONFLICT(kind, key) DO UPDATE SET "
            "parent = excluded.parent, updated_at = excluded.updated_at, deleted_at = NULL",
            (name, parent, time.time()),
        )
        _bump_version(conn)


def tombstone_memory_type(conn: sqlite3.Connection, name: str) -> bool:
    """Tombstone an overlay memory type. Built-in and env types are not in the table."""
    now = time.time()
    with conn:
        cursor = conn.execute(
            "UPDATE ontology_entries SET deleted_at = ?, updated_at = ? "
            "WHERE kind = 'memory_type' AND key = ? AND deleted_at IS NULL",
            (now, now, name),
        )
        if cursor.rowcount:
            _bump_version(conn)
    return bool(cursor.rowcount)


class OverlaySource:
    """What ``models.ontology`` polls. Opens its own short-lived connection per
    read, so it is safe from any thread and sees writes by other processes."""

    def __init__(self, db_path: str):
        # Read-only: a source left behind by a storage that was never closed
        # must not create an empty database where the file used to be.
        self.uri = Path(db_path).resolve().as_uri() + "?mode=ro"

    def _read(self, query):
        conn = sqlite3.connect(self.uri, uri=True, timeout=5)
        try:
            return query(conn)
        finally:
            conn.close()

    def read_ontology_version(self) -> int:
        return self._read(read_ontology_version)

    def read_memory_type_overlay(self) -> List[Tuple[str, Optional[str]]]:
        return self._read(read_memory_type_overlay)
