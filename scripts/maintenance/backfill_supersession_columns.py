#!/usr/bin/env python3
"""Backfill the migration-011 supersession columns for memories versioned before #1348.

Before #1348, ``update_memory_versioned`` recorded a supersession only in the old
row's ``metadata`` JSON (``metadata["superseded_by"] = <new_hash>``), never in the
``superseded_by`` column that default retrieval filters on. Those old versions stay
searchable and ``get_memory_history`` cannot link the chain.

This one-off walks such rows and copies the metadata trace into the real columns:

* ``superseded_by`` on the old row (only when the winner still exists — see #1352:
  pointing at a deleted winner would strand the row invisibly, so those are skipped
  and reported instead).
* ``parent_id`` / ``version`` when the metadata carries them but the columns are empty.

Approach and field semantics follow @timkjr's production backfill reported on #1348
(walking the chains, leaving deleted-winner rows visible).

Safe by default: prints a plan and writes nothing unless ``--apply`` is given.
The database is taken from ``MCP_MEMORY_SQLITE_PATH`` (or the standard default).
Stop the service before applying so writes do not race the running process.
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_HASH_MIN_LEN = 16  # generate_content_hash is a 64-char sha256; guard against junk


def resolve_db_path(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.getenv("MCP_MEMORY_SQLITE_PATH")
    if env:
        return Path(env).expanduser().resolve()
    return Path("~/.local/share/mcp-memory/sqlite_vec.db").expanduser().resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--db", help="SQLite path (default: MCP_MEMORY_SQLITE_PATH or standard location).")
    p.add_argument("--apply", action="store_true", help="Write the changes. Without it, this is a dry run.")
    return p.parse_args(argv)


def _valid_hash(value: object) -> bool:
    return isinstance(value, str) and len(value) >= _HASH_MIN_LEN


def backfill(conn: sqlite3.Connection, apply: bool) -> dict[str, int]:
    stats = {"scanned": 0, "superseded_filled": 0, "winner_gone": 0,
             "parent_filled": 0, "version_filled": 0}

    rows = conn.execute(
        "SELECT content_hash, metadata, superseded_by, parent_id, version "
        "FROM memories WHERE deleted_at IS NULL AND metadata LIKE '%superseded_by%'"
    ).fetchall()

    updates: list[tuple] = []
    orphans: list[str] = []

    for content_hash, metadata, col_superseded, col_parent, col_version in rows:
        stats["scanned"] += 1
        try:
            meta = json.loads(metadata) if metadata else {}
        except (json.JSONDecodeError, TypeError):
            continue

        new_superseded = None
        winner = meta.get("superseded_by")
        if _valid_hash(winner) and not col_superseded:
            exists = conn.execute(
                "SELECT 1 FROM memories WHERE content_hash = ? AND deleted_at IS NULL",
                (winner,),
            ).fetchone()
            if exists:
                new_superseded = winner
                stats["superseded_filled"] += 1
            else:
                # #1352: pointing the column at a deleted winner would hide this row
                # with no way back. Leave it visible; report for operator review.
                stats["winner_gone"] += 1
                orphans.append(content_hash)

        new_parent = None
        meta_parent = meta.get("parent_id")
        if _valid_hash(meta_parent) and not col_parent:
            new_parent = meta_parent
            stats["parent_filled"] += 1

        new_version = None
        meta_version = meta.get("version")
        if isinstance(meta_version, int) and meta_version > 1 and (col_version or 1) <= 1:
            new_version = meta_version
            stats["version_filled"] += 1

        if new_superseded is not None or new_parent is not None or new_version is not None:
            updates.append((new_superseded, new_parent, new_version, content_hash))

    if orphans:
        log.warning("%d row(s) point at a deleted/missing winner in metadata; left "
                    "visible (see #1352). Example hashes: %s",
                    len(orphans), ", ".join(h[:8] for h in orphans[:5]))

    if apply and updates:
        for new_superseded, new_parent, new_version, content_hash in updates:
            sets, params = [], []
            if new_superseded is not None:
                sets.append("superseded_by = ?"); params.append(new_superseded)
            if new_parent is not None:
                sets.append("parent_id = ?"); params.append(new_parent)
            if new_version is not None:
                sets.append("version = ?"); params.append(new_version)
            params.append(content_hash)
            conn.execute(f"UPDATE memories SET {', '.join(sets)} WHERE content_hash = ?", params)
        conn.commit()

    return stats


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    db_path = resolve_db_path(args.db)
    if not db_path.exists():
        log.error("Database not found: %s", db_path)
        return 1

    log.info("Database: %s", db_path)
    log.info("Mode: %s", "APPLY (writing)" if args.apply else "DRY RUN (no writes)")

    conn = sqlite3.connect(str(db_path))
    try:
        stats = backfill(conn, apply=args.apply)
    finally:
        conn.close()

    log.info("Scanned %d candidate row(s).", stats["scanned"])
    log.info("superseded_by to fill (winner exists): %d", stats["superseded_filled"])
    log.info("parent_id to fill: %d | version to fill: %d",
             stats["parent_filled"], stats["version_filled"])
    log.info("winner deleted/missing (skipped, left visible): %d", stats["winner_gone"])
    if not args.apply:
        log.info("Dry run only. Re-run with --apply (service stopped) to write.")
    else:
        log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
