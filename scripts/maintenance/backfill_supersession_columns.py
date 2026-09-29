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
    """Resolve the SQLite-vec DB the way the service does.

    Prefer the service's own resolved ``config.storage.SQLITE_VEC_PATH`` (which honors
    MCP_MEMORY_SQLITE_PATH, MCP_MEMORY_SQLITEVEC_PATH, MCP_MEMORY_BASE_DIR and the
    per-OS default), so the script never backfills a different store than the one the
    service uses. ``--db`` overrides everything for ad-hoc runs.
    """
    if explicit:
        return Path(explicit).expanduser().resolve()
    try:
        from mcp_memory_service.config.storage import SQLITE_VEC_PATH  # type: ignore
        if SQLITE_VEC_PATH:
            return Path(SQLITE_VEC_PATH).expanduser().resolve()
    except Exception as exc:  # config import shouldn't hard-fail the script
        log.debug("Could not import service config (%s); falling back to env vars.", exc)
    for env_var in ("MCP_MEMORY_SQLITE_PATH", "MCP_MEMORY_SQLITEVEC_PATH"):
        value = os.getenv(env_var)
        if value:
            return Path(value).expanduser().resolve()
    base = os.getenv("MCP_MEMORY_BASE_DIR")
    if base:
        return (Path(base).expanduser() / "sqlite_vec.db").resolve()
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
        "SELECT content_hash, metadata, superseded_by, version "
        "FROM memories WHERE deleted_at IS NULL AND metadata LIKE '%superseded_by%'"
    ).fetchall()

    # content_hash -> new superseded_by value to set on the OLD (loser) row.
    old_updates: list[tuple[str, str]] = []
    # winner_hash -> (parent_hash, new_version) to reconnect the NEW row's history.
    # The legacy trace only lives on the old row (old.metadata.superseded_by = winner);
    # parent_id/version were never written anywhere. We derive the inverse link:
    # if OLD is superseded by NEW, then NEW.parent_id = OLD and NEW.version = OLD.version + 1,
    # so get_memory_history (which walks parent_id) can reconnect the chain (Greptile P1.2).
    new_links: dict[str, tuple[str, int]] = {}
    orphans: list[str] = []

    for content_hash, metadata, col_superseded, col_version in rows:
        stats["scanned"] += 1
        try:
            meta = json.loads(metadata) if metadata else {}
        except (json.JSONDecodeError, TypeError):
            continue

        winner = meta.get("superseded_by")
        if not (_valid_hash(winner) and not col_superseded):
            continue

        winner_row = conn.execute(
            "SELECT parent_id, version FROM memories WHERE content_hash = ? AND deleted_at IS NULL",
            (winner,),
        ).fetchone()
        if winner_row is None:
            # #1352: pointing the column at a deleted winner would hide this row
            # with no way back. Leave it visible; report for operator review.
            stats["winner_gone"] += 1
            orphans.append(content_hash)
            continue

        # OLD row: set the real superseded_by column so default retrieval drops it.
        old_updates.append((winner, content_hash))
        stats["superseded_filled"] += 1

        # NEW row: reconnect history only if it isn't already linked. Longest chain wins
        # so multi-step chains (v1->v2->v3) get monotonic versions.
        w_parent, w_version = winner_row
        if not w_parent:
            old_version = col_version or 1
            candidate_version = old_version + 1
            existing = new_links.get(winner)
            if existing is None or candidate_version > existing[1]:
                new_links[winner] = (content_hash, candidate_version)

    if orphans:
        log.warning("%d row(s) point at a deleted/missing winner in metadata; left "
                    "visible (see #1352). Example hashes: %s",
                    len(orphans), ", ".join(h[:8] for h in orphans[:5]))

    stats["parent_filled"] = len(new_links)
    stats["version_filled"] = len(new_links)

    if apply:
        for winner, old_hash in old_updates:
            conn.execute(
                "UPDATE memories SET superseded_by = ? WHERE content_hash = ?",
                (winner, old_hash),
            )
        for winner, (parent_hash, new_version) in new_links.items():
            conn.execute(
                "UPDATE memories SET parent_id = ?, version = ? WHERE content_hash = ?",
                (parent_hash, new_version, winner),
            )
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
