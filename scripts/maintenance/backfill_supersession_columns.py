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
    # Edges of the legacy supersession forest: old_hash -> winner_hash (winner is the
    # newer version). The legacy trace only lives on the old row
    # (old.metadata.superseded_by = winner); parent_id/version were never written.
    # We reconnect history by deriving the inverse link (NEW.parent_id = OLD) and, for
    # the version, by numbering each chain from its root so multi-step chains
    # v1->v2->v3 get 1,2,3 instead of colliding at 2 (Greptile P1.2 + P1.4).
    edges: dict[str, str] = {}  # old_hash -> winner_hash
    # cache of which content_hashes exist (live) and their already-set parent_id/version
    live_cache: dict[str, object] = {}

    def _live(h: str):
        """Return (parent_id, version) for a live hash, or "__missing__" if not found."""
        if h not in live_cache:
            r = conn.execute(
                "SELECT parent_id, version FROM memories WHERE content_hash = ? AND deleted_at IS NULL",
                (h,),
            ).fetchone()
            live_cache[h] = (r if r else "__missing__")
        return live_cache[h]

    orphans: list[str] = []

    for content_hash, metadata, col_superseded, _col_version in rows:
        stats["scanned"] += 1
        try:
            meta = json.loads(metadata) if metadata else {}
        except (json.JSONDecodeError, TypeError):
            continue

        winner = meta.get("superseded_by")
        if not (_valid_hash(winner) and not col_superseded):
            continue

        if _live(winner) == "__missing__":
            # #1352: pointing the column at a deleted winner would hide this row
            # with no way back. Leave it visible; report for operator review.
            stats["winner_gone"] += 1
            orphans.append(content_hash)
            continue

        # OLD row: set the real superseded_by column so default retrieval drops it.
        old_updates.append((winner, content_hash))
        stats["superseded_filled"] += 1
        edges[content_hash] = winner

    # Walk each chain and assign versions using anchor-aware numbering (P1.5 fix).
    # For each chain ordered root->tip:
    # - Find PIVOT = first node with existing version (anchor from #1348) 
    # - Number forward (pivot+1, +2...) and backward (pivot-1, -2...) from pivot
    # - If backward numbering would create version < 1, leave those nodes with version NULL
    # - Nodes already linked (have parent_id) are NEVER overwritten (idempotency)
    winners = set(edges.values())
    roots = [old for old in edges if old not in winners]
    new_links: dict[str, tuple[str, int]] = {}  # winner -> (parent, version)
    seen: set[str] = set()
    
    for root in roots:
        # Build the complete chain from root to tip (including final winner)
        chain = []
        node = root
        while True:
            chain.append(node)
            if node not in edges:
                break
            nxt = edges[node]
            if nxt in seen:  # defensive: cycle / diamond, stop
                break
            seen.add(node)
            node = nxt
            
        # Add the final winner to the chain if it exists
        if chain and chain[-1] in edges:
            final_winner = edges[chain[-1]]
            if _live(final_winner) != "__missing__":
                chain.append(final_winner)
        
        # Find pivot: first node in chain that has existing parent_id (anchor)
        pivot_idx = None
        pivot_version = None
        
        for i, node_hash in enumerate(chain):
            live_info = _live(node_hash)
            if live_info != "__missing__":
                parent_id, version = live_info
                if parent_id:  # This node already has parent_id (anchor from #1348)
                    pivot_idx = i
                    pivot_version = version
                    break
        
        # If no anchor found, pivot is root with version 1 (pure legacy chain - P1.4 case)
        if pivot_idx is None:
            pivot_idx = 0
            pivot_version = 1
        
        # Number the chain from pivot, but only process edges (old->winner relationships)
        for i, node_hash in enumerate(chain[:-1]):  # Exclude last node since it has no outgoing edge
            if node_hash in edges:  # This is an old node, process its winner
                winner_hash = edges[node_hash]
                winner_idx = i + 1  # Winner is at next position in chain
                
                live_info = _live(winner_hash)
                if live_info == "__missing__":
                    continue  # Skip missing winners
                
                parent_id, existing_version = live_info
                
                # Skip nodes already linked (idempotency) 
                if parent_id:
                    continue
                
                # Calculate version based on winner's position relative to pivot
                if winner_idx == pivot_idx:
                    # This winner IS the pivot
                    new_version = pivot_version
                elif winner_idx < pivot_idx:
                    # Winner is before pivot: count backward
                    new_version = pivot_version - (pivot_idx - winner_idx)
                else:
                    # Winner is after pivot: count forward  
                    new_version = pivot_version + (winner_idx - pivot_idx)
                
                # Check if this version would collide with any existing version in the chain
                version_collision = False
                if new_version >= 1:
                    for check_i, check_node_hash in enumerate(chain):
                        if check_i == winner_idx:
                            continue  # Skip self
                        check_live_info = _live(check_node_hash)
                        if check_live_info != "__missing__":
                            _, check_version = check_live_info
                            if check_version == new_version:
                                version_collision = True
                                break
                
                # Only set version if >= 1 and no collision, otherwise leave NULL and warn
                if new_version >= 1 and not version_collision:
                    new_links[winner_hash] = (node_hash, new_version)
                else:
                    # Set only parent_id, leave version NULL
                    new_links[winner_hash] = (node_hash, None)
                    if new_version < 1:
                        log.warning("chain has inconsistent anchor version; left version unset for 1 row(s)")
                    elif version_collision:
                        log.warning("version collision detected; left version unset for 1 row(s)")
    
    # Filter new_links into those with versions and those without
    links_with_version = {k: v for k, v in new_links.items() if v[1] is not None}
    links_without_version = {k: v for k, v in new_links.items() if v[1] is None}

    if orphans:
        log.warning("%d row(s) point at a deleted/missing winner in metadata; left "
                    "visible (see #1352). Example hashes: %s",
                    len(orphans), ", ".join(h[:8] for h in orphans[:5]))

    stats["parent_filled"] = len(new_links)
    stats["version_filled"] = len(links_with_version)

    if apply:
        for winner, old_hash in old_updates:
            conn.execute(
                "UPDATE memories SET superseded_by = ? WHERE content_hash = ?",
                (winner, old_hash),
            )
        # Update links that have both parent_id and version
        for winner, (parent_hash, new_version) in links_with_version.items():
            conn.execute(
                "UPDATE memories SET parent_id = ?, version = ? WHERE content_hash = ?",
                (parent_hash, new_version, winner),
            )
        # Update links that have only parent_id (version stays NULL due to version < 1)
        for winner, (parent_hash, _) in links_without_version.items():
            conn.execute(
                "UPDATE memories SET parent_id = ?, version = NULL WHERE content_hash = ?",
                (parent_hash, winner),
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
