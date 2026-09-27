#!/usr/bin/env python3
"""Backfill the last_accessed COLUMN from metadata['last_accessed_at'] (issue #1239).

The retrieve() write-back historically only wrote metadata JSON, leaving the
last_accessed column NULL — so staleness/decay measured age, not disuse. This
one-shot migration repairs existing rows. Idempotent; dry-run by default.
"""
import argparse, json, sqlite3


def backfill(db_path: str, apply: bool) -> dict:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT content_hash, metadata, last_accessed FROM memories WHERE deleted_at IS NULL"
    ).fetchall()
    stats = {"total": len(rows), "updated": 0, "already_set": 0, "no_metadata_ts": 0}
    for r in rows:
        if r["last_accessed"] is not None:
            stats["already_set"] += 1
            continue
        try:
            meta = json.loads(r["metadata"]) if r["metadata"] else {}
        except (json.JSONDecodeError, TypeError):
            meta = {}
        la = meta.get("last_accessed_at")
        if la is None:
            stats["no_metadata_ts"] += 1
            continue
        if apply:
            cur.execute(
                "UPDATE memories SET last_accessed = ? WHERE content_hash = ?",
                (int(float(la)), r["content_hash"]),
            )
        stats["updated"] += 1
    if apply:
        conn.commit()
    conn.close()
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/home/claudio/local-data/mcp/sqlite_vec.db")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    s = backfill(args.db, args.apply)
    mode = "APPLIED" if args.apply else "DRY-RUN"
    print(f"[{mode}] total={s['total']} updated={s['updated']} "
          f"already_set={s['already_set']} no_metadata_ts={s['no_metadata_ts']}")
    if not args.apply:
        print("Re-run with --apply to write.")
