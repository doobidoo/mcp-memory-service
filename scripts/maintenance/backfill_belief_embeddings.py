#!/usr/bin/env python3
"""Backfill belief embeddings (R4).

Computes and stores a serialize_float32 embedding for every belief whose
`embedding` column is still NULL (migration 021 adds the column nullable).
Idempotent: safe to re-run — only touches NULL rows.

Usage:
    VIRTUAL_ENV=.venv PYTHONPATH=src MCP_MEMORY_USE_ONNX=1 \
        .venv/bin/python scripts/maintenance/backfill_belief_embeddings.py [DB_PATH]

If DB_PATH is omitted, uses the live default (~/local-data/mcp/sqlite_vec.db).
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage  # noqa: E402

DEFAULT_DB = os.path.expanduser("~/local-data/mcp/sqlite_vec.db")


async def main(db_path: str) -> int:
    storage = SqliteVecMemoryStorage(db_path)
    await storage.initialize()

    if getattr(storage, "embedding_model", None) is None:
        print("ERROR: no embedding model available; cannot backfill.", file=sys.stderr)
        return 1

    from sqlite_vec import serialize_float32

    conn = storage.conn
    rows = conn.execute(
        "SELECT belief_hash, content FROM beliefs WHERE embedding IS NULL"
    ).fetchall()
    print(f"beliefs pending embedding: {len(rows)}")

    done = 0
    for belief_hash, content in rows:
        try:
            vec = storage._generate_embedding(content or "")
            conn.execute(
                "UPDATE beliefs SET embedding = ? WHERE belief_hash = ?",
                (serialize_float32(vec), belief_hash),
            )
            done += 1
        except Exception as e:  # pragma: no cover
            print(f"  skip {belief_hash[:12]}: {e}", file=sys.stderr)
    conn.commit()

    remaining = conn.execute(
        "SELECT COUNT(*) FROM beliefs WHERE embedding IS NULL"
    ).fetchone()[0]
    print(f"backfilled: {done} | still NULL: {remaining}")
    return 0


if __name__ == "__main__":
    db = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DB
    raise SystemExit(asyncio.run(main(db)))
