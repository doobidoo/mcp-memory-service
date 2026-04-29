#!/usr/bin/env python3
"""Migrate memories from sqlite-vec to pgvector, preserving content + embeddings.

Reads the local sqlite-vec database (the one currently mounted into the
running mcp-memory-service container), reads each memory + its existing
768-dim embedding, and copies both into pgvector. Embeddings are copied
verbatim — no re-embedding — so the migration is fast and the resulting
recall behavior is byte-identical to the source.

The script is **idempotent**: re-running it skips rows whose content_hash
already exists in the target. Soft-deleted memories (deleted_at IS NOT NULL)
are migrated with their tombstone preserved.

Usage:
    # Dry run — counts what would happen, writes nothing
    python scripts/team/migrate_sqlite_to_pgvector.py \\
        --sqlite /path/to/sqlite_vec.db \\
        --dsn 'postgresql://mcp_memory:PASS@localhost:5432/mcp_memory' \\
        --dry-run

    # Real run
    python scripts/team/migrate_sqlite_to_pgvector.py \\
        --sqlite /path/to/sqlite_vec.db \\
        --dsn 'postgresql://mcp_memory:PASS@localhost:5432/mcp_memory'

    # From inside the running container (db is at /app/data/sqlite_vec.db):
    python -m scripts.team.migrate_sqlite_to_pgvector \\
        --sqlite /app/data/sqlite_vec.db \\
        --dsn "$MCP_PGVECTOR_DSN"

The script connects to the target database, ensures the `vector` extension
and the schema/tables exist (matching the live backend's schema), then
streams rows in batches.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sqlite3
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger("migrate_sqlite_to_pgvector")


# --------------------------------------------------------------------- helpers

def _deserialize_float32_blob(blob: bytes) -> List[float]:
    """sqlite-vec stores embeddings as packed little-endian float32. Decode
    back into a Python list[float] suitable for pgvector."""
    if not blob:
        return []
    if len(blob) % 4 != 0:
        raise ValueError(f"Embedding blob length {len(blob)} not a multiple of 4")
    count = len(blob) // 4
    return list(struct.unpack(f"<{count}f", blob))


def _split_tags(tags_field: Optional[str]) -> List[str]:
    """sqlite-vec stores tags as a comma-joined string; pgvector wants text[]."""
    if not tags_field:
        return []
    return [t.strip() for t in tags_field.split(",") if t.strip()]


def _parse_metadata(metadata_field: Optional[str]) -> Dict[str, Any]:
    if not metadata_field:
        return {}
    try:
        loaded = json.loads(metadata_field)
        if isinstance(loaded, dict):
            return loaded
        return {}
    except json.JSONDecodeError:
        return {}


def _epoch_to_tombstone(ts: Optional[float]) -> Optional[datetime]:
    if ts is None:
        return None
    return datetime.fromtimestamp(float(ts), tz=timezone.utc)


# --------------------------------------------------------------------- reader

def _read_source(sqlite_path: Path, batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    """Yield batches of source rows from sqlite_vec.

    Each row dict carries the raw fields plus the decoded embedding. The
    inner JOIN to memory_embeddings means we silently skip any orphan
    memory rows that lack an embedding — they wouldn't be searchable in
    pgvector either, and re-running the live service would re-embed them.
    """
    if not sqlite_path.exists():
        raise FileNotFoundError(f"sqlite source not found: {sqlite_path}")

    con = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        cur = con.execute(
            """
            SELECT m.id, m.content_hash, m.content, m.tags, m.memory_type, m.metadata,
                   m.created_at, m.updated_at, m.created_at_iso, m.updated_at_iso,
                   m.deleted_at,
                   e.content_embedding AS embedding_blob
            FROM memories m
            JOIN memory_embeddings e ON e.rowid = m.id
            ORDER BY m.id
            """
        )
        batch: List[Dict[str, Any]] = []
        while True:
            rows = cur.fetchmany(batch_size)
            if not rows:
                break
            for row in rows:
                batch.append(dict(row))
            yield batch
            batch = []
    finally:
        con.close()


# --------------------------------------------------------------------- writer

async def _ensure_target_schema(conn, schema: str, dim: int) -> None:
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    if schema != "public":
        await conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}";')

    await conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS "{schema}"."memories" (
            id              BIGSERIAL PRIMARY KEY,
            content_hash    TEXT        NOT NULL UNIQUE,
            content         TEXT        NOT NULL,
            tags            TEXT[]      NOT NULL DEFAULT '{{}}',
            memory_type     TEXT,
            metadata        JSONB       NOT NULL DEFAULT '{{}}'::jsonb,
            created_at      DOUBLE PRECISION NOT NULL,
            updated_at      DOUBLE PRECISION NOT NULL,
            created_at_iso  TEXT,
            updated_at_iso  TEXT,
            deleted_at      TIMESTAMPTZ
        );
        """
    )
    await conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS "{schema}"."memory_embeddings" (
            memory_id     BIGINT PRIMARY KEY
                          REFERENCES "{schema}"."memories"(id) ON DELETE CASCADE,
            content_hash  TEXT NOT NULL UNIQUE,
            embedding     vector({dim}) NOT NULL
        );
        """
    )


async def _write_batch(conn, schema: str, batch: List[Dict[str, Any]]) -> Tuple[int, int]:
    """Write a batch inside a single transaction.

    Returns (inserted, skipped). Skipped means content_hash already in
    target — re-runs converge.
    """
    inserted = 0
    skipped = 0

    async with conn.transaction():
        for row in batch:
            embedding = _deserialize_float32_blob(row["embedding_blob"])
            tags = _split_tags(row["tags"])
            metadata = _parse_metadata(row["metadata"])
            tombstone = _epoch_to_tombstone(row["deleted_at"])

            inserted_row = await conn.fetchrow(
                f"""
                INSERT INTO "{schema}"."memories"
                    (content_hash, content, tags, memory_type, metadata,
                     created_at, updated_at, created_at_iso, updated_at_iso, deleted_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (content_hash) DO NOTHING
                RETURNING id
                """,
                row["content_hash"], row["content"], tags, row["memory_type"],
                json.dumps(metadata),
                row["created_at"], row["updated_at"],
                row["created_at_iso"], row["updated_at_iso"],
                tombstone,
            )

            if inserted_row is None:
                skipped += 1
                continue

            await conn.execute(
                f"""
                INSERT INTO "{schema}"."memory_embeddings" (memory_id, content_hash, embedding)
                VALUES ($1, $2, $3)
                ON CONFLICT (memory_id) DO NOTHING
                """,
                inserted_row["id"], row["content_hash"], embedding,
            )
            inserted += 1

    return inserted, skipped


async def _migrate(sqlite_path: Path, dsn: str, schema: str, batch_size: int, dry_run: bool) -> None:
    # Peek at the source to learn the embedding dimension before touching pg.
    con = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        cur = con.execute("SELECT content_embedding FROM memory_embeddings LIMIT 1")
        first = cur.fetchone()
        if first is None or not first[0]:
            logger.warning("Source has no embeddings — nothing to migrate.")
            return
        dim = len(_deserialize_float32_blob(first[0]))
        total = con.execute(
            "SELECT COUNT(*) FROM memories m JOIN memory_embeddings e ON e.rowid = m.id"
        ).fetchone()[0]
    finally:
        con.close()

    logger.info("Source: %s (rows=%d, embedding_dim=%d)", sqlite_path, total, dim)

    if dry_run:
        logger.info("Dry run requested — no writes will be issued.")
        # Re-iterate purely to give a representative summary.
        seen = 0
        for batch in _read_source(sqlite_path, batch_size):
            seen += len(batch)
        logger.info("Would migrate %d rows.", seen)
        return

    # Real run: connect, ensure schema, stream batches.
    import asyncpg  # local import so dry-run works without asyncpg installed
    from pgvector.asyncpg import register_vector

    conn = await asyncpg.connect(dsn=dsn)
    try:
        await register_vector(conn)
        await conn.execute(f'SET search_path TO "{schema}", public;')
        await _ensure_target_schema(conn, schema, dim)

        inserted_total = 0
        skipped_total = 0
        start = time.monotonic()

        for batch_idx, batch in enumerate(_read_source(sqlite_path, batch_size), start=1):
            inserted, skipped = await _write_batch(conn, schema, batch)
            inserted_total += inserted
            skipped_total += skipped
            logger.info(
                "Batch %d: inserted=%d skipped=%d (running totals inserted=%d skipped=%d)",
                batch_idx, inserted, skipped, inserted_total, skipped_total,
            )

        elapsed = time.monotonic() - start
        logger.info(
            "Done. inserted=%d skipped=%d total=%d elapsed=%.1fs",
            inserted_total, skipped_total, inserted_total + skipped_total, elapsed,
        )
    finally:
        await conn.close()


# --------------------------------------------------------------------- cli

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Migrate mcp-memory-service data from sqlite-vec to pgvector.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--sqlite", required=True, type=Path,
        help="Path to the source sqlite_vec.db file",
    )
    p.add_argument(
        "--dsn", required=True,
        help="postgresql:// DSN of the target database (must have pgvector available)",
    )
    p.add_argument(
        "--schema", default="public",
        help="Target schema (default: public)",
    )
    p.add_argument(
        "--batch-size", type=int, default=200,
        help="Rows per transaction (default: 200)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Read the source and report counts without writing to the target",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable DEBUG logging",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        asyncio.run(_migrate(
            sqlite_path=args.sqlite,
            dsn=args.dsn,
            schema=args.schema,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
        ))
        return 0
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130
    except Exception as e:
        logger.error("Migration failed: %s", e, exc_info=args.verbose if hasattr(args, 'verbose') else False)
        return 1


if __name__ == "__main__":
    sys.exit(main())
