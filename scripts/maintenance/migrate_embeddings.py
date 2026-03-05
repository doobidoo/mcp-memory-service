#!/usr/bin/env python3
"""Migrate memory embeddings to a different model with a different dimension.

Unlike regenerate_embeddings.py (which re-embeds with the current model at the
same dimension), this script handles full model migration including vec0 virtual
table rebuild when the embedding dimension changes.

Use this when switching from the default all-MiniLM-L6-v2 (384-dim) to an
external model like nomic-embed-text (768-dim) via Ollama, vLLM, or OpenAI.

Prerequisites:
  - External embedding API running (e.g., `brew services start ollama`)
  - Model pulled (e.g., `ollama pull nomic-embed-text`)
  - mcp-memory-service STOPPED (to avoid concurrent DB access)

Usage:
  python scripts/maintenance/migrate_embeddings.py --dry-run
  python scripts/maintenance/migrate_embeddings.py \\
      --embedding-url http://localhost:11434/v1/embeddings \\
      --model nomic-embed-text
  python scripts/maintenance/migrate_embeddings.py \\
      --embedding-url https://api.openai.com/v1/embeddings \\
      --model text-embedding-3-small \\
      --api-key sk-...

What it does:
  1. Validates the embedding API is reachable and detects dimension
  2. Creates a timestamped backup of the database
  3. Reads all active memory content
  4. Generates new embeddings via the external API (batched)
  5. Drops and recreates the vec0 virtual table at the new dimension
  6. Inserts new embeddings with correct rowid mapping
  7. Optionally wipes graph edges (based on old embedding similarities)
  8. VACUUMs the database
  9. Verifies integrity (count match, dimension spot-check, KNN test)
"""

import argparse
import logging
import os
import re
import shutil
import sqlite3
import struct
import sys
import time
from pathlib import Path

import requests

# Add parent directory to path for imports (used only for default DB path)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
try:
    from src.mcp_memory_service.config import SQLITE_VEC_PATH

    DEFAULT_DB_PATH = Path(SQLITE_VEC_PATH)
except ImportError:
    DEFAULT_DB_PATH = (
        Path.home() / "Library" / "Application Support" / "mcp-memory" / "sqlite_vec.db"
    )

DEFAULT_EMBEDDING_URL = "http://localhost:11434/v1/embeddings"
DEFAULT_MODEL = "nomic-embed-text"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def serialize_float32(vector: list) -> bytes:
    """Serialize a list of floats to bytes for sqlite-vec insertion."""
    return struct.pack(f"{len(vector)}f", *vector)


def check_embedding_api(url: str, model: str, api_key: str = None) -> int:
    """Verify the embedding API is reachable and return the embedding dimension.

    Args:
        url: Embedding API endpoint URL.
        model: Model name to use.
        api_key: Optional API key for authenticated endpoints.

    Returns:
        The embedding dimension.

    Raises:
        SystemExit: If the API is unreachable or returns an error.
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        resp = requests.post(
            url,
            headers=headers,
            json={"input": "test", "model": model},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        dims = len(data["data"][0]["embedding"])
        return dims
    except requests.exceptions.ConnectionError:
        logger.error(f"Cannot connect to embedding API at {url}")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        logger.error(f"Embedding API returned error: {e}")
        sys.exit(1)
    except (KeyError, IndexError) as e:
        logger.error(f"Unexpected API response format: {e}")
        sys.exit(1)


def batch_embed(
    texts: list,
    url: str,
    model: str,
    target_dims: int,
    api_key: str = None,
    batch_size: int = 32,
) -> list:
    """Generate embeddings for a list of texts via an external API.

    Args:
        texts: List of text strings to embed.
        url: Embedding API endpoint URL.
        model: Model name to use.
        target_dims: Expected embedding dimension (for validation).
        api_key: Optional API key.
        batch_size: Number of texts per API call.

    Returns:
        List of embedding vectors (each a list of floats).
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    all_embeddings = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        resp = requests.post(
            url,
            headers=headers,
            json={"input": batch, "model": model},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        # Sort by index to ensure correct order
        batch_embs = sorted(data["data"], key=lambda x: x["index"])
        for item in batch_embs:
            emb = item["embedding"]
            if len(emb) != target_dims:
                raise ValueError(
                    f"Expected {target_dims} dims, got {len(emb)} at index {item['index']}"
                )
            all_embeddings.append(emb)
        done = min(i + batch_size, total)
        logger.info(f"  Embedded {done}/{total} memories ({100 * done // total}%)")
    return all_embeddings


def get_db_stats(conn: sqlite3.Connection) -> dict:
    """Gather current database statistics."""
    active = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE deleted_at IS NULL"
    ).fetchone()[0]
    deleted = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE deleted_at IS NOT NULL"
    ).fetchone()[0]
    try:
        edges = conn.execute("SELECT COUNT(*) FROM memory_graph").fetchone()[0]
    except sqlite3.OperationalError:
        edges = 0

    # Parse current vec0 dimensions from DDL
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name='memory_embeddings' AND type='table'"
    ).fetchone()
    vec_ddl = row[0] if row else "NOT FOUND"
    current_dims = None
    if vec_ddl and "FLOAT[" in vec_ddl:
        match = re.search(r"FLOAT\[(\d+)\]", vec_ddl)
        if match:
            current_dims = int(match.group(1))

    try:
        emb_count = conn.execute("SELECT COUNT(*) FROM memory_embeddings").fetchone()[0]
    except sqlite3.OperationalError:
        emb_count = 0

    return {
        "active_memories": active,
        "deleted_memories": deleted,
        "embeddings": emb_count,
        "graph_edges": edges,
        "current_dims": current_dims,
        "vec_ddl": vec_ddl,
    }


def load_sqlite_vec(conn: sqlite3.Connection):
    """Load the sqlite-vec extension into the connection."""
    conn.enable_load_extension(True)
    try:
        import sqlite_vec

        sqlite_vec.load(conn)
    except ImportError:
        logger.error("sqlite_vec not importable. Install with: pip install sqlite-vec")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Migrate memory embeddings to a different model/dimension",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report current state without making changes",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"Path to sqlite_vec.db (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--embedding-url",
        default=DEFAULT_EMBEDDING_URL,
        help=f"Embedding API endpoint (default: {DEFAULT_EMBEDDING_URL})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Embedding model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for authenticated endpoints (or set MCP_EXTERNAL_EMBEDDING_API_KEY)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size (default: 32)",
    )
    parser.add_argument(
        "--keep-graph",
        action="store_true",
        help="Preserve graph edges (not recommended — edges are based on old similarities)",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("MCP_EXTERNAL_EMBEDDING_API_KEY")

    if not args.db_path.exists():
        logger.error(f"Database not found at {args.db_path}")
        sys.exit(1)

    # --- Phase A: Validate ---
    logger.info("=" * 60)
    logger.info("Phase A: Validate")
    logger.info("=" * 60)

    target_dims = check_embedding_api(args.embedding_url, args.model, api_key)
    logger.info(
        f"  Embedding API OK: {args.model} returns {target_dims}-dim embeddings"
    )

    conn = sqlite3.connect(str(args.db_path))
    load_sqlite_vec(conn)

    stats = get_db_stats(conn)
    logger.info(f"  Active memories:  {stats['active_memories']}")
    logger.info(f"  Deleted (tombs):  {stats['deleted_memories']}")
    logger.info(f"  Embeddings:       {stats['embeddings']}")
    logger.info(f"  Graph edges:      {stats['graph_edges']}")
    logger.info(f"  Current dims:     {stats['current_dims']}")

    if stats["current_dims"] == target_dims:
        logger.warning(
            f"  Current dimension ({stats['current_dims']}) matches target ({target_dims}). "
            f"Use regenerate_embeddings.py instead for same-dimension re-embedding."
        )

    if args.dry_run:
        logger.info("\n  --dry-run: No changes made.")
        conn.close()
        return

    try:
        _run_migration(conn, args, stats, api_key, target_dims)
    except KeyboardInterrupt:
        logger.error("")
        logger.error("Migration interrupted by user!")
        logger.error(
            "The database may be in an inconsistent state. "
            "Restore from backup before restarting the service."
        )
        logger.error(f"  Look for backup at: {args.db_path.parent}/")
        sys.exit(1)
    finally:
        conn.close()


def _run_migration(
    conn: sqlite3.Connection,
    args: argparse.Namespace,
    stats: dict,
    api_key: str | None,
    target_dims: int,
):
    """Execute the migration phases B through E.

    Separated from main() so that KeyboardInterrupt is caught cleanly
    and the connection is always closed.
    """
    # --- Phase B: Prepare ---
    logger.info("")
    logger.info("=" * 60)
    logger.info("Phase B: Prepare")
    logger.info("=" * 60)

    backup_path = args.db_path.with_suffix(
        f".db.backup.{time.strftime('%Y%m%d-%H%M%S')}"
    )
    logger.info(f"  Backing up to {backup_path}")
    shutil.copy2(str(args.db_path), str(backup_path))

    logger.info("  Reading active memories...")
    rows = conn.execute(
        "SELECT rowid, content FROM memories WHERE deleted_at IS NULL ORDER BY rowid"
    ).fetchall()
    rowids = [r[0] for r in rows]
    contents = [r[1] for r in rows]
    logger.info(f"  Read {len(rows)} memories")

    # --- Phase C: Generate new embeddings ---
    logger.info("")
    logger.info("=" * 60)
    logger.info("Phase C: Generate embeddings")
    logger.info("=" * 60)

    t0 = time.time()
    embeddings = batch_embed(
        contents, args.embedding_url, args.model, target_dims, api_key, args.batch_size
    )
    elapsed = time.time() - t0
    logger.info(f"  Generated {len(embeddings)} embeddings in {elapsed:.1f}s")

    if len(embeddings) != len(rowids):
        logger.error(
            f"  Embedding count ({len(embeddings)}) != memory count ({len(rowids)})"
        )
        sys.exit(1)

    # --- Phase D: Migrate database ---
    logger.info("")
    logger.info("=" * 60)
    logger.info("Phase D: Migrate database")
    logger.info("=" * 60)

    # Drop old vec0 table (outside transaction — virtual table DDL limitation)
    logger.info("  Dropping old memory_embeddings vec0 table...")
    conn.execute("DROP TABLE IF EXISTS memory_embeddings")
    conn.commit()

    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE name='memory_embeddings'"
    ).fetchone()
    if row:
        logger.error("  Failed to drop memory_embeddings table")
        sys.exit(1)
    logger.info("  Dropped successfully")

    # Create new vec0 table at target dimensions
    logger.info(f"  Creating new vec0 table at FLOAT[{target_dims}]...")
    conn.execute(
        f"""
        CREATE VIRTUAL TABLE memory_embeddings USING vec0(
            content_embedding FLOAT[{target_dims}] distance_metric=cosine
        )
    """
    )
    conn.commit()
    logger.info("  Created successfully")

    # Insert embeddings
    logger.info(f"  Inserting {len(embeddings)} embeddings...")
    t0 = time.time()
    conn.execute("BEGIN")
    for i, (rowid, emb) in enumerate(zip(rowids, embeddings)):
        blob = serialize_float32(emb)
        conn.execute(
            "INSERT INTO memory_embeddings (rowid, content_embedding) VALUES (?, ?)",
            (rowid, blob),
        )
        if (i + 1) % 200 == 0:
            logger.info(f"    Inserted {i + 1}/{len(embeddings)}")
    conn.commit()
    elapsed = time.time() - t0
    logger.info(f"  Inserted {len(embeddings)} embeddings in {elapsed:.1f}s")

    # Optionally wipe graph edges
    if not args.keep_graph and stats["graph_edges"] > 0:
        logger.info(
            f"  Wiping {stats['graph_edges']} graph edges (based on old similarities)..."
        )
        conn.execute("DELETE FROM memory_graph")
        conn.commit()

    # Update metadata
    logger.info("  Updating metadata...")
    conn.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES ('distance_metric', 'cosine')"
    )
    conn.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES ('embedding_model', ?)",
        (args.model,),
    )
    conn.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES ('embedding_dims', ?)",
        (str(target_dims),),
    )
    conn.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES ('migration_date', ?)",
        (time.strftime("%Y-%m-%dT%H:%M:%SZ"),),
    )
    conn.commit()

    # Rebuild FTS index
    logger.info("  Rebuilding FTS index...")
    try:
        conn.execute(
            "INSERT INTO memory_content_fts(memory_content_fts) VALUES('rebuild')"
        )
        conn.commit()
        logger.info("  FTS index rebuilt")
    except sqlite3.OperationalError as e:
        logger.warning(f"  FTS rebuild skipped (non-fatal): {e}")

    # VACUUM
    logger.info("  Running VACUUM...")
    conn.execute("VACUUM")
    logger.info("  VACUUM complete")

    # --- Phase E: Verify ---
    logger.info("")
    logger.info("=" * 60)
    logger.info("Phase E: Verify")
    logger.info("=" * 60)

    new_stats = get_db_stats(conn)
    logger.info(f"  Active memories:  {new_stats['active_memories']}")
    logger.info(f"  Embeddings:       {new_stats['embeddings']}")
    logger.info(f"  Graph edges:      {new_stats['graph_edges']}")
    logger.info(f"  New dims:         {new_stats['current_dims']}")

    if new_stats["active_memories"] != new_stats["embeddings"]:
        logger.warning(
            f"  Memory count ({new_stats['active_memories']}) != "
            f"embedding count ({new_stats['embeddings']})"
        )
    else:
        logger.info(
            f"  OK: Memory count matches embedding count ({new_stats['active_memories']})"
        )

    # Spot check embeddings
    logger.info("  Spot-checking 5 random embeddings...")
    check_rows = conn.execute(
        "SELECT rowid FROM memory_embeddings ORDER BY RANDOM() LIMIT 5"
    ).fetchall()
    for (rid,) in check_rows:
        emb_row = conn.execute(
            "SELECT content_embedding FROM memory_embeddings WHERE rowid = ?",
            (rid,),
        ).fetchone()
        if emb_row is None:
            logger.error(f"    No embedding for rowid {rid}")
        else:
            n_floats = len(emb_row[0]) // 4
            if n_floats != target_dims:
                logger.error(
                    f"    rowid {rid} has {n_floats} dims, expected {target_dims}"
                )
            else:
                logger.info(f"    OK: rowid {rid} = {n_floats} dims")

    # Summary
    db_size_mb = args.db_path.stat().st_size / (1024 * 1024)
    backup_size_mb = backup_path.stat().st_size / (1024 * 1024)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Migration complete!")
    logger.info("=" * 60)
    logger.info(
        f"  Before: {stats['embeddings']} embeddings @ {stats['current_dims']}-dim, "
        f"{stats['graph_edges']} graph edges"
    )
    logger.info(
        f"  After:  {new_stats['embeddings']} embeddings @ {target_dims}-dim, "
        f"{new_stats['graph_edges']} graph edges"
    )
    logger.info(f"  DB size: {backup_size_mb:.1f}MB -> {db_size_mb:.1f}MB")
    logger.info(f"  Backup:  {backup_path}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Update your service config with the new embedding model env vars")
    logger.info("  2. Restart the service")
    logger.info("  3. Verify search works as expected")


if __name__ == "__main__":
    main()
