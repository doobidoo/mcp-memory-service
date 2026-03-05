#!/usr/bin/env python3
"""Migrate mcp-memory-service embeddings from one model to another.

Handles dimension changes (e.g., 384-dim all-MiniLM-L6-v2 to 768-dim
nomic-embed-text) by dropping and recreating the vec0 virtual table at the
new dimension, re-embedding all active memories, and optionally wiping graph
edges that were computed from old similarity scores.

Prerequisites:
  - The target embedding API must be running and reachable.
  - mcp-memory-service should be STOPPED to avoid concurrent writes.

Usage:
  # Preview current state (no changes):
  python scripts/maintenance/migrate_embeddings.py \\
      --url http://localhost:11434/v1/embeddings \\
      --model nomic-embed-text \\
      --dry-run

  # Run the migration:
  python scripts/maintenance/migrate_embeddings.py \\
      --url http://localhost:11434/v1/embeddings \\
      --model nomic-embed-text

  # Keep graph edges (if you trust existing relationships):
  python scripts/maintenance/migrate_embeddings.py \\
      --url http://localhost:11434/v1/embeddings \\
      --model nomic-embed-text \\
      --keep-graph

What it does:
  1. Validates the target embedding API is reachable
  2. Reports current database state
  3. Backs up the database file
  4. Reads all active memory content
  5. Generates new embeddings via the target API (batched)
  6. Drops and recreates the vec0 virtual table at the new dimension
  7. Inserts new embeddings with correct rowid mapping
  8. Optionally wipes graph edges (recommended when changing models)
  9. Updates metadata with new model info
  10. Rebuilds FTS index, VACUUMs the database
  11. Verifies integrity (count match, dimension spot-check, KNN test)
"""

import argparse
import os
import platform
import re
import shutil
import sqlite3
import struct
import subprocess
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: 'requests' package is required. Install with: pip install requests")
    sys.exit(1)


def default_db_path():
    """Return the platform-appropriate default database path."""
    system = platform.system()
    if system == "Darwin":
        base = Path.home() / "Library" / "Application Support" / "mcp-memory"
    elif system == "Windows":
        base = Path(os.environ.get("APPDATA", Path.home())) / "mcp-memory"
    else:
        base = (
            Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
            / "mcp-memory"
        )
    return base / "sqlite_vec.db"


def serialize_float32(vector: list) -> bytes:
    """Serialize a list of floats to bytes for sqlite-vec insertion."""
    return struct.pack(f"{len(vector)}f", *vector)


def check_embedding_api(url: str, model: str, api_key: str = None) -> int:
    """Verify the embedding API is reachable and return embedding dimension."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        resp = requests.post(
            url,
            json={"input": "dimension probe", "model": model},
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        dims = len(data["data"][0]["embedding"])
        return dims
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Cannot connect to embedding API at {url}")
        print("  Ensure your embedding service is running.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Embedding API check failed: {e}")
        sys.exit(1)


def batch_embed(
    texts: list,
    url: str,
    model: str,
    expected_dims: int,
    api_key: str = None,
    batch_size: int = 32,
) -> list:
    """Generate embeddings via an OpenAI-compatible API, batched."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    all_embeddings = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        batch_num = i // batch_size + 1
        try:
            resp = requests.post(
                url,
                json={"input": batch, "model": model},
                headers=headers,
                timeout=120,
            )
            resp.raise_for_status()
        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"Embedding API timed out on batch {batch_num} "
                f"(memories {i + 1}-{min(i + batch_size, total)}). "
                "No database changes have been made yet. "
                "Consider reducing --batch-size or checking API availability."
            )
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(
                f"Embedding API error on batch {batch_num} "
                f"(memories {i + 1}-{min(i + batch_size, total)}): {exc}\n"
                "No database changes have been made yet."
            )
        data = resp.json()
        batch_embs = sorted(data["data"], key=lambda x: x["index"])
        for item in batch_embs:
            emb = item["embedding"]
            if len(emb) != expected_dims:
                raise ValueError(
                    f"Expected {expected_dims} dims, got {len(emb)} "
                    f"(batch starting at index {i})"
                )
            all_embeddings.append(emb)
        done = min(i + batch_size, total)
        print(f"  Embedded {done}/{total} memories ({100 * done // total}%)")
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

    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name='memory_embeddings' AND type='table'"
    ).fetchone()
    vec_ddl = row[0] if row else "NOT FOUND"

    # Parse current dimension from DDL like "FLOAT[384]"
    current_dims = None
    if vec_ddl and vec_ddl != "NOT FOUND":
        m = re.search(r"FLOAT\[(\d+)\]", vec_ddl)
        if m:
            current_dims = int(m.group(1))

    try:
        emb_count = conn.execute("SELECT COUNT(*) FROM memory_embeddings").fetchone()[0]
    except sqlite3.OperationalError:
        emb_count = 0

    return {
        "active_memories": active,
        "deleted_memories": deleted,
        "embeddings": emb_count,
        "graph_edges": edges,
        "vec_ddl": vec_ddl,
        "current_dims": current_dims,
    }


def load_sqlite_vec(conn: sqlite3.Connection):
    """Load the sqlite-vec extension."""
    conn.enable_load_extension(True)

    try:
        import sqlite_vec

        sqlite_vec.load(conn)
        return
    except ImportError:
        pass

    # Try common venv locations for mcp-memory-service
    candidates = [
        # uv tools
        Path.home() / ".local" / "share" / "uv" / "tools" / "mcp-memory-service",
        # pipx
        Path.home() / ".local" / "pipx" / "venvs" / "mcp-memory-service",
    ]
    for base in candidates:
        if not base.exists():
            continue
        for site_pkg in base.rglob("site-packages"):
            sys.path.insert(0, str(site_pkg))
            try:
                import sqlite_vec

                sqlite_vec.load(conn)
                return
            except ImportError:
                sys.path.pop(0)

    print(
        "ERROR: Cannot load sqlite-vec extension.\n"
        "  Install it with: pip install sqlite-vec\n"
        "  Or run this script from the mcp-memory-service virtualenv."
    )
    sys.exit(1)


def check_service_not_running():
    """Warn if mcp-memory-service appears to be running."""
    system = platform.system()
    try:
        if system == "Darwin":
            result = subprocess.run(
                ["launchctl", "print", f"gui/{os.getuid()}/com.mcp-memory-service"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print("  WARNING: mcp-memory-service appears to be running (launchd).")
                print(
                    "  Recommendation: stop it before migrating to avoid data corruption."
                )
                print("  macOS: launchctl bootout gui/$(id -u)/com.mcp-memory-service")
                return False
        elif system == "Linux":
            result = subprocess.run(
                ["systemctl", "--user", "is-active", "mcp-memory.service"],
                capture_output=True,
                text=True,
            )
            if result.stdout.strip() == "active":
                print("  WARNING: mcp-memory-service appears to be running (systemd).")
                print(
                    "  Recommendation: stop it before migrating to avoid data corruption."
                )
                print("  Linux: systemctl --user stop mcp-memory.service")
                return False
    except FileNotFoundError:
        pass  # launchctl/systemctl not available — skip check
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Migrate memory embeddings to a different model/dimension.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Migrate to nomic-embed-text via Ollama:
  %(prog)s --url http://localhost:11434/v1/embeddings --model nomic-embed-text

  # Migrate to OpenAI text-embedding-3-small:
  %(prog)s --url https://api.openai.com/v1/embeddings \\
           --model text-embedding-3-small --api-key sk-...

  # Preview without changes:
  %(prog)s --url http://localhost:11434/v1/embeddings --model nomic-embed-text --dry-run
""",
    )
    parser.add_argument(
        "--url",
        required=True,
        help="OpenAI-compatible embeddings API URL",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Embedding model name to pass to the API",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for authenticated endpoints (or set MCP_EXTERNAL_EMBEDDING_API_KEY)",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Path to sqlite_vec.db (auto-detected if not set)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of texts per embedding API call (default: 32)",
    )
    parser.add_argument(
        "--keep-graph",
        action="store_true",
        help="Preserve graph edges (by default they are wiped since similarity scores change)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report current state without making any changes",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("MCP_EXTERNAL_EMBEDDING_API_KEY")
    db_path = args.db_path or default_db_path()

    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}")
        print("  Use --db-path to specify the correct location.")
        sys.exit(1)

    # --- Phase 1: Validate ---
    print("=" * 60)
    print("Phase 1: Validate")
    print("=" * 60)

    target_dims = check_embedding_api(args.url, args.model, api_key)
    print(f"  Target model: {args.model} ({target_dims}-dim)")

    conn = sqlite3.connect(str(db_path))
    load_sqlite_vec(conn)

    stats = get_db_stats(conn)
    print(f"  Database: {db_path}")
    print(f"  Active memories: {stats['active_memories']}")
    print(f"  Deleted (tombstones): {stats['deleted_memories']}")
    print(f"  Current embeddings: {stats['embeddings']}")
    print(f"  Current dimension: {stats['current_dims'] or 'unknown'}")
    print(f"  Graph edges: {stats['graph_edges']}")

    if stats["current_dims"] == target_dims:
        print(f"\n  Note: Current and target dimensions are both {target_dims}.")
        print("  The migration will still re-embed with the new model.")

    if args.dry_run:
        print("\n  --dry-run: No changes will be made.")
        conn.close()
        return

    if stats["active_memories"] == 0:
        print("\n  No active memories to migrate.")
        conn.close()
        return

    # Check service status
    print()
    service_stopped = check_service_not_running()
    if not service_stopped:
        response = input("\n  Continue anyway? (y/N): ").strip().lower()
        if response != "y":
            print("  Migration cancelled.")
            conn.close()
            sys.exit(0)
    else:
        print("  Service check OK.")

    # Phases 2-5 perform destructive operations (DROP TABLE, etc.).
    # Wrap in try/except so Ctrl+C gives a clear recovery message.
    backup_path = None
    try:
        _run_migration(args, conn, db_path, stats, target_dims, api_key)
    except KeyboardInterrupt:
        print("\n\n  Migration interrupted!")
        print("  Your database may be in an inconsistent state.")
        print(f"  Check for .pre-migration backups in {db_path.parent}")
        sys.exit(1)
    finally:
        conn.close()


def _run_migration(args, conn, db_path, stats, target_dims, api_key):
    """Execute the destructive migration phases (2-5)."""
    # --- Phase 2: Prepare ---
    print("\n" + "=" * 60)
    print("Phase 2: Backup and read memories")
    print("=" * 60)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    backup_path = db_path.with_suffix(f".db.pre-migration.{timestamp}")
    print(f"  Backing up to {backup_path}")
    shutil.copy2(str(db_path), str(backup_path))

    print("  Reading active memories...")
    rows = conn.execute(
        "SELECT rowid, content FROM memories WHERE deleted_at IS NULL ORDER BY rowid"
    ).fetchall()
    rowids = [r[0] for r in rows]
    contents = [r[1] for r in rows]
    print(f"  Read {len(rows)} memories")

    # --- Phase 3: Generate new embeddings ---
    print("\n" + "=" * 60)
    print("Phase 3: Generate embeddings")
    print("=" * 60)

    t0 = time.time()
    embeddings = batch_embed(
        contents,
        args.url,
        args.model,
        target_dims,
        api_key=api_key,
        batch_size=args.batch_size,
    )
    elapsed = time.time() - t0
    print(f"  Generated {len(embeddings)} embeddings in {elapsed:.1f}s")

    if len(embeddings) != len(rowids):
        print(
            f"  ERROR: Embedding count ({len(embeddings)}) != "
            f"memory count ({len(rowids)})"
        )
        sys.exit(1)

    # --- Phase 4: Migrate database ---
    print("\n" + "=" * 60)
    print("Phase 4: Migrate database")
    print("=" * 60)

    # Drop old vec0 table (virtual table DDL must be outside transactions)
    print("  Dropping old memory_embeddings table...")
    conn.execute("DROP TABLE IF EXISTS memory_embeddings")
    conn.commit()

    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE name='memory_embeddings'"
    ).fetchone()
    if row:
        print("  ERROR: Failed to drop memory_embeddings table")
        sys.exit(1)

    print(f"  Creating new vec0 table at FLOAT[{target_dims}]...")
    conn.execute(
        f"CREATE VIRTUAL TABLE memory_embeddings USING vec0("
        f"content_embedding FLOAT[{target_dims}] distance_metric=cosine"
        f")"
    )
    conn.commit()

    print(f"  Inserting {len(embeddings)} embeddings...")
    t0 = time.time()
    conn.execute("BEGIN")
    data_to_insert = (
        (rowid, serialize_float32(emb))
        for rowid, emb in zip(rowids, embeddings)
    )
    conn.executemany(
        "INSERT INTO memory_embeddings (rowid, content_embedding) VALUES (?, ?)",
        data_to_insert,
    )
    conn.commit()
    elapsed = time.time() - t0
    print(f"  Inserted {len(embeddings)} embeddings in {elapsed:.1f}s")

    if not args.keep_graph:
        try:
            print(f"  Wiping {stats['graph_edges']} graph edges...")
            conn.execute("DELETE FROM memory_graph")
            conn.commit()
        except sqlite3.OperationalError:
            print("  No memory_graph table found (skipping graph wipe)")
    else:
        print(f"  Keeping {stats['graph_edges']} graph edges (--keep-graph)")

    print("  Updating metadata...")
    for key, value in [
        ("distance_metric", "cosine"),
        ("embedding_model", args.model),
        ("embedding_dims", str(target_dims)),
        ("migration_date", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
        ("migration_from_dims", str(stats["current_dims"] or "unknown")),
    ]:
        conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            (key, value),
        )
    conn.commit()

    print("  Rebuilding FTS index...")
    try:
        conn.execute(
            "INSERT INTO memory_content_fts(memory_content_fts) VALUES('rebuild')"
        )
        conn.commit()
    except sqlite3.OperationalError as e:
        print(f"  FTS rebuild skipped (non-fatal): {e}")

    print("  Running VACUUM...")
    conn.execute("VACUUM")

    # --- Phase 5: Verify ---
    print("\n" + "=" * 60)
    print("Phase 5: Verify")
    print("=" * 60)

    new_stats = get_db_stats(conn)
    print(f"  Active memories: {new_stats['active_memories']}")
    print(f"  Embeddings: {new_stats['embeddings']}")
    print(f"  Graph edges: {new_stats['graph_edges']}")
    print(f"  Vec table dimension: {new_stats['current_dims']}")

    if new_stats["active_memories"] != new_stats["embeddings"]:
        print(
            f"  WARNING: Memory count ({new_stats['active_memories']}) != "
            f"embedding count ({new_stats['embeddings']})"
        )
    else:
        print(f"  OK: Counts match ({new_stats['active_memories']})")

    # Spot-check dimensions
    print("  Spot-checking 5 random embeddings...")
    check_rows = conn.execute(
        "SELECT rowid FROM memory_embeddings ORDER BY RANDOM() LIMIT 5"
    ).fetchall()
    all_ok = True
    for (rid,) in check_rows:
        emb_row = conn.execute(
            "SELECT content_embedding FROM memory_embeddings WHERE rowid = ?",
            (rid,),
        ).fetchone()
        if emb_row is None:
            print(f"    ERROR: No embedding for rowid {rid}")
            all_ok = False
        else:
            n_floats = len(emb_row[0]) // 4
            if n_floats != target_dims:
                print(
                    f"    ERROR: rowid {rid} has {n_floats} dims, "
                    f"expected {target_dims}"
                )
                all_ok = False
            else:
                print(f"    OK: rowid {rid} = {n_floats} dims")
    if all_ok:
        print("  All spot checks passed.")

    # Test KNN search
    print("  Testing KNN search...")
    try:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        test_resp = requests.post(
            args.url,
            json={"input": "test search query", "model": args.model},
            headers=headers,
            timeout=15,
        )
        test_resp.raise_for_status()
        test_emb = test_resp.json()["data"][0]["embedding"]
        test_blob = serialize_float32(test_emb)
        knn_results = conn.execute(
            "SELECT me.rowid, m.content_hash, distance "
            "FROM memory_embeddings me "
            "JOIN memories m ON m.rowid = me.rowid "
            "WHERE content_embedding MATCH ? "
            "  AND k = 3 "
            "  AND m.deleted_at IS NULL "
            "ORDER BY distance",
            (test_blob,),
        ).fetchall()
        print(f"  KNN search returned {len(knn_results)} results:")
        for rid, hash_val, dist in knn_results:
            sim = 1.0 - dist
            print(f"    rowid={rid} hash={hash_val[:16]}... similarity={sim:.4f}")
    except Exception as e:
        print(f"  KNN search test skipped (non-fatal): {e}")
        print("  Migration data is committed. Verify manually after restart.")

    # Summary
    db_size_mb = db_path.stat().st_size / (1024 * 1024)
    backup_size_mb = backup_path.stat().st_size / (1024 * 1024)
    print("\n" + "=" * 60)
    print("Migration complete!")
    print("=" * 60)
    print(
        f"  Before: {stats['embeddings']} embeddings "
        f"@ {stats['current_dims'] or '?'}-dim, "
        f"{stats['graph_edges']} graph edges"
    )
    print(
        f"  After:  {new_stats['embeddings']} embeddings "
        f"@ {target_dims}-dim, "
        f"{new_stats['graph_edges']} graph edges"
    )
    print(f"  DB size: {backup_size_mb:.1f}MB -> {db_size_mb:.1f}MB")
    print(f"  Backup:  {backup_path}")
    print()
    print("Next steps:")
    print("  1. Update your service configuration with the new model settings:")
    print(f"     MCP_EXTERNAL_EMBEDDING_URL={args.url}")
    print(f"     MCP_EXTERNAL_EMBEDDING_MODEL={args.model}")
    print("  2. Restart the memory service")
    print("  3. Run a few test searches to verify quality")


if __name__ == "__main__":
    main()
