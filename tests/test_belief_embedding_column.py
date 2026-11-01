"""Tests for migration 020 (beliefs.embedding column) + partial-rollback recovery.

PR 1 of the belief-embedding arc (#1728): adds a nullable `beliefs.embedding` column
so later PRs can store a per-belief embedding (producer, PR 2) and dedup near-identical
beliefs (PR 3). This PR is pure DDL — no code writes or reads the column yet.

Covers the migration (additive, reversible) and the recovery path the runner must
provide so an upgrade after a documented Part-A-only rollback (column retained,
registry row removed) does not re-run `ADD COLUMN` and fail duplicate-column.
"""

import sqlite3
from pathlib import Path

from mcp_memory_service.storage.migration_runner import MigrationRunner


MIGRATIONS = Path(__file__).parent.parent / "src" / "mcp_memory_service" / "storage" / "migrations"


def _belief_table_only(conn):
    """Base beliefs table WITHOUT the embedding column (pre-020 shape)."""
    conn.executescript((MIGRATIONS / "012_add_belief_store.sql").read_text())


def _with_020(conn):
    _belief_table_only(conn)
    conn.executescript((MIGRATIONS / "020_add_belief_embedding.sql").read_text())


def _bookkeeping(conn, version):
    """Minimal metadata + migration_registry the runner/rollback own."""
    conn.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES ('schema_version', ?)", (str(version),))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS migration_registry "
        "(version INTEGER PRIMARY KEY, name TEXT, filename TEXT, applied_at TEXT, checksum TEXT)"
    )
    conn.execute("INSERT OR IGNORE INTO migration_registry (version) VALUES (?)", (version,))
    conn.commit()


# ── Migration 020: additive, idempotent, reversible ──

def test_migration_020_adds_column_and_pending_index():
    conn = sqlite3.connect(":memory:")
    _with_020(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(beliefs)")}
    assert "embedding" in cols
    idx = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='beliefs'"
    )}
    assert "idx_beliefs_embedding_pending" in idx


def test_migration_020_rollback_deregisters_and_drops():
    conn = sqlite3.connect(":memory:")
    _with_020(conn)
    _bookkeeping(conn, 20)
    down = (MIGRATIONS / "rollback" / "020_add_belief_embedding.down.sql").read_text()
    conn.executescript(down)
    # Part B dropped the column (SQLite here is >= 3.35).
    assert "embedding" not in {r[1] for r in conn.execute("PRAGMA table_info(beliefs)")}
    # Part A deregistered the migration and rolled the schema version back.
    assert conn.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()[0] == "19"
    assert conn.execute("SELECT COUNT(*) FROM migration_registry WHERE version=20").fetchone()[0] == 0


# ── Greptile P1: upgrade after a Part-A-only rollback must not fail duplicate-column ──

def test_recovery_after_partial_rollback_restamps_without_readd():
    """Part-A-only rollback leaves the `embedding` column but removes the v20 registry
    row. The runner's baseline recovery must detect the retained column, re-stamp v20
    (restoring the pending index) and bump schema_version — so run_pending does NOT
    re-run `ADD COLUMN embedding` (which would fail duplicate-column)."""
    conn = sqlite3.connect(":memory:")
    _with_020(conn)
    _bookkeeping(conn, 20)
    # Simulate Part-A-only rollback: drop index + deregister v20, KEEP the column.
    conn.execute("DROP INDEX IF EXISTS idx_beliefs_embedding_pending")
    conn.execute("DELETE FROM migration_registry WHERE version=20")
    conn.execute("UPDATE metadata SET value='19' WHERE key='schema_version'")
    conn.commit()

    runner = MigrationRunner(MIGRATIONS)
    recovered = runner._stamp_baseline(conn)

    assert 20 in recovered, "v20 should be recovered from the retained embedding column"
    # Registry row restored -> run_pending will skip the ADD COLUMN.
    assert conn.execute("SELECT COUNT(*) FROM migration_registry WHERE version=20").fetchone()[0] == 1
    # Secondary artifact (the pending index) was repaired.
    idx = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='beliefs'"
    )}
    assert "idx_beliefs_embedding_pending" in idx
    # schema_version advanced back to at least 20.
    assert int(conn.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()[0]) >= 20
