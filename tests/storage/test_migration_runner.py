"""Unit tests for MigrationRunner.

Tests the versioned SQL migration runner with registry tracking.
"""

import sqlite3
from pathlib import Path

import pytest

from mcp_memory_service.storage.migration_runner import MigrationRunner


@pytest.fixture
def setup_migrations(tmp_path):
    """Create a migrations directory with test files and a DB connection."""
    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()

    (migrations_dir / "001_first.sql").write_text(
        "CREATE TABLE IF NOT EXISTS first_table (id INTEGER PRIMARY KEY);"
    )
    (migrations_dir / "002_second.sql").write_text(
        "CREATE TABLE IF NOT EXISTS second_table (id INTEGER);"
    )

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    return migrations_dir, conn


@pytest.mark.unit
def test_migration_runner_sync_executes_sql_files(setup_migrations):
    """Test that migration runner executes SQL files."""
    migrations_dir, conn = setup_migrations

    runner = MigrationRunner(migrations_dir)
    result = runner.run_pending(conn)

    assert result["error"] is None
    assert len(result["applied"]) == 2

    # Verify table was created
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='first_table'"
    )
    assert cursor.fetchone() is not None


@pytest.mark.unit
def test_migration_runner_sync_multiple_files(setup_migrations):
    """Test that migration runner executes multiple files in order."""
    migrations_dir, conn = setup_migrations

    runner = MigrationRunner(migrations_dir)
    result = runner.run_pending(conn)

    assert result["error"] is None
    assert len(result["applied"]) == 2

    # Verify both tables were created
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    table_names = [row[0] for row in cursor.fetchall()]
    assert "first_table" in table_names
    assert "second_table" in table_names


@pytest.mark.unit
def test_migration_runner_sync_invalid_sql(tmp_path):
    """Test that migration runner handles invalid SQL gracefully."""
    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()
    (migrations_dir / "001_invalid.sql").write_text("INVALID SQL STATEMENT;")

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

    runner = MigrationRunner(migrations_dir)
    result = runner.run_pending(conn)

    assert result["error"] is not None


@pytest.mark.unit
def test_migration_runner_sync_idempotent(setup_migrations):
    """Test that migrations are idempotent (can run multiple times)."""
    migrations_dir, conn = setup_migrations

    runner = MigrationRunner(migrations_dir)

    # Run first time
    result1 = runner.run_pending(conn)
    assert result1["error"] is None
    assert len(result1["applied"]) == 2

    # Run second time (should skip all)
    result2 = runner.run_pending(conn)
    assert result2["error"] is None
    assert len(result2["applied"]) == 0
    assert len(result2["skipped"]) == 2


@pytest.mark.unit
def test_stamp_baseline_recovers_v16_after_partial_rollback(tmp_path):
    """Greptile P3: a partial rollback that keeps v15 registered but removes the v16
    registry row (while retaining the hlc columns) must NOT make the next upgrade re-run
    the unconditional ADD COLUMN and fail with duplicate-column.

    _stamp_baseline must probe versions ABSENT from the registry (not early-return just
    because the registry is non-empty), so it detects the retained hlc_physical column and
    stamps v16 as applied, letting run_pending skip the forward ADD COLUMN.
    """
    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()
    # Minimal stand-ins mirroring the real 015/016 shapes and their unconditional DDL.
    (migrations_dir / "015_add_sync_events.sql").write_text(
        "CREATE TABLE sync_events (seq INTEGER PRIMARY KEY, content_hash TEXT, agent_id TEXT, event_id TEXT);"
    )
    (migrations_dir / "016_add_hlc_to_sync_events.sql").write_text(
        "ALTER TABLE sync_events ADD COLUMN hlc_physical INTEGER;\n"
        "ALTER TABLE sync_events ADD COLUMN hlc_logical INTEGER;\n"
        "CREATE INDEX IF NOT EXISTS idx_sync_events_hlc "
        "ON sync_events(hlc_physical, hlc_logical, agent_id, event_id);"
    )

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

    runner = MigrationRunner(migrations_dir)
    first = runner.run_pending(conn)
    assert first["error"] is None
    assert {m["version"] for m in first["applied"]} == {15, 16}

    # Simulate a partial manual rollback of 016 on old SQLite: drop the index + the registry
    # row + schema_version, but keep the hlc columns (cannot DROP COLUMN before 3.35).
    conn.execute("DROP INDEX IF EXISTS idx_sync_events_hlc")
    conn.execute("DELETE FROM migration_registry WHERE version = 16")
    conn.execute("UPDATE metadata SET value = '15' WHERE key = 'schema_version'")
    conn.commit()
    # Sanity: v15 still registered, v16 gone, column retained, index gone.
    regd = {r[0] for r in conn.execute("SELECT version FROM migration_registry").fetchall()}
    assert 15 in regd and 16 not in regd
    cols = [r[1] for r in conn.execute("PRAGMA table_info(sync_events)").fetchall()]
    assert "hlc_physical" in cols
    idx = conn.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_sync_events_hlc'").fetchone()
    assert idx is None, "precondition: the index was dropped by the rollback"

    # Re-upgrade must succeed: the baseline probe detects the retained column, REPAIRS the
    # dropped index, and stamps v16 — the forward ADD COLUMN never re-runs (duplicate-column).
    second = runner.run_pending(conn)
    assert second["error"] is None, f"re-upgrade failed: {second['error']}"
    regd2 = {r[0] for r in conn.execute("SELECT version FROM migration_registry").fetchall()}
    assert 16 in regd2, "v16 must be recovered (stamped) after the partial rollback"
    idx2 = conn.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_sync_events_hlc'").fetchone()
    assert idx2 is not None, "v16 recovery must repair the dropped HLC index"
