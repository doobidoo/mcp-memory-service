"""Regression test for #1628.

A sqlite-vec database written before v8.7.0 has no `metadata` table. The runner
keeps `schema_version` there and writes it after every migration, so on such a
database migration 008 applied its DDL, the write of `schema_version` failed with
`no such table: metadata`, the rollback took 008 down with it, and every later
migration — including 017's `embedding_pending` column on `memories` — was
skipped. `retrieve()` and `search_by_tag()` then returned nothing on intact data.

`SqliteVecMemoryStorage.initialize()` was fixed for this in #1622. `memory migrate`
never goes through `initialize()`: it opens the file with a plain
`sqlite3.connect()` and calls `MigrationRunner.run_pending()` directly, so it kept
failing. These tests drive that command, not the runner in isolation.
"""

import sqlite3
from pathlib import Path

from click.testing import CliRunner

from mcp_memory_service.cli.main import cli

# Reuse the pre-v8.7 database shape from the #1622 regression test.
from test_1622_legacy_metadata_table import _build_pre_v8_7_database

MIGRATIONS_DIR = (
    Path(__file__).parents[2] / "src" / "mcp_memory_service" / "storage" / "migrations"
)
EXPECTED_VERSIONS = {
    int(path.name[:3]) for path in MIGRATIONS_DIR.glob("[0-9][0-9][0-9]_*.sql")
}


def _tables(db_path: Path) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        return {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
    finally:
        conn.close()


def test_cli_migrate_upgrades_a_pre_v8_7_database(tmp_path):
    db_path = tmp_path / "legacy.db"
    _build_pre_v8_7_database(str(db_path))

    # Precondition: the fixture really is the shape the bug needs.
    assert "metadata" not in _tables(db_path)

    result = CliRunner().invoke(cli, ["migrate", "--db-path", str(db_path)])
    assert result.exit_code == 0, result.output
    assert "no such table: metadata" not in result.output

    conn = sqlite3.connect(db_path)
    try:
        version = conn.execute(
            "SELECT value FROM metadata WHERE key='schema_version'"
        ).fetchone()
        assert int(version[0]) == max(EXPECTED_VERSIONS)

        applied = {row[0] for row in conn.execute(
            "SELECT version FROM migration_registry"
        )}
        assert applied == EXPECTED_VERSIONS

        # Migration 017 adds this to `memories`, not to the vec0 table.
        columns = {row[1] for row in conn.execute("PRAGMA table_info(memories)")}
        assert "embedding_pending" in columns
    finally:
        conn.close()


def test_cli_migrate_dry_run_creates_no_tables(tmp_path):
    db_path = tmp_path / "legacy.db"
    _build_pre_v8_7_database(str(db_path))

    result = CliRunner().invoke(cli, ["migrate", "--db-path", str(db_path), "--dry-run"])
    assert result.exit_code == 0, result.output

    tables = _tables(db_path)
    assert "metadata" not in tables
    assert "migration_registry" not in tables
