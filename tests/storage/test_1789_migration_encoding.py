"""Issue #1789: migrations are UTF-8 files and must not depend on the locale code page.

Python on Windows (before 3.15) decodes ``Path.read_text()`` without an encoding with
the locale code page. Under cp950 (Traditional Chinese) or cp949 (Korean), 008 and 010
could not be decoded, the runner stopped there, and a fresh database never got the
columns later migrations add, so retrieval failed. These tests simulate that locale.
"""

import hashlib
import os
import sqlite3
import tempfile
from pathlib import Path

import pytest

from mcp_memory_service.storage import migration_runner
from mcp_memory_service.storage.migration_runner import MigrationRunner

MIGRATIONS_DIR = Path(migration_runner.__file__).parent / "migrations"


@pytest.fixture
def cp950_locale(monkeypatch):
    """Make an encoding-less ``Path.read_text()`` decode as a zh-TW Windows machine does."""
    real_read_text = Path.read_text

    def read_text(self, encoding=None, *args, **kwargs):
        return real_read_text(self, encoding or "cp950", *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)


def test_shipped_migrations_do_not_decode_as_cp950():
    """Guard for the premise: without this the cp950 test below proves nothing."""
    failing = []
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        try:
            path.read_bytes().decode("cp950")
        except UnicodeDecodeError:
            failing.append(path.name)
    assert failing, "no shipped migration is undecodable as cp950 any more"


@pytest.mark.asyncio
async def test_fresh_database_gets_every_migration_under_cp950(cp950_locale, monkeypatch):
    from mcp_memory_service.models.memory import Memory
    from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
    from mcp_memory_service.utils.hashing import generate_content_hash

    monkeypatch.setenv("MCP_SEMANTIC_DEDUP_ENABLED", "false")
    shipped = sorted(int(p.name[:3]) for p in MIGRATIONS_DIR.glob("*.sql"))
    with tempfile.TemporaryDirectory() as tmp:
        storage = SqliteVecMemoryStorage(os.path.join(tmp, "cp950.db"))
        await storage.initialize()
        try:
            applied = sorted(
                row[0] for row in storage.conn.execute("SELECT version FROM migration_registry")
            )
            assert applied == shipped
            columns = {row[1] for row in storage.conn.execute("PRAGMA table_info(memories)")}
            assert {"embedding_pending", "superseded_by", "superseded_at"} <= columns

            content = "a memory stored on a cp950 machine"
            ok, msg = await storage.store(
                Memory(content=content, content_hash=generate_content_hash(content))
            )
            assert ok, msg
            results = await storage.retrieve("memory stored", n_results=3)
            assert [r.memory.content for r in results] == [content]
        finally:
            await storage.close()


@pytest.fixture
def applied_utf8_migration(tmp_path):
    """One applied migration whose comment holds UTF-8 punctuation, like 008 and 010."""
    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()
    path = migrations_dir / "001_first.sql"
    path.write_bytes("-- graph — edges\nCREATE TABLE first_table (id INTEGER);\n".encode())
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    runner = MigrationRunner(migrations_dir)
    assert runner.run_pending(conn)["error"] is None
    return runner, conn, path


def _store_checksum(conn, checksum):
    conn.execute("UPDATE migration_registry SET checksum = ? WHERE version = 1", (checksum,))


def test_check_accepts_checksum_recorded_under_cp1252(applied_utf8_migration, monkeypatch):
    """A Western Windows machine applied the file decoded as cp1252 and stored that hash."""
    runner, conn, path = applied_utf8_migration
    _store_checksum(conn, hashlib.sha256(path.read_bytes().decode("cp1252").encode()).hexdigest())
    monkeypatch.setattr("locale.getpreferredencoding", lambda do_setlocale=True: "cp1252")

    report = runner.check(conn)
    assert report["checksum_mismatches"] == []
    assert report["healthy"] is True


def test_check_still_reports_a_modified_migration(applied_utf8_migration, monkeypatch):
    runner, conn, _ = applied_utf8_migration
    _store_checksum(conn, hashlib.sha256(b"something else").hexdigest())
    monkeypatch.setattr("locale.getpreferredencoding", lambda do_setlocale=True: "cp1252")

    report = runner.check(conn)
    assert [m["version"] for m in report["checksum_mismatches"]] == [1]
    assert report["healthy"] is False
