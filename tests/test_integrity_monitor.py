"""Tests for database integrity health monitoring."""

import asyncio
import json
import os
import sqlite3
import tempfile

import pytest


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database with a memories table."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """CREATE TABLE memories (
            content_hash TEXT PRIMARY KEY,
            content TEXT,
            created_at REAL,
            updated_at REAL,
            metadata TEXT,
            tags TEXT,
            type TEXT
        )"""
    )
    conn.execute(
        "INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("hash1", "test memory", 1.0, 1.0, "{}", "test", "note"),
    )
    conn.commit()
    conn.close()

    yield db_path

    os.unlink(db_path)
    for suffix in ("-wal", "-shm"):
        path = db_path + suffix
        if os.path.exists(path):
            os.unlink(path)


@pytest.fixture
def monitor(temp_db, monkeypatch):
    """Create an IntegrityMonitor with test config."""
    monkeypatch.setenv("MCP_MEMORY_INTEGRITY_CHECK_ENABLED", "true")
    monkeypatch.setenv("MCP_MEMORY_INTEGRITY_CHECK_INTERVAL", "60")

    # Reload config to pick up test values
    import importlib
    from mcp_memory_service import config
    importlib.reload(config)

    from mcp_memory_service.health.integrity import IntegrityMonitor
    return IntegrityMonitor(temp_db)


class TestIntegrityCheck:
    """Tests for PRAGMA integrity_check."""

    def test_healthy_database(self, monitor):
        is_healthy, detail = monitor.check_integrity()
        assert is_healthy is True
        assert detail == "ok"

    def test_missing_database(self, tmp_path):
        from mcp_memory_service.health.integrity import IntegrityMonitor
        m = IntegrityMonitor(str(tmp_path / "nonexistent.db"))
        # SQLite creates the file on connect, so it will be "healthy" but empty
        is_healthy, detail = m.check_integrity()
        assert is_healthy is True

    def test_check_increments_counters(self, monitor):
        assert monitor.total_checks == 0
        asyncio.get_event_loop().run_until_complete(monitor.run_check())
        assert monitor.total_checks == 1
        assert monitor.last_check_healthy is True


class TestWalRepair:
    """Tests for WAL checkpoint repair."""

    def test_repair_on_healthy_db(self, monitor):
        success, detail = monitor.attempt_wal_repair()
        assert success is True
        assert "successful" in detail


class TestMemoryExport:
    """Tests for emergency memory export."""

    def test_export_memories(self, monitor, tmp_path):
        export_path = str(tmp_path / "export.json")
        success, count = monitor.export_memories(export_path)
        assert success is True
        assert count == 1

        with open(export_path) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["hash"] == "hash1"
        assert data[0]["content"] == "test memory"

    def test_export_empty_database(self, tmp_path):
        from mcp_memory_service.health.integrity import IntegrityMonitor

        db_path = str(tmp_path / "empty.db")
        conn = sqlite3.connect(db_path)
        conn.execute(
            """CREATE TABLE memories (
                content_hash TEXT PRIMARY KEY, content TEXT,
                created_at REAL, updated_at REAL,
                metadata TEXT, tags TEXT, type TEXT
            )"""
        )
        conn.close()

        m = IntegrityMonitor(db_path)
        export_path = str(tmp_path / "export.json")
        success, count = m.export_memories(export_path)
        assert success is True
        assert count == 0


class TestRunCheck:
    """Tests for the full check + repair flow."""

    def test_healthy_check_result(self, monitor):
        result = asyncio.get_event_loop().run_until_complete(monitor.run_check())
        assert result["healthy"] is True
        assert result["repaired"] is False
        assert result["exported"] is False
        assert result["check_ms"] < 1000  # Should be fast


class TestGetStatus:
    """Tests for status reporting."""

    def test_initial_status(self, monitor):
        status = monitor.get_status()
        assert status["running"] is False
        assert status["total_checks"] == 0
        assert status["total_auto_repairs"] == 0
        assert status["total_unrecoverable"] == 0

    def test_status_after_check(self, monitor):
        asyncio.get_event_loop().run_until_complete(monitor.run_check())
        status = monitor.get_status()
        assert status["total_checks"] == 1
        assert status["last_check_healthy"] is True
