"""
Log-injection tests for health/integrity.py (#1146).

The integrity monitor logs what the corruption check printed, the repair detail
and the paths it exports to. The first two come out of the database engine and
the paths from the configured database location, so a newline in one must not
reach the log as a line break.
"""

import logging

import pytest

from mcp_memory_service.health import integrity
from mcp_memory_service.health.integrity import IntegrityMonitor

FORGED = "FORGED admin authenticated"


def _assert_clean(caplog, expected):
    messages = [record.getMessage() for record in caplog.records]
    assert any(expected in m for m in messages)
    assert not any(f"\n{FORGED}" in m for m in messages)


def _monitor(tmp_path):
    return IntegrityMonitor(str(tmp_path / "memories.db"))


@pytest.mark.asyncio
async def test_corruption_and_repair_logs_do_not_carry_newlines(caplog, tmp_path, monkeypatch):
    monitor = _monitor(tmp_path)

    async def corrupt():
        return False, f"page 3 is broken\n{FORGED}"

    async def repaired():
        return True, f"wal checkpoint ok\n{FORGED}"

    monkeypatch.setattr(monitor, "check_integrity", corrupt)
    monkeypatch.setattr(monitor, "attempt_wal_repair", repaired)

    with caplog.at_level(logging.DEBUG):
        result = await monitor.run_check()

    assert result["repaired"]
    _assert_clean(caplog, "Database corruption detected: page 3 is broken")
    _assert_clean(caplog, "Auto-repair successful: wal checkpoint ok")


@pytest.mark.asyncio
async def test_export_path_and_failure_logs_do_not_carry_newlines(caplog, tmp_path, monkeypatch):
    monitor = _monitor(tmp_path)

    async def corrupt():
        return False, "broken"

    async def not_repaired():
        return False, "no luck"

    async def exported(path):
        return True, 3

    monkeypatch.setattr(monitor, "check_integrity", corrupt)
    monkeypatch.setattr(monitor, "attempt_wal_repair", not_repaired)
    monkeypatch.setattr(monitor, "export_memories", exported)
    monkeypatch.setattr(monitor, "db_path", str(tmp_path / f"db\n{FORGED}" / "memories.db"))

    with caplog.at_level(logging.DEBUG):
        await monitor.run_check()

    _assert_clean(caplog, "Memories exported to")


@pytest.mark.asyncio
async def test_export_failure_log_does_not_carry_newlines(caplog, tmp_path, monkeypatch):
    monitor = _monitor(tmp_path)

    def broken(*_args, **_kwargs):
        raise RuntimeError(f"disk gone\n{FORGED}")

    monkeypatch.setattr(integrity.sqlite3, "connect", broken)

    with caplog.at_level(logging.DEBUG):
        ok, _count = await monitor.export_memories(str(tmp_path / "out.json"))

    assert not ok
    _assert_clean(caplog, "Memory export failed: disk gone")


@pytest.mark.asyncio
async def test_successful_export_log_does_not_carry_newlines(caplog, tmp_path):
    import sqlite3

    db_path = tmp_path / "memories.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE memories (content_hash TEXT, content TEXT, created_at REAL, "
        "metadata TEXT, tags TEXT, type TEXT)"
    )
    conn.execute("INSERT INTO memories VALUES ('h', 'hello', 1.0, '{}', '', 'note')")
    conn.commit()
    conn.close()

    monitor = IntegrityMonitor(str(db_path))

    with caplog.at_level(logging.DEBUG):
        ok, count = await monitor.export_memories(str(tmp_path / f"out\n{FORGED}.json"))

    assert ok and count == 1
    _assert_clean(caplog, "Exported 1 memories to")
