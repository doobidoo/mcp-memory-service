"""The backup scheduler's log lines hold one record each, whatever a filename carries.

``test_log_injection_guard.py`` checks the source of backup/scheduler.py. This runs
the service and reads what the logger emitted: a newline inside a backup filename,
or inside an exception, must come out as a literal ``\\n`` in one record, not as a
second line that reads like a forged entry. Part of #1146.
"""

import logging
import shutil
import sqlite3

import pytest

import mcp_memory_service.backup.scheduler as scheduler_module
from mcp_memory_service.backup.scheduler import BackupService

LOGGER = "mcp_memory_service.backup.scheduler"
FORGED = "memory_backup_evil\nINFO forged line.db"


@pytest.fixture
def service(tmp_path, monkeypatch):
    backups_dir = tmp_path / "backups"
    backups_dir.mkdir()
    db_path = tmp_path / "memory.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE t (x)")
    (backups_dir / FORGED).write_bytes(b"")
    monkeypatch.setattr(scheduler_module, "BACKUP_MAX_COUNT", 0)
    monkeypatch.setattr(scheduler_module, "BACKUP_RETENTION", 3650)
    return BackupService(backups_dir=str(backups_dir), db_path=str(db_path))


def _messages(caplog):
    return [record.getMessage() for record in caplog.records if record.name == LOGGER]


@pytest.mark.asyncio
async def test_newline_in_backup_filename_stays_inside_one_record(service, caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert (await service.restore_backup(FORGED))["success"]
        result = await service.cleanup_old_backups()

    assert [entry["filename"] for entry in result["removed"]] == [FORGED]
    messages = _messages(caplog)
    restored = [m for m in messages if m.startswith("Restored database from backup: ")]
    removed = [m for m in messages if m.startswith("Removed old backup: ")]
    assert restored == ["Restored database from backup: memory_backup_evil\\nINFO forged line.db"]
    assert removed == ["Removed old backup: memory_backup_evil\\nINFO forged line.db (exceeds max count (0))"]
    assert not any("\n" in m for m in messages)
    assert "\nINFO forged" not in caplog.text


@pytest.mark.asyncio
async def test_newline_in_exception_stays_inside_one_record(service, caplog, monkeypatch):
    def explode(*args, **kwargs):
        raise OSError("disk full\nINFO forged line")

    monkeypatch.setattr(shutil, "copy2", explode)
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        result = await service.restore_backup(FORGED)

    assert result == {"success": False, "error": "disk full\nINFO forged line"}
    assert _messages(caplog) == ["Failed to restore backup: disk full\\nINFO forged line"]
    assert "\nINFO forged" not in caplog.text
