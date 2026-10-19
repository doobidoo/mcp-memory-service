"""
Log-injection tests for sync/exporter.py (#1146).

The exporter logs the output path it is given (the ``--output`` argument of
``scripts/sync/export_memories.py``) when the export starts and when it ends.
A newline in that path must not reach the log as a line break.

The hostile path is never written to disk: Windows rejects a newline in a file
name, so the file operations are stubbed and only the logging is exercised.
"""

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import mock_open

import pytest

from mcp_memory_service.sync import exporter as exporter_module
from mcp_memory_service.sync.exporter import MemoryExporter

FORGED = "FORGED export verified"
LOGGER = "mcp_memory_service.sync.exporter"


class _NoDiskPath(type(Path())):
    """A real Path whose directory and stat calls never touch the filesystem."""

    def mkdir(self, *args, **kwargs):
        return None

    def stat(self, *args, **kwargs):
        return SimpleNamespace(st_size=0)


def _assert_clean(caplog, expected):
    messages = [record.getMessage() for record in caplog.records]
    assert any(expected in m for m in messages), messages
    assert not any(f"\n{FORGED}" in m for m in messages), messages


@pytest.fixture
def exporter(monkeypatch: pytest.MonkeyPatch) -> MemoryExporter:
    """An exporter over an empty store; no backend is touched."""
    instance = MemoryExporter(storage=object())

    async def _no_memories(_filter_tags):
        return []

    monkeypatch.setattr(instance, "_get_filtered_memories", _no_memories)
    return instance


@pytest.mark.asyncio
async def test_export_logs_do_not_carry_newlines_from_the_output_path(exporter, monkeypatch, caplog):
    output_file = _NoDiskPath("exports", f"export\n{FORGED}.json")
    opened = mock_open()
    monkeypatch.setattr(exporter_module, "open", opened, raising=False)

    with caplog.at_level(logging.INFO, logger=LOGGER):
        result = await exporter.export_to_json(output_file)

    assert result["exported_count"] == 0
    opened.assert_called_once_with(output_file, "w", encoding="utf-8")
    _assert_clean(caplog, "Starting memory export to")
    _assert_clean(caplog, "Export completed: 0 memories written to")
    _assert_clean(caplog, "File size:")
