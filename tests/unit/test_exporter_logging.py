"""
Log-injection tests for sync/exporter.py (#1146).

The exporter logs the output path it is given (the ``--output`` argument of
``scripts/sync/export_memories.py``) when the export starts and when it ends.
A newline in that path must not reach the log as a line break.
"""

import logging
from pathlib import Path

import pytest

from mcp_memory_service.sync.exporter import MemoryExporter

FORGED = "FORGED export verified"
LOGGER = "mcp_memory_service.sync.exporter"


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
async def test_export_logs_do_not_carry_newlines_from_the_output_path(exporter, tmp_path: Path, caplog):
    output_file = tmp_path / f"export\n{FORGED}.json"

    with caplog.at_level(logging.INFO, logger=LOGGER):
        result = await exporter.export_to_json(output_file)

    assert result["exported_count"] == 0
    assert output_file.exists()
    _assert_clean(caplog, "Starting memory export to")
    _assert_clean(caplog, "Export completed: 0 memories written to")
    _assert_clean(caplog, "File size:")
