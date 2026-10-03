"""
Log-injection tests for server/handlers/utility.py and documents.py (#1146).

These handlers log the error they hit. Ingestion errors echo file paths and
parser messages, and health and cache errors echo backend messages, so a newline
in one must not reach the log as a line break.
"""

import logging

import pytest

from mcp_memory_service.server.handlers import documents, utility

FORGED = "FORGED admin authenticated"


def _messages(caplog):
    return [record.getMessage() for record in caplog.records]


class _FailingServer:
    def __init__(self, message):
        self._message = message
        self.query_times = []

    async def _ensure_storage_initialized(self):
        raise RuntimeError(self._message)

    def get_average_query_time(self):
        return 0.0


def _assert_clean(caplog, expected):
    messages = _messages(caplog)
    assert any(expected in m for m in messages)
    assert not any(f"\n{FORGED}" in m for m in messages)


@pytest.mark.asyncio
async def test_document_ingestion_error_log_does_not_carry_newlines(caplog):
    server = _FailingServer(f"disk gone\n{FORGED}")

    with caplog.at_level(logging.DEBUG):
        await documents.handle_ingest_document(server, {"file_path": "/tmp/x.txt"})

    _assert_clean(caplog, "Error in document ingestion: disk gone")


@pytest.mark.asyncio
async def test_directory_ingestion_error_log_does_not_carry_newlines(caplog):
    server = _FailingServer(f"disk gone\n{FORGED}")

    with caplog.at_level(logging.DEBUG):
        await documents.handle_ingest_directory(server, {"directory_path": "/tmp"})

    _assert_clean(caplog, "Error in directory ingestion: disk gone")


@pytest.mark.asyncio
async def test_health_check_init_error_log_does_not_carry_newlines(caplog):
    server = _FailingServer(f"backend down\n{FORGED}")

    with caplog.at_level(logging.DEBUG):
        await utility.handle_check_database_health(server, {})

    _assert_clean(caplog, "Storage initialization failed during health check: backend down")


@pytest.mark.asyncio
async def test_cache_stats_error_log_does_not_carry_newlines(caplog, monkeypatch):
    from mcp_memory_service.utils import cache_manager

    def boom(*_args, **_kwargs):
        raise RuntimeError(f"stats broke\n{FORGED}")

    monkeypatch.setattr(cache_manager, "calculate_cache_stats_dict", boom)

    with caplog.at_level(logging.DEBUG):
        await utility.handle_get_cache_stats(object(), {})

    _assert_clean(caplog, "Error in get_cache_stats: stats broke")
