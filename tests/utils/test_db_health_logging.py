"""
Log-injection tests for utils/db_utils.py and utils/health_check.py (#1146).

Both modules log the error a backend raised while it was validated or checked.
Backend messages can echo configuration or request data, so a newline in one
must not reach the log as a line break.
"""

import logging

import pytest

from mcp_memory_service.utils import db_utils, health_check

FORGED = "FORGED admin authenticated"


def _assert_clean(caplog, expected):
    messages = [record.getMessage() for record in caplog.records]
    assert any(expected in m for m in messages)
    assert not any(f"\n{FORGED}" in m for m in messages)


class SqliteVecMemoryStorage:
    """Named like the real class: db_utils dispatches on the class name."""

    def is_initialized(self):
        raise RuntimeError(f"state unknown\n{FORGED}")

    def get_stats(self):
        raise RuntimeError(f"stats broke\n{FORGED}")


@pytest.mark.asyncio
async def test_validate_database_logs_do_not_carry_newlines(caplog):
    with caplog.at_level(logging.DEBUG):
        await db_utils.validate_database(SqliteVecMemoryStorage())

    _assert_clean(caplog, "Error checking initialization status: state unknown")


@pytest.mark.asyncio
async def test_database_stats_and_repair_logs_do_not_carry_newlines(caplog):
    with caplog.at_level(logging.DEBUG):
        await db_utils.get_database_stats(SqliteVecMemoryStorage())
        await db_utils.repair_database(SqliteVecMemoryStorage())

    _assert_clean(caplog, "Error calling get_stats method: stats broke")


class _BrokenConn:
    def execute(self, *_args, **_kwargs):
        raise RuntimeError(f"locked\n{FORGED}")


def test_embedding_integrity_log_does_not_carry_newlines(caplog):
    with caplog.at_level(logging.DEBUG):
        assert health_check._check_embedding_integrity(_BrokenConn()) == {}

    _assert_clean(caplog, "Embedding integrity check failed: locked")


class _Storage:
    def __init__(self):
        self.client = object()
        self.conn = object()
        self.primary = self

    async def get_stats(self):
        raise RuntimeError(f"api down\n{FORGED}")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "checker, expected",
    [
        (health_check.CloudflareHealthChecker, "Cloudflare health check error: api down"),
        (health_check.MilvusHealthChecker, "Milvus health check error: api down"),
    ],
)
async def test_checker_error_logs_do_not_carry_newlines(caplog, checker, expected):
    with caplog.at_level(logging.DEBUG):
        ok, _message, _stats = await checker().check_health(_Storage())

    assert not ok
    _assert_clean(caplog, expected)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "checker, expected",
    [
        (health_check.SqliteHealthChecker, "SQLite health check error: locked"),
        (health_check.HybridHealthChecker, "Hybrid health check error: locked"),
    ],
)
async def test_locked_checker_error_logs_do_not_carry_newlines(caplog, monkeypatch, checker, expected):
    async def broken(*_args, **_kwargs):
        raise RuntimeError(f"locked\n{FORGED}")

    monkeypatch.setattr(health_check, "_run_locked", broken)

    with caplog.at_level(logging.DEBUG):
        ok, _message, _stats = await checker().check_health(_Storage())

    assert not ok
    _assert_clean(caplog, expected)
