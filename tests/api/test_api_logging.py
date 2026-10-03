"""
Log-injection tests for api/client.py and api/operations.py (#1146).

The code execution API logs the error that a backend, the consolidator or the
scheduler raised. Those errors can echo configuration or request data, so a
newline in one must not reach the log as a line break.
"""

import logging
from types import SimpleNamespace

import pytest

from mcp_memory_service.api import client, operations

FORGED = "FORGED admin authenticated"


def _assert_clean(caplog, expected):
    messages = [record.getMessage() for record in caplog.records]
    assert any(expected in m for m in messages)
    assert not any(f"\n{FORGED}" in m for m in messages)


@pytest.mark.asyncio
async def test_storage_init_failure_log_does_not_carry_newlines(caplog, monkeypatch):
    async def broken(_path):
        raise RuntimeError(f"disk gone\n{FORGED}")

    monkeypatch.setattr(client, "_storage_instance", None)
    monkeypatch.setattr(client, "create_storage_instance", broken)

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(RuntimeError):
            await client.get_storage_async()

    _assert_clean(caplog, "Failed to initialize storage backend: disk gone")


def test_health_failure_log_does_not_carry_newlines(caplog, monkeypatch):
    async def broken():
        raise RuntimeError(f"storage gone\n{FORGED}")

    monkeypatch.setattr(operations, "get_storage_async", broken)

    with caplog.at_level(logging.DEBUG):
        # health() is the synchronous public API: it wraps the coroutine itself.
        info = operations.health()

    assert info.status == "error"
    _assert_clean(caplog, "Health check failed: storage gone")


@pytest.mark.asyncio
async def test_consolidation_failure_log_does_not_carry_newlines(caplog, monkeypatch):
    async def broken(_horizon):
        raise RuntimeError(f"consolidator gone\n{FORGED}")

    monkeypatch.setattr(operations, "get_consolidator", lambda: SimpleNamespace(consolidate=broken))

    with caplog.at_level(logging.DEBUG):
        result = await operations._consolidate_async("weekly")

    assert result.status == "failed"
    _assert_clean(caplog, "Consolidation failed: consolidator gone")


@pytest.mark.asyncio
async def test_scheduler_status_failure_log_does_not_carry_newlines(caplog, monkeypatch):
    class Inner:
        def get_jobs(self):
            raise RuntimeError(f"scheduler gone\n{FORGED}")

    monkeypatch.setattr(operations, "get_scheduler", lambda: SimpleNamespace(scheduler=Inner()))

    with caplog.at_level(logging.DEBUG):
        status = await operations._scheduler_status_async()

    assert not status.running
    _assert_clean(caplog, "Failed to get scheduler status: scheduler gone")
