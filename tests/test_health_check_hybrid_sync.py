"""Tests for the hybrid health check's Cloudflare sync status."""

import sqlite3
from types import SimpleNamespace

import pytest

from mcp_memory_service.utils.health_check import HybridHealthChecker


def _storage(secondary=True, sync_service=None):
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE memories (id INTEGER PRIMARY KEY, deleted_at REAL)")
    return SimpleNamespace(
        primary=SimpleNamespace(conn=conn, db_path=":memory:"),
        secondary=object() if secondary else None,
        sync_service=sync_service,
    )


def _sync_service(is_running, cloudflare_available, consecutive_failures=0):
    return SimpleNamespace(
        is_running=is_running,
        sync_stats={"cloudflare_available": cloudflare_available},
        consecutive_failures=consecutive_failures,
    )


async def _cloudflare_stats(storage):
    ok, message, stats = await HybridHealthChecker().check_health(storage)
    assert ok, message
    return stats


async def test_syncing_when_running_and_cloudflare_available():
    stats = await _cloudflare_stats(_storage(sync_service=_sync_service(True, True)))
    assert stats["cloudflare_sync"] == "syncing"
    assert stats["cloudflare_available"] is True
    assert stats["consecutive_failures"] == 0


async def test_sync_failing_when_running_and_cloudflare_unavailable():
    stats = await _cloudflare_stats(_storage(sync_service=_sync_service(True, False, 3)))
    assert stats["cloudflare_sync"] == "sync_failing"
    assert stats["cloudflare_available"] is False
    assert stats["consecutive_failures"] == 3


async def test_configured_when_sync_service_not_running():
    stats = await _cloudflare_stats(_storage(sync_service=_sync_service(False, True)))
    assert stats["cloudflare_sync"] == "configured"
    assert "cloudflare_available" not in stats


async def test_not_configured_without_secondary():
    stats = await _cloudflare_stats(_storage(secondary=False))
    assert stats["cloudflare_sync"] == "not_configured"
    assert "cloudflare_available" not in stats
