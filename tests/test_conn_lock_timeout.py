"""Connection-lock wait budget (MCP_MEMORY_LOCK_TIMEOUT).

Every storage operation serializes on ``_conn_lock`` because the sqlite-vec
extension is not thread-safe. Before the wait budget existed, one slow
operation parked the whole ``asyncio.to_thread`` pool on ``acquire()`` and the
HTTP server stopped answering — in production a single vec0 MATCH scan held
the lock for minutes on a CPU-saturated host while every worker and every
request queued behind it.

These tests bypass ``__init__`` (same trick the upstream suite uses for
``SqliteVecMemoryStorage.__new__``) and exercise only the lock discipline, so
they need no database, no sqlite-vec extension, and no embedding model.
"""

import asyncio
import threading
import time

import pytest

from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage


def _storage_with_timeout(timeout: float) -> SqliteVecMemoryStorage:
    s = SqliteVecMemoryStorage.__new__(SqliteVecMemoryStorage)
    s._conn_lock = threading.Lock()
    s._conn_lock_timeout = timeout
    return s


def test_default_lock_timeout_is_30s(monkeypatch):
    monkeypatch.delenv("MCP_MEMORY_LOCK_TIMEOUT", raising=False)
    s = SqliteVecMemoryStorage.__new__(SqliteVecMemoryStorage)
    assert s._get_conn_lock_timeout() == 30.0


def test_lock_timeout_env_override_and_invalid(monkeypatch):
    monkeypatch.setenv("MCP_MEMORY_LOCK_TIMEOUT", "5")
    s = SqliteVecMemoryStorage.__new__(SqliteVecMemoryStorage)
    assert s._get_conn_lock_timeout() == 5.0
    monkeypatch.setenv("MCP_MEMORY_LOCK_TIMEOUT", "not-a-number")
    assert s._get_conn_lock_timeout() == 30.0
    monkeypatch.setenv("MCP_MEMORY_LOCK_TIMEOUT", "0")
    assert s._get_conn_lock_timeout() == 0.0


def test_operation_runs_when_lock_free():
    s = _storage_with_timeout(2.0)
    assert asyncio.run(s._run_in_thread(lambda: "ok")) == "ok"


def test_held_lock_fails_fast_within_budget():
    """A lock holder that outlives the budget must fail the waiter fast —
    never park it for the holder's full duration (the production freeze)."""
    s = _storage_with_timeout(1.0)
    release = threading.Event()

    def hold():
        with s._conn_lock:
            release.wait(timeout=10)

    threading.Thread(target=hold, daemon=True).start()
    try:
        assert s._conn_lock.locked()
        t0 = time.monotonic()
        with pytest.raises(TimeoutError, match="connection lock not acquired"):
            asyncio.run(s._run_in_thread(lambda: "never"))
        elapsed = time.monotonic() - t0
        assert elapsed < 5.0, f"waiter blocked {elapsed:.1f}s instead of failing at the 1s budget"
    finally:
        release.set()


def test_operations_recover_after_holder_exits():
    s = _storage_with_timeout(1.0)
    release = threading.Event()

    def hold():
        with s._conn_lock:
            release.wait(timeout=10)

    threading.Thread(target=hold, daemon=True).start()
    try:
        with pytest.raises(TimeoutError):
            asyncio.run(s._run_in_thread(lambda: "x"))
    finally:
        release.set()
    assert asyncio.run(s._run_in_thread(lambda: "recovered")) == "recovered"
