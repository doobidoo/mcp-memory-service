"""Regression tests for rollback handlers in shared SQLite storage (#1328)."""

import asyncio
import sqlite3
import os
import threading

import pytest
import pytest_asyncio

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage


@pytest_asyncio.fixture
async def storage(temp_db_path):
    storage = SqliteVecMemoryStorage(os.path.join(temp_db_path, "rollback.db"))
    await storage.initialize()
    yield storage
    await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["delete", "update_memories_batch", "mark_superseded_batch"])
async def test_error_rollback_uses_locked_executor(storage, monkeypatch, operation):
    """Error cleanup must use _run_in_thread, not rollback on the event loop."""
    calls = []

    async def fail_operation(_operation):
        raise sqlite3.OperationalError("synthetic failure")

    async def record_locked(operation, *args):
        calls.append(operation)
        return operation(*args)

    monkeypatch.setattr(storage, "_execute_with_retry", fail_operation)
    monkeypatch.setattr(storage, "_run_in_thread", record_locked)

    if operation == "delete":
        result = await storage.delete("missing-hash")
        assert result[0] is False
    elif operation == "update_memories_batch":
        memory = Memory(content="rollback test", content_hash="rollback-test-hash")
        assert await storage.update_memories_batch([memory]) == [False]
    else:
        assert await storage.mark_superseded_batch([("winner", "loser")]) == 0

    assert calls, f"{operation} rolled back without the locked executor"


@pytest.mark.asyncio
async def test_error_rollback_waits_for_connection_lock(storage, monkeypatch):
    """Rollback must wait for an in-flight connection operation to release the lock."""
    lock_held = threading.Event()
    release_lock = threading.Event()

    def hold_connection_lock():
        with storage._conn_lock:
            lock_held.set()
            assert release_lock.wait(timeout=2)

    holder = threading.Thread(target=hold_connection_lock)
    holder.start()
    assert await asyncio.to_thread(lock_held.wait, 1)

    async def fail_operation(_operation):
        raise sqlite3.OperationalError("synthetic failure")

    monkeypatch.setattr(storage, "_execute_with_retry", fail_operation)
    rollback_task = asyncio.create_task(storage.delete("missing-hash"))
    await asyncio.sleep(0.05)
    assert not rollback_task.done(), "rollback bypassed the connection lock"

    release_lock.set()
    result = await asyncio.wait_for(rollback_task, timeout=2)
    holder.join(timeout=2)
    assert not holder.is_alive()
    assert result[0] is False
