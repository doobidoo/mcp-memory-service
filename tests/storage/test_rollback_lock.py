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


@pytest.mark.asyncio
async def test_remote_tombstone_failure_cannot_rollback_a_concurrent_store(
    storage, monkeypatch
):
    """A failed tombstone transaction must roll back before releasing the DB lock."""
    deleted_memory = Memory(
        content="memory whose remote tombstone event will fail",
        content_hash="remote-delete-target",
        updated_at=1.0,
    )
    concurrent_memory = Memory(
        content="independent memory stored while tombstone handling fails",
        content_hash="concurrent-store-target",
        updated_at=2.0,
    )
    assert (await storage.store(deleted_memory))[0]

    hook_entered = threading.Event()
    release_failure = threading.Event()
    original_append = storage._append_sync_event

    def fail_tombstone_event(conn, operation, content_hash, payload, **kwargs):
        if operation == "delete" and content_hash == deleted_memory.content_hash:
            hook_entered.set()
            assert release_failure.wait(timeout=3)
            raise RuntimeError("synthetic sync-event failure")
        return original_append(conn, operation, content_hash, payload, **kwargs)

    monkeypatch.setattr(storage, "_append_sync_event", fail_tombstone_event)

    original_run = storage._run_in_thread
    store_inserted = asyncio.Event()
    delete_finished = asyncio.Event()
    rollback_finished = asyncio.Event()

    async def coordinate_shared_connection(operation, *args):
        is_connection_method = getattr(operation, "__self__", None) is storage.conn
        operation_name = getattr(operation, "__name__", "")

        if is_connection_method and operation_name == "rollback":
            await asyncio.wait_for(store_inserted.wait(), timeout=3)
            result = await original_run(operation, *args)
            rollback_finished.set()
            return result

        if is_connection_method and operation_name == "commit" and store_inserted.is_set():
            waiters = [
                asyncio.create_task(delete_finished.wait()),
                asyncio.create_task(rollback_finished.wait()),
            ]
            done, pending = await asyncio.wait(
                waiters, timeout=3, return_when=asyncio.FIRST_COMPLETED
            )
            for waiter in pending:
                waiter.cancel()
            assert done, "neither tombstone rollback nor completion released the store"
        result = await original_run(operation, *args)
        if operation_name == "insert_memory_and_embedding":
            store_inserted.set()
        return result

    monkeypatch.setattr(storage, "_run_in_thread", coordinate_shared_connection)
    tombstone_task = asyncio.create_task(
        storage.delete_if_not_updated_after(deleted_memory.content_hash, 10.0)
    )
    assert await asyncio.to_thread(hook_entered.wait, 3)
    store_task = asyncio.create_task(storage.store(concurrent_memory))
    release_failure.set()

    try:
        tombstone_result = await asyncio.wait_for(tombstone_task, timeout=5)
    finally:
        delete_finished.set()
    store_result = await asyncio.wait_for(store_task, timeout=5)

    assert tombstone_result == "failed"
    assert store_result[0] is True
    assert store_inserted.is_set()
    assert await storage.get_by_hash(concurrent_memory.content_hash) is not None
