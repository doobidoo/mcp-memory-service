"""Tests for health checker connection lock compliance (#1363)."""

import asyncio
import os
import threading
from typing import Any, Dict

import pytest
import pytest_asyncio

from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.health_check import SqliteHealthChecker, HybridHealthChecker, _collect_sqlite_stats


@pytest_asyncio.fixture
async def storage(temp_db_path):
    """Empty SQLite storage for health check testing."""
    storage = SqliteVecMemoryStorage(os.path.join(temp_db_path, "health.db"))
    await storage.initialize()
    yield storage
    await storage.close()


@pytest.mark.asyncio
async def test_sqlite_health_checker_waits_for_connection_lock(storage):
    """SqliteHealthChecker must wait for _conn_lock before reading the database."""
    lock_held = threading.Event()
    release_lock = threading.Event()

    def hold_connection_lock():
        with storage._conn_lock:
            lock_held.set()
            assert release_lock.wait(timeout=2)

    # Hold the connection lock from a worker thread
    holder = threading.Thread(target=hold_connection_lock)
    holder.start()
    assert await asyncio.to_thread(lock_held.wait, 1)

    # Start health check as async task
    checker = SqliteHealthChecker()
    health_task = asyncio.create_task(checker.check_health(storage))
    
    # Health check should be blocked waiting for the lock
    await asyncio.sleep(0.05)
    assert not health_task.done(), "health check bypassed the connection lock"

    # Release lock and verify success
    release_lock.set()
    is_valid, message, stats = await asyncio.wait_for(health_task, timeout=2)
    holder.join(timeout=2)
    assert not holder.is_alive()
    
    # Verify correct health check results
    assert is_valid is True
    assert "validation successful" in message
    assert stats["total_memories"] == 0
    assert stats["backend"] == "sqlite-vec"


@pytest.mark.asyncio
async def test_sqlite_health_checker_uses_run_in_thread(storage, monkeypatch):
    """SqliteHealthChecker must route SQLite work through _run_in_thread."""
    run_in_thread_calls = []

    async def record_run_in_thread(operation, *args):
        run_in_thread_calls.append((operation, args))
        # Call the original _run_in_thread to complete the health check
        return await storage.__class__._run_in_thread(storage, operation, *args)

    monkeypatch.setattr(storage, "_run_in_thread", record_run_in_thread)

    checker = SqliteHealthChecker()
    is_valid, message, stats = await checker.check_health(storage)

    # Verify health check succeeded
    assert is_valid is True
    assert stats["backend"] == "sqlite-vec"
    
    # Verify that _run_in_thread was called for SQLite operations
    assert len(run_in_thread_calls) > 0, "health check did not use _run_in_thread"
    
    # At least one call should be for the stats collection
    operation_names = [call[0].__name__ if callable(call[0]) else str(call[0]) for call in run_in_thread_calls]
    assert any("collect" in name.lower() for name in operation_names), f"Expected stats collection call, got: {operation_names}"


@pytest.mark.asyncio
async def test_hybrid_health_checker_waits_for_connection_lock(storage):
    """HybridHealthChecker must wait for primary storage _conn_lock."""
    # Create a mock hybrid storage with the test SQLite storage as primary
    class MockHybridStorage:
        def __init__(self, primary_storage):
            self.primary = primary_storage
            self.secondary = None
            self.sync_service = None

    hybrid_storage = MockHybridStorage(storage)
    
    lock_held = threading.Event()
    release_lock = threading.Event()

    def hold_connection_lock():
        with storage._conn_lock:
            lock_held.set()
            assert release_lock.wait(timeout=2)

    # Hold the connection lock from a worker thread
    holder = threading.Thread(target=hold_connection_lock)
    holder.start()
    assert await asyncio.to_thread(lock_held.wait, 1)

    # Start health check as async task
    checker = HybridHealthChecker()
    health_task = asyncio.create_task(checker.check_health(hybrid_storage))
    
    # Health check should be blocked waiting for the lock
    await asyncio.sleep(0.05)
    assert not health_task.done(), "hybrid health check bypassed the connection lock"

    # Release lock and verify success
    release_lock.set()
    is_valid, message, stats = await asyncio.wait_for(health_task, timeout=2)
    holder.join(timeout=2)
    assert not holder.is_alive()
    
    # Verify correct health check results
    assert is_valid is True
    assert "validation successful" in message
    assert stats["total_memories"] == 0
    assert stats["backend"] == "hybrid"


def test_collect_sqlite_stats_synchronous_helper(temp_db_path):
    """_collect_sqlite_stats must be a synchronous helper that returns correct stats."""
    import sqlite3
    
    # Create a minimal test database
    db_path = os.path.join(temp_db_path, "test_stats.db")
    conn = sqlite3.connect(db_path)
    
    # Create the required tables (minimal schema)
    conn.execute("""
        CREATE TABLE memories (
            rowid INTEGER PRIMARY KEY,
            content TEXT,
            deleted_at TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE memory_embeddings (
            rowid INTEGER PRIMARY KEY,
            embedding BLOB
        )
    """)
    conn.commit()
    
    # Create a mock storage object with required attributes
    class MockStorage:
        def __init__(self, db_path):
            self.db_path = db_path
            self.embedding_model_name = "test-model"
    
    mock_storage = MockStorage(db_path)
    
    # Call the helper function (this should fail with ImportError initially)
    stats = _collect_sqlite_stats(conn, mock_storage)
    
    # Verify the expected structure
    assert isinstance(stats, dict)
    assert stats["backend"] == "sqlite-vec"
    assert stats["total_memories"] == 0
    assert "has_embedding_tables" in stats
    assert stats["has_embedding_tables"] is True
    
    conn.close()


def test_collect_sqlite_stats_handles_missing_tables(temp_db_path):
    """_collect_sqlite_stats must handle missing tables gracefully."""
    import sqlite3
    
    # Create database without required tables
    db_path = os.path.join(temp_db_path, "empty_stats.db")
    conn = sqlite3.connect(db_path)
    
    class MockStorage:
        def __init__(self, db_path):
            self.db_path = db_path
            self.embedding_model_name = "test-model"
    
    mock_storage = MockStorage(db_path)
    
    # This should raise LookupError for missing tables
    with pytest.raises(LookupError):
        _collect_sqlite_stats(conn, mock_storage)
    
    conn.close()