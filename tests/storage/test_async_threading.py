"""Tests for async threading behavior in sqlite_vec storage."""

import asyncio
import time
import pytest
import sqlite3


class TestExecuteWithRetryThreading:
    """Verify _execute_with_retry offloads work to a thread."""

    @pytest.mark.asyncio
    async def test_execute_with_retry_does_not_block_loop(self, tmp_path):
        """DB operations inside _execute_with_retry must not block the event loop.

        NOTE: As of the sqlite-vec thread-safety fix, _execute_with_retry holds
        a per-storage threading.Lock around the operation so that the (non-thread-safe)
        sqlite-vec extension cannot be entered concurrently from two worker threads.
        That means two _execute_with_retry calls now serialize against each other —
        but the event loop must STILL stay responsive while a slow op runs.

        Strategy: launch one slow (300ms) DB op in a worker thread and, from the
        main coroutine, await asyncio.sleep(0.05) repeatedly. The sleeps must
        complete on schedule even while the worker is sleeping inside the lock.
        """
        from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage

        storage = SqliteVecMemoryStorage.__new__(SqliteVecMemoryStorage)
        storage.conn = sqlite3.connect(str(tmp_path / "test.db"), check_same_thread=False)
        storage.conn.execute("CREATE TABLE t (id INTEGER)")

        def slow_op():
            time.sleep(0.3)
            return "ok"

        async def loop_responsiveness_probe():
            # If the event loop is blocked, these sleeps will overshoot wildly.
            ticks = []
            for _ in range(5):
                t = time.monotonic()
                await asyncio.sleep(0.05)
                ticks.append(time.monotonic() - t)
            return ticks

        result, ticks = await asyncio.gather(
            storage._execute_with_retry(slow_op),
            loop_responsiveness_probe(),
        )

        assert result == "ok"
        # Each 50ms sleep should complete in well under 200ms even on slow CI;
        # if the loop were blocked by slow_op, at least one would balloon to ~300ms+.
        assert max(ticks) < 0.2, (
            f"Event loop was blocked while DB op ran — sleep ticks: {ticks}"
        )
        storage.conn.close()

    @pytest.mark.asyncio
    async def test_execute_with_retry_returns_result(self, tmp_path):
        """_execute_with_retry should return the operation's result."""
        from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage

        storage = SqliteVecMemoryStorage.__new__(SqliteVecMemoryStorage)
        # check_same_thread=False required because asyncio.to_thread runs in a worker thread
        storage.conn = sqlite3.connect(str(tmp_path / "test.db"), check_same_thread=False)
        storage.conn.execute("CREATE TABLE t (val TEXT)")
        storage.conn.execute("INSERT INTO t VALUES ('hello')")

        def query():
            return storage.conn.execute("SELECT val FROM t").fetchone()[0]

        result = await storage._execute_with_retry(query)
        assert result == "hello"
        storage.conn.close()

    @pytest.mark.asyncio
    async def test_execute_with_retry_retries_on_locked(self, tmp_path):
        """Should retry with backoff on OperationalError('database is locked')."""
        from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage

        storage = SqliteVecMemoryStorage.__new__(SqliteVecMemoryStorage)
        storage.conn = sqlite3.connect(str(tmp_path / "test.db"), check_same_thread=False)

        call_count = 0

        def flaky_op():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise sqlite3.OperationalError("database is locked")
            return "success"

        result = await storage._execute_with_retry(flaky_op, max_retries=5, initial_delay=0.01)
        assert result == "success"
        assert call_count == 3
        storage.conn.close()
