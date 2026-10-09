"""
Log-injection tests for storage/mixins/delete.py (#1146).

The delete operations log the hash, tags and error text of the call they serve.
Those come from API and MCP callers or from the backend, so a newline in one
must not reach the log as a line break.
"""

import logging
from datetime import date

import pytest

from mcp_memory_service.storage.mixins.delete import DeleteMixin

FORGED = "FORGED admin authenticated"
LOGGER = "mcp_memory_service.storage.mixins.delete"


class _Conn:
    def execute(self, *_args, **_kwargs):
        raise RuntimeError(f"disk gone\n{FORGED}")

    def rollback(self):
        pass


class _Store(DeleteMixin):
    def __init__(self):
        self.conn = _Conn()

    async def _execute_with_retry(self, operation):
        return operation()

    async def _run_in_thread(self, operation, *args):
        return operation(*args)


def _assert_clean(caplog, expected):
    messages = [record.getMessage() for record in caplog.records]
    assert any(expected in m for m in messages)
    assert not any(f"\n{FORGED}" in m for m in messages)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call, expected",
    [
        (lambda s: s.delete("abc"), "Failed to delete memory: disk gone"),
        (lambda s: s.is_deleted("abc"), "Failed to check if memory is deleted: disk gone"),
        (lambda s: s.purge_deleted(), "Failed to purge deleted memories: disk gone"),
        (lambda s: s.delete_by_tag("t"), "Failed to delete by tag: disk gone"),
        (lambda s: s.delete_by_tags(["t"]), "Failed to delete by tags: disk gone"),
        (
            lambda s: s.delete_by_timeframe(date(2026, 1, 1), date(2026, 1, 2)),
            "Error deleting by timeframe: disk gone",
        ),
        (lambda s: s.delete_before_date(date(2026, 1, 1)), "Error deleting before date: disk gone"),
        (lambda s: s.cleanup_duplicates(), "Failed to cleanup duplicates: disk gone"),
    ],
)
async def test_error_logs_do_not_carry_newlines(caplog, call, expected):
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        await call(_Store())

    _assert_clean(caplog, expected)


class _Cursor:
    def __init__(self, row=None, rowcount=1):
        self._row = row
        self.rowcount = rowcount

    def fetchone(self):
        return self._row

    def fetchall(self):
        return [self._row] if self._row else []


class _WorkingConn:
    """A connection whose statements succeed, except for the embedding delete."""

    def execute(self, sql, *_args):
        if "memory_embeddings" in sql and sql.lstrip().startswith("DELETE"):
            raise RuntimeError(f"corrupted blob\n{FORGED}")
        if sql.lstrip().startswith("SELECT"):
            return _Cursor(row=(1, "abc"))
        return _Cursor()

    def commit(self):
        pass

    def rollback(self):
        pass


class _WorkingStore(_Store):
    def __init__(self):
        self.conn = _WorkingConn()


@pytest.mark.asyncio
async def test_success_and_embedding_fallback_logs_do_not_carry_newlines(caplog):
    store = _WorkingStore()

    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        ok, _message = await store.delete(f"abc\n{FORGED}")
        count, _message = await store.delete_by_tag(f"t\n{FORGED}")
        count_tags, _message, _hashes = await store.delete_by_tags([f"t\n{FORGED}"])

    assert ok
    assert count == 1
    assert count_tags == 1
    _assert_clean(caplog, "Soft-deleted memory: abc")
    _assert_clean(caplog, "Could not delete embedding for memory abc")
    _assert_clean(caplog, "Soft-deleted 1 memories with tag: t")
    _assert_clean(caplog, "Soft-deleted 1 memories matching tags: ['t")
