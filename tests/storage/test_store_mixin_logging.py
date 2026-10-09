"""
Log-injection tests for storage/mixins/store.py (#1146).

The store operations log the content hash of the memory and the error a backend
or the embedding model raised. Both can carry request data, so a newline in one
must not reach the log as a line break.
"""

import asyncio
import logging
from types import SimpleNamespace

import pytest

from mcp_memory_service.storage.mixins.store import StoreMixin

FORGED = "FORGED admin authenticated"
LOGGER = "mcp_memory_service.storage.mixins.store"


class _Cursor:
    def fetchone(self):
        return None


class _Conn:
    def execute(self, *_args, **_kwargs):
        return _Cursor()

    def rollback(self):
        pass


class _Store(StoreMixin):
    semantic_dedup_enabled = False

    def __init__(self):
        self.conn = _Conn()
        self.embedding_model = SimpleNamespace(encode=self._encode)

    @staticmethod
    def _encode(*_args, **_kwargs):
        raise RuntimeError(f"model gone\n{FORGED}")

    def _generate_embedding(self, _content):
        raise RuntimeError(f"model gone\n{FORGED}")

    async def _execute_with_retry(self, operation):
        return operation()

    async def _run_in_thread(self, operation, *args):
        return operation(*args)


def _memory():
    return SimpleNamespace(content="hello", content_hash=f"abc\n{FORGED}")


def _assert_clean(caplog, expected):
    messages = [record.getMessage() for record in caplog.records]
    assert any(expected in m for m in messages)
    assert not any(f"\n{FORGED}" in m for m in messages)


@pytest.mark.asyncio
async def test_embedding_failure_log_does_not_carry_newlines(caplog):
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        ok, _message = await _Store().store(_memory())

    assert not ok
    _assert_clean(caplog, "Failed to generate embedding for memory abc")
    _assert_clean(caplog, "model gone")


@pytest.mark.asyncio
async def test_batch_embedding_failure_log_does_not_carry_newlines(caplog):
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        results = await _Store().store_batch([_memory()])

    assert not results[0][0]
    _assert_clean(caplog, "Batch embedding generation failed: model gone")


class _Cur:
    lastrowid = 1

    def __init__(self, rows=()):
        self._rows = list(rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows


class _OkConn:
    """Statements succeed; the embedding delete of a tombstone purge fails."""

    def __init__(self, fail_on=None):
        self._fail_on = fail_on

    def execute(self, sql, *_args):
        if self._fail_on and self._fail_on in sql:
            raise RuntimeError(f"disk gone\n{FORGED}")
        if sql.lstrip().startswith("SELECT id FROM memories"):
            return _Cur(rows=[(1,)])
        return _Cur()

    def commit(self):
        pass

    def rollback(self):
        pass


def _full_memory(content_hash):
    return SimpleNamespace(
        content="hello",
        content_hash=content_hash,
        tags=[],
        metadata={},
        memory_type="note",
        created_at=0.0,
        updated_at=0.0,
        created_at_iso="",
        updated_at_iso="",
    )


class _WorkingStore(_Store):
    def __init__(self, conn):
        super().__init__()
        self.conn = conn
        self._savepoint_lock = asyncio.Lock()
        self.embedding_model = SimpleNamespace(encode=lambda *_a, **_k: [[0.1, 0.2]])

    def _generate_embedding(self, _content):
        return [0.1, 0.2]

    def _detect_conflicts(self, *_args):
        return []


@pytest.mark.asyncio
async def test_success_log_does_not_carry_newlines(caplog):
    store = _WorkingStore(_OkConn())

    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        ok, _message = await store.store(_full_memory(f"abc\n{FORGED}"))

    assert ok
    _assert_clean(caplog, "Successfully stored memory: abc")


def test_tombstone_purge_log_does_not_carry_newlines(caplog):
    store = _WorkingStore(_OkConn(fail_on="DELETE FROM memory_embeddings"))

    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        store._purge_tombstone("abc")

    _assert_clean(caplog, "Could not delete embedding rowid=1 during tombstone purge: disk gone")


@pytest.mark.asyncio
async def test_batch_transaction_failure_log_does_not_carry_newlines(caplog):
    store = _WorkingStore(_OkConn(fail_on="SAVEPOINT"))

    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        results = await store.store_batch([_full_memory("abc")])

    assert not results[0][0]
    _assert_clean(caplog, "Batch transaction failed: disk gone")
