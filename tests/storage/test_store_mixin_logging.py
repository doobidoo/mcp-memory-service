"""
Log-injection tests for storage/mixins/store.py (#1146).

The store operations log the content hash of the memory and the error a backend
or the embedding model raised. Both can carry request data, so a newline in one
must not reach the log as a line break.
"""

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
