"""
Log-injection tests for web/api/memories.py (#1146).

The store and delete endpoints log the error they hit. Storage errors can echo
the content or hash a request carried, so a newline in the message must not
reach the log as a line break.
"""

import logging
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from mcp_memory_service.web.api import memories

FORGED = "FORGED admin authenticated"
LOGGER = "mcp_memory_service.web.api.memories"


def _messages(caplog):
    return [record.getMessage() for record in caplog.records]


@pytest.mark.asyncio
async def test_delete_error_log_does_not_carry_newlines(caplog):
    class Storage:
        async def get_by_hash(self, *_args, **_kwargs):
            raise RuntimeError(f"backend down\n{FORGED}")

    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        with pytest.raises(HTTPException) as err:
            await memories.delete_memory(
                content_hash="abc", store="default", storage=Storage(), user=None
            )

    assert err.value.status_code == 500
    messages = _messages(caplog)
    assert any("Failed to delete memory: backend down" in m for m in messages)
    assert not any(f"\n{FORGED}" in m for m in messages)


@pytest.mark.asyncio
async def test_delete_broadcast_failure_log_does_not_carry_newlines(caplog, monkeypatch):
    class Storage:
        async def get_by_hash(self, *_args, **_kwargs):
            return SimpleNamespace(content_hash="abc")

        async def delete(self, _content_hash):
            return True, "deleted"

    async def broadcast_event(_event):
        raise RuntimeError(f"sse down\n{FORGED}")

    monkeypatch.setattr(memories.sse_manager, "broadcast_event", broadcast_event)

    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        response = await memories.delete_memory(
            content_hash="abc", store="default", storage=Storage(), user=None
        )

    assert response.success
    messages = _messages(caplog)
    assert any("Failed to broadcast memory_deleted event: sse down" in m for m in messages)
    assert not any(f"\n{FORGED}" in m for m in messages)
