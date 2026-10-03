"""
Log-injection tests for web/sse.py (#1146).

The SSE manager logs the client address of each connection and the text of the
errors it hits. The address comes from the request and the error text can echo
request data, so a newline in either must not reach the log as a line break.
"""

import asyncio
import logging
from types import SimpleNamespace

import pytest

from mcp_memory_service.web.sse import SSEEvent, SSEManager

FORGED = "FORGED admin authenticated"
LOGGER = "mcp_memory_service.web.sse"


def _messages(caplog):
    return [record.getMessage() for record in caplog.records]


def _request(host):
    return SimpleNamespace(headers={}, client=SimpleNamespace(host=host))


@pytest.mark.asyncio
async def test_connection_log_does_not_carry_client_ip_newlines(caplog):
    manager = SSEManager(replay_buffer_size=0)

    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        await manager.add_connection("conn-1", _request(f"10.0.0.1\n{FORGED}"))

    messages = _messages(caplog)
    assert any("SSE connection added: conn-1 from 10.0.0.1" in m for m in messages)
    assert not any(f"\n{FORGED}" in m for m in messages)


@pytest.mark.asyncio
async def test_send_failure_log_does_not_carry_error_newlines(caplog):
    manager = SSEManager(replay_buffer_size=0)

    class BrokenQueue:
        async def put(self, _event):
            raise RuntimeError(f"queue closed\n{FORGED}")

    manager.connections["conn-1"] = {
        "queue": BrokenQueue(),
        "request": _request("10.0.0.1"),
        "connected_at": 0.0,
        "last_heartbeat": 0.0,
        "user_agent": "test",
        "client_ip": "10.0.0.1",
    }

    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        await manager.broadcast_event(SSEEvent(event_type="memory_stored", data={}))

    messages = _messages(caplog)
    assert any("Failed to send event to conn-1: queue closed" in m for m in messages)
    assert not any(f"\n{FORGED}" in m for m in messages)


@pytest.mark.asyncio
async def test_replay_log_does_not_carry_last_event_id_newlines(caplog):
    manager = SSEManager(replay_buffer_size=10)
    resumed_from = f"evt-1\n{FORGED}"
    await manager.broadcast_event(
        SSEEvent(event_type="memory_stored", data={}, event_id=resumed_from)
    )
    await manager.broadcast_event(SSEEvent(event_type="memory_stored", data={}))

    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        await manager.add_connection("conn-1", _request("10.0.0.1"), last_event_id=resumed_from)

    messages = _messages(caplog)
    assert any("SSE replayed 1 event(s) to conn-1 after Last-Event-ID=evt-1" in m for m in messages)
    assert not any(f"\n{FORGED}" in m for m in messages)
