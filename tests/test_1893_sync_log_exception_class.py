"""
Regression tests for #1893: sync failure logs dropped the exception type.

`str(exc)` is empty for several httpx timeout/transport errors, so lines like
"Delta-sync capability probe failed for peer hub: " ended after the colon and the
cause was lost. The class name is the one part that is always there.
"""

import logging
from unittest.mock import MagicMock

import httpx
import pytest

import mcp_memory_service.storage.sync.orchestrator as orch
from mcp_memory_service.consolidation.scheduler import ConsolidationScheduler
from mcp_memory_service.storage.remote_http import RemoteHTTPStorage


class _FakeConsolidator:
    def __init__(self, storage=None):
        self.storage = storage
        self.health_monitor = None


def _scheduler():
    return ConsolidationScheduler(
        _FakeConsolidator(storage=MagicMock()), schedule_config={}, enabled=False
    )


def _peer_failing_with(exc):
    peer = MagicMock()

    async def _init():
        return None

    async def _close():
        return None

    async def _check(storage):
        raise exc

    peer.initialize = _init
    peer.close = _close
    peer.check_sync_compatibility = _check
    return peer


async def test_request_failure_logs_exception_class(caplog):
    """RemoteHTTPStorage._request names the exception even with an empty message."""
    storage = RemoteHTTPStorage(base_url="https://example.invalid", api_key="k")

    async def _raise(method, url, **kwargs):
        raise httpx.ReadTimeout("")

    storage.client.request = _raise

    with caplog.at_level(logging.ERROR):
        with pytest.raises(httpx.ReadTimeout):
            await storage._request("GET", "/api/sync/capabilities")

    await storage.client.aclose()

    messages = [r.getMessage() for r in caplog.records]
    assert any("ReadTimeout" in m for m in messages), messages
    assert not any("example.invalid" in m or "k" == m for m in messages), \
        "the url and the api key must stay out of the log"


async def test_probe_failure_logs_exception_class(caplog, monkeypatch):
    """The startup capability probe names the exception even with an empty message."""
    sch = _scheduler()
    peer = _peer_failing_with(httpx.ConnectTimeout(""))
    monkeypatch.setattr(sch, "_resolve_sync_peers", lambda: [("hub", peer)])

    with caplog.at_level(logging.WARNING):
        await sch._probe_sync_peers()

    messages = [r.getMessage() for r in caplog.records]
    assert any(
        "Delta-sync capability probe failed for peer hub" in m and "ConnectTimeout" in m
        for m in messages
    ), messages


async def test_cycle_failure_logs_exception_class(caplog, monkeypatch):
    """The sync cycle names the exception even with an empty message."""

    async def _boom(storage, peer, peer_id, limit):
        raise httpx.RemoteProtocolError("")

    monkeypatch.setattr(orch, "sync_from_peer", _boom)

    sch = _scheduler()
    peer = MagicMock()

    async def _init():
        return None

    async def _close():
        return None

    async def _check(storage):
        return {"negotiated_sync_protocol": 1}

    peer.initialize = _init
    peer.close = _close
    peer.check_sync_compatibility = _check
    monkeypatch.setattr(sch, "_resolve_sync_peers", lambda: [("hub", peer)])

    with caplog.at_level(logging.ERROR):
        await sch._run_sync_cycle()

    messages = [r.getMessage() for r in caplog.records]
    assert any(
        "Delta-sync cycle failed for peer hub" in m and "RemoteProtocolError" in m
        for m in messages
    ), messages
