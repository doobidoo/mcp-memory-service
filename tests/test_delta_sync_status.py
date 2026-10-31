"""Issue #1555: delta-sync visibility for non-hybrid backends."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mcp_memory_service.api import client as api_client
from mcp_memory_service.consolidation.scheduler import ConsolidationScheduler
from mcp_memory_service.storage.sync.orchestrator import PushResult, SyncResult
from mcp_memory_service.web.api.sync import require_read_access, router
from mcp_memory_service.web.dependencies import get_storage


@pytest.fixture
def scheduler(monkeypatch):
    monkeypatch.setenv("MCP_SYNC_PEERS", "hub")
    monkeypatch.setenv("MCP_SYNC_PEER_URL", "https://user:secret@hub.test/memory?token=secret#secret")
    monkeypatch.setenv("MCP_SYNC_SCHEDULE", "15m")
    return ConsolidationScheduler(SimpleNamespace(health_monitor=None), {}, enabled=True)


def test_config_visible_before_scheduler_start_without_credentials(scheduler):
    status = scheduler.get_delta_sync_status()
    assert status["configured"] is True
    assert status["enabled"] is False
    assert status["peers"][0]["url"] == "https://hub.test/memory"
    assert "secret" not in str(status)
    assert status["peers"][0]["health"] == "pending"


@pytest.mark.asyncio
async def test_actual_next_run_and_disabled_configuration(scheduler, monkeypatch):
    scheduler._schedule_sync_job()
    # Avoid the network probe; only the interval job is relevant here.
    scheduler.scheduler.remove_job("delta_sync_probe")
    scheduler.scheduler.start()
    try:
        status = scheduler.get_delta_sync_status()
        assert status["enabled"] is True
        assert status["status"] == "pending"
        assert status["next_run_at"] == scheduler.scheduler.get_job("delta_sync").next_run_time.isoformat()
        scheduler.scheduler.pause_job("delta_sync")
        assert scheduler.get_delta_sync_status()["next_run_at"] is None
        assert scheduler.get_delta_sync_status()["enabled"] is False
        scheduler.scheduler.resume_job("delta_sync")
        scheduler.scheduler.pause()
        assert scheduler.get_delta_sync_status()["next_run_at"] is None
    finally:
        scheduler.scheduler.shutdown(wait=False)
    monkeypatch.delenv("MCP_SYNC_PEERS")
    monkeypatch.setenv("MCP_SYNC_SCHEDULE", "disabled")
    assert ConsolidationScheduler.delta_sync_configuration()["configured"] is False


@pytest.mark.asyncio
async def test_cycle_records_results_partial_failure_and_recovery(scheduler):
    peer = SimpleNamespace(initialize=AsyncMock(), close=AsyncMock(),
                           check_sync_compatibility=AsyncMock(return_value={}))
    pull = AsyncMock(return_value=SyncResult(3, 1, 42, events_received=4))
    push = AsyncMock(return_value=PushResult(2, 0, 30))

    async def observe_pull(*args, **kwargs):
        assert scheduler.get_delta_sync_status()["status"] == "syncing"
        return pull.return_value

    await scheduler._run_sync_peer_cycle(None, "hub", peer, observe_pull, push)
    result = scheduler.get_delta_sync_status()["peers"][0]
    assert result["health"] == "synced"
    assert result["pull"] == {"received": 4, "applied": 3, "cursor": 42}
    assert result["push"] == {"pushed": 2, "cursor": 30}
    assert result["last_cycle_at"] and result["last_sync_at"]
    # Snapshots cannot mutate the scheduler's nested state.
    result["pull"]["applied"] = -1
    assert scheduler.get_delta_sync_status()["peers"][0]["pull"]["applied"] == 3

    push.side_effect = RuntimeError("sensitive transport error")
    await scheduler._run_sync_peer_cycle(None, "hub", peer, pull, push)
    failed = scheduler.get_delta_sync_status()["peers"][0]
    assert failed["health"] == "error"
    assert failed["pull"]["applied"] == 3
    assert failed["push"] is None
    assert failed["last_sync_at"] == result["last_sync_at"]
    assert failed["last_error"] and "sensitive" not in failed["last_error"]
    push.side_effect = None
    await scheduler._run_sync_peer_cycle(None, "hub", peer, pull, push)
    assert scheduler.get_delta_sync_status()["peers"][0]["last_error"] is None


def test_sqlite_status_api_with_and_without_scheduler(scheduler, monkeypatch):
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_storage] = lambda: object()
    app.dependency_overrides[require_read_access] = lambda: None
    for current in (None, scheduler):
        monkeypatch.setattr(api_client, "get_scheduler", lambda current=current: current)
        with TestClient(app) as client:
            response = client.get("/api/sync/status")
        assert response.status_code == 200
        status = response.json()
        assert status["is_hybrid"] is False
        assert status["delta_sync"]["configured"] is True
        assert status["delta_sync"]["peers"][0]["peer_id"] == "hub"
        assert "secret" not in response.text
