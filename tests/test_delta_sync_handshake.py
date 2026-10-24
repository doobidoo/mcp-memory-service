"""Capability-handshake tests for scheduled delta sync (#1505).

Exercise the real SQLite-vec storage and orchestrator. Only the remote HTTP transport
is simulated, so these tests fail if the scheduler bypasses the compatibility check.
"""

import os
import time

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mcp_memory_service.consolidation.scheduler import ConsolidationScheduler
from mcp_memory_service.storage.remote_http import RemoteHTTPStorage
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.storage.sync.apply import KNOWN_SCHEMA_VERSION
from mcp_memory_service.web.api.health import router as health_router
from mcp_memory_service.web.api.sync import router as sync_router
from mcp_memory_service.web.dependencies import get_storage
from mcp_memory_service.web.oauth.middleware import require_read_access


class _Consolidator:
    def __init__(self, storage):
        self.storage = storage
        self.health_monitor = None


@pytest_asyncio.fixture
async def local_storage(temp_db_path, monkeypatch):
    monkeypatch.setenv("MCP_AGENT_ID", "handshake-test")
    monkeypatch.setenv("MCP_SYNC_EVENTLOG", "true")
    monkeypatch.setenv("MCP_SEMANTIC_DEDUP_ENABLED", "false")
    storage = SqliteVecMemoryStorage(db_path=os.path.join(temp_db_path, "handshake.db"))
    await storage.initialize()
    yield storage
    await storage.close()


def _capabilities(storage, **overrides):
    data = {
        "sync_protocol": 2,
        "sync_protocol_min": 1,
        "event_schema_version": KNOWN_SCHEMA_VERSION,
        "embedding_model": storage.embedding_model_name,
        "embedding_dimension": storage.embedding_dimension,
        "version": "11.14.0",
    }
    data.update(overrides)
    return data


def _delete_event():
    content_hash = "a" * 64
    return {
        "seq": 1,
        "schema_version": KNOWN_SCHEMA_VERSION,
        "event_id": "peer-delete-1",
        "op": "delete",
        "content_hash": content_hash,
        "agent_id": "peer-agent",
        "hlc_physical": 1,
        "hlc_logical": 0,
        "embedding_model": None,
        "embedding_dim": None,
        "payload": {"content_hash": content_hash, "deleted_at": time.time()},
    }


async def _remote_peer(capabilities, *, capability_status=200):
    requested = []
    event = _delete_event()

    def handler(request):
        requested.append((request.method, request.url.path))
        if request.url.path == "/api/sync/capabilities":
            return httpx.Response(capability_status, json=capabilities, request=request)
        if request.url.path == "/api/sync/baseline":
            return httpx.Response(
                200, json={"events": [], "watermark": 0}, request=request
            )
        if request.url.path == "/api/sync/events" and request.method == "GET":
            return httpx.Response(
                200,
                json={"events": [event], "next_seq": 1, "has_more": False},
                request=request,
            )
        if request.url.path == "/api/sync/events" and request.method == "POST":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"event_id": event["event_id"], "status": "skipped_duplicate"}
                    ],
                    "applied": 0,
                    "skipped": 1,
                    "failed": 0,
                },
                request=request,
            )
        return httpx.Response(404, json={"detail": "not found"}, request=request)

    peer = RemoteHTTPStorage("https://peer.test")
    await peer.client.aclose()
    peer.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return peer, requested


def _scheduler(storage, peer):
    scheduler = ConsolidationScheduler(
        _Consolidator(storage), schedule_config={}, enabled=False
    )
    scheduler._resolve_sync_peers = lambda: [("peer-test", peer)]
    return scheduler


@pytest.mark.asyncio
async def test_overlapping_protocol_ranges_allow_real_delta_sync(local_storage):
    peer, requested = await _remote_peer(_capabilities(local_storage))
    scheduler = _scheduler(local_storage, peer)

    await scheduler._run_sync_cycle()

    assert ("GET", "/api/sync/events") in requested
    assert scheduler.get_sync_peer_status()[0]["status"] == "compatible"
    assert scheduler.get_sync_peer_status()[0]["negotiated_sync_protocol"] == 1
    row = local_storage.conn.execute(
        "SELECT last_seq_seen FROM sync_cursor WHERE peer_id = ?", ("peer-test",)
    ).fetchone()
    assert row[0] == 1


@pytest.mark.asyncio
async def test_startup_probe_checks_capabilities_without_syncing(local_storage):
    peer, requested = await _remote_peer(_capabilities(local_storage))
    scheduler = _scheduler(local_storage, peer)

    await scheduler._probe_sync_peers()

    assert requested == [("GET", "/api/sync/capabilities")]
    status = scheduler.get_sync_peer_status()[0]
    assert status["status"] == "compatible"
    assert status["last_sync_at"] is None


@pytest.mark.asyncio
async def test_non_overlapping_protocol_ranges_stop_before_event_requests(
    local_storage,
):
    peer, requested = await _remote_peer(
        _capabilities(local_storage, sync_protocol=2, sync_protocol_min=2)
    )
    scheduler = _scheduler(local_storage, peer)

    await scheduler._run_sync_cycle()

    assert not any(
        path in {"/api/sync/baseline", "/api/sync/events"} for _, path in requested
    )
    status = scheduler.get_sync_peer_status()[0]
    assert status["status"] == "incompatible"
    assert "no common sync protocol" in status["message"]


@pytest.mark.asyncio
async def test_legacy_peer_404_stops_with_upgrade_message(local_storage):
    peer, requested = await _remote_peer({}, capability_status=404)
    scheduler = _scheduler(local_storage, peer)

    await scheduler._run_sync_cycle()

    assert not any(
        path in {"/api/sync/baseline", "/api/sync/events"} for _, path in requested
    )
    status = scheduler.get_sync_peer_status()[0]
    assert status["status"] == "incompatible"
    assert "peer does not advertise a sync protocol" in status["message"]
    assert "upgrade it" in status["message"]


@pytest.mark.asyncio
async def test_embedding_model_mismatch_refuses_delta_sync(local_storage):
    peer, requested = await _remote_peer(
        _capabilities(local_storage, embedding_model="different-embedding-model")
    )
    scheduler = _scheduler(local_storage, peer)

    await scheduler._run_sync_cycle()

    assert not any(
        path in {"/api/sync/baseline", "/api/sync/events"} for _, path in requested
    )
    status = scheduler.get_sync_peer_status()[0]
    assert status["status"] == "incompatible"
    assert "embedding model mismatch" in status["message"].lower()


@pytest.mark.asyncio
async def test_embedding_dimension_mismatch_refuses_delta_sync(local_storage):
    peer, requested = await _remote_peer(
        _capabilities(
            local_storage, embedding_dimension=local_storage.embedding_dimension + 1
        )
    )
    scheduler = _scheduler(local_storage, peer)

    await scheduler._run_sync_cycle()

    assert not any(
        path in {"/api/sync/baseline", "/api/sync/events"} for _, path in requested
    )
    status = scheduler.get_sync_peer_status()[0]
    assert status["status"] == "incompatible"
    assert "embedding dimension mismatch" in status["message"].lower()


@pytest.mark.asyncio
async def test_event_schema_mismatch_refuses_delta_sync(local_storage):
    peer, requested = await _remote_peer(
        _capabilities(local_storage, event_schema_version=KNOWN_SCHEMA_VERSION + 1)
    )
    scheduler = _scheduler(local_storage, peer)

    await scheduler._run_sync_cycle()

    assert not any(
        path in {"/api/sync/baseline", "/api/sync/events"} for _, path in requested
    )
    status = scheduler.get_sync_peer_status()[0]
    assert status["status"] == "incompatible"
    assert "event schema mismatch" in status["message"].lower()


@pytest.mark.asyncio
async def test_capabilities_endpoint_and_sync_status_expose_versions_and_peer_result(
    local_storage, monkeypatch
):
    from mcp_memory_service.api import client as api_client

    scheduler = _scheduler(local_storage, None)
    scheduler._set_sync_peer_status(
        "peer-test",
        "compatible",
        "Peer capabilities are compatible",
        {
            "negotiated_sync_protocol": 1,
            "event_schema_version": KNOWN_SCHEMA_VERSION,
            "version": "11.14.0",
        },
    )
    monkeypatch.setattr(api_client, "get_scheduler", lambda: scheduler)
    app = FastAPI()
    app.include_router(sync_router, prefix="/api")
    app.include_router(health_router, prefix="/api")
    app.dependency_overrides[get_storage] = lambda: local_storage
    app.dependency_overrides[require_read_access] = lambda: None
    with TestClient(app) as client:
        response = client.get("/api/sync/capabilities")
        status_response = client.get("/api/sync/status")
        model_health_response = client.get("/api/health/model")

    assert response.status_code == 200
    data = response.json()
    assert data["sync_protocol"] == data["sync_protocol_min"] == 1
    assert data["event_schema_version"] == KNOWN_SCHEMA_VERSION
    assert data["embedding_model"] == local_storage.embedding_model_name
    assert data["embedding_dimension"] == local_storage.embedding_dimension
    assert data["version"]
    assert status_response.status_code == 200
    status = status_response.json()
    assert status["is_hybrid"] is False
    assert status["delta_sync_peers"][0]["peer_id"] == "peer-test"
    assert status["delta_sync_peers"][0]["negotiated_sync_protocol"] == 1
    assert model_health_response.status_code == 200
    assert model_health_response.json()["embedding_model"] == data["embedding_model"]
    assert (
        model_health_response.json()["embedding_dimension"]
        == data["embedding_dimension"]
    )
