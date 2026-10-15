"""REST API tests for explicit tombstone conflict resolution."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from mcp_memory_service.web.api import sync


class ResolutionStorage:
    def __init__(self, result):
        self.result = result
        self.calls = []
        self.secondary_backend_type = "cloudflare"

    async def resolve_tombstone_conflict(self, content_hash, action):
        self.calls.append((content_hash, action))
        return self.result


def _client_for(storage):
    app = FastAPI()
    app.include_router(sync.router, prefix="/api")
    app.dependency_overrides[sync.get_storage] = lambda: storage
    app.dependency_overrides[sync.require_write_access] = lambda: None
    return TestClient(app)


def test_tombstone_conflict_endpoint_returns_resolution_result():
    storage = ResolutionStorage(
        {
            "success": True,
            "status": "resolved",
            "action": "keep_local",
            "message": "The local memory was kept and uploaded.",
        }
    )

    with _client_for(storage) as client:
        response = client.post(
            "/api/sync/tombstone-conflicts/hash-123",
            json={"action": "keep_local"},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "resolved"
    assert storage.calls == [("hash-123", "keep_local")]


def test_tombstone_conflict_endpoint_rejects_unknown_action():
    storage = ResolutionStorage({})

    with _client_for(storage) as client:
        response = client.post(
            "/api/sync/tombstone-conflicts/hash-123",
            json={"action": "overwrite"},
        )

    assert response.status_code == 422
    assert storage.calls == []


def test_tombstone_conflict_endpoint_is_unavailable_without_hybrid_storage():
    with _client_for(object()) as client:
        response = client.post(
            "/api/sync/tombstone-conflicts/hash-123",
            json={"action": "accept_delete"},
        )

    assert response.status_code == 404


def test_tombstone_conflict_endpoint_is_unavailable_for_http_secondary():
    storage = ResolutionStorage({})
    storage.secondary_backend_type = "http"

    with _client_for(storage) as client:
        response = client.post(
            "/api/sync/tombstone-conflicts/hash-123",
            json={"action": "accept_delete"},
        )

    assert response.status_code == 404
    assert storage.calls == []
