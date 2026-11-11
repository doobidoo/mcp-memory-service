"""REST API for the ontology overlay (#1897, phase 5)."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mcp_memory_service.models import ontology
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.web.api.ontology import router
from mcp_memory_service.web.dependencies import get_storage
from mcp_memory_service.web.oauth.middleware import AuthenticationResult, get_current_user


@pytest.fixture
async def storage(temp_db_path, monkeypatch):
    monkeypatch.setenv("MCP_SYNC_EVENTLOG", "on")
    monkeypatch.delenv("MCP_CUSTOM_MEMORY_TYPES", raising=False)
    monkeypatch.setattr(ontology, "_OVERLAY_POLL_SECONDS", 0)
    s = SqliteVecMemoryStorage(f"{temp_db_path}/test_1897_api.db")
    await s.initialize()
    yield s
    await s.close()
    ontology.clear_ontology_caches()


def _client(storage, scope="read write"):
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_storage] = lambda: storage
    # Pin the caller so the tests do not depend on ambient auth configuration.
    app.dependency_overrides[get_current_user] = lambda: AuthenticationResult(
        authenticated=True, client_id="test", scope=scope, auth_method="none")
    return TestClient(app)


async def test_round_trip_through_every_kind(storage):
    c = _client(storage)
    assert c.post("/api/ontology/memory-types", json={"name": "Legal"}).json() == {
        "name": "legal", "parent": None}
    assert c.post("/api/ontology/memory-types",
                  json={"name": "postmortem", "parent": "learning"}).status_code == 201
    assert c.put("/api/ontology/retention/legal", json={"days": 400}).status_code == 200
    assert c.post("/api/ontology/relationships", json={
        "name": "cites", "valid_patterns": ["research -> any"]}).status_code == 201
    assert c.put("/api/ontology/type-patterns/decision/error",
                 json={"relationship": "fixes", "confidence": 0.8}).json()["key"] == "decision → error"

    body = c.get("/api/ontology").json()
    assert {"name": "postmortem", "parent": "learning"} in body["memory_types"]
    assert {"name": "legal", "parent": None} in body["memory_types"]
    assert {"name": "observation", "parent": None} in body["memory_types"]
    assert body["retention"]["legal"] == 400
    assert body["relationships"]["cites"]["valid_patterns"] == ["research → any"]
    assert {"source": "decision", "target": "error", "relationship": "fixes",
            "confidence": 0.8} in body["type_patterns"]
    assert {(r["kind"], r["key"]) for r in body["stored"]} == {
        ("memory_type", "legal"), ("memory_type", "postmortem"), ("retention", "legal"),
        ("relationship", "cites"), ("type_pattern", "decision → error")}
    assert body["writable"] is True

    for path in ("/api/ontology/type-patterns/decision/error", "/api/ontology/relationships/cites",
                 "/api/ontology/retention/legal", "/api/ontology/memory-types/postmortem"):
        assert c.delete(path).status_code == 200
        assert c.delete(path).status_code == 404
    assert [(r["kind"], r["key"]) for r in c.get("/api/ontology").json()["stored"]] == [
        ("memory_type", "legal")]


async def test_writes_are_sync_events(storage):
    _client(storage).post("/api/ontology/memory-types", json={"name": "legal"})
    assert storage.conn.execute(
        "SELECT op FROM sync_events WHERE content_hash = 'ontology:memory_type:legal'"
    ).fetchall() == [("ontology_upsert",)]


@pytest.mark.parametrize("method, path, body, code", [
    ("post", "/api/ontology/memory-types", {"name": "observation"}, 400),   # built-in
    ("post", "/api/ontology/memory-types", {"name": "x", "parent": "nope"}, 400),
    ("post", "/api/ontology/memory-types", {"name": 5}, 422),
    ("put", "/api/ontology/retention/decision", {"days": 90}, 400),          # config has it
    ("put", "/api/ontology/retention/insight", {"days": 0}, 400),
    ("put", "/api/ontology/retention/insight", {"days": "90"}, 422),
    ("put", "/api/ontology/retention/insight", {"days": True}, 422),
    ("post", "/api/ontology/relationships", {"name": "fixes", "valid_patterns": ["any → any"]}, 400),
    ("post", "/api/ontology/relationships", {"name": "x", "valid_patterns": "any → any"}, 422),
    ("put", "/api/ontology/type-patterns/learning/error", {"relationship": "fixes", "confidence": 0.8}, 400),
    ("put", "/api/ontology/type-patterns/decision/error", {"relationship": "has_entity", "confidence": 0.8}, 400),
    ("put", "/api/ontology/type-patterns/decision/error", {"relationship": "fixes", "confidence": 1.5}, 400),
])
async def test_invalid_writes_are_refused(storage, method, path, body, code):
    assert getattr(_client(storage), method)(path, json=body).status_code == code
    assert _client(storage).get("/api/ontology").json()["stored"] == []


@pytest.mark.parametrize("scope", ["read", "read write:projectA"])
async def test_writes_need_the_global_write_scope(storage, scope):
    c = _client(storage, scope=scope)
    assert c.get("/api/ontology").status_code == 200
    assert c.post("/api/ontology/memory-types", json={"name": "legal"}).status_code == 403
    assert c.delete("/api/ontology/memory-types/legal").status_code == 403


def test_other_backends_read_but_cannot_write():
    class CloudflareLike:
        pass

    c = _client(CloudflareLike())
    body = c.get("/api/ontology").json()
    assert body["writable"] is False and body["stored"] == []
    assert "observation" in [t["name"] for t in body["memory_types"]]
    assert c.post("/api/ontology/memory-types", json={"name": "legal"}).status_code == 501
