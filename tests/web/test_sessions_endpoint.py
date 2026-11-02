"""Tests for POST /api/sessions HTTP endpoint."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mcp_memory_service.web.api.memories import router
from mcp_memory_service.web.dependencies import get_memory_service
from mcp_memory_service.web.oauth.middleware import (
    AuthenticationResult,
    get_current_user,
)


def _make_app(mock_svc):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_memory_service] = lambda: mock_svc
    # Pin an authenticated user with the global "write" scope so these tests are
    # hermetic: they must not depend on ambient auth configuration. Without the
    # override, a dev .env that sets MCP_API_KEY turns every request into 401
    # while CI (no key) passes.
    app.dependency_overrides[get_current_user] = lambda: AuthenticationResult(
        authenticated=True,
        client_id="test-contributor",
        scope="write",
        auth_method="none",
    )
    return app


def _make_memory_service(content_hash="abc123"):
    svc = MagicMock()
    svc.store_memory = AsyncMock(return_value={
        "success": True,
        "memory": {
            "content": "[user] Hello\n[assistant] Hi",
            "content_hash": content_hash,
            "tags": [f"session:test-42"],
            "memory_type": "session",
            "metadata": {},
            "created_at": 1234567890.0,
            "created_at_iso": "2024-01-01T00:00:00Z",
            "updated_at": None,
            "updated_at_iso": None,
        },
    })
    return svc



def test_store_session_success():
    svc = _make_memory_service()
    client = TestClient(_make_app(svc))

    resp = client.post("/sessions", json={
        "turns": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ],
        "session_id": "test-42",
    })

    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["session_id"] == "test-42"
    assert data["content_hash"] == "abc123"
    assert data["turn_count"] == 2


def test_store_session_autogenerates_session_id():
    svc = _make_memory_service()
    client = TestClient(_make_app(svc))

    resp = client.post("/sessions", json={
        "turns": [{"role": "user", "content": "Hello"}],
    })

    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"]  # non-empty auto-generated UUID


def test_store_session_content_has_speaker_labels():
    svc = _make_memory_service()
    client = TestClient(_make_app(svc))

    client.post("/sessions", json={
        "turns": [
            {"role": "user", "content": "What is Redis?"},
            {"role": "assistant", "content": "Redis is a key-value store."},
        ],
    })

    call_kwargs = svc.store_memory.call_args.kwargs
    content = call_kwargs["content"]
    assert "[user] What is Redis?" in content
    assert "[assistant] Redis is a key-value store." in content


def test_store_session_memory_type_is_session():
    svc = _make_memory_service()
    client = TestClient(_make_app(svc))

    client.post("/sessions", json={
        "turns": [{"role": "user", "content": "Test"}],
    })

    call_kwargs = svc.store_memory.call_args.kwargs
    assert call_kwargs["memory_type"] == "session"


def test_store_session_tags_include_session_prefix():
    svc = _make_memory_service()
    client = TestClient(_make_app(svc))

    client.post("/sessions", json={
        "turns": [{"role": "user", "content": "Test"}],
        "session_id": "my-sess",
        "tags": ["project:foo"],
    })

    call_kwargs = svc.store_memory.call_args.kwargs
    assert "session:my-sess" in call_kwargs["tags"]
    assert "project:foo" in call_kwargs["tags"]


def test_store_session_rejects_empty_turns():
    svc = _make_memory_service()
    client = TestClient(_make_app(svc))

    resp = client.post("/sessions", json={"turns": []})

    assert resp.status_code == 422
    svc.store_memory.assert_not_called()


def test_store_session_rejects_missing_turns():
    svc = _make_memory_service()
    client = TestClient(_make_app(svc))

    resp = client.post("/sessions", json={})

    assert resp.status_code == 422
    svc.store_memory.assert_not_called()
