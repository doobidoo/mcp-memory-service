# Copyright 2024 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for POST /api/harvest endpoint (Issue #630)."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from mcp_memory_service.harvest.models import HarvestCandidate, HarvestResult


@pytest.fixture
def project_dir(tmp_path, monkeypatch):
    """Create a Claude-projects-style session directory under a redirected HOME.

    The endpoint constrains project_path to ``~/.claude/projects/`` to prevent
    path traversal (CodeQL #383). Tests therefore redirect ``Path.home()`` to
    ``tmp_path`` so the fake project dir lives inside the allowed tree.
    """
    projects_root = tmp_path / ".claude" / "projects"
    project = projects_root / "test-project"
    project.mkdir(parents=True)
    session = project / "session-abc123.jsonl"
    session.write_text(
        json.dumps({"type": "user", "message": {"content": "hello world"}}) + "\n"
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    yield project


@pytest.fixture
def authed_client(monkeypatch):
    """FastAPI TestClient with auth mocked to allow write access."""
    monkeypatch.setenv("MCP_API_KEY", "")
    monkeypatch.setenv("MCP_OAUTH_ENABLED", "false")
    monkeypatch.setenv("MCP_ALLOW_ANONYMOUS_ACCESS", "true")

    from mcp_memory_service.web.app import app
    from mcp_memory_service.web.oauth.middleware import (
        AuthenticationResult,
        get_current_user,
        require_read_access,
        require_write_access,
    )

    async def _user():
        return AuthenticationResult(
            authenticated=True, client_id="test", scope="read write", auth_method="test"
        )

    app.dependency_overrides[get_current_user] = _user
    app.dependency_overrides[require_read_access] = _user
    app.dependency_overrides[require_write_access] = _user

    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


@pytest.fixture
def unauth_client(monkeypatch):
    """TestClient with auth dependencies overridden to reject requests (401)."""
    from fastapi import HTTPException, status
    from mcp_memory_service.web.oauth import middleware
    from mcp_memory_service.web.app import app
    from mcp_memory_service.web.oauth.middleware import (
        get_current_user,
        require_read_access,
        require_write_access,
    )

    async def _reject():
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="auth required")

    app.dependency_overrides[get_current_user] = _reject
    app.dependency_overrides[require_read_access] = _reject
    app.dependency_overrides[require_write_access] = _reject

    # In CI, `MCP_ALLOW_ANONYMOUS_ACCESS=true` is exported for the entire
    # pytest run, and earlier tests (e.g. `tests/web/test_middleware.py`)
    # may have reloaded the middleware module so its route-captured
    # function objects no longer match the post-reload identifiers used
    # above. When that happens, the `dependency_overrides` registered here
    # don't intercept the request — and the route falls through to the
    # live middleware, which honours the env var and returns 200 instead
    # of the expected 401. To make this fixture robust against that
    # reload-poisoning, also force the middleware module-level constants
    # that `_reject` is meant to simulate. `monkeypatch` reverts both on
    # teardown.
    monkeypatch.setattr(middleware, "ALLOW_ANONYMOUS_ACCESS", False, raising=False)
    monkeypatch.setattr(middleware, "API_KEY", "", raising=False)
    monkeypatch.setattr(middleware, "OAUTH_ENABLED", False, raising=False)

    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


@pytest.mark.integration
def test_harvest_dry_run_happy_path(authed_client, project_dir):
    """Dry-run harvest returns candidates wrapped from SessionHarvester."""
    fake_result = HarvestResult(
        candidates=[
            HarvestCandidate(
                content="Decided to use SQLite-Vec for local storage",
                memory_type="decision",
                tags=["decision"],
                confidence=0.9,
                source_line="",
            )
        ],
        session_id="session-abc123",
        total_messages=42,
        found=1,
        by_type={"decision": 1},
    )

    with patch(
        "mcp_memory_service.web.api.harvest.SessionHarvester"
    ) as mock_cls:
        instance = mock_cls.return_value
        instance.harvest.return_value = [fake_result]

        response = authed_client.post(
            "/api/harvest",
            json={
                "sessions": 1,
                "use_llm": False,
                "dry_run": True,
                "min_confidence": 0.6,
                "project_path": project_dir.name,
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["dry_run"] is True
    assert len(data["results"]) == 1
    r0 = data["results"][0]
    assert r0["session_id"] == "session-abc123"
    assert r0["total_messages"] == 42
    assert r0["found"] == 1
    assert r0["by_type"] == {"decision": 1}
    assert r0["candidates"][0]["type"] == "decision"
    assert r0["candidates"][0]["confidence"] == 0.9


@pytest.mark.integration
def test_harvest_requires_auth(unauth_client, project_dir):
    """Without credentials the endpoint returns 401."""
    response = unauth_client.post(
        "/api/harvest",
        json={"dry_run": True, "project_path": project_dir.name},
    )
    assert response.status_code == 401


@pytest.mark.integration
def test_harvest_rejects_invalid_types(authed_client, project_dir):
    """Types not in HARVEST_TYPES are rejected with 400."""
    response = authed_client.post(
        "/api/harvest",
        json={
            "dry_run": True,
            "project_path": project_dir.name,
            "types": ["decision", "not-a-real-type"],
        },
    )
    assert response.status_code == 400
    assert "Invalid types" in response.json()["detail"]


@pytest.mark.integration
def test_harvest_rejects_negative_sessions(authed_client, project_dir):
    """Pydantic validation rejects sessions < 1."""
    response = authed_client.post(
        "/api/harvest",
        json={"sessions": -1, "dry_run": True, "project_path": project_dir.name},
    )
    assert response.status_code == 422


def test_harvest_missing_project_dir_returns_404(authed_client, tmp_path, monkeypatch):
    """Non-existent project name under the allowed tree returns 404."""
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".claude" / "projects").mkdir(parents=True)
    response = authed_client.post(
        "/api/harvest",
        json={"dry_run": True, "project_path": "does-not-exist"},
    )
    assert response.status_code == 404


def test_harvest_requires_project_path(authed_client):
    """HTTP endpoint requires explicit project_path (no CWD fallback)."""
    response = authed_client.post("/api/harvest", json={"dry_run": True})
    assert response.status_code == 400
    assert "project_path is required" in response.json()["detail"]


@pytest.mark.parametrize("bad_path", ["../evil", "/abs/path", "../../etc", "foo/../../bar"])
def test_harvest_rejects_traversal_attempts(authed_client, tmp_path, monkeypatch, bad_path):
    """Absolute paths and parent-dir traversal are rejected with 400 (CodeQL #383/#384)."""
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".claude" / "projects").mkdir(parents=True)
    response = authed_client.post(
        "/api/harvest",
        json={"dry_run": True, "project_path": bad_path},
    )
    assert response.status_code == 400
    assert "claude/projects" in response.json()["detail"]


@pytest.mark.integration
def test_harvest_propagates_harvester_failure_as_500(authed_client, project_dir):
    """Unexpected harvester exceptions surface as 500."""
    with patch(
        "mcp_memory_service.web.api.harvest.SessionHarvester"
    ) as mock_cls:
        mock_cls.return_value.harvest.side_effect = RuntimeError("boom")
        response = authed_client.post(
            "/api/harvest",
            json={"dry_run": True, "project_path": project_dir.name},
        )
    assert response.status_code == 500
    assert response.json()["detail"] == "Harvest failed"
