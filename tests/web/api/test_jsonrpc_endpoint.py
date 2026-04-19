# Copyright 2024 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for JSON-RPC handling on the /mcp endpoint."""

import pytest
import pytest_asyncio
import tempfile
import os
from fastapi.testclient import TestClient

from mcp_memory_service.web.dependencies import set_storage
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, "test.db")


@pytest_asyncio.fixture
async def initialized_storage(temp_db, monkeypatch):
    monkeypatch.setenv("MCP_SEMANTIC_DEDUP_ENABLED", "false")
    storage = SqliteVecMemoryStorage(temp_db)
    await storage.initialize()
    yield storage
    await storage.close()


@pytest.fixture
def test_app(initialized_storage, monkeypatch):
    # Patch module-level auth config directly instead of reloading modules.
    # Reloading (importlib.reload) creates new function objects, which
    # desynchronizes FastAPI dependency_overrides keys from the references
    # the app was initialized with, and causes cross-test state pollution.
    from mcp_memory_service.web.oauth import middleware
    monkeypatch.setattr(middleware, "API_KEY", None)
    monkeypatch.setattr(middleware, "OAUTH_ENABLED", False)
    monkeypatch.setattr(middleware, "ALLOW_ANONYMOUS_ACCESS", True)

    from mcp_memory_service.web.app import app
    from mcp_memory_service.web.oauth.middleware import (
        get_current_user, require_read_access, AuthenticationResult,
    )

    set_storage(initialized_storage)

    async def mock_user():
        return AuthenticationResult(
            authenticated=True, client_id="test", scope="read write admin", auth_method="test",
        )

    app.dependency_overrides[get_current_user] = mock_user
    app.dependency_overrides[require_read_access] = mock_user

    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


@pytest.mark.integration
def test_initialized_notification_returns_202_with_empty_body(test_app):
    """
    JSON-RPC 2.0 requires servers never respond to notifications (messages
    without `id`). MCP Streamable HTTP further requires HTTP 202 Accepted
    with no body in this case. Regression for:
    clients like Codex's rmcp that treat a JSON-RPC error response to
    `notifications/initialized` as a handshake failure.
    """
    response = test_app.post(
        "/mcp",
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
    )

    assert response.status_code == 202, (
        f"Notifications must get 202 Accepted, got {response.status_code} "
        f"with body: {response.text!r}"
    )
    assert response.content == b"", (
        f"Notifications must get empty body, got: {response.text!r}"
    )


@pytest.mark.integration
def test_initialize_request_still_returns_200_with_result(test_app):
    """Sanity check: regular requests (with `id`) continue to work."""
    response = test_app.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"},
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["jsonrpc"] == "2.0"
    assert body["id"] == 1
    assert "result" in body
    assert "protocolVersion" in body["result"]
