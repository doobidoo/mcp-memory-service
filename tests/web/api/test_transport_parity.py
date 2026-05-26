"""Regression: HTTP /mcp tool surface must match stdio's surface.

The HTTP MCP shim was forked from `server_impl.py` around v4 and drifted
out of sync until the v10 unification (commit `ea820bef`). Both transports
now delegate to `MemoryServer.list_tools()` so the surfaces stay aligned
by construction. This test exists to catch a re-introduction of the drift
mechanism (e.g. someone adding back a hardcoded tool list in
`web/api/mcp.py`).
"""

import os
import tempfile

import pytest
import pytest_asyncio
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
            authenticated=True, client_id="test", scope="read write",
            auth_method="test",
        )

    app.dependency_overrides[get_current_user] = mock_user
    app.dependency_overrides[require_read_access] = mock_user

    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_http_mcp_surface_matches_stdio(test_app):
    """`tools/list` over HTTP must advertise the same names + schemas as
    `MemoryServer.list_tools()` directly. If this fails, the shim has
    started carrying its own tool definitions again."""
    import mcp_memory_service.server  # bootstrap circular import
    from mcp_memory_service.web.api.mcp import _get_memory_server

    server = _get_memory_server()
    stdio_tools = await server.list_tools()
    stdio_index = {t.name: t for t in stdio_tools}

    response = test_app.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
    )
    assert response.status_code == 200
    http_tools = response.json()["result"]["tools"]
    http_names = {t["name"] for t in http_tools}

    assert http_names == set(stdio_index.keys()), (
        f"HTTP and stdio tool sets diverged. "
        f"HTTP-only: {http_names - set(stdio_index.keys())}, "
        f"stdio-only: {set(stdio_index.keys()) - http_names}"
    )

    for http_tool in http_tools:
        stdio_tool = stdio_index[http_tool["name"]]
        assert http_tool["description"] == stdio_tool.description, (
            f"Description for {http_tool['name']} differs between transports"
        )
        assert http_tool["inputSchema"] == stdio_tool.inputSchema, (
            f"inputSchema for {http_tool['name']} differs between transports"
        )
