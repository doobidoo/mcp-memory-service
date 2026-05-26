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
    `MemoryServer.list_tools()` directly, minus the local-only tools that
    must not be exposed remotely. If this fails, the shim has started
    carrying its own tool definitions again."""
    import mcp_memory_service.server  # bootstrap circular import
    from mcp_memory_service.web.api.mcp import _get_memory_server

    server = _get_memory_server()
    stdio_tools = await server.list_tools()
    stdio_index = {t.name: t for t in stdio_tools}
    expected_http_names = set(stdio_index.keys()) - server.local_only_tools()

    response = test_app.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
    )
    assert response.status_code == 200
    http_tools = response.json()["result"]["tools"]
    http_names = {t["name"] for t in http_tools}

    assert http_names == expected_http_names, (
        f"HTTP and stdio tool sets diverged beyond the local_only carve-out. "
        f"HTTP-only: {http_names - expected_http_names}, "
        f"missing-from-http: {expected_http_names - http_names}"
    )

    for http_tool in http_tools:
        stdio_tool = stdio_index[http_tool["name"]]
        assert http_tool["description"] == stdio_tool.description, (
            f"Description for {http_tool['name']} differs between transports"
        )
        assert http_tool["inputSchema"] == stdio_tool.inputSchema, (
            f"inputSchema for {http_tool['name']} differs between transports"
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_local_only_tools_excluded_from_http_list(test_app):
    """Local-only tools (memory_harvest, memory_ingest) must not appear
    in HTTP `tools/list` — they read caller-supplied filesystem paths and
    are stdio-only by policy."""
    import mcp_memory_service.server  # noqa: F401
    from mcp_memory_service.web.api.mcp import _get_memory_server

    server = _get_memory_server()
    local_only = server.local_only_tools()
    assert local_only, "Test premise broken: nothing flagged local-only"

    response = test_app.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
    )
    assert response.status_code == 200
    http_names = {t["name"] for t in response.json()["result"]["tools"]}
    leaked = http_names & local_only
    assert not leaked, f"local-only tools leaked into HTTP tools/list: {leaked}"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_local_only_tools_rejected_on_http_call(test_app):
    """Calling a local-only tool over HTTP — or via one of its deprecated
    aliases — must return JSON-RPC -32601 (method not found). The
    handler must not run."""
    import mcp_memory_service.server  # noqa: F401
    from mcp_memory_service.web.api.mcp import _get_memory_server
    from mcp_memory_service.compat import DEPRECATED_TOOLS

    server = _get_memory_server()
    local_only = server.local_only_tools()

    targets = list(local_only)
    targets += [
        old for old, (new, _) in DEPRECATED_TOOLS.items() if new in local_only
    ]
    assert targets, "Test premise broken: no local-only names to check"

    for name in targets:
        response = test_app.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": name, "arguments": {}},
            },
        )
        assert response.status_code == 200, f"{name}: unexpected HTTP status"
        body = response.json()
        assert "error" in body, f"{name}: expected error, got {body}"
        assert body["error"]["code"] == -32601, (
            f"{name}: expected -32601 method-not-found, got {body['error']}"
        )
