"""Rate limiting on the /mcp FastAPI shim path (#1096).

The shim calls ``server.call_tool()`` outside any SDK request context, so
it resolves the caller identity from its own HTTP request (``X-Agent-ID``
header, read before the arguments rewrite) and passes it explicitly.
"""

import pytest
from fastapi.testclient import TestClient
from mcp import types

from mcp_memory_service.server import MemoryServer
import mcp_memory_service.web.api.mcp as mcp_shim


async def _echo_handler(_server, arguments):
    return [types.TextContent(type="text", text="ok")]


@pytest.fixture
def shim_client(monkeypatch):
    monkeypatch.setattr(
        "mcp_memory_service.tools.routing.resolve_handler",
        lambda _name: _echo_handler,
    )
    server = MemoryServer()
    monkeypatch.setattr(mcp_shim, "_memory_server", server)

    from mcp_memory_service.web.oauth import middleware

    monkeypatch.setattr(middleware, "API_KEY", None)
    monkeypatch.setattr(middleware, "OAUTH_ENABLED", False)
    monkeypatch.setattr(middleware, "ALLOW_ANONYMOUS_ACCESS", True)

    from mcp_memory_service.web.app import app
    from mcp_memory_service.web.oauth.middleware import (
        get_current_user,
        require_read_access,
        AuthenticationResult,
    )

    async def mock_user():
        return AuthenticationResult(
            authenticated=True,
            client_id="test",
            scope="read write admin",
            auth_method="test",
        )

    app.dependency_overrides[get_current_user] = mock_user
    app.dependency_overrides[require_read_access] = mock_user
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


def _clear_env(monkeypatch):
    for var in (
        "MCP_RATE_LIMIT_PER_MINUTE",
        "MCP_RATE_LIMIT_memory_search",
        "MCP_RATE_LIMIT_MEMORY_SEARCH",
    ):
        monkeypatch.delenv(var, raising=False)


def _tools_call(client, agent_id=None, arguments=None):
    headers = {}
    if agent_id is not None:
        headers["X-Agent-ID"] = agent_id
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "memory_search", "arguments": arguments or {}},
    }
    return client.post("/mcp", json=body, headers=headers)


@pytest.mark.integration
def test_shim_enforces_per_agent_limit(shim_client, monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("MCP_RATE_LIMIT_PER_MINUTE", "2")

    for _ in range(2):
        response = _tools_call(shim_client, agent_id="agent-a")
        assert response.status_code == 200
        assert "rate_limit_exceeded" not in response.text
    response = _tools_call(shim_client, agent_id="agent-a")
    assert response.status_code == 200
    assert "rate_limit_exceeded" in response.text
    # A different agent is unaffected: windows are isolated.
    response = _tools_call(shim_client, agent_id="agent-b")
    assert "rate_limit_exceeded" not in response.text


@pytest.mark.integration
def test_shim_identity_comes_from_header_not_arguments(shim_client, monkeypatch):
    """The limiter resolves identity before the shim rewrites arguments.

    On /mcp the shim copies ``X-Agent-ID`` into ``arguments["agent_id"]``
    when the call has none; the limiter must use the header anyway, so an
    ``agent_id`` argument acting as an author filter never changes buckets.
    """
    _clear_env(monkeypatch)
    monkeypatch.setenv("MCP_RATE_LIMIT_PER_MINUTE", "1")

    response = _tools_call(
        shim_client, agent_id="agent-a", arguments={"agent_id": "author-x"}
    )
    assert "rate_limit_exceeded" not in response.text
    # Same header, different author filter -> same caller, throttled.
    response = _tools_call(
        shim_client, agent_id="agent-a", arguments={"agent_id": "author-y"}
    )
    assert "rate_limit_exceeded" in response.text
    # Different header, same author filter -> different caller, allowed.
    response = _tools_call(
        shim_client, agent_id="agent-b", arguments={"agent_id": "author-x"}
    )
    assert "rate_limit_exceeded" not in response.text


@pytest.mark.integration
def test_shim_without_header_falls_back_to_ip_bucket(shim_client, monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("MCP_RATE_LIMIT_PER_MINUTE", "1")

    for _ in range(2):
        response = _tools_call(shim_client)
    assert "rate_limit_exceeded" in response.text
