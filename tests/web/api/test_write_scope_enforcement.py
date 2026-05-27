"""
Regression tests for GHSA-2r68-g678-7qr3:
OAuth read-only clients must NOT reach store_memory or delete_memory
through the /mcp tools/call endpoint.

Before the fix, mcp_endpoint used Depends(require_read_access) for the
entire endpoint but dispatched tools/call without checking per-tool scope,
allowing a read-only token to call mutating tools that the REST layer
correctly rejected with 403.
"""

import pytest
import tempfile
import os
from fastapi.testclient import TestClient

from mcp_memory_service.web.dependencies import set_storage
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
import pytest_asyncio


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


def _make_client(initialized_storage, monkeypatch, *, scope: str) -> TestClient:
    """Return a TestClient whose injected user has exactly the given scope string."""
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

    set_storage(initialized_storage)

    async def mock_user():
        return AuthenticationResult(
            authenticated=True,
            client_id="test-client",
            scope=scope,
            auth_method="test",
        )

    app.dependency_overrides[get_current_user] = mock_user
    app.dependency_overrides[require_read_access] = mock_user

    return TestClient(app)


@pytest.fixture
def read_only_client(initialized_storage, monkeypatch):
    client = _make_client(initialized_storage, monkeypatch, scope="read")
    yield client
    from mcp_memory_service.web.app import app
    app.dependency_overrides.clear()


@pytest.fixture
def read_write_client(initialized_storage, monkeypatch):
    client = _make_client(initialized_storage, monkeypatch, scope="read write")
    yield client
    from mcp_memory_service.web.app import app
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# GHSA-2r68-g678-7qr3: read-only token must be rejected for write tools
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_store_memory_rejected_with_read_only_scope(read_only_client):
    """read scope cannot call store_memory through /mcp (GHSA-2r68-g678-7qr3)."""
    response = read_only_client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "store_memory",
                "arguments": {"content": "should not be stored", "tags": ["poc"]},
            },
        },
    )
    assert response.status_code == 403
    body = response.json()
    assert "error" in body
    assert body["error"]["code"] == -32003
    assert "write" in body["error"]["message"].lower()


@pytest.mark.integration
def test_delete_memory_rejected_with_read_only_scope(read_only_client):
    """read scope cannot call delete_memory through /mcp (GHSA-2r68-g678-7qr3)."""
    response = read_only_client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "delete_memory",
                "arguments": {"content_hash": "deadbeef" * 8},
            },
        },
    )
    assert response.status_code == 403
    body = response.json()
    assert "error" in body
    assert body["error"]["code"] == -32003


@pytest.mark.integration
def test_store_memory_allowed_with_write_scope(read_write_client):
    """read+write scope can call store_memory through /mcp."""
    response = read_write_client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "store_memory",
                "arguments": {"content": "legitimate write", "tags": ["test"]},
            },
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert "result" in body
    assert "error" not in body


@pytest.mark.integration
def test_read_tools_allowed_with_read_only_scope(read_only_client):
    """read scope can still call retrieve_memory, search_by_tag, list_memories."""
    for tool, args in [
        ("retrieve_memory", {"query": "test query"}),
        ("search_by_tag", {"tags": ["test"]}),
        ("check_database_health", {}),
        ("list_memories", {}),
    ]:
        response = read_only_client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 10,
                "method": "tools/call",
                "params": {"name": tool, "arguments": args},
            },
        )
        assert response.status_code == 200, f"{tool} should be readable but got {response.status_code}"
        body = response.json()
        assert "result" in body, f"{tool} returned error: {body}"
        assert "error" not in body


# ---------------------------------------------------------------------------
# v10 tool names: the same scope semantics applied to the canonical surface
# (the pre-v10 names above route via compat.transform_deprecated_call).
# ---------------------------------------------------------------------------

V10_WRITE_TOOLS = [
    ("memory_store", {"content": "should not be stored", "metadata": {"tags": ["poc"]}}),
    ("memory_store_session", {"turns": [{"role": "user", "content": "x"}]}),
    ("memory_delete", {"content_hash": "deadbeef" * 8}),
    ("memory_cleanup", {}),
    ("memory_update", {"content_hash": "deadbeef" * 8, "updates": {"tags": ["x"]}}),
    ("memory_quality", {"action": "rate", "content_hash": "deadbeef" * 8, "rating": "1"}),
    ("memory_resolve", {"winner_hash": "a" * 64, "loser_hash": "b" * 64}),
    ("mistake_note_add", {
        "error_pattern": "p", "context_signature": "c",
        "incorrect_action": "i", "correct_action": "c",
    }),
]

V10_READ_TOOLS = [
    ("memory_search", {"query": "test"}),
    ("memory_list", {}),
    ("memory_health", {}),
    ("memory_stats", {}),
    ("memory_graph", {"action": "connected", "hash": "deadbeef" * 8}),
    ("memory_conflicts", {}),
    ("mistake_note_search", {"query": "test"}),
]


@pytest.mark.integration
@pytest.mark.parametrize("tool,args", V10_WRITE_TOOLS)
def test_v10_write_tool_rejected_with_read_only_scope(read_only_client, tool, args):
    """v10 write tools must require the OAuth 'write' scope (GHSA-2r68-g678-7qr3).

    Covers the surface exposed by the unification refactor — the pre-v10
    cases above exercise the compat-rewrite path; these exercise direct v10
    dispatch via the shared MemoryServer.
    """
    response = read_only_client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool, "arguments": args},
        },
    )
    assert response.status_code == 403, (
        f"{tool} should be blocked for read-only scope, got {response.status_code}: {response.text}"
    )
    body = response.json()
    assert body["error"]["code"] == -32003, f"{tool}: unexpected error: {body}"


@pytest.mark.integration
@pytest.mark.parametrize("tool,args", V10_READ_TOOLS)
def test_v10_read_tool_allowed_with_read_only_scope(read_only_client, tool, args):
    """v10 read tools must be callable with only the OAuth 'read' scope.

    Confirms the derive-from-`readOnlyHint` model classifies these as read.
    A failure here usually means a tool's annotation lost `readOnlyHint=True`
    (which is what made memory_conflicts / mistake_note_search over-strict
    earlier — they now carry the annotation).
    """
    response = read_only_client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool, "arguments": args},
        },
    )
    assert response.status_code == 200, (
        f"{tool} should be allowed for read-only scope, got {response.status_code}: {response.text}"
    )
    body = response.json()
    assert "result" in body, f"{tool} returned error: {body}"


@pytest.mark.integration
def test_tools_call_without_name_returns_invalid_params(read_only_client):
    """A `tools/call` with no `name` must be rejected as -32602 Invalid params.

    Before the fix it reached `MemoryServer.call_tool(None, {})` which raised
    ValueError and surfaced as HTTP 200 with an internal-error string —
    confusing for the client and (by short-circuiting the scope check on a
    falsy name) a potential bypass surface for future regressions.
    """
    response = read_only_client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"arguments": {}},
        },
    )
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == -32602, f"unexpected error: {body}"
