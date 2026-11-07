"""Bootstrap reads must not consume another agent's fresh-start signal."""

import sqlite3
from types import SimpleNamespace

import pytest
import pytest_asyncio
from mcp import types
from mcp.server.lowlevel.server import request_ctx
from mcp.shared.context import RequestContext

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.server import MemoryServer
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from tests.web.api.test_write_scope_enforcement import _make_client


@pytest_asyncio.fixture
async def bootstrap_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_BOOTSTRAP_ENABLED", "true")
    monkeypatch.setenv("MCP_BOOTSTRAP_FRESH_START_INTERVAL", "10")
    monkeypatch.setenv("MCP_SYNC_EVENTLOG", "true")
    monkeypatch.setenv("MCP_SEMANTIC_DEDUP_ENABLED", "false")
    db_path = str(tmp_path / "bootstrap.db")
    storage = SqliteVecMemoryStorage(db_path)
    await storage.initialize()
    counter = Memory(
        content="__session_counter_claude-code",
        content_hash="b" * 64,
        memory_type="observation",
        tags=["session-counter", "internal"],
        metadata={"counter_key": "__session_counter_claude-code", "session_count": 10},
    )
    ok, message = await storage.store(counter)
    assert ok, message
    yield storage, db_path
    await storage.close()


def _snapshot(db_path):
    with sqlite3.connect(db_path) as connection:
        return (
            connection.execute("SELECT * FROM memories ORDER BY id").fetchall(),
            connection.execute("SELECT * FROM sync_events ORDER BY rowid").fetchall(),
        )


@pytest.mark.integration
@pytest.mark.parametrize("scope,status", [("read", 403), ("read write", 200)])
def test_bootstrap_tool_requires_write_scope(bootstrap_storage, monkeypatch, scope, status):
    storage, db_path = bootstrap_storage
    client = _make_client(storage, monkeypatch, scope=scope)
    from mcp_memory_service.web.app import app

    try:
        before = _snapshot(db_path)
        response = client.post("/mcp", json={
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "get_bootstrap_profile", "arguments": {
                "agent_ids": ["claude-code"],
            }},
        })
        assert response.status_code == status, response.text
        if scope == "read":
            assert response.json()["error"]["code"] == -32003
            assert _snapshot(db_path) == before
        else:
            assert "FRESH START" in response.json()["result"]["content"][0]["text"]
            assert _snapshot(db_path) != before
    finally:
        app.dependency_overrides.clear()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["read", "read write"])
async def test_bootstrap_resource_preserves_fresh_start_signal(bootstrap_storage, scope):
    storage, db_path = bootstrap_storage
    server = MemoryServer(storage=storage)
    before = _snapshot(db_path)

    token = request_ctx.set(RequestContext(
        request_id=1, meta=None, session=None, lifespan_context=None,
        request=SimpleNamespace(scope={"mcp_auth_scope": scope}),
    ))
    try:
        handler = server.server.request_handlers[types.ReadResourceRequest]
        for _ in range(2):
            result = await handler(types.ReadResourceRequest(
                method="resources/read",
                params=types.ReadResourceRequestParams(uri="memory://agent/claude-code/bootstrap"),
            ))
            assert "FRESH START" in result.root.contents[0].text
            assert _snapshot(db_path) == before
    finally:
        request_ctx.reset(token)

    # The legitimate write-capable boot still sees and consumes the signal.
    result = await server.handle_get_bootstrap_profile({"agent_ids": ["claude-code"]})
    assert "FRESH START" in result[0].text
    assert await server._get_session_count("claude-code") == 0
