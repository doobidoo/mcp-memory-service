"""MCP transport consistency: the /mcp shim must not serve a weaker protocol
than stdio.

Pins three lists that used to drift independently (the memory_graph schema
enum, the handler's accepted actions and the write-scope exemption), and
checks the HTTP shim carries tool annotations and answers ``ping``.
"""

import pytest
import pytest_asyncio
import tempfile
import os
from fastapi.testclient import TestClient

from mcp_memory_service.web.dependencies import set_storage
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.tools.registry import TOOL_REGISTRY
from mcp_memory_service.server.handlers.graph import (
    GRAPH_ACTIONS,
    GRAPH_READ_ACTIONS,
)


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
            authenticated=True, client_id="test", scope="read write admin", auth_method="test",
        )

    app.dependency_overrides[get_current_user] = mock_user
    app.dependency_overrides[require_read_access] = mock_user
    yield app
    app.dependency_overrides.clear()


def _graph_tool_def():
    return next(td for td in TOOL_REGISTRY if td.name == "memory_graph")


def test_graph_schema_enum_lists_every_handler_action():
    schema = _graph_tool_def().input_schema
    enum = schema["properties"]["action"]["enum"]
    assert enum == GRAPH_ACTIONS
    # the advertised description covers every action too
    description = _graph_tool_def().description
    for action in GRAPH_ACTIONS:
        assert action in description, f"{action} missing from the tool description"


def test_read_actions_exclude_only_entity_extraction():
    # extract_entities stores entity links (GHSA-2mw6-h564-86wv)
    assert set(GRAPH_READ_ACTIONS) == set(GRAPH_ACTIONS) - {"extract_entities"}


@pytest.mark.asyncio
async def test_write_scope_follows_the_action_table(initialized_storage):
    from mcp_memory_service.server_impl import MemoryServer

    server = MemoryServer(
        storage=initialized_storage, consolidator=None, consolidation_scheduler=None
    )
    for action in GRAPH_READ_ACTIONS:
        requires_write = await server.tool_requires_write(
            "memory_graph", {"action": action}
        )
        assert not requires_write, f"{action} is read-only but required write scope"

    assert await server.tool_requires_write(
        "memory_graph", {"action": "extract_entities"}
    )
    assert await server.tool_requires_write("memory_graph", {"action": "bogus"})
    assert await server.tool_requires_write("memory_graph", None)


def test_http_tools_list_carries_annotations(test_app):
    client = TestClient(test_app)
    response = client.post(
        "/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    )
    tools = response.json()["result"]["tools"]

    assert tools, "tools/list returned no tools"
    missing = [t["name"] for t in tools if not t.get("annotations")]
    assert not missing, f"HTTP tools/list dropped annotations for: {missing}"

    by_name = {t["name"]: t for t in tools}
    assert by_name["memory_search"]["annotations"].get("readOnlyHint") is True
    assert by_name["memory_delete"]["annotations"].get("destructiveHint") is True
    assert by_name["memory_graph"]["annotations"].get("readOnlyHint") is True


def test_http_ping_answers_empty_result(test_app):
    client = TestClient(test_app)
    response = client.post(
        "/mcp", json={"jsonrpc": "2.0", "id": 7, "method": "ping"}
    )
    body = response.json()
    assert "error" not in body
    assert body["result"] == {}
