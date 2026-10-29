"""Per-store grants on the native MCP transports (SSE, Streamable HTTP).

The SDK dispatch gate must apply the same rule as the /mcp shim: a
read:<store>/write:<store> grant authorizes a call only when the tool confines
itself to that store, the call does not reach across partitions, and the
backend can prove confinement. Everything else needs the global grant.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mcp import types
from mcp.server.lowlevel.server import request_ctx
from mcp.shared.context import RequestContext

# Via the package: the direct server_impl import races the package __init__.
from mcp_memory_service.server import MemoryServer


@pytest.fixture
def server(monkeypatch):
    srv = MemoryServer()
    srv.remote_transport = True
    srv.storage = AsyncMock()
    srv.storage.get_by_hash_is_store_scoped = True
    srv.storage.supports_store_scoped_access = True
    monkeypatch.setattr(srv, "_ensure_storage_initialized", AsyncMock(return_value=srv.storage))
    srv.handled = []

    async def handler(_server, arguments):
        srv.handled.append(arguments)
        return [types.TextContent(type="text", text="handled")]

    monkeypatch.setattr("mcp_memory_service.tools.routing.resolve_handler", lambda name: handler)
    return srv


async def _call(server, name, arguments, scope):
    ctx = RequestContext(
        request_id=1, meta=None, session=None, lifespan_context=None,
        request=SimpleNamespace(scope={"mcp_auth_scope": scope}, headers={}, client=None),
    )
    token = request_ctx.set(ctx)
    try:
        handler = server.server.request_handlers[types.CallToolRequest]
        req = types.CallToolRequest(
            method="tools/call",
            params=types.CallToolRequestParams(name=name, arguments=arguments),
        )
        return (await handler(req)).root
    finally:
        request_ctx.reset(token)


@pytest.mark.asyncio
async def test_store_grant_cannot_reach_a_tool_that_ignores_store(server):
    result = await _call(server, "memory_quality",
                         {"action": "rate", "content_hash": "h", "rating": "-1", "store": "projectA"},
                         "read:projectA write:projectA")
    assert result.isError
    assert server.handled == []


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", ["include_beliefs", "fallback"])
async def test_store_grant_can_search_with_store_scoped_options(server, flag):
    result = await _call(
        server,
        "memory_search",
        {"query": "x", "store": "projectA", flag: True},
        "read:projectA",
    )
    assert not result.isError
    assert server.handled == [{"query": "x", "store": "projectA", flag: True}]


@pytest.mark.asyncio
@pytest.mark.parametrize("tool, arguments", [
    ("memory_explore", {"query": "x", "store": "projectA"}),
    ("memory_detail", {"entity_id": "e", "store": "projectA"}),
])
async def test_store_grant_can_read_the_scoped_entity_graph(server, tool, arguments):
    """Graph tools are eligible only when they explicitly name a store."""
    result = await _call(server, tool, arguments, "read:projectA")
    assert not result.isError
    assert server.handled == [arguments]


@pytest.mark.asyncio
@pytest.mark.parametrize("tool, arguments", [
    ("memory_explore", {"query": "x", "store": "projectA"}),
    ("memory_detail", {"entity_id": "e", "store": "projectA"}),
])
async def test_graph_store_grant_requires_store_scoped_backend_capability(
    server, tool, arguments
):
    server.storage.supports_store_scoped_access = False
    result = await _call(server, tool, arguments, "read:projectA")
    assert result.isError
    assert server.handled == []


@pytest.mark.asyncio
async def test_store_grant_refused_on_a_backend_that_cannot_confine(server):
    server.storage.get_by_hash_is_store_scoped = False
    result = await _call(server, "memory_delete", {"content_hash": "h", "store": "projectA"},
                         "read:projectA write:projectA")
    assert result.isError
    assert server.handled == []


@pytest.mark.asyncio
async def test_store_grant_still_works_for_a_confined_tool(server):
    result = await _call(server, "memory_delete", {"content_hash": "h", "store": "projectA"},
                         "read:projectA write:projectA")
    assert not result.isError
    assert server.handled == [{"content_hash": "h", "store": "projectA"}]


@pytest.mark.asyncio
async def test_global_grant_is_unchanged(server):
    server.storage.get_by_hash_is_store_scoped = False
    result = await _call(server, "memory_quality",
                         {"action": "rate", "content_hash": "h", "rating": "-1", "store": "projectA"},
                         "read write")
    assert not result.isError
    assert len(server.handled) == 1
