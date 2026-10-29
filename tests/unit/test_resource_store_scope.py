"""MCP resource reads must honor per-store grants (#1683)."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mcp import types
from mcp.server.lowlevel.server import request_ctx
from mcp.shared.context import RequestContext

from mcp_memory_service.server import MemoryServer


@pytest.fixture
def server(monkeypatch):
    srv = MemoryServer()
    srv.storage = AsyncMock()
    srv.storage.supports_store_scoped_access = True
    srv.storage.get_by_hash_is_store_scoped = True
    srv.storage.count_all_memories = AsyncMock(return_value=2)
    srv.storage.get_all_tags_with_counts = AsyncMock(return_value=[{"tag": "local", "count": 2}])
    srv.storage.get_all_tags = AsyncMock(return_value=["global-secret"])
    srv.storage.get_all_memories = AsyncMock(return_value=[])
    srv.storage.retrieve = AsyncMock(return_value=[])
    srv.storage.get_stats = AsyncMock(return_value={"total_memories": 99})
    monkeypatch.setattr(srv, "_ensure_storage_initialized", AsyncMock(return_value=srv.storage))
    return srv


async def _invoke(server, request_type, request, scope):
    context = RequestContext(
        request_id=1,
        meta=None,
        session=None,
        lifespan_context=None,
        request=SimpleNamespace(
            scope={"mcp_auth_scope": scope}, headers={}, client=None
        ),
    )
    token = request_ctx.set(context)
    try:
        handler = server.server.request_handlers[request_type]
        return await handler(request)
    finally:
        request_ctx.reset(token)


@pytest.mark.asyncio
async def test_store_scoped_stats_use_only_partition_counts(server):
    response = await _invoke(
        server,
        types.ReadResourceRequest,
        types.ReadResourceRequest(
            params=types.ReadResourceRequestParams(uri="memory://store/projectA/stats")
        ),
        "read:projectA",
    )

    payload = json.loads(response.root.contents[0].text)
    assert payload == {"store": "projectA", "total_memories": 2, "unique_tags": 1}
    server.storage.count_all_memories.assert_awaited_once_with(store="projectA")
    server.storage.get_all_tags_with_counts.assert_awaited_once_with(store="projectA")
    server.storage.get_stats.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("resource_path", "method", "expected_args", "expected_kwargs"),
    [
        ("recent/5", "get_all_memories", (), {"limit": 5, "store": "projectA"}),
        (
            "tag/local",
            "get_all_memories",
            (),
            {"tags": ["local"], "store": "projectA"},
        ),
        (
            "search/needle",
            "retrieve",
            ("needle",),
            {"n_results": 10, "store": "projectA"},
        ),
    ],
)
async def test_store_scoped_reads_pass_partition_to_storage(
    server, resource_path, method, expected_args, expected_kwargs
):
    await _invoke(
        server,
        types.ReadResourceRequest,
        types.ReadResourceRequest(
            params=types.ReadResourceRequestParams(
                uri=f"memory://store/projectA/{resource_path}"
            )
        ),
        "read:projectA",
    )

    if method == "retrieve":
        server.storage.retrieve.assert_awaited_once_with(
            *expected_args, **expected_kwargs
        )
    else:
        server.storage.get_all_memories.assert_awaited_once_with(
            *expected_args, **expected_kwargs
        )


@pytest.mark.asyncio
async def test_store_scoped_grant_cannot_read_global_or_other_store_uri(server):
    with pytest.raises(PermissionError, match="store-scoped resources require"):
        await _invoke(
            server,
            types.ReadResourceRequest,
            types.ReadResourceRequest(
                params=types.ReadResourceRequestParams(uri="memory://stats")
            ),
            "read:projectA",
        )

    with pytest.raises(PermissionError, match="not covered"):
        await _invoke(
            server,
            types.ReadResourceRequest,
            types.ReadResourceRequest(
                params=types.ReadResourceRequestParams(uri="memory://store/projectB/stats")
            ),
            "read:projectA",
        )

    server.storage.get_stats.assert_not_awaited()


@pytest.mark.asyncio
async def test_store_scoped_resource_listing_uses_partition_tags(server):
    response = await _invoke(
        server,
        types.ListResourcesRequest,
        types.ListResourcesRequest(),
        "read:projectA",
    )

    uris = {str(resource.uri) for resource in response.root.resources}
    assert "memory://store/projectA/stats" in uris
    assert "memory://store/projectA/tag/local" in uris
    assert "memory://tag/global-secret" not in uris
    server.storage.get_all_tags_with_counts.assert_awaited_once_with(store="projectA")
    server.storage.get_all_tags.assert_not_awaited()


@pytest.mark.asyncio
async def test_scoped_listing_fails_closed_when_backend_cannot_confine(server):
    server.storage.supports_store_scoped_access = False
    response = await _invoke(
        server,
        types.ListResourcesRequest,
        types.ListResourcesRequest(),
        "read:projectA",
    )

    uris = {str(resource.uri) for resource in response.root.resources}
    assert uris == {"memory://onboarding/generic"}
    server.storage.get_all_tags_with_counts.assert_not_awaited()
