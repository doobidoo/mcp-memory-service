"""Regression tests for retrieval plugins on the REST search surface."""

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.services.memory_service import MemoryService
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash
from mcp_memory_service.web.dependencies import get_memory_service, set_storage
from mcp_memory_service.web.oauth.middleware import (
    AuthenticationResult,
    get_current_user,
    require_read_access,
)


@pytest_asyncio.fixture
async def web_plugin_context(temp_db_path):
    storage = SqliteVecMemoryStorage(f"{temp_db_path}/web-plugin.db")
    await storage.initialize()
    target_content = "REST retrieval plugin target"
    target = Memory(
        content=target_content,
        content_hash=generate_content_hash(target_content),
        tags=["plugin-test"],
        memory_type="note",
    )
    await storage.store(target)

    service = MemoryService(storage)
    calls = []
    injected_content = "row injected by retrieval plugin"

    async def inject_row(query, results):
        calls.append((query, list(results)))
        return results + [
            {
                "content": injected_content,
                "content_hash": generate_content_hash(injected_content),
                "tags": ["plugin-injected"],
                "memory_type": "note",
                "metadata": {},
                "created_at": 1.0,
                "created_at_iso": "1970-01-01T00:00:01Z",
                "updated_at": 1.0,
                "updated_at_iso": "1970-01-01T00:00:01Z",
                "similarity_score": 1.0,
                "relevance_reason": "Injected by test plugin",
            }
        ]

    service._plugin_registry.ctx.on("on_retrieve", inject_row)

    from mcp_memory_service.web.app import app

    async def mock_user():
        return AuthenticationResult(
            authenticated=True,
            client_id="test",
            scope="read write",
            auth_method="test",
        )

    set_storage(storage)
    app.dependency_overrides[get_memory_service] = lambda: service
    app.dependency_overrides[get_current_user] = mock_user
    app.dependency_overrides[require_read_access] = mock_user
    yield TestClient(app), target, calls, injected_content
    app.dependency_overrides.clear()
    await storage.close()


@pytest.mark.integration
@pytest.mark.parametrize(
    ("method", "path", "payload"),
    [
        ("post", "/api/search", {"query": "REST retrieval", "n_results": 5}),
        ("post", "/api/search/by-tag", {"tags": ["plugin-test"]}),
        ("post", "/api/search/by-time", {"query": "today", "n_results": 5}),
        ("get", "/api/search/similar/{content_hash}?n_results=5", None),
    ],
)
def test_rest_search_fires_plugin_once_and_returns_injected_row(
    web_plugin_context, method, path, payload
):
    client, target, calls, injected_content = web_plugin_context
    path = path.format(content_hash=target.content_hash)

    response = (
        getattr(client, method)(path, json=payload)
        if payload is not None
        else getattr(client, method)(path)
    )

    assert response.status_code == 200, response.text
    assert len(calls) == 1
    assert all("content" in row for row in calls[0][1])
    assert response.json()["results"][-1]["memory"]["content"] == injected_content


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("drop_newest", [False, True])
async def test_tag_search_limit_keeps_newest_matches_when_plugin_reorders(temp_db_path, drop_newest):
    """limit keeps the newest rows even if a plugin re-sorts, adds or drops rows (#1324)."""
    storage = SqliteVecMemoryStorage(f"{temp_db_path}/web-plugin-limit.db")
    await storage.initialize()
    hashes = []
    for index in range(3):
        content = f"Scoped plugin memory {index}"
        memory = Memory(
            content=content,
            content_hash=generate_content_hash(content),
            tags=["plugin-limit"],
            memory_type="note",
            created_at=1_700_000_000.0 + index,
        )
        await storage.store(memory)
        hashes.append(memory.content_hash)

    service = MemoryService(storage)
    injected_content = "row injected by reordering plugin"

    async def reverse_and_inject(query, results):
        # Oldest first, the way a score-based re-sort can end up, plus an extra row,
        # and optionally without the newest match.
        if drop_newest:
            results = [row for row in results if row["content_hash"] != hashes[2]]
        return list(reversed(results)) + [
            {
                "content": injected_content,
                "content_hash": generate_content_hash(injected_content),
                "tags": ["plugin-injected"],
                "memory_type": "note",
                "metadata": {},
                "created_at": 1.0,
                "created_at_iso": "1970-01-01T00:00:01Z",
                "updated_at": 1.0,
                "updated_at_iso": "1970-01-01T00:00:01Z",
                "similarity_score": 1.0,
                "relevance_reason": "Injected by test plugin",
            }
        ]

    service._plugin_registry.ctx.on("on_retrieve", reverse_and_inject)

    from mcp_memory_service.web.app import app

    async def mock_user():
        return AuthenticationResult(
            authenticated=True,
            client_id="test",
            scope="read write",
            auth_method="test",
        )

    set_storage(storage)
    app.dependency_overrides[get_memory_service] = lambda: service
    app.dependency_overrides[get_current_user] = mock_user
    app.dependency_overrides[require_read_access] = mock_user
    try:
        response = TestClient(app).post(
            "/api/search/by-tag", json={"tags": ["plugin-limit"], "limit": 2}
        )
        assert response.status_code == 200, response.text
        data = response.json()
        returned = [row["memory"]["content_hash"] for row in data["results"]]
        if drop_newest:
            # The page is still full: the next newest match takes the dropped one's place.
            assert returned == [hashes[1], hashes[0]]
            assert data["total_found"] == 3
        else:
            assert returned == [hashes[2], hashes[1]]
            # The two remaining matches and the injected row.
            assert data["total_found"] == 4
    finally:
        app.dependency_overrides.clear()
        await storage.close()


@pytest.mark.asyncio
async def test_no_plugin_preserves_http_result_rows():
    from mcp_memory_service.plugins.context import PluginContext
    from mcp_memory_service.plugins.registry import PluginRegistry
    from mcp_memory_service.web.api.memories import MemoryResponse
    from mcp_memory_service.web.api.search import (
        SearchResult,
        _apply_retrieve_plugins,
    )

    service = MemoryService.__new__(MemoryService)
    service._plugin_registry = PluginRegistry(
        PluginContext(storage=None, service=service)
    )
    original = SearchResult(
        memory=MemoryResponse(
            content="unchanged",
            content_hash="hash",
            tags=["test"],
            memory_type="note",
            metadata={"source": "test"},
            created_at=1.0,
            created_at_iso="1970-01-01T00:00:01Z",
            updated_at=2.0,
            updated_at_iso="1970-01-01T00:00:02Z",
        ),
        similarity_score=0.75,
        relevance_reason="baseline",
    )

    assert await _apply_retrieve_plugins(service, "query", [original]) == [original]
