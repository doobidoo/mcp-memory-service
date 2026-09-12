"""A retrieval plugin that returns a malformed row must not turn a REST search into a 500."""

import logging

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

PARTIAL_CONTENT = "partial row injected by retrieval plugin"
PARTIAL_HASH = generate_content_hash(PARTIAL_CONTENT)


@pytest_asyncio.fixture
async def partial_row_plugin_context(temp_db_path):
    storage = SqliteVecMemoryStorage(f"{temp_db_path}/web-plugin-partial.db")
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

    async def inject_partial_row(query, results):
        # Only the three fields a minimal plugin is likely to fill in; the
        # MCP path (handle_memory_search) accepts this shape already.
        return results + [
            {"content": PARTIAL_CONTENT, "content_hash": PARTIAL_HASH, "tags": ["plugin-injected"]}
        ]

    service._plugin_registry.ctx.on("on_retrieve", inject_partial_row)

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
    yield TestClient(app), target
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
def test_rest_search_drops_malformed_plugin_row(
    partial_row_plugin_context, caplog, method, path, payload
):
    client, target = partial_row_plugin_context
    path = path.format(content_hash=target.content_hash)

    with caplog.at_level(logging.WARNING, logger="mcp_memory_service.web.api.search"):
        response = (
            getattr(client, method)(path, json=payload)
            if payload is not None
            else getattr(client, method)(path)
        )

    assert response.status_code == 200, response.text
    hashes = [row["memory"]["content_hash"] for row in response.json()["results"]]
    assert PARTIAL_HASH not in hashes
    assert set(hashes) <= {target.content_hash}
    assert PARTIAL_HASH in caplog.text
