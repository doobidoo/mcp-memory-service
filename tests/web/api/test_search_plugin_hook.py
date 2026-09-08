# Copyright 2024 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wiring tests: REST search endpoints route through on_retrieve plugin hook.

Each test registers a recording plugin via MemoryService, seeds a memory,
calls one search endpoint, and asserts the plugin fired once with the final
result set.
"""

import pytest
import pytest_asyncio
import tempfile
import os
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from mcp_memory_service.web.dependencies import set_storage
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.models.memory import Memory


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, "test_search_plugin.db")


@pytest_asyncio.fixture
async def initialized_storage(temp_db, monkeypatch):
    monkeypatch.setenv('MCP_SEMANTIC_DEDUP_ENABLED', 'false')
    storage = SqliteVecMemoryStorage(temp_db)
    await storage.initialize()
    # Seed a test memory
    memory = Memory(
        content="test memory for plugin wiring",
        content_hash="test_plugin_hash_001",
        tags=["test", "plugin"],
        memory_type="note",
        metadata={},
    )
    await storage.store(memory)
    yield storage
    await storage.close()


@pytest.fixture
def recording_plugin_calls():
    """Shared list that the recording plugin appends to."""
    return []


@pytest.fixture
def test_app(initialized_storage, recording_plugin_calls, monkeypatch):
    monkeypatch.setenv('MCP_API_KEY', '')
    monkeypatch.setenv('MCP_OAUTH_ENABLED', 'false')
    monkeypatch.setenv('MCP_ALLOW_ANONYMOUS_ACCESS', 'true')
    monkeypatch.setenv('INCLUDE_HOSTNAME', 'false')

    from mcp_memory_service.web.app import app
    from mcp_memory_service.web.oauth.middleware import (
        get_current_user, require_write_access, require_read_access,
        AuthenticationResult
    )
    from mcp_memory_service.services.memory_service import MemoryService
    from mcp_memory_service.web.dependencies import get_memory_service

    set_storage(initialized_storage)

    # Create a single MemoryService with a recording plugin
    ms = MemoryService(initialized_storage)

    async def recording_plugin(query, results, **kwargs):
        recording_plugin_calls.append({"query": query, "count": len(results)})
        return results  # pass through unchanged

    ms._plugin_registry.ctx.on("on_retrieve", recording_plugin)

    def override_memory_service():
        return ms

    async def mock_get_current_user():
        return AuthenticationResult(
            authenticated=True,
            client_id="test_client",
            scope="read write admin",
            auth_method="test"
        )

    app.dependency_overrides[get_current_user] = mock_get_current_user
    app.dependency_overrides[require_write_access] = mock_get_current_user
    app.dependency_overrides[require_read_access] = mock_get_current_user
    app.dependency_overrides[get_memory_service] = override_memory_service

    client = TestClient(app)
    yield client

    app.dependency_overrides.clear()


@pytest.mark.integration
def test_semantic_search_routes_through_plugin(test_app, recording_plugin_calls):
    """POST /api/search fires on_retrieve plugin once."""
    resp = test_app.post("/api/search", json={"query": "test memory", "n_results": 5})
    assert resp.status_code == 200
    assert len(recording_plugin_calls) == 1
    assert recording_plugin_calls[0]["count"] >= 1


@pytest.mark.integration
def test_tag_search_routes_through_plugin(test_app, recording_plugin_calls):
    """POST /api/search/by-tag fires on_retrieve plugin once."""
    resp = test_app.post("/api/search/by-tag", json={"tags": ["test"]})
    assert resp.status_code == 200
    assert len(recording_plugin_calls) == 1
    assert recording_plugin_calls[0]["count"] >= 1


@pytest.mark.integration
def test_time_search_routes_through_plugin(test_app, recording_plugin_calls):
    """POST /api/search/by-time fires on_retrieve plugin once."""
    resp = test_app.post("/api/search/by-time", json={"query": "today", "n_results": 5})
    assert resp.status_code == 200
    assert len(recording_plugin_calls) == 1


@pytest.mark.integration
def test_similar_search_routes_through_plugin(test_app, recording_plugin_calls):
    """GET /api/search/similar/{hash} fires on_retrieve plugin once."""
    resp = test_app.get("/api/search/similar/test_plugin_hash_001?n_results=5")
    assert resp.status_code == 200
    assert len(recording_plugin_calls) == 1
    # count may be 0 when embeddings are unavailable (hash fallback);
    # the important assertion is that the plugin was called at all.