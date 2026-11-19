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

"""
Tests for GET /api/search/similar/{content_hash} endpoint.
Verifies that target memories are looked up directly by hash rather than through
vector search, preventing false 404 errors in multi-memory databases.
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
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_search_similar.db")
        yield db_path


@pytest_asyncio.fixture
async def initialized_storage(temp_db, monkeypatch):
    """Create and initialize a real SQLite storage backend."""
    monkeypatch.setenv("MCP_SEMANTIC_DEDUP_ENABLED", "false")
    monkeypatch.setenv("MCP_MEMORY_ALLOW_HASH_EMBEDDINGS", "1")
    monkeypatch.setenv("MCP_MEMORY_ONNX_ALLOW_DOWNLOAD", "0")

    storage = SqliteVecMemoryStorage(temp_db)
    await storage.initialize()
    yield storage
    await storage.close()


@pytest.fixture
def test_app(initialized_storage, monkeypatch):
    """Create a FastAPI test application with initialized storage."""
    monkeypatch.setenv("MCP_API_KEY", "")
    monkeypatch.setenv("MCP_OAUTH_ENABLED", "false")
    monkeypatch.setenv("MCP_ALLOW_ANONYMOUS_ACCESS", "true")
    monkeypatch.setenv("INCLUDE_HOSTNAME", "false")

    from mcp_memory_service.web.app import app
    from mcp_memory_service.web.oauth.middleware import (
        get_current_user,
        require_write_access,
        require_read_access,
        require_store_read,
        AuthenticationResult,
    )

    set_storage(initialized_storage)

    async def mock_get_current_user():
        return AuthenticationResult(
            authenticated=True,
            client_id="test_client",
            scope="read write admin",
            auth_method="test",
        )

    app.dependency_overrides[get_current_user] = mock_get_current_user
    app.dependency_overrides[require_write_access] = mock_get_current_user
    app.dependency_overrides[require_read_access] = mock_get_current_user
    app.dependency_overrides[require_store_read] = mock_get_current_user

    client = TestClient(app)
    yield client

    app.dependency_overrides.clear()


@pytest.mark.integration
def test_search_similar_with_multiple_memories(test_app):
    """GET /api/search/similar/{content_hash} succeeds when multiple memories exist in the DB."""
    # Seed 3 distinct memories
    res1 = test_app.post(
        "/api/memories",
        json={"content": "Alpha service handles authentication and token refresh.", "tags": ["auth"]},
    )
    assert res1.status_code == 200
    hash1 = res1.json()["content_hash"]

    res2 = test_app.post(
        "/api/memories",
        json={"content": "Beta service validates session tokens and user permissions.", "tags": ["auth", "security"]},
    )
    assert res2.status_code == 200
    hash2 = res2.json()["content_hash"]

    res3 = test_app.post(
        "/api/memories",
        json={"content": "Gamma database stores user preferences and configuration.", "tags": ["storage"]},
    )
    assert res3.status_code == 200
    hash3 = res3.json()["content_hash"]

    # Call similar search for hash1
    response = test_app.get(f"/api/search/similar/{hash1}")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    data = response.json()
    assert data["search_type"] == "similar"
    assert len(data["results"]) == 2
    # Ensure the target memory itself is filtered out and the remaining memories are returned
    result_hashes = [r["memory"]["content_hash"] for r in data["results"]]
    assert hash1 not in result_hashes
    assert hash2 in result_hashes
    assert hash3 in result_hashes


@pytest.mark.integration
def test_search_similar_returns_404_for_unknown_hash(test_app):
    """GET /api/search/similar/{content_hash} returns 404 for nonexistent memory hash."""
    fake_hash = "0" * 64
    response = test_app.get(f"/api/search/similar/{fake_hash}")
    assert response.status_code == 404
    assert response.json()["detail"] == "Memory not found"
