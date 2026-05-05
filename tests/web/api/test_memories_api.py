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
Tests for Memory CRUD API endpoints.

Tests the store_memory endpoint for:
- X-Agent-ID header auto-tagging (agent:<id> appended to tags)
"""

import pytest
import pytest_asyncio
import tempfile
import os
from fastapi.testclient import TestClient

from mcp_memory_service.web.dependencies import set_storage
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_memories.db")
        yield db_path


@pytest_asyncio.fixture
async def initialized_storage(temp_db, monkeypatch):
    """Create and initialize a real SQLite storage backend."""
    monkeypatch.setenv('MCP_SEMANTIC_DEDUP_ENABLED', 'false')

    storage = SqliteVecMemoryStorage(temp_db)
    await storage.initialize()
    yield storage
    await storage.close()


@pytest.fixture
def test_app(initialized_storage, monkeypatch):
    """Create a FastAPI test application with initialized storage."""
    # Auth is bypassed via `app.dependency_overrides` below — we deliberately
    # do NOT `importlib.reload` the middleware module here. Reloading rebinds
    # the dependency functions to fresh objects while the FastAPI route graph
    # still holds the *original* references captured at app-import time;
    # subsequent tests (e.g. `test_harvest_api::test_harvest_requires_auth`)
    # then register overrides keyed by the post-reload objects, miss the
    # route-captured originals, and fall through to the live middleware,
    # which under the CI-wide `MCP_ALLOW_ANONYMOUS_ACCESS=true` silently
    # returns 200 instead of 401. See PR #844 / issue #843 follow-up.
    monkeypatch.setenv('MCP_API_KEY', '')
    monkeypatch.setenv('MCP_OAUTH_ENABLED', 'false')
    monkeypatch.setenv('MCP_ALLOW_ANONYMOUS_ACCESS', 'true')
    monkeypatch.setenv('INCLUDE_HOSTNAME', 'false')

    from mcp_memory_service.web.app import app
    from mcp_memory_service.web.oauth.middleware import (
        get_current_user, require_write_access, require_read_access,
        AuthenticationResult
    )

    set_storage(initialized_storage)

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

    client = TestClient(app)
    yield client

    app.dependency_overrides.clear()


@pytest.mark.integration
def test_store_memory_with_agent_id_header_appends_agent_tag(test_app):
    """X-Agent-ID header auto-appends agent:<id> tag to stored memory."""
    response = test_app.post(
        "/api/memories",
        json={
            "content": "Researcher found that the API rate limit is 100 req/min",
            "tags": ["api", "rate-limit"],
        },
        headers={"X-Agent-ID": "researcher"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True

    stored_tags = data["memory"]["tags"]
    assert "agent:researcher" in stored_tags
    assert "api" in stored_tags
    assert "rate-limit" in stored_tags


@pytest.mark.integration
def test_store_memory_without_agent_id_header_no_agent_tag(test_app):
    """Without X-Agent-ID header, no agent: tag is added."""
    response = test_app.post(
        "/api/memories",
        json={
            "content": "Regular memory without agent context",
            "tags": ["general"],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True

    stored_tags = data["memory"]["tags"]
    assert not any(tag.startswith("agent:") for tag in stored_tags)


@pytest.mark.integration
def test_store_memory_agent_id_not_duplicated_if_already_in_tags(test_app):
    """X-Agent-ID header does not duplicate an agent: tag already in tags."""
    response = test_app.post(
        "/api/memories",
        json={
            "content": "Memory with pre-existing agent tag",
            "tags": ["agent:researcher", "other"],
        },
        headers={"X-Agent-ID": "researcher"},
    )

    assert response.status_code == 200
    data = response.json()
    stored_tags = data["memory"]["tags"]
    assert stored_tags.count("agent:researcher") == 1
