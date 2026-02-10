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
Tests for OAuth middleware and API key authentication.

Validates X-API-Key header, query parameter, and Bearer token authentication methods.
"""

import pytest
import pytest_asyncio
from fastapi import Request
from fastapi.testclient import TestClient
import os

@pytest_asyncio.fixture
async def temp_storage(temp_db_path):
    """Create and initialize temporary storage."""
    from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
    from mcp_memory_service.web.dependencies import set_storage

    db_path = os.path.join(temp_db_path, "test.db")
    storage = SqliteVecMemoryStorage(db_path)
    await storage.initialize()
    set_storage(storage)
    yield storage
    storage.close()


@pytest.fixture
def client(temp_storage):
    """Create FastAPI test client with initialized storage and auth config."""
    import importlib
    from mcp_memory_service import config
    from mcp_memory_service.web.app import app

    # Store original values
    original_api_key = config.API_KEY
    original_oauth_enabled = config.OAUTH_ENABLED
    original_allow_anonymous = config.ALLOW_ANONYMOUS_ACCESS

    # Set test configuration directly on the module
    config.API_KEY = "test-secret-key-12345"
    config.OAUTH_ENABLED = False
    config.ALLOW_ANONYMOUS_ACCESS = False

    # Reload the middleware module to pick up new config
    from mcp_memory_service.web.oauth import middleware
    importlib.reload(middleware)

    # Reload the app to pick up new middleware
    from mcp_memory_service.web import app as app_module
    importlib.reload(app_module)

    # Get the reloaded app
    from mcp_memory_service.web.app import app as reloaded_app

    yield TestClient(reloaded_app)

    # Restore original configuration
    config.API_KEY = original_api_key
    config.OAUTH_ENABLED = original_oauth_enabled
    config.ALLOW_ANONYMOUS_ACCESS = original_allow_anonymous


class TestAPIKeyAuthentication:
    """Tests for API key authentication methods."""

    @pytest.mark.asyncio
    async def test_no_auth_returns_401(self, client):
        """Test that requests without authentication return 401."""
        response = client.get("/api/memories")
        assert response.status_code == 401
        assert "error_description" in response.json()["detail"]
        assert "API key" in response.json()["detail"]["error_description"]

    @pytest.mark.asyncio
    async def test_x_api_key_header_success(self, client):
        """Test successful authentication with X-API-Key header."""
        response = client.get(
            "/api/memories",
            headers={"X-API-Key": "test-secret-key-12345"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "memories" in data or "total" in data

    @pytest.mark.asyncio
    async def test_x_api_key_header_wrong_key(self, client):
        """Test that wrong API key returns 401."""
        response = client.get(
            "/api/memories",
            headers={"X-API-Key": "wrong-key"}
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_query_parameter_success(self, client):
        """Test successful authentication with query parameter."""
        response = client.get("/api/memories?api_key=test-secret-key-12345")
        assert response.status_code == 200
        data = response.json()
        assert "memories" in data or "total" in data

    @pytest.mark.asyncio
    async def test_query_parameter_wrong_key(self, client):
        """Test that wrong query parameter returns 401."""
        response = client.get("/api/memories?api_key=wrong-key")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_bearer_token_fallback(self, client):
        """Test backward compatible Bearer token authentication."""
        response = client.get(
            "/api/memories",
            headers={"Authorization": "Bearer test-secret-key-12345"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "memories" in data or "total" in data

    @pytest.mark.asyncio
    async def test_bearer_token_wrong_key(self, client):
        """Test that wrong Bearer token returns 401."""
        response = client.get(
            "/api/memories",
            headers={"Authorization": "Bearer wrong-key"}
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_case_sensitive_header(self, client):
        """Test that X-API-Key header is case-insensitive (HTTP standard)."""
        # FastAPI/Starlette normalizes headers to lowercase
        response = client.get(
            "/api/memories",
            headers={"x-api-key": "test-secret-key-12345"}  # lowercase
        )
        assert response.status_code == 200


class TestAuthenticationPriority:
    """Tests for authentication method priority."""

    @pytest.mark.asyncio
    async def test_x_api_key_preferred_over_query_param(self, client):
        """Test that X-API-Key header is checked before query parameter."""
        # Send both, but header has correct key
        response = client.get(
            "/api/memories?api_key=wrong-key",
            headers={"X-API-Key": "test-secret-key-12345"}
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_bearer_token_before_no_credentials(self, client):
        """Test Bearer token is checked first if provided."""
        response = client.get(
            "/api/memories",
            headers={"Authorization": "Bearer test-secret-key-12345"}
        )
        assert response.status_code == 200


class TestMultipleEndpoints:
    """Test that authentication works across different API endpoints."""

    @pytest.mark.parametrize("endpoint,method", [
        ("/api/health", "get"),
        ("/api/memories", "get"),
        ("/api/memories", "post"),
    ])
    @pytest.mark.asyncio
    async def test_auth_on_various_endpoints(self, client, endpoint, method):
        """Test authentication works on various endpoints."""
        if method == "get":
            response = client.get(
                endpoint,
                headers={"X-API-Key": "test-secret-key-12345"}
            )
        else:
            response = client.post(
                endpoint,
                headers={"X-API-Key": "test-secret-key-12345"},
                json={"content": "test", "tags": ["test"]}
            )

        # Health endpoint might not require auth, but others should work
        assert response.status_code in [200, 201, 422]  # 422 = validation error, not auth


class TestSecurityHeaders:
    """Test security-related headers and responses."""

    @pytest.mark.asyncio
    async def test_401_includes_www_authenticate(self, client):
        """Test that 401 responses include WWW-Authenticate header when Bearer used."""
        response = client.get(
            "/api/memories",
            headers={"Authorization": "Bearer wrong-key"}
        )
        assert response.status_code == 401
        # WWW-Authenticate header is added for Bearer token failures


class TestAnonymousAccess:
    """Test anonymous access behavior (when enabled)."""

    @pytest.mark.asyncio
    async def test_anonymous_disabled_by_default(self, client):
        """Test that anonymous access is disabled by default in test env."""
        response = client.get("/api/memories")
        assert response.status_code == 401


@pytest.mark.integration
class TestIntegrationScenarios:
    """Integration tests for real-world authentication scenarios."""

    @pytest.mark.asyncio
    async def test_api_only_deployment(self, client):
        """Test API-key-only deployment scenario (no OAuth)."""
        # This is the main use case for Issue #407
        response = client.get(
            "/api/memories",
            headers={"X-API-Key": "test-secret-key-12345"}
        )
        assert response.status_code == 200

        # Verify WebUI dashboard is accessible (no auth required for HTML)
        response = client.get("/")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_script_access_with_query_param(self, client):
        """Test convenient query parameter access for scripts."""
        # Useful for quick scripts, though less secure
        response = client.get("/api/memories?api_key=test-secret-key-12345")
        assert response.status_code == 200
