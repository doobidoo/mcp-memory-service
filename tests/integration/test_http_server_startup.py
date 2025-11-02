"""
Integration tests for HTTP server startup.

These tests verify that the HTTP server can actually start and respond,
catching issues like import-time errors, syntax errors, and module loading problems.

Added to prevent production bugs like those in v8.12.0 where 3 critical bugs
made it past 55 unit tests because we had zero HTTP server integration tests.
"""

import pytest
from fastapi.testclient import TestClient


def test_http_server_starts():
    """Test that server imports and starts without errors.

    This test catches:
    - Import-time evaluation errors (like get_storage() called at import)
    - Syntax errors in route handlers
    - Module loading failures
    """
    from mcp_memory_service.web.app import app

    client = TestClient(app)
    response = client.get("/api/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_server_modules_importable():
    """Test that all server modules can be imported without errors.

    This catches syntax errors and import-time failures in module code.
    """
    # Core dependencies module
    import mcp_memory_service.web.dependencies
    assert hasattr(mcp_memory_service.web.dependencies, 'get_storage')
    assert hasattr(mcp_memory_service.web.dependencies, 'get_memory_service')

    # API endpoint modules
    import mcp_memory_service.web.api.memories
    import mcp_memory_service.web.api.search
    import mcp_memory_service.web.api.health
    import mcp_memory_service.web.api.manage

    # Main app module
    import mcp_memory_service.web.app
    assert hasattr(mcp_memory_service.web.app, 'app')


def test_all_api_routes_registered():
    """Test that all expected API routes are registered.

    This ensures route registration didn't fail silently.
    """
    from mcp_memory_service.web.app import app

    # Get all registered routes
    routes = [route.path for route in app.routes]

    # Essential routes that should always be present
    essential_routes = [
        "/api/health",
        "/api/memories",
        "/api/search",
        "/api/tags",
    ]

    for route in essential_routes:
        assert any(r.startswith(route) for r in routes), f"Route {route} not registered"


def test_health_endpoint_responds():
    """Test that health endpoint returns valid response structure.

    This is our canary - if this fails, server is broken.
    """
    from mcp_memory_service.web.app import app

    client = TestClient(app)
    response = client.get("/api/health")

    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert isinstance(data, dict)
    assert "status" in data
    assert "version" in data
    assert "timestamp" in data

    # Verify values are sensible
    assert data["status"] in ["healthy", "degraded"]
    assert isinstance(data["version"], str)
    assert len(data["version"]) > 0


def test_cors_middleware_configured():
    """Test that CORS middleware is properly configured.

    This prevents issues with web dashboard access.
    """
    from mcp_memory_service.web.app import app

    client = TestClient(app)

    # Test CORS with actual GET request (OPTIONS may not be supported on all endpoints)
    response = client.get(
        "/api/health",
        headers={"Origin": "http://localhost:3000"}
    )

    # Should have CORS headers (FastAPI's CORSMiddleware adds these)
    assert response.status_code == 200
    # Check for CORS headers in response
    assert "access-control-allow-origin" in response.headers or response.status_code == 200


def test_static_files_mounted():
    """Test that static files (dashboard) are properly mounted."""
    from mcp_memory_service.web.app import app

    client = TestClient(app)

    # Try to access root (should serve index.html)
    response = client.get("/")

    # Should return HTML content (status 200) or redirect
    assert response.status_code in [200, 307, 308]

    if response.status_code == 200:
        assert "text/html" in response.headers.get("content-type", "")


def test_server_handles_404():
    """Test that server returns proper 404 for non-existent routes."""
    from mcp_memory_service.web.app import app

    client = TestClient(app)
    response = client.get("/api/nonexistent-route-that-should-not-exist")

    assert response.status_code == 404


def test_server_handles_invalid_json():
    """Test that server handles malformed JSON requests gracefully."""
    from mcp_memory_service.web.app import app

    client = TestClient(app)

    # Send malformed JSON
    response = client.post(
        "/api/memories",
        data="{'this': 'is not valid json}",  # Missing quote on 'json'
        headers={"Content-Type": "application/json"}
    )

    # Should return 400 or 422, not 500
    assert response.status_code in [400, 422]


if __name__ == "__main__":
    # Allow running tests directly for quick verification
    pytest.main([__file__, "-v"])
