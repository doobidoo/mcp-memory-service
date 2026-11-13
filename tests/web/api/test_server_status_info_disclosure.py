"""GHSA-hm53-3f6c-cm5h: /api/server/status must not hand host fingerprinting
data (pid, OS, kernel, Python version) to callers below admin scope."""

import pytest
from fastapi.testclient import TestClient

from mcp_memory_service.web.api import server as server_module
from mcp_memory_service.web.oauth.middleware import AuthenticationResult

HOST_FIELDS = ("pid", "platform", "platform_version", "python_version")


@pytest.fixture
def status_for():
    from mcp_memory_service.web.app import app

    def fetch(scope):
        async def user():
            return AuthenticationResult(authenticated=True, client_id="t", scope=scope, auth_method="test")
        # Override the exact object the route holds (see test_server_management.py).
        app.dependency_overrides[server_module.require_read_access] = user
        resp = TestClient(app).get("/api/server/status")
        assert resp.status_code == 200
        return resp.json()

    yield fetch
    app.dependency_overrides.clear()


@pytest.mark.parametrize("scope", ["read", "read write"])
def test_non_admin_gets_no_host_details(status_for, scope):
    data = status_for(scope)
    assert data["version"] and "uptime_seconds" in data
    for field in HOST_FIELDS:
        assert data.get(field) is None, field


def test_admin_keeps_host_details(status_for):
    # The dashboard compares pid across an (admin-only) restart.
    data = status_for("read write admin")
    assert isinstance(data["pid"], int)
    assert data["platform"] and data["python_version"]
