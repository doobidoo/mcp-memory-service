"""The system-operation route only knows the operations the dashboard uses (#1974).

``optimize_db`` and ``rebuild_index`` were placeholders behind the dashboard's
System Maintenance buttons and always answered 501. They are gone, so the route
answers them like any other unknown operation.
"""

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.web.dependencies import set_storage


@pytest_asyncio.fixture
async def storage(temp_db_path, monkeypatch):
    monkeypatch.setenv("MCP_SEMANTIC_DEDUP_ENABLED", "false")
    storage = SqliteVecMemoryStorage(f"{temp_db_path}/test.db")
    await storage.initialize()
    yield storage
    await storage.close()


@pytest.fixture
def client(storage, monkeypatch):
    monkeypatch.setenv("MCP_API_KEY", "")
    monkeypatch.setenv("MCP_OAUTH_ENABLED", "false")
    monkeypatch.setenv("MCP_ALLOW_ANONYMOUS_ACCESS", "true")

    from mcp_memory_service.web.app import app
    from mcp_memory_service.web.oauth.middleware import (
        AuthenticationResult,
        get_current_user,
        require_read_access,
        require_write_access,
    )

    set_storage(storage)

    async def authenticated_user():
        return AuthenticationResult(
            authenticated=True,
            client_id="test_client",
            scope="read write admin",
            auth_method="test",
        )

    app.dependency_overrides[get_current_user] = authenticated_user
    app.dependency_overrides[require_write_access] = authenticated_user
    app.dependency_overrides[require_read_access] = authenticated_user
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.mark.parametrize("operation", ["optimize_db", "rebuild_index"])
def test_removed_placeholder_operations_are_unknown(client, operation):
    response = client.post(f"/api/manage/system/{operation}")

    assert response.status_code == 400
    assert response.json()["detail"] == f"Unknown operation: {operation}"


def test_unknown_operation_is_rejected(client):
    response = client.post("/api/manage/system/no_such_operation")

    assert response.status_code == 400
    assert response.json()["detail"] == "Unknown operation: no_such_operation"


def test_cleanup_duplicates_is_still_routed(client):
    response = client.post("/api/manage/system/cleanup_duplicates")

    # Still dispatched to the duplicate cleanup, not refused as unknown.
    assert response.status_code != 400
    assert response.json().get("detail") != "Unknown operation: cleanup_duplicates"
    direct = client.post("/api/manage/cleanup-duplicates")
    assert response.status_code == direct.status_code
