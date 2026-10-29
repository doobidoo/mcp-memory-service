"""
Wire format of the Dynamic Client Registration response.

RFC 7591 section 3.2.1 lists every member of the client information response
except client_id as optional, and an optional member that has no value is
omitted rather than sent as JSON null. A public client (token_endpoint_auth_method
"none") is issued no secret, so its response must not carry client_secret at
all. Strict clients reject a null there. Claude Code refuses the response and
never opens the authorization page.

These tests go through FastAPI on purpose. The null only appears when the
response model is serialized, so calling register_client() directly would not
see it.
"""

from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mcp_memory_service.web.oauth import registration
from mcp_memory_service.web.oauth.rate_limit import auth_rate_limit
from mcp_memory_service.web.oauth.storage.memory import MemoryOAuthStorage


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(registration.router, prefix="/oauth")
    app.dependency_overrides[auth_rate_limit] = lambda: None
    with patch.object(registration, "get_oauth_storage", return_value=MemoryOAuthStorage()), \
         patch.object(registration, "DCR_REGISTRATION_KEY", None):
        yield TestClient(app)


def _register(client, **overrides):
    body = {
        "client_name": "Claude Code",
        "redirect_uris": ["http://localhost:53123/callback"],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
    }
    body.update(overrides)
    response = client.post("/oauth/register", json=body)
    assert response.status_code == 201, response.text
    return response.json()


def test_public_client_response_omits_client_secret(client):
    data = _register(client)

    assert "client_secret" not in data
    assert data["token_endpoint_auth_method"] == "none"


def test_response_never_contains_null_members(client):
    data = _register(client, client_name=None)

    assert None not in data.values(), data
    assert "client_name" not in data
    assert "scope" not in data


def test_confidential_client_still_receives_its_secret(client):
    data = _register(client, token_endpoint_auth_method="client_secret_basic")

    assert isinstance(data["client_secret"], str) and data["client_secret"]


def test_requested_scope_is_echoed(client):
    data = _register(client, scope="read write")

    assert data["scope"] == "read write"
