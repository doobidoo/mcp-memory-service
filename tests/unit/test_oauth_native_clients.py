import pytest
from unittest.mock import patch

from fastapi import HTTPException

from mcp_memory_service.web.oauth.authorization import (
    _handle_authorization_code_grant,
    validate_redirect_uri,
)
from mcp_memory_service.web.oauth.models import RegisteredClient
from mcp_memory_service.web.oauth.storage.memory import MemoryOAuthStorage


@pytest.mark.asyncio
async def test_validate_redirect_uri_accepts_loopback_alias_and_dynamic_port():
    storage = MemoryOAuthStorage()
    client = RegisteredClient(
        client_id="native-client",
        client_secret="unused-secret",
        redirect_uris=["http://127.0.0.1/mcp/oauth/callback"],
        grant_types=["authorization_code"],
        response_types=["code"],
        token_endpoint_auth_method="none",
        client_name="Native Client",
        created_at=0,
    )
    await storage.store_client(client)

    with patch(
        "mcp_memory_service.web.oauth.authorization.get_oauth_storage",
        return_value=storage,
    ):
        validated = await validate_redirect_uri(
            "native-client",
            "http://localhost:43188/mcp/oauth/callback",
        )

    # For native loopback clients, the runtime callback URI must be preserved so
    # the redirect lands on the ephemeral port the client actually opened.
    assert validated == "http://localhost:43188/mcp/oauth/callback"


@pytest.mark.asyncio
async def test_validate_redirect_uri_rejects_unregistered_loopback_path():
    storage = MemoryOAuthStorage()
    client = RegisteredClient(
        client_id="native-client",
        client_secret="unused-secret",
        redirect_uris=["http://127.0.0.1/mcp/oauth/callback"],
        grant_types=["authorization_code"],
        response_types=["code"],
        token_endpoint_auth_method="none",
        client_name="Native Client",
        created_at=0,
    )
    await storage.store_client(client)

    with patch(
        "mcp_memory_service.web.oauth.authorization.get_oauth_storage",
        return_value=storage,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await validate_redirect_uri(
                "native-client",
                "http://127.0.0.1:43188/other/callback",
            )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["error"] == "invalid_redirect_uri"


@pytest.mark.asyncio
async def test_public_pkce_client_can_exchange_code_without_secret():
    storage = MemoryOAuthStorage()
    client = RegisteredClient(
        client_id="public-native-client",
        client_secret="server-generated-but-unused",
        redirect_uris=["http://127.0.0.1/mcp/oauth/callback"],
        grant_types=["authorization_code"],
        response_types=["code"],
        token_endpoint_auth_method="none",
        client_name="OpenCode",
        created_at=0,
    )
    await storage.store_client(client)
    await storage.store_authorization_code(
        code="auth-code-123",
        client_id="public-native-client",
        redirect_uri="http://127.0.0.1:19876/mcp/oauth/callback",
        scope="read write",
        expires_in=300,
        code_challenge="E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM",
        code_challenge_method="S256",
    )

    with patch(
        "mcp_memory_service.web.oauth.authorization.get_oauth_storage",
        return_value=storage,
    ):
        response = await _handle_authorization_code_grant(
            final_client_id="public-native-client",
            final_client_secret=None,
            code="auth-code-123",
            redirect_uri="http://127.0.0.1:19876/mcp/oauth/callback",
            code_verifier="dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk",
        )

    assert response.token_type == "Bearer"
    assert response.access_token
