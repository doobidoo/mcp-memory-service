import json
from urllib.parse import parse_qs, urlparse

import pytest
from unittest.mock import patch

from fastapi import HTTPException

from mcp_memory_service.web.oauth.authorization import (
    _handle_authorization_code_grant,
    authorize_post,
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


@pytest.mark.parametrize(
    "client_state",
    [
        "plain_state_123",
        "eyJ0Ijp7ImEiOjF9fQ==",                       # base64url with '=' padding
        "abc+def/ghi==",                              # standard base64 (+ / =)
        "hdr.eyJzdWIiOiJ4In0.c2ln",                   # JWT-shaped
        "S" * 200,                                    # > 128 chars
    ],
)
@pytest.mark.asyncio
async def test_authorize_post_returns_state_verbatim(client_state):
    """RFC 6749 §4.1.2: state MUST be reflected back to the client unchanged.
    """
    storage = MemoryOAuthStorage()
    await storage.store_client(
        RegisteredClient(
            client_id="cursor-native",
            client_secret="unused",
            redirect_uris=["cursor://127.0.0.1:51234/callback"],
            grant_types=["authorization_code"],
            response_types=["code"],
            token_endpoint_auth_method="none",
            client_name="Cursor",
            created_at=0,
        )
    )

    with patch(
        "mcp_memory_service.web.oauth.authorization.get_oauth_storage",
        return_value=storage,
    ), patch(
        "mcp_memory_service.web.oauth.authorization.API_KEY", "test-api-key"
    ) as api_key:
        response = await authorize_post(
            request=None,  # unused on the success path
            response_type="code",
            client_id="cursor-native",
            redirect_uri="cursor://127.0.0.1:51234/callback",
            scope="read write",
            state=client_state,
            code_challenge="E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM",
            code_challenge_method="S256",
            api_key=api_key,
        )

    # Success body embeds the callback URL as a JS string literal:
    #   <script>window.location.href = "<redirect_url>";</script>
    body = response.body.decode()
    js_literal = body.split("window.location.href = ", 1)[1].split(";</script>", 1)[0]
    redirect_url = json.loads(js_literal.replace("<\\/", "</"))

    qs = parse_qs(urlparse(redirect_url).query)
    assert "code" in qs, f"authorization code missing: {redirect_url}"
    assert qs.get("state")[0] == client_state
