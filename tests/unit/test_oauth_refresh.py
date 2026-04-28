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
Tests for the OAuth 2.1 refresh_token grant (RFC 6749 §6, OAuth 2.1 §4.3.1).

Covers issuance (only when offline_access is requested), rotation
(old refresh token rejected after use), expiry, scope narrowing, client
binding, and the public-client path. Also includes a regression check
that the existing authorization_code flow without offline_access keeps
its single-token response shape.
"""

import time
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from mcp_memory_service.web.oauth.authorization import (
    _handle_authorization_code_grant,
    _handle_refresh_token_grant,
    create_refresh_token,
)
from mcp_memory_service.web.oauth.discovery import router as discovery_router
from mcp_memory_service.web.oauth.models import RegisteredClient
from mcp_memory_service.web.oauth.storage.memory import MemoryOAuthStorage


CONFIDENTIAL_CLIENT_ID = "conf-client"
CONFIDENTIAL_CLIENT_SECRET = "conf-secret"
PUBLIC_CLIENT_ID = "pub-client"


async def _register_confidential_client(storage: MemoryOAuthStorage) -> None:
    await storage.store_client(
        RegisteredClient(
            client_id=CONFIDENTIAL_CLIENT_ID,
            client_secret=CONFIDENTIAL_CLIENT_SECRET,
            redirect_uris=["https://example.com/cb"],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="client_secret_basic",
            client_name="Confidential Client",
            created_at=time.time(),
        )
    )


async def _register_public_client(storage: MemoryOAuthStorage) -> None:
    await storage.store_client(
        RegisteredClient(
            client_id=PUBLIC_CLIENT_ID,
            client_secret="unused",
            redirect_uris=["http://127.0.0.1/cb"],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="none",
            client_name="Public PKCE Client",
            created_at=time.time(),
        )
    )


async def _issue_via_auth_code(
    storage: MemoryOAuthStorage,
    scope: str,
    client_id: str = CONFIDENTIAL_CLIENT_ID,
    client_secret: str | None = CONFIDENTIAL_CLIENT_SECRET,
    code_verifier: str | None = None,
):
    """Drive authorization_code grant to issuance so we have a real refresh token."""
    code = "auth-code-" + str(time.time_ns())
    await storage.store_authorization_code(
        code=code,
        client_id=client_id,
        redirect_uri="https://example.com/cb",
        scope=scope,
        expires_in=300,
    )
    return await _handle_authorization_code_grant(
        final_client_id=client_id,
        final_client_secret=client_secret,
        code=code,
        redirect_uri="https://example.com/cb",
        code_verifier=code_verifier,
    )


# ===========================================================================
# Issuance — refresh token only when offline_access was requested
# ===========================================================================


@pytest.mark.asyncio
async def test_refresh_token_issued_when_offline_access_requested():
    storage = MemoryOAuthStorage()
    await _register_confidential_client(storage)

    with patch(
        "mcp_memory_service.web.oauth.authorization.get_oauth_storage",
        return_value=storage,
    ):
        resp = await _issue_via_auth_code(storage, scope="read write offline_access")

    assert resp.access_token
    assert resp.refresh_token is not None
    assert resp.refresh_token != resp.access_token
    assert "offline_access" in (resp.scope or "")


@pytest.mark.asyncio
async def test_refresh_token_not_issued_without_offline_access():
    """Regression: clients that don't opt in keep the single-token response shape."""
    storage = MemoryOAuthStorage()
    await _register_confidential_client(storage)

    with patch(
        "mcp_memory_service.web.oauth.authorization.get_oauth_storage",
        return_value=storage,
    ):
        resp = await _issue_via_auth_code(storage, scope="read write")

    assert resp.access_token
    assert resp.refresh_token is None
    assert resp.scope == "read write"


# ===========================================================================
# Refresh grant — happy path and rotation
# ===========================================================================


@pytest.mark.asyncio
async def test_refresh_grant_returns_new_access_token():
    storage = MemoryOAuthStorage()
    await _register_confidential_client(storage)

    with patch(
        "mcp_memory_service.web.oauth.authorization.get_oauth_storage",
        return_value=storage,
    ):
        issued = await _issue_via_auth_code(storage, scope="read write offline_access")
        refreshed = await _handle_refresh_token_grant(
            final_client_id=CONFIDENTIAL_CLIENT_ID,
            final_client_secret=CONFIDENTIAL_CLIENT_SECRET,
            refresh_token_value=issued.refresh_token,
            requested_scope=None,
        )

    assert refreshed.access_token
    assert refreshed.access_token != issued.access_token
    assert refreshed.scope == issued.scope


@pytest.mark.asyncio
async def test_refresh_grant_rotates_refresh_token():
    """Each refresh must issue a new, distinct refresh token (OAuth 2.1 §4.3.1)."""
    storage = MemoryOAuthStorage()
    await _register_confidential_client(storage)

    with patch(
        "mcp_memory_service.web.oauth.authorization.get_oauth_storage",
        return_value=storage,
    ):
        issued = await _issue_via_auth_code(storage, scope="offline_access read")
        refreshed = await _handle_refresh_token_grant(
            final_client_id=CONFIDENTIAL_CLIENT_ID,
            final_client_secret=CONFIDENTIAL_CLIENT_SECRET,
            refresh_token_value=issued.refresh_token,
            requested_scope=None,
        )

    assert refreshed.refresh_token is not None
    assert refreshed.refresh_token != issued.refresh_token

    # The new token's parent_token points to the original for audit.
    new_record = await storage.get_refresh_token(refreshed.refresh_token)
    assert new_record is not None
    assert new_record["parent_token"] == issued.refresh_token


@pytest.mark.asyncio
async def test_old_refresh_token_rejected_after_rotation():
    """After a successful refresh, the presented token must be unusable."""
    storage = MemoryOAuthStorage()
    await _register_confidential_client(storage)

    with patch(
        "mcp_memory_service.web.oauth.authorization.get_oauth_storage",
        return_value=storage,
    ):
        issued = await _issue_via_auth_code(storage, scope="offline_access read")
        await _handle_refresh_token_grant(
            final_client_id=CONFIDENTIAL_CLIENT_ID,
            final_client_secret=CONFIDENTIAL_CLIENT_SECRET,
            refresh_token_value=issued.refresh_token,
            requested_scope=None,
        )

        with pytest.raises(HTTPException) as excinfo:
            await _handle_refresh_token_grant(
                final_client_id=CONFIDENTIAL_CLIENT_ID,
                final_client_secret=CONFIDENTIAL_CLIENT_SECRET,
                refresh_token_value=issued.refresh_token,
                requested_scope=None,
            )

    assert excinfo.value.status_code == 400
    assert excinfo.value.detail["error"] == "invalid_grant"


# ===========================================================================
# Rejection paths
# ===========================================================================


@pytest.mark.asyncio
async def test_expired_refresh_token_rejected():
    storage = MemoryOAuthStorage()
    await _register_confidential_client(storage)

    # Insert a refresh token with negative TTL so it's already expired on read.
    await storage.store_refresh_token(
        token="rt-expired",
        client_id=CONFIDENTIAL_CLIENT_ID,
        scope="offline_access",
        expires_in=-1,
    )

    with patch(
        "mcp_memory_service.web.oauth.authorization.get_oauth_storage",
        return_value=storage,
    ):
        with pytest.raises(HTTPException) as excinfo:
            await _handle_refresh_token_grant(
                final_client_id=CONFIDENTIAL_CLIENT_ID,
                final_client_secret=CONFIDENTIAL_CLIENT_SECRET,
                refresh_token_value="rt-expired",
                requested_scope=None,
            )

    assert excinfo.value.status_code == 400
    assert excinfo.value.detail["error"] == "invalid_grant"


@pytest.mark.asyncio
async def test_invalid_refresh_token_rejected():
    storage = MemoryOAuthStorage()
    await _register_confidential_client(storage)

    with patch(
        "mcp_memory_service.web.oauth.authorization.get_oauth_storage",
        return_value=storage,
    ):
        with pytest.raises(HTTPException) as excinfo:
            await _handle_refresh_token_grant(
                final_client_id=CONFIDENTIAL_CLIENT_ID,
                final_client_secret=CONFIDENTIAL_CLIENT_SECRET,
                refresh_token_value="never-issued",
                requested_scope=None,
            )

    assert excinfo.value.status_code == 400
    assert excinfo.value.detail["error"] == "invalid_grant"


@pytest.mark.asyncio
async def test_refresh_token_client_mismatch_rejected():
    storage = MemoryOAuthStorage()
    await _register_confidential_client(storage)
    # Second client registered under a different id — refresh must not cross.
    other_id = "other-client"
    await storage.store_client(
        RegisteredClient(
            client_id=other_id,
            client_secret="other-secret",
            redirect_uris=["https://example.com/cb"],
            grant_types=["refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="client_secret_basic",
            client_name="Other",
            created_at=time.time(),
        )
    )

    with patch(
        "mcp_memory_service.web.oauth.authorization.get_oauth_storage",
        return_value=storage,
    ):
        issued = await _issue_via_auth_code(storage, scope="offline_access")
        with pytest.raises(HTTPException) as excinfo:
            await _handle_refresh_token_grant(
                final_client_id=other_id,
                final_client_secret="other-secret",
                refresh_token_value=issued.refresh_token,
                requested_scope=None,
            )

    assert excinfo.value.status_code == 400
    assert excinfo.value.detail["error"] == "invalid_grant"


@pytest.mark.asyncio
async def test_refresh_scope_narrowing_allowed():
    storage = MemoryOAuthStorage()
    await _register_confidential_client(storage)

    with patch(
        "mcp_memory_service.web.oauth.authorization.get_oauth_storage",
        return_value=storage,
    ):
        issued = await _issue_via_auth_code(storage, scope="read write offline_access")
        refreshed = await _handle_refresh_token_grant(
            final_client_id=CONFIDENTIAL_CLIENT_ID,
            final_client_secret=CONFIDENTIAL_CLIENT_SECRET,
            refresh_token_value=issued.refresh_token,
            requested_scope="read",
        )

    assert refreshed.scope == "read"


@pytest.mark.asyncio
async def test_refresh_scope_broadening_rejected():
    storage = MemoryOAuthStorage()
    await _register_confidential_client(storage)

    with patch(
        "mcp_memory_service.web.oauth.authorization.get_oauth_storage",
        return_value=storage,
    ):
        issued = await _issue_via_auth_code(storage, scope="read offline_access")
        with pytest.raises(HTTPException) as excinfo:
            await _handle_refresh_token_grant(
                final_client_id=CONFIDENTIAL_CLIENT_ID,
                final_client_secret=CONFIDENTIAL_CLIENT_SECRET,
                refresh_token_value=issued.refresh_token,
                requested_scope="read write admin",
            )

    assert excinfo.value.status_code == 400
    assert excinfo.value.detail["error"] == "invalid_scope"


@pytest.mark.asyncio
async def test_confidential_client_without_secret_rejected():
    """Refresh from a confidential client with no client authentication fails."""
    storage = MemoryOAuthStorage()
    await _register_confidential_client(storage)

    with patch(
        "mcp_memory_service.web.oauth.authorization.get_oauth_storage",
        return_value=storage,
    ):
        issued = await _issue_via_auth_code(storage, scope="offline_access")
        with pytest.raises(HTTPException) as excinfo:
            await _handle_refresh_token_grant(
                final_client_id=CONFIDENTIAL_CLIENT_ID,
                final_client_secret=None,
                refresh_token_value=issued.refresh_token,
                requested_scope=None,
            )

    assert excinfo.value.status_code == 401
    assert excinfo.value.detail["error"] == "invalid_client"


# ===========================================================================
# Public client (PKCE) path — no client secret required on refresh
# ===========================================================================


@pytest.mark.asyncio
async def test_public_client_refresh_without_secret():
    """Public clients authenticate via the rotating refresh token itself."""
    storage = MemoryOAuthStorage()
    await _register_public_client(storage)

    # For this unit test we bypass PKCE on the issuance side by pre-seeding
    # a refresh token directly — the surface under test is the refresh grant.
    rt, _ = None, None
    with patch(
        "mcp_memory_service.web.oauth.authorization.get_oauth_storage",
        return_value=storage,
    ):
        rt, _ = await create_refresh_token(
            client_id=PUBLIC_CLIENT_ID,
            scope="read offline_access",
        )
        refreshed = await _handle_refresh_token_grant(
            final_client_id=PUBLIC_CLIENT_ID,
            final_client_secret=None,
            refresh_token_value=rt,
            requested_scope=None,
        )

    assert refreshed.access_token
    assert refreshed.refresh_token is not None
    assert refreshed.refresh_token != rt


# ===========================================================================
# Discovery advertises refresh_token and offline_access
# ===========================================================================


def _discovery_client() -> TestClient:
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(discovery_router)
    return TestClient(app)


def test_discovery_advertises_refresh_token_grant():
    client = _discovery_client()
    response = client.get("/.well-known/oauth-authorization-server/mcp")
    assert response.status_code == 200
    body = response.json()
    assert "refresh_token" in body["grant_types_supported"]


def test_discovery_advertises_offline_access_scope():
    client = _discovery_client()
    response = client.get("/.well-known/oauth-authorization-server/mcp")
    assert response.status_code == 200
    body = response.json()
    assert "offline_access" in body["scopes_supported"]


def test_protected_resource_metadata_advertises_offline_access():
    client = _discovery_client()
    response = client.get("/.well-known/oauth-protected-resource")
    assert response.status_code == 200
    body = response.json()
    assert "offline_access" in body["scopes_supported"]


# ===========================================================================
# Review feedback regression — chain revoke + public client client_id
# ===========================================================================


@pytest.mark.asyncio
async def test_replay_revokes_chain():
    """
    OAuth 2.1 §4.3.1: replay of an already-revoked refresh token signals
    possible compromise. The server cannot tell the legitimate client from
    an attacker, so it MUST revoke every other live token in the rotation
    chain (compromise mitigation).
    """
    storage = MemoryOAuthStorage()
    await _register_confidential_client(storage)

    with patch(
        "mcp_memory_service.web.oauth.authorization.get_oauth_storage",
        return_value=storage,
    ):
        # Issue an initial refresh token (R0).
        first = await _issue_via_auth_code(storage, scope="read offline_access")
        r0 = first.refresh_token
        assert r0 is not None

        # Rotate once (R0 -> R1).
        second = await _handle_refresh_token_grant(
            final_client_id=CONFIDENTIAL_CLIENT_ID,
            final_client_secret=CONFIDENTIAL_CLIENT_SECRET,
            refresh_token_value=r0,
            requested_scope=None,
        )
        r1 = second.refresh_token
        assert r1 is not None and r1 != r0

        # Rotate again (R1 -> R2). R2 is the only live token at this point.
        third = await _handle_refresh_token_grant(
            final_client_id=CONFIDENTIAL_CLIENT_ID,
            final_client_secret=CONFIDENTIAL_CLIENT_SECRET,
            refresh_token_value=r1,
            requested_scope=None,
        )
        r2 = third.refresh_token
        assert r2 is not None

        # Sanity: R2 should still be live before the replay.
        assert await storage.get_refresh_token(r2) is not None

        # Replay R0 (already revoked by the first rotation).
        with pytest.raises(HTTPException) as exc_info:
            await _handle_refresh_token_grant(
                final_client_id=CONFIDENTIAL_CLIENT_ID,
                final_client_secret=CONFIDENTIAL_CLIENT_SECRET,
                refresh_token_value=r0,
                requested_scope=None,
            )
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["error"] == "invalid_grant"

        # The live descendant R2 must now also be revoked
        # (chain-wide revocation triggered by the replay).
        assert await storage.get_refresh_token(r2) is None


@pytest.mark.asyncio
async def test_public_client_must_send_client_id():
    """
    RFC 6749 §6: refresh requests MUST include client_id when the client
    is not authenticating (public clients). The token's internal binding
    is not a substitute for the explicit parameter.
    """
    storage = MemoryOAuthStorage()
    await _register_public_client(storage)

    with patch(
        "mcp_memory_service.web.oauth.authorization.get_oauth_storage",
        return_value=storage,
    ):
        # Bypass PKCE on issuance — we are testing the refresh grant surface.
        refresh_token, _ = await create_refresh_token(
            client_id=PUBLIC_CLIENT_ID,
            scope="read offline_access",
        )
        assert refresh_token is not None

        # Public client omitting client_id — should be rejected with
        # invalid_request / 400.
        with pytest.raises(HTTPException) as exc_info:
            await _handle_refresh_token_grant(
                final_client_id=None,
                final_client_secret=None,
                refresh_token_value=refresh_token,
                requested_scope=None,
            )
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail["error"] == "invalid_request"
        assert "client_id" in exc_info.value.detail["error_description"]

