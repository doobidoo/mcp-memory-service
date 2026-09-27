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
Security-hardening tests for the OAuth endpoints.

Covers: reflected-XSS fix on /authorize, PKCE enforcement for public clients,
code_challenge_method validation, input length caps (code_verifier, code,
client_secret, refresh_token), JWT/api-key length guards, client_secret hashing
at rest, the consolidated redirect-URI validator, the request body-size cap,
and the in-process auth rate limiter.
"""

import pytest
from unittest.mock import patch

from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from mcp_memory_service.web.oauth.authorization import (
    authorize_get,
    _handle_authorization_code_grant,
    _handle_client_credentials_grant,
    token as token_endpoint,
)
from mcp_memory_service.web.oauth.models import RegisteredClient
from mcp_memory_service.web.oauth.storage.memory import MemoryOAuthStorage
from mcp_memory_service.web.oauth import redirect as redirect_mod
from mcp_memory_service.web.oauth import limits
from mcp_memory_service.web.oauth import rate_limit
from mcp_memory_service.web.oauth.storage.base import (
    hash_client_secret,
    is_hashed_secret,
    verify_client_secret,
)
from mcp_memory_service.web.body_limit import BodySizeLimitMiddleware, _limit_for_path


AUTH_PATCH = "mcp_memory_service.web.oauth.authorization.get_oauth_storage"


async def _store_public_client(storage, client_id="pub", auth_method="none"):
    await storage.store_client(
        RegisteredClient(
            client_id=client_id,
            client_secret="unused",
            redirect_uris=["http://127.0.0.1/cb"],
            grant_types=["authorization_code"],
            response_types=["code"],
            token_endpoint_auth_method=auth_method,
            client_name="C",
            created_at=0,
        )
    )


# ---------------------------------------------------------------------------
# Reflected XSS on /authorize (issue #4 / Claude #1)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_authorize_page_does_not_reflect_raw_query_xss():
    storage = MemoryOAuthStorage()
    await _store_public_client(storage, auth_method="client_secret_basic")
    payload = '"><script>alert(1)</script>'
    with patch(AUTH_PATCH, return_value=storage):
        resp = await authorize_get(
            request=None,
            response_type="code",
            client_id="pub",
            redirect_uri=None,
            scope=payload,
            state=payload,
            code_challenge=None,
            code_challenge_method=None,
        )
    body = resp.body.decode()
    # The injected markup must never appear unescaped in the rendered page.
    assert "<script>alert(1)</script>" not in body
    assert "&lt;script&gt;" in body or "%3Cscript%3E" in body


# ---------------------------------------------------------------------------
# PKCE mandatory for public clients (Claude #3) + method validation (Claude #4)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_authorize_rejects_public_client_without_pkce():
    storage = MemoryOAuthStorage()
    await _store_public_client(storage, auth_method="none")
    with patch(AUTH_PATCH, return_value=storage):
        with pytest.raises(HTTPException) as exc:
            await authorize_get(
                request=None, response_type="code", client_id="pub",
                redirect_uri=None, scope=None, state=None,
                code_challenge=None, code_challenge_method=None,
            )
    assert exc.value.status_code == 400
    assert "code_challenge" in exc.value.detail["error_description"]


@pytest.mark.asyncio
async def test_authorize_allows_confidential_client_without_pkce():
    storage = MemoryOAuthStorage()
    await _store_public_client(storage, auth_method="client_secret_basic")
    with patch(AUTH_PATCH, return_value=storage):
        resp = await authorize_get(
            request=None, response_type="code", client_id="pub",
            redirect_uri=None, scope=None, state=None,
            code_challenge=None, code_challenge_method=None,
        )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_authorize_rejects_non_s256_challenge_method():
    storage = MemoryOAuthStorage()
    await _store_public_client(storage, auth_method="none")
    challenge = "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
    with patch(AUTH_PATCH, return_value=storage):
        with pytest.raises(HTTPException) as exc:
            await authorize_get(
                request=None, response_type="code", client_id="pub",
                redirect_uri=None, scope=None, state=None,
                code_challenge=challenge, code_challenge_method="plain",
            )
    assert exc.value.status_code == 400
    assert "S256" in exc.value.detail["error_description"]


# ---------------------------------------------------------------------------
# code_verifier length (issue #3)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("verifier", ["short", "x" * 5000])
async def test_token_grant_rejects_bad_code_verifier_length(verifier):
    storage = MemoryOAuthStorage()
    await _store_public_client(storage, auth_method="none")
    with patch(AUTH_PATCH, return_value=storage):
        with pytest.raises(HTTPException) as exc:
            await _handle_authorization_code_grant(
                final_client_id="pub", final_client_secret=None,
                code="somecode", redirect_uri=None, code_verifier=verifier,
            )
    assert exc.value.status_code == 400
    assert "code_verifier" in exc.value.detail["error_description"]


# ---------------------------------------------------------------------------
# Opaque-input length caps on /token (issue #3)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value,max_len",
    [
        ("refresh_token", "r" * (limits.MAX_REFRESH_TOKEN_LEN + 1), limits.MAX_REFRESH_TOKEN_LEN),
        ("client_secret", "s" * (limits.MAX_CLIENT_SECRET_LEN + 1), limits.MAX_CLIENT_SECRET_LEN),
        ("code", "c" * (limits.MAX_AUTHORIZATION_CODE_LEN + 1), limits.MAX_AUTHORIZATION_CODE_LEN),
    ],
)
async def test_token_endpoint_rejects_oversized_inputs(field, value, max_len):
    storage = MemoryOAuthStorage()

    class _Req:
        headers = {}

    kwargs = dict(
        request=_Req(), grant_type="authorization_code", code="c", redirect_uri=None,
        client_id="pub", client_secret=None, code_verifier=None,
        refresh_token=None, scope=None,
    )
    kwargs[field] = value
    with patch(AUTH_PATCH, return_value=storage):
        with pytest.raises(HTTPException) as exc:
            await token_endpoint(**kwargs)
    assert exc.value.status_code == 400
    assert field in exc.value.detail["error_description"]


# ---------------------------------------------------------------------------
# JWT / API-key length guards in middleware (issue #5)
# ---------------------------------------------------------------------------

def test_validate_jwt_token_rejects_oversized():
    from mcp_memory_service.web.oauth.middleware import validate_jwt_token
    oversized = "a." + "b" * (limits.MAX_JWT_LEN) + ".c"
    assert validate_jwt_token(oversized) is None


def test_authenticate_api_key_rejects_oversized():
    from mcp_memory_service.web.oauth import middleware
    with patch.object(middleware, "API_KEY", "real-key"):
        result = middleware.authenticate_api_key("x" * (limits.MAX_API_KEY_LEN + 1))
    assert result.authenticated is False


# ---------------------------------------------------------------------------
# client_secret hashing at rest (Claude #2)
# ---------------------------------------------------------------------------

def test_secret_hash_helpers():
    h = hash_client_secret("topsecret")
    assert is_hashed_secret(h)
    assert not is_hashed_secret("topsecret")
    assert verify_client_secret(h, "topsecret")
    assert not verify_client_secret(h, "wrong")
    # Legacy plaintext rows still verify (constant-time) pending upgrade.
    assert verify_client_secret("legacyplain", "legacyplain")
    assert not verify_client_secret("legacyplain", "other")


@pytest.mark.asyncio
async def test_legacy_plaintext_secret_upgraded_on_authenticate():
    storage = MemoryOAuthStorage()
    # Simulate a pre-existing plaintext row by writing directly past store_client.
    storage._clients["legacy"] = RegisteredClient(
        client_id="legacy", client_secret="plainsecret",
        redirect_uris=[], token_endpoint_auth_method="client_secret_basic",
        created_at=0,
    )
    assert await storage.authenticate_client("legacy", "plainsecret") is True
    # After a successful auth the stored secret must be hashed.
    assert is_hashed_secret(storage._clients["legacy"].client_secret)


# ---------------------------------------------------------------------------
# Consolidated redirect-URI validator (issue #1)
# ---------------------------------------------------------------------------

def test_registration_validator_rejects_dangerous_scheme():
    with pytest.raises(HTTPException):
        redirect_mod.validate_registration_redirect_uris(["javascript:alert(1)"])


def test_registration_validator_accepts_custom_and_https():
    redirect_mod.validate_registration_redirect_uris(
        ["cursor://callback", "https://app.example.com/cb", "http://localhost/cb"]
    )


def test_resolve_redirect_uri_returns_stored_value_and_rejects_unregistered():
    registered = ["https://app.example.com/cb"]
    assert (
        redirect_mod.resolve_registered_redirect_uri(registered, "https://app.example.com/cb")
        == "https://app.example.com/cb"
    )
    with pytest.raises(HTTPException):
        redirect_mod.resolve_registered_redirect_uri(registered, "https://evil.example/cb")


def test_build_redirect_url_no_longer_rejects_schemes():
    # The dead denylist pass was removed; building from an already-validated URI
    # just assembles the query string.
    url = redirect_mod.build_redirect_url("myapp://cb", {"code": "abc"})
    assert url == "myapp://cb?code=abc"


# ---------------------------------------------------------------------------
# Body size cap (issues #2 and #6)
# ---------------------------------------------------------------------------

def test_limit_for_path_tiers():
    from mcp_memory_service.config import HTTP_MAX_BODY_BYTES, OAUTH_MAX_BODY_BYTES
    assert _limit_for_path("/oauth/token") == OAUTH_MAX_BODY_BYTES
    assert _limit_for_path("/api/memories") == HTTP_MAX_BODY_BYTES
    assert _limit_for_path("/api/documents/upload") is None  # exempt


def _body_limit_client():
    app = FastAPI()
    app.add_middleware(BodySizeLimitMiddleware)

    @app.post("/oauth/echo")
    async def oauth_echo(request: Request):
        return {"len": len(await request.body())}

    @app.post("/api/documents/echo")
    async def docs_echo(request: Request):
        return {"len": len(await request.body())}

    @app.post("/api/other")
    async def other_echo(request: Request):
        return {"len": len(await request.body())}

    return TestClient(app)


def test_body_limit_rejects_oversized_oauth_body():
    client = _body_limit_client()
    from mcp_memory_service.config import OAUTH_MAX_BODY_BYTES
    big = b"x" * (OAUTH_MAX_BODY_BYTES + 1)
    resp = client.post("/oauth/echo", content=big)
    assert resp.status_code == 413
    small = client.post("/oauth/echo", content=b"x" * 100)
    assert small.status_code == 200


def test_body_limit_exempts_documents():
    client = _body_limit_client()
    from mcp_memory_service.config import HTTP_MAX_BODY_BYTES
    big = b"x" * (HTTP_MAX_BODY_BYTES + 1024)
    resp = client.post("/api/documents/echo", content=big)
    assert resp.status_code == 200


def test_body_limit_global_cap():
    client = _body_limit_client()
    from mcp_memory_service.config import HTTP_MAX_BODY_BYTES
    big = b"x" * (HTTP_MAX_BODY_BYTES + 1)
    resp = client.post("/api/other", content=big)
    assert resp.status_code == 413


# ---------------------------------------------------------------------------
# Rate limiting + concurrency (issue #8)
# ---------------------------------------------------------------------------

def test_sliding_window_rate_limiter():
    limiter = rate_limit._SlidingWindowRateLimiter(max_per_window=3, window_seconds=60)
    assert limiter.check("ip", now=100.0)
    assert limiter.check("ip", now=100.1)
    assert limiter.check("ip", now=100.2)
    assert not limiter.check("ip", now=100.3)  # 4th in window blocked
    # After the window slides, requests are allowed again.
    assert limiter.check("ip", now=200.0)


@pytest.mark.asyncio
async def test_auth_rate_limit_dependency_raises_429_over_limit():
    rate_limit._reset_for_tests()

    class _Req:
        class client:
            host = "1.2.3.4"

    with patch.object(rate_limit, "_rate_limiter",
                      rate_limit._SlidingWindowRateLimiter(max_per_window=2)):
        # First two acquire/release cleanly.
        for _ in range(2):
            gen = rate_limit.auth_rate_limit(_Req())
            await gen.__anext__()
            await gen.aclose()
        # Third is over the limit -> 429.
        gen = rate_limit.auth_rate_limit(_Req())
        with pytest.raises(HTTPException) as exc:
            await gen.__anext__()
        assert exc.value.status_code == 429

    rate_limit._reset_for_tests()


# ---------------------------------------------------------------------------
# Proxy-aware client keying (Finding #1)
# ---------------------------------------------------------------------------

class _ProxyReq:
    """Minimal stand-in for a Starlette Request for _client_key."""
    def __init__(self, headers=None, peer="1.2.3.4"):
        self.headers = headers or {}
        self._peer = peer

    @property
    def client(self):
        peer = self._peer
        return type("C", (), {"host": peer})()


def test_client_key_uses_socket_peer_by_default():
    # With no trusted proxy header configured, a client-supplied
    # X-Forwarded-For must be ignored (otherwise it could be spoofed to
    # evade the per-IP limit).
    req = _ProxyReq(headers={"X-Forwarded-For": "9.9.9.9"}, peer="1.2.3.4")
    with patch.object(rate_limit, "OAUTH_TRUST_PROXY_HEADER", ""):
        assert rate_limit._client_key(req) == "1.2.3.4"


def test_client_key_trusts_configured_proxy_header():
    # When configured, the left-most forwarded entry (the originating client)
    # is used so each real client gets its own bucket behind a proxy.
    req = _ProxyReq(headers={"X-Forwarded-For": "9.9.9.9, 10.0.0.1"}, peer="1.2.3.4")
    with patch.object(rate_limit, "OAUTH_TRUST_PROXY_HEADER", "X-Forwarded-For"):
        assert rate_limit._client_key(req) == "9.9.9.9"


def test_client_key_falls_back_when_proxy_header_absent():
    # Configured header but request didn't carry it -> socket peer.
    req = _ProxyReq(headers={}, peer="1.2.3.4")
    with patch.object(rate_limit, "OAUTH_TRUST_PROXY_HEADER", "X-Forwarded-For"):
        assert rate_limit._client_key(req) == "1.2.3.4"


# ---------------------------------------------------------------------------
# Body-size middleware: no duplicate response.start (Finding #2)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_body_limit_no_double_start_when_response_already_started():
    from mcp_memory_service.config import OAUTH_MAX_BODY_BYTES

    async def app(scope, receive, send):
        # Streaming-style handler: flush headers BEFORE reading the body.
        await send({"type": "http.response.start", "status": 200, "headers": []})
        while True:
            msg = await receive()
            if msg["type"] == "http.disconnect" or not msg.get("more_body"):
                break
        await send({"type": "http.response.body", "body": b"late"})

    middleware = BodySizeLimitMiddleware(app)
    scope = {"type": "http", "path": "/oauth/token", "headers": []}

    chunks = [{
        "type": "http.request",
        "body": b"x" * (OAUTH_MAX_BODY_BYTES + 10),
        "more_body": False,
    }]

    async def receive():
        return chunks.pop(0) if chunks else {"type": "http.disconnect"}

    sent = []

    async def send(message):
        sent.append(message)

    # Must not raise (a duplicate http.response.start would in a real server).
    await middleware(scope, receive, send)

    starts = [m for m in sent if m["type"] == "http.response.start"]
    assert len(starts) == 1, "headers already sent -> 413 must not be injected"
    assert starts[0]["status"] == 200


# ---------------------------------------------------------------------------
# client_credentials authorization bypass (reported privately, 2026-08-22)
#
# Chain that was possible: register a public client (auth method "none") limited
# to authorization_code, receive a usable client_secret from the registration
# response anyway, then present that secret at the token endpoint with
# grant_type=client_credentials. The grant authenticated the secret without
# checking either the registered grant types or the registered auth method, and
# issued a read+write bearer token — no owner API key involved.
# ---------------------------------------------------------------------------

REG_PATCH = "mcp_memory_service.web.oauth.registration.get_oauth_storage"


@pytest.fixture
def gated_dcr(monkeypatch):
    """Precondition for the client_credentials grant since GHSA-6mvm-q4j3-27qg.

    The grant is refused outright while Dynamic Client Registration is open,
    because a self-registered client is not evidence of owner consent. These
    tests are about what happens *after* that policy gate, so they have to
    pass it first. Both spellings are patched: registration.py binds the
    constant at import, authorization.py reads it per call.
    """
    import mcp_memory_service.config as config  # inline import: patched per test
    from mcp_memory_service.web.oauth import registration as registration_mod

    monkeypatch.setattr(config, "DCR_REGISTRATION_KEY", "test-registration-key")
    monkeypatch.setattr(registration_mod, "DCR_REGISTRATION_KEY", "test-registration-key")


async def _register(storage, **kwargs):
    from mcp_memory_service.web.oauth.models import ClientRegistrationRequest
    from mcp_memory_service.web.oauth.registration import register_client

    with patch(REG_PATCH, return_value=storage):
        return await register_client(ClientRegistrationRequest(**kwargs), None)


@pytest.mark.asyncio
async def test_registration_issues_no_secret_to_public_client():
    storage = MemoryOAuthStorage()
    resp = await _register(
        storage,
        client_name="attacker",
        redirect_uris=["https://example.test/cb"],
        grant_types=["authorization_code"],
        token_endpoint_auth_method="none",
    )
    assert resp.client_secret is None, "a public client must never be issued a secret"

    stored = await storage.get_client(resp.client_id)
    assert not stored.client_secret
    # And the stored value must not be usable as one.
    assert await storage.authenticate_client(resp.client_id, "") is False


@pytest.mark.asyncio
async def test_registration_still_issues_secret_to_confidential_client(gated_dcr):
    storage = MemoryOAuthStorage()
    resp = await _register(
        storage,
        client_name="legit",
        redirect_uris=["https://example.test/cb"],
        grant_types=["client_credentials"],
        token_endpoint_auth_method="client_secret_basic",
    )
    assert resp.client_secret
    assert await storage.authenticate_client(resp.client_id, resp.client_secret) is True


@pytest.mark.asyncio
async def test_client_credentials_rejects_unregistered_grant():
    """A client that registered only authorization_code may not use this grant."""
    storage = MemoryOAuthStorage()
    secret = "s3cret-value"
    await storage.store_client(
        RegisteredClient(
            client_id="conf",
            client_secret=secret,
            redirect_uris=["https://example.test/cb"],
            grant_types=["authorization_code"],
            response_types=["code"],
            token_endpoint_auth_method="client_secret_basic",
            client_name="C",
            created_at=0,
        )
    )
    with patch(AUTH_PATCH, return_value=storage):
        with pytest.raises(HTTPException) as exc:
            await _handle_client_credentials_grant("conf", secret)
    assert exc.value.status_code == 400
    assert exc.value.detail["error"] == "unauthorized_client"


@pytest.mark.asyncio
async def test_client_credentials_rejects_public_client_with_secret(gated_dcr):
    """Auth method is read off the stored client, not inferred from the request."""
    storage = MemoryOAuthStorage()
    secret = "leaked-from-registration"
    await storage.store_client(
        RegisteredClient(
            client_id="pubc",
            client_secret=secret,
            # Deliberately permissive on the grant so the auth-method check is
            # what has to reject this, not the grant check.
            grant_types=["authorization_code", "client_credentials"],
            response_types=["code"],
            token_endpoint_auth_method="none",
            client_name="C",
            created_at=0,
        )
    )
    with patch(AUTH_PATCH, return_value=storage):
        with pytest.raises(HTTPException) as exc:
            await _handle_client_credentials_grant("pubc", secret)
    assert exc.value.status_code == 401
    assert exc.value.detail["error"] == "invalid_client"


@pytest.mark.asyncio
async def test_client_credentials_allows_properly_registered_client(gated_dcr):
    storage = MemoryOAuthStorage()
    secret = "s3cret-value"
    await storage.store_client(
        RegisteredClient(
            client_id="cc",
            client_secret=secret,
            grant_types=["client_credentials"],
            response_types=["code"],
            token_endpoint_auth_method="client_secret_basic",
            client_name="C",
            created_at=0,
        )
    )
    with patch(AUTH_PATCH, return_value=storage):
        resp = await _handle_client_credentials_grant("cc", secret)
    assert resp.access_token
    assert resp.scope == "read write"


@pytest.mark.asyncio
async def test_full_reported_chain_is_broken_end_to_end(gated_dcr):
    """The reported chain, start to finish, must not yield a token."""
    storage = MemoryOAuthStorage()
    reg = await _register(
        storage,
        client_name="attacker",
        redirect_uris=["https://example.test/cb"],
        grant_types=["authorization_code"],
        token_endpoint_auth_method="none",
    )

    # Step 1 already fails: there is no secret to carry into step 2.
    assert reg.client_secret is None

    # Even handed the stored value directly, the grant refuses it.
    stored = await storage.get_client(reg.client_id)
    with patch(AUTH_PATCH, return_value=storage):
        with pytest.raises(HTTPException) as exc:
            await _handle_client_credentials_grant(
                reg.client_id, stored.client_secret or "anything"
            )
    assert exc.value.status_code == 401
