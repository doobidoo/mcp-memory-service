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

"""GHSA-w8p9-5v6q-h9ch: the authorize page has to say who is asking.

With MCP_DCR_REGISTRATION_KEY unset anyone can register a client with their
own redirect URI. The owner then authorises on a page that named neither the
client, nor where the code would go, nor the scope, and the scope could be
``admin``. The owner's API key was the only control, and the owner had nothing
to decide with.

Three things are pinned here: the page shows the destination, the
self-reported name and the scope that will be granted; ``admin`` is granted
only to a client whose registration was gated and declared it; the page
cannot be framed.
"""

import base64
import hashlib
import re
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mcp_memory_service.web.oauth import authorization, discovery, registration
from mcp_memory_service.web.oauth.middleware import validate_jwt_token
from mcp_memory_service.web.oauth.models import RegisteredClient
from mcp_memory_service.web.oauth.storage.memory import MemoryOAuthStorage

OWNER_KEY = "owner-api-key"
ATTACKER_CB = "https://attacker.example/cb"
VERIFIER = "v" * 64
CHALLENGE = (
    base64.urlsafe_b64encode(hashlib.sha256(VERIFIER.encode()).digest())
    .rstrip(b"=")
    .decode()
)


@pytest.fixture
def storage(monkeypatch):
    store = MemoryOAuthStorage()
    monkeypatch.setattr(authorization, "get_oauth_storage", lambda: store)
    monkeypatch.setattr(registration, "get_oauth_storage", lambda: store)
    monkeypatch.setattr(authorization, "API_KEY", OWNER_KEY)
    return store


def _set_dcr_key(monkeypatch, key):
    for module in (authorization, registration, discovery):
        monkeypatch.setattr(module, "DCR_REGISTRATION_KEY", key)


@pytest.fixture
def open_dcr(monkeypatch):
    _set_dcr_key(monkeypatch, None)


@pytest.fixture
def gated_dcr(monkeypatch):
    _set_dcr_key(monkeypatch, "registration-key")


@pytest.fixture
def http(storage):
    app = FastAPI()
    app.include_router(registration.router, prefix="/oauth")
    app.include_router(authorization.router, prefix="/oauth")
    app.include_router(discovery.router)
    return TestClient(app, follow_redirects=False)


async def _store_client(storage, allowed_scope, name="Totally Not Evil"):
    await storage.store_client(
        RegisteredClient(
            client_id="c1",
            client_secret="",
            redirect_uris=[ATTACKER_CB],
            token_endpoint_auth_method="none",
            client_name=name,
            created_at=time.time(),
            allowed_scope=allowed_scope,
        )
    )
    return "c1"


def _authorize_params(client_id, scope="read write admin"):
    return {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": ATTACKER_CB,
        "scope": scope,
        "state": "x",
        "code_challenge": CHALLENGE,
        "code_challenge_method": "S256",
    }


def _granted_scope(http, client_id, scope="read write admin"):
    """Run authorize + token as the owner would and return the token's scope."""
    resp = http.post(
        "/oauth/authorize",
        params=_authorize_params(client_id, scope),
        data={"api_key": OWNER_KEY},
    )
    assert resp.status_code == 200, resp.text
    code = re.search(r"code=([^&\"]+)", resp.text).group(1)
    token = http.post(
        "/oauth/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "client_id": client_id,
            "redirect_uri": ATTACKER_CB,
            "code_verifier": VERIFIER,
        },
    )
    assert token.status_code == 200, token.text
    # The signed claim is what the API checks, not the response field.
    return validate_jwt_token(token.json()["access_token"])["scope"]


def test_reported_attack_no_longer_yields_admin(http, open_dcr):
    """The advisory's steps 2-6, end to end."""
    reg = http.post(
        "/oauth/register",
        json={
            "client_name": "Totally Not Evil",
            "redirect_uris": [ATTACKER_CB],
            "token_endpoint_auth_method": "none",
        },
    )
    assert reg.status_code == 201, reg.text
    client_id = reg.json()["client_id"]

    page = http.get("/oauth/authorize", params=_authorize_params(client_id))
    assert page.status_code == 200
    assert "attacker.example" in page.text
    assert "Totally Not Evil" in page.text
    assert "read, write" in page.text
    assert "admin" not in page.text

    assert _granted_scope(http, client_id) == "read write"


def test_open_registration_cannot_declare_admin(http, storage, open_dcr):
    reg = http.post(
        "/oauth/register",
        json={
            "redirect_uris": [ATTACKER_CB],
            "token_endpoint_auth_method": "none",
            "scope": "read write admin",
        },
    )
    assert reg.status_code == 201, reg.text
    assert reg.json()["scope"] == "read write"
    assert _granted_scope(http, reg.json()["client_id"]) == "read write"


@pytest.mark.asyncio
async def test_client_registered_before_the_fix_gets_no_admin(http, storage, open_dcr):
    """A stored admin ceiling from an open registration proves nothing."""
    client_id = await _store_client(storage, "read write admin")
    assert _granted_scope(http, client_id) == "read write"


@pytest.mark.asyncio
async def test_admin_only_request_falls_back_to_default(http, storage, open_dcr):
    client_id = await _store_client(storage, None)
    assert _granted_scope(http, client_id, scope="admin") == "read write"


@pytest.mark.asyncio
async def test_gated_client_that_declared_admin_keeps_it(http, storage, gated_dcr):
    client_id = await _store_client(storage, "read write admin")
    page = http.get("/oauth/authorize", params=_authorize_params(client_id))
    assert "read, write, admin" in page.text
    assert _granted_scope(http, client_id) == "read write admin"


@pytest.mark.asyncio
async def test_gated_client_without_declared_admin_gets_none(http, storage, gated_dcr):
    client_id = await _store_client(storage, None)
    assert _granted_scope(http, client_id) == "read write"


@pytest.mark.asyncio
async def test_refresh_token_issued_before_the_fix_loses_admin(http, storage, open_dcr):
    client_id = await _store_client(storage, None)
    await storage.store_refresh_token(
        token="old-refresh",
        client_id=client_id,
        scope="read write admin offline_access",
        expires_in=3600,
    )
    resp = http.post(
        "/oauth/token",
        data={
            "grant_type": "refresh_token",
            "refresh_token": "old-refresh",
            "client_id": client_id,
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["scope"] == "read write offline_access"


@pytest.mark.asyncio
async def test_refresh_narrowed_to_admin_does_not_widen_to_the_default(
    http, storage, open_dcr
):
    # Stripping admin leaves nothing; an empty scope would be defaulted to the
    # global "read write", which is more than read:projectA.
    client_id = await _store_client(storage, None)
    await storage.store_refresh_token(
        token="old-refresh",
        client_id=client_id,
        scope="read:projectA admin offline_access",
        expires_in=3600,
    )
    resp = http.post(
        "/oauth/token",
        data={
            "grant_type": "refresh_token",
            "refresh_token": "old-refresh",
            "client_id": client_id,
            "scope": "admin",
        },
    )
    assert resp.status_code == 400, resp.text
    assert "access_token" not in resp.text


BACKSLASH_CB = "https://attacker.example\\@claude.ai/cb"


def test_registration_refuses_a_redirect_uri_browsers_read_differently(http, open_dcr):
    # urlsplit reads the host as claude.ai; a browser navigates to attacker.example.
    resp = http.post(
        "/oauth/register",
        json={
            "redirect_uris": [BACKSLASH_CB],
            "token_endpoint_auth_method": "none",
        },
    )
    assert resp.status_code in (400, 422), resp.text


@pytest.mark.asyncio
async def test_stored_backslash_redirect_uri_is_refused_at_authorize(
    http, storage, open_dcr
):
    client_id = await _store_client(storage, None)
    (await storage.get_client(client_id)).redirect_uris[:] = [BACKSLASH_CB]
    params = {**_authorize_params(client_id), "redirect_uri": BACKSLASH_CB}
    page = http.get("/oauth/authorize", params=params)
    granted = http.post("/oauth/authorize", params=params, data={"api_key": OWNER_KEY})
    for resp in (page, granted):
        assert resp.status_code == 400, resp.text
        assert "code=" not in resp.text


@pytest.mark.asyncio
async def test_page_cannot_be_framed_and_wrong_key_keeps_the_details(
    http, storage, open_dcr
):
    client_id = await _store_client(storage, None)
    page = http.get("/oauth/authorize", params=_authorize_params(client_id))
    denied = http.post(
        "/oauth/authorize",
        params=_authorize_params(client_id),
        data={"api_key": "wrong"},
    )
    assert denied.status_code == 403
    for resp in (page, denied):
        assert resp.headers["X-Frame-Options"] == "DENY"
        assert "frame-ancestors 'none'" in resp.headers["Content-Security-Policy"]
        assert "attacker.example" in resp.text


@pytest.mark.asyncio
async def test_client_name_is_escaped_and_bounded(http, storage, open_dcr):
    client_id = await _store_client(storage, None, name="<script>x</script>" + "A" * 500)
    page = http.get("/oauth/authorize", params=_authorize_params(client_id))
    assert "<script>x" not in page.text
    assert "&lt;script&gt;x" in page.text
    assert "A" * 81 not in page.text


@pytest.mark.parametrize(
    "uri, shown",
    [
        ("https://claude.ai/api/mcp/auth_callback", "claude.ai"),
        # userinfo must not pass for the host
        ("https://claude.ai@evil.example/cb", "evil.example"),
        ("http://127.0.0.1:54321/callback", "127.0.0.1:54321"),
        ("cursor://anysphere.cursor-mcp/oauth/callback", "anysphere.cursor-mcp"),
    ],
)
def test_redirect_destination(uri, shown):
    assert authorization._redirect_destination(uri) == shown


def test_admin_is_advertised_only_while_registration_is_gated(http, monkeypatch):
    _set_dcr_key(monkeypatch, None)
    for path in (
        "/.well-known/oauth-protected-resource",
        "/.well-known/oauth-authorization-server/mcp",
    ):
        assert "admin" not in http.get(path).json()["scopes_supported"]
    _set_dcr_key(monkeypatch, "registration-key")
    assert "admin" in http.get("/.well-known/oauth-protected-resource").json()["scopes_supported"]
