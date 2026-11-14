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
OAuth 2.1 Authorization Server implementation for MCP Memory Service.

Implements OAuth 2.1 authorization code flow and token endpoints.
"""

import html
import time
import json
import hashlib
import logging
import base64
import secrets
from typing import Optional, Tuple
from urllib.parse import urlencode, urlsplit
from fastapi import APIRouter, Depends, HTTPException, status, Form, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
import jwt

from ...config import (
    OAUTH_ISSUER,
    OAUTH_ACCESS_TOKEN_EXPIRE_MINUTES,
    OAUTH_AUTHORIZATION_CODE_EXPIRE_MINUTES,
    OAUTH_REFRESH_TOKEN_EXPIRE_DAYS,
    API_KEY,
    DCR_REGISTRATION_KEY,
    get_jwt_algorithm,
    get_jwt_signing_key,
)
from .models import RegisteredClient, TokenResponse
from .middleware import scope_within_ceiling
from .storage import get_oauth_storage
from .rate_limit import auth_rate_limit
from .redirect import (
    build_redirect_url as _build_redirect_url,
    reject_ambiguous_redirect_uri,
    resolve_registered_redirect_uri,
)
from .limits import (
    PKCE_CODE_VERIFIER_MIN_LEN,
    PKCE_CODE_VERIFIER_MAX_LEN,
    PKCE_CODE_CHALLENGE_MIN_LEN,
    PKCE_CODE_CHALLENGE_MAX_LEN,
    SUPPORTED_CODE_CHALLENGE_METHODS,
    MAX_AUTHORIZATION_CODE_LEN,
    MAX_CLIENT_SECRET_LEN,
    MAX_REFRESH_TOKEN_LEN,
    MAX_SCOPE_LEN,
    MAX_STATE_LEN,
    reject_if_too_long,
)

OFFLINE_ACCESS_SCOPE = "offline_access"

logger = logging.getLogger(__name__)

router = APIRouter()



def parse_basic_auth(
    authorization_header: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse HTTP Basic authentication header.

    Returns:
        Tuple of (client_id, client_secret) or (None, None) if not valid
    """
    if not authorization_header:
        return None, None

    try:
        # Check if it's Basic authentication
        if not authorization_header.startswith("Basic "):
            return None, None

        # Extract and decode the credentials
        encoded_credentials = authorization_header[6:]  # Remove 'Basic ' prefix
        decoded_credentials = base64.b64decode(encoded_credentials).decode("utf-8")

        # Split username:password
        if ":" not in decoded_credentials:
            return None, None

        client_id, client_secret = decoded_credentials.split(":", 1)
        return client_id, client_secret

    except Exception:
        logger.debug("Failed to parse Basic auth header")
        return None, None


def create_access_token(client_id: str, scope: Optional[str] = None) -> tuple[str, int]:
    """
    Create a JWT access token for the given client.

    Uses RS256 with RSA key pair if available, otherwise falls back to HS256.

    Returns:
        Tuple of (token, expires_in_seconds)
    """
    expires_in = OAUTH_ACCESS_TOKEN_EXPIRE_MINUTES * 60
    expire_time = time.time() + expires_in

    payload = {
        "iss": OAUTH_ISSUER,
        "sub": client_id,
        "aud": "mcp-memory-service",
        "exp": expire_time,
        "iat": time.time(),
        "scope": scope or "read write",
    }

    algorithm = get_jwt_algorithm()
    signing_key = get_jwt_signing_key()

    logger.debug("Creating JWT token")
    token = jwt.encode(payload, signing_key, algorithm=algorithm)
    return token, expires_in


def _scope_has_offline_access(scope: Optional[str]) -> bool:
    """Return True if the scope string explicitly requests offline_access.

    Per OIDC convention (also adopted by MCP SEP-2207), refresh tokens are
    only issued when the client asks for them via the ``offline_access``
    scope. Clients that don't opt in keep their existing single-token flow.
    """
    if not scope:
        return False
    return OFFLINE_ACCESS_SCOPE in scope.split()


def _is_scope_subset(requested: Optional[str], original: Optional[str]) -> bool:
    """Return True if every token in ``requested`` appears in ``original``.

    Used by the refresh grant to enforce RFC 6749 §6: clients may narrow
    the granted scope on refresh but must not broaden it.
    """
    if not requested:
        return True
    if not original:
        return False
    original_set = set(original.split())
    return set(requested.split()).issubset(original_set)


async def create_refresh_token(
    client_id: str,
    scope: Optional[str] = None,
    parent_token: Optional[str] = None,
) -> tuple[str, int]:
    """
    Create and persist an opaque refresh token for the given client.

    Refresh tokens are DB-backed (not JWT) so they can be revoked on
    rotation or logout — a property JWTs cannot provide without an
    additional blocklist.

    Args:
        client_id: Authenticated client the token is bound to.
        scope: Space-separated granted scope (preserved from the original
            authorization for rotation parity).
        parent_token: The refresh token this one supersedes, recorded
            on the row for rotation-chain audit.

    Returns:
        Tuple of (token, expires_in_seconds).
    """
    expires_in = OAUTH_REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60
    token = get_oauth_storage().generate_refresh_token()
    await get_oauth_storage().store_refresh_token(
        token=token,
        client_id=client_id,
        scope=scope,
        expires_in=expires_in,
        parent_token=parent_token,
    )
    return token, expires_in


def _without(scope: Optional[str], token: str) -> Optional[str]:
    """``scope`` with ``token`` removed, or None when nothing is left."""
    return " ".join(t for t in (scope or "").split() if t != token) or None


def _may_hold_admin(client: Optional[RegisteredClient]) -> bool:
    """Whether ``admin`` may be granted to this client.

    Only to a client whose registered ceiling names it, and only while
    registration is gated by MCP_DCR_REGISTRATION_KEY. With registration open
    anyone can register a client and pick its ceiling, so the ceiling says
    nothing about who is asking (GHSA-w8p9-5v6q-h9ch).
    """
    if not DCR_REGISTRATION_KEY or client is None:
        return False
    return "admin" in (client.allowed_scope or "").split()


def _grantable_scope(scope: Optional[str], client: Optional[RegisteredClient]) -> Optional[str]:
    """Drop ``admin`` from ``scope`` unless this client may hold it.

    The scope is narrowed rather than refused because clients commonly request
    every scope the server advertises.
    """
    if _may_hold_admin(client):
        return scope
    return _without(scope, "admin")


def _bind_scope_to_ceiling(
    scope: Optional[str], ceiling: Optional[str]
) -> Tuple[Optional[str], bool]:
    """Apply a client's registered scope ceiling. Returns ``(scope, denied)``.

    offline_access is a protocol scope (refresh-token request), not a
    data-access grant, so it is exempt from the data-scope ceiling. A client
    with no ceiling (``None``) is unaffected.
    """
    if ceiling is None:
        return scope, False
    data_scope = _without(scope, OFFLINE_ACCESS_SCOPE)
    if data_scope:
        return scope, not scope_within_ceiling(data_scope, ceiling)
    # No data scope requested. create_access_token() defaults an empty scope to
    # the global "read write", which would ESCAPE a store-scoped ceiling. Bind
    # the grant to the ceiling instead (never conferring admin), preserving an
    # explicit offline_access.
    ceil = _without(ceiling, "admin")
    offline = OFFLINE_ACCESS_SCOPE if _scope_has_offline_access(scope) else None
    bound = " ".join(t for t in (ceil, offline) if t) or None
    return bound, not ceil  # an admin-only ceiling has nothing to grant here


async def _resolve_authorize_request(
    client_id: str, redirect_uri: Optional[str], scope: Optional[str]
) -> Tuple[RegisteredClient, str, Optional[str], bool]:
    """Resolve what an authorize request would grant, and to whom.

    Returns ``(client, validated_redirect_uri, scope, denied)``. GET and POST
    both go through here, so the page shows exactly what the POST will issue.
    """
    validated_redirect_uri = await validate_redirect_uri(client_id, redirect_uri)
    # Also here, not only at registration: a client registered before the
    # check existed still has its URI in the store.
    reject_ambiguous_redirect_uri(validated_redirect_uri)
    client = await get_oauth_storage().get_client(client_id)
    scope, denied = _bind_scope_to_ceiling(
        _grantable_scope(scope, client), client.allowed_scope
    )
    return client, validated_redirect_uri, scope, denied


def _callback_url(redirect_uri: str, params: dict[str, str], state: Optional[str]) -> str:
    """The client's callback URL with ``params``, echoing ``state`` when one was sent."""
    if state:
        params = {**params, "state": state}
    return _build_redirect_url(redirect_uri, params)


def _scope_denied_redirect(redirect_uri: str, state: Optional[str]) -> RedirectResponse:
    logger.warning("Authorization denied: requested scope exceeds client ceiling")
    error_params = {
        "error": "invalid_scope",
        "error_description": "Requested scope exceeds the scope this client is registered for",
    }
    return RedirectResponse(url=_callback_url(redirect_uri, error_params, state))


async def validate_redirect_uri(client_id: str, redirect_uri: Optional[str]) -> str:
    """Validate redirect URI against registered client and return the trusted value."""
    client = await get_oauth_storage().get_client(client_id)
    if not client:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_client",
                "error_description": "Invalid client_id",
            },
        )

    return resolve_registered_redirect_uri(client.redirect_uris, redirect_uri)


async def _validate_pkce_authorize_params(
    client_id: str,
    code_challenge: Optional[str],
    code_challenge_method: Optional[str],
) -> None:
    """
    Enforce PKCE policy at the authorization endpoint.

    * ``code_challenge_method`` (when supplied) must be S256 — "plain" and any
      other method are rejected (OAuth 2.1 §7.5.2).
    * ``code_challenge`` (when supplied) must be a sane length.
    * Public clients (token_endpoint_auth_method=none) MUST supply a
      code_challenge — PKCE is mandatory for them (OAuth 2.1 §7.5.2).
    """
    if code_challenge_method is not None and code_challenge_method not in SUPPORTED_CODE_CHALLENGE_METHODS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_request",
                "error_description": "Unsupported code_challenge_method; only S256 is supported",
            },
        )

    if code_challenge is not None:
        if not (PKCE_CODE_CHALLENGE_MIN_LEN <= len(code_challenge) <= PKCE_CODE_CHALLENGE_MAX_LEN):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "invalid_request",
                    "error_description": (
                        f"code_challenge must be {PKCE_CODE_CHALLENGE_MIN_LEN}-"
                        f"{PKCE_CODE_CHALLENGE_MAX_LEN} characters"
                    ),
                },
            )

    if code_challenge is None:
        client = await get_oauth_storage().get_client(client_id)
        if client and client.token_endpoint_auth_method == "none":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "invalid_request",
                    "error_description": "code_challenge is required for public clients (PKCE)",
                },
            )


def _authorize_form_query(
    response_type: str,
    client_id: str,
    redirect_uri: Optional[str],
    scope: Optional[str],
    state: Optional[str],
    code_challenge: Optional[str],
    code_challenge_method: Optional[str],
) -> str:
    """Build the POST form-action query string from validated parameters only.

    The raw inbound query string is NEVER reflected back into the page — it is
    attacker-controlled and reflecting it caused a reflected-XSS sink. Instead we
    re-serialize the known, server-parsed parameters via ``urlencode`` and
    HTML-escape the result for the attribute context.
    """
    params = {"response_type": response_type, "client_id": client_id}
    if redirect_uri:
        params["redirect_uri"] = redirect_uri
    if scope:
        params["scope"] = scope
    if state:
        params["state"] = state
    if code_challenge:
        params["code_challenge"] = code_challenge
    if code_challenge_method:
        params["code_challenge_method"] = code_challenge_method
    return html.escape(urlencode(params), quote=True)


_MAX_DISPLAYED_CLIENT_NAME_LEN = 80


def _redirect_destination(redirect_uri: str) -> str:
    """The part of a redirect URI the owner has to recognise.

    The hostname only, so userinfo cannot dress the URI up
    (``https://claude.ai@evil.example/cb`` shows ``evil.example``). A native
    client's custom scheme has no host; the whole URI is shown then.
    """
    parts = urlsplit(redirect_uri)
    if not parts.hostname:
        return redirect_uri
    return f"{parts.hostname}:{parts.port}" if parts.port else parts.hostname


def _authorize_page_response(
    query_string: str,
    client: RegisteredClient,
    redirect_uri: str,
    scope: Optional[str],
    error: Optional[str] = None,
    status_code: int = 200,
) -> HTMLResponse:
    """Render the consent page, with headers that forbid framing it."""
    page = _build_authorize_page(
        query_string,
        error,
        client_name=client.client_name,
        destination=_redirect_destination(redirect_uri),
        # create_access_token() turns an empty scope into "read write".
        scope=scope or "read write",
    )
    return HTMLResponse(
        page,
        status_code=status_code,
        headers={
            "X-Frame-Options": "DENY",
            "Content-Security-Policy": "frame-ancestors 'none'",
        },
    )


def _build_authorize_page(
    query_string: str,
    error: Optional[str] = None,
    *,
    client_name: Optional[str],
    destination: str,
    scope: str,
) -> str:
    """Build the HTML authorization/login page.

    ``query_string`` MUST already be HTML-attribute-safe (see
    ``_authorize_form_query``). Everything else is HTML-escaped here:
    ``client_name`` and ``destination`` come from a client registration,
    which anyone can make while registration is open.
    """
    name = (client_name or "").strip()[:_MAX_DISPLAYED_CLIENT_NAME_LEN] or "(none given)"
    safe_name = html.escape(name, quote=True)
    safe_destination = html.escape(destination, quote=True)
    safe_scope = html.escape(", ".join(scope.split()), quote=True)
    error_html = ""
    if error:
        safe_error = html.escape(error, quote=True)
        error_html = f'<div style="color:#ef4444;background:#fef2f2;border:1px solid #fecaca;padding:12px;border-radius:8px;margin-bottom:16px;font-size:14px;">{safe_error}</div>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MCP Memory Service - Authorize</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #e2e8f0; display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; }}
  .card {{ background: #1e293b; border-radius: 12px; padding: 32px; width: 100%; max-width: 400px; box-shadow: 0 4px 24px rgba(0,0,0,0.3); }}
  h1 {{ font-size: 20px; margin: 0 0 8px; color: #f8fafc; }}
  p {{ font-size: 14px; color: #94a3b8; margin: 0 0 24px; }}
  dl {{ margin: 0 0 16px; padding: 12px; border: 1px solid #334155; border-radius: 8px; font-size: 14px; }}
  dt {{ font-size: 12px; color: #94a3b8; }}
  dd {{ margin: 2px 0 10px; color: #f8fafc; overflow-wrap: anywhere; }}
  dd:last-child {{ margin-bottom: 0; }}
  dd.dest {{ font-size: 17px; font-weight: 600; }}
  p.warn {{ color: #fbbf24; margin: 0 0 16px; }}
  label {{ display: block; font-size: 13px; font-weight: 500; color: #cbd5e1; margin-bottom: 6px; }}
  input[type=password] {{ width: 100%; padding: 10px 12px; border: 1px solid #334155; border-radius: 8px; background: #0f172a; color: #f8fafc; font-size: 15px; box-sizing: border-box; }}
  input[type=password]:focus {{ outline: none; border-color: #3b82f6; box-shadow: 0 0 0 3px rgba(59,130,246,0.2); }}
  button {{ width: 100%; padding: 10px; margin-top: 16px; background: #3b82f6; color: white; border: none; border-radius: 8px; font-size: 15px; font-weight: 500; cursor: pointer; }}
  button:hover {{ background: #2563eb; }}
</style>
</head>
<body>
<div class="card">
  <h1>MCP Memory Service</h1>
  <p>An application is asking for access to your memories.</p>
  <dl>
    <dt>Access will be sent to</dt>
    <dd class="dest">{safe_destination}</dd>
    <dt>Application name (chosen by the application, not verified)</dt>
    <dd>{safe_name}</dd>
    <dt>Access it will get</dt>
    <dd>{safe_scope}</dd>
  </dl>
  <p class="warn">Only continue if you started this connection yourself and recognise the destination above. If you got here from a link someone sent you, close this page.</p>
  {error_html}
  <form method="POST" action="authorize?{query_string}">
    <label for="api_key">API Key</label>
    <input type="password" id="api_key" name="api_key" required autofocus autocomplete="current-password">
    <button type="submit">Authorize</button>
  </form>
</div>
</body>
</html>"""


def _validate_authorize_param_lengths(
    scope: Optional[str], state: Optional[str]
) -> None:
    """Bound the free-form authorize parameters that get echoed/stored."""
    reject_if_too_long(scope, MAX_SCOPE_LEN, "scope")
    reject_if_too_long(state, MAX_STATE_LEN, "state")


@router.get("/authorize", dependencies=[Depends(auth_rate_limit)])
async def authorize_get(
    request: Request,
    response_type: str = Query(..., description="OAuth response type"),
    client_id: str = Query(..., description="OAuth client identifier"),
    redirect_uri: Optional[str] = Query(None, description="Redirection URI"),
    scope: Optional[str] = Query(None, description="Requested scope"),
    state: Optional[str] = Query(None, description="Opaque value for CSRF protection"),
    code_challenge: Optional[str] = Query(None, description="PKCE code challenge"),
    code_challenge_method: Optional[str] = Query(
        None, description="PKCE code challenge method (S256)"
    ),
):
    """
    OAuth 2.1 Authorization endpoint (GET).

    Shows a login page where the user must enter their API key
    to approve the authorization request.
    """
    logger.info("Authorization page requested")

    _validate_authorize_param_lengths(scope, state)

    # Validate client and redirect_uri before showing the form
    client, validated_redirect_uri, granted_scope, denied = await _resolve_authorize_request(
        client_id, redirect_uri, scope
    )

    if response_type != "code":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "unsupported_response_type",
                "error_description": "Only 'code' response type is supported",
            },
        )

    await _validate_pkce_authorize_params(client_id, code_challenge, code_challenge_method)

    # Show login form — re-serialize the validated params (never reflect the raw
    # query). The form carries the scope shown on the page, not the one asked for.
    form_query = _authorize_form_query(
        response_type, client_id, redirect_uri, granted_scope, state,
        code_challenge, code_challenge_method,
    )
    if denied:
        return _scope_denied_redirect(validated_redirect_uri, state)
    return _authorize_page_response(form_query, client, validated_redirect_uri, granted_scope)


def _redirect_page(redirect_url: str) -> HTMLResponse:
    """Send the browser to the client's callback after a granted authorization.

    Uses HTML meta-refresh + JS redirect for maximum popup compatibility.
    Some OAuth clients (Claude.ai) use popups where HTTP 302 from a form POST
    can be unreliable across cross-origin boundaries.
    """
    # HTML-attribute-escape the URL for the meta refresh tag.
    # For the <script> context, json.dumps() produces a valid JS string
    # literal but does NOT escape the ``</script>`` sequence, which would
    # close the script element in HTML parsing before JS parsing begins.
    # Replace ``</`` with ``<\/`` so the redirect URL cannot break out of
    # the script element even if the allowlist regresses.
    # ``validate_redirect_uri`` already allowlists the URI; both escapes
    # are defense-in-depth.
    safe_url_attr = html.escape(redirect_url, quote=True)
    safe_url_js = json.dumps(redirect_url).replace("</", "<\\/")
    return HTMLResponse(f"""<!DOCTYPE html>
<html><head>
<meta http-equiv=\"refresh\" content=\"0;url={safe_url_attr}\">
<script>window.location.href = {safe_url_js};</script>
</head><body>Redirecting...</body></html>""")


async def _grant_authorization_code(
    client_id: str,
    redirect_uri: str,
    scope: Optional[str],
    state: Optional[str],
    code_challenge: Optional[str],
    code_challenge_method: Optional[str],
) -> HTMLResponse:
    """Store an authorization code and send the browser to the client's callback.

    ``redirect_uri`` is the validated one, so a failure here can be reported
    to the client as well.
    """
    try:
        auth_code = get_oauth_storage().generate_authorization_code()
        await get_oauth_storage().store_authorization_code(
            code=auth_code,
            client_id=client_id,
            redirect_uri=redirect_uri,
            scope=scope,
            expires_in=OAUTH_AUTHORIZATION_CODE_EXPIRE_MINUTES * 60,
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method,
        )
        logger.info("Authorization granted, redirecting to callback")
        return _redirect_page(_callback_url(redirect_uri, {"code": auth_code}, state))
    except HTTPException:
        raise
    except Exception:
        logger.error("Authorization error occurred", exc_info=True)
        error_params = {
            "error": "server_error",
            "error_description": "Internal server error",
        }
        return RedirectResponse(url=_callback_url(redirect_uri, error_params, state))


@router.post("/authorize", dependencies=[Depends(auth_rate_limit)])
async def authorize_post(
    request: Request,
    response_type: str = Query(..., description="OAuth response type"),
    client_id: str = Query(..., description="OAuth client identifier"),
    redirect_uri: Optional[str] = Query(None, description="Redirection URI"),
    scope: Optional[str] = Query(None, description="Requested scope"),
    state: Optional[str] = Query(None, description="Opaque value for CSRF protection"),
    code_challenge: Optional[str] = Query(None, description="PKCE code challenge"),
    code_challenge_method: Optional[str] = Query(
        None, description="PKCE code challenge method (S256)"
    ),
    api_key: str = Form(..., description="API key for authorization"),
):
    """
    OAuth 2.1 Authorization endpoint (POST).

    Validates the API key and issues an authorization code if correct.
    """
    logger.info("Authorization form submitted")

    _validate_authorize_param_lengths(scope, state)
    await _validate_pkce_authorize_params(client_id, code_challenge, code_challenge_method)

    # Resolved before the key check so a wrong key re-renders the same details.
    client, validated_redirect_uri, scope, denied = await _resolve_authorize_request(
        client_id, redirect_uri, scope
    )

    form_query = _authorize_form_query(
        response_type, client_id, redirect_uri, scope, state,
        code_challenge, code_challenge_method,
    )

    # Validate API key
    if not API_KEY or not secrets.compare_digest(api_key.encode(), API_KEY.encode()):
        logger.warning("Authorization denied: invalid API key")
        return _authorize_page_response(
            form_query, client, validated_redirect_uri, scope,
            error="Invalid API key. Please try again.", status_code=403,
        )

    if denied:
        return _scope_denied_redirect(validated_redirect_uri, state)

    return await _grant_authorization_code(
        client_id, validated_redirect_uri, scope, state,
        code_challenge, code_challenge_method,
    )


async def _handle_authorization_code_grant(
    final_client_id: str,
    final_client_secret: Optional[str],
    code: Optional[str],
    redirect_uri: Optional[str],
    code_verifier: Optional[str] = None,
) -> TokenResponse:
    """Handle OAuth authorization_code grant type."""
    if not code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_request",
                "error_description": "Missing required parameter: code",
            },
        )

    # RFC 7636 §4.1: code_verifier must be 43-128 characters. Validate before
    # any hashing so a caller cannot push unbounded input into hashlib.sha256.
    if code_verifier is not None and not (
        PKCE_CODE_VERIFIER_MIN_LEN <= len(code_verifier) <= PKCE_CODE_VERIFIER_MAX_LEN
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_grant",
                "error_description": (
                    f"code_verifier must be {PKCE_CODE_VERIFIER_MIN_LEN}-"
                    f"{PKCE_CODE_VERIFIER_MAX_LEN} characters"
                ),
            },
        )

    if not final_client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_request",
                "error_description": "Missing required parameter: client_id",
            },
        )

    # Authenticate client — but allow public clients using PKCE (OAuth 2.1 §2.1)
    # Public clients (e.g. claude.ai) may not send a client_secret; they prove
    # identity via PKCE code_verifier instead.
    if final_client_secret:
        if not await get_oauth_storage().authenticate_client(
            final_client_id, final_client_secret
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "error": "invalid_client",
                    "error_description": "Client authentication failed",
                },
            )
    else:
        # Public client — verify it exists and is actually a public client
        client = await get_oauth_storage().get_client(final_client_id)
        if not client or client.token_endpoint_auth_method != "none":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "error": "invalid_client",
                    "error_description": "Client authentication failed",
                },
            )
        # PKCE is mandatory for public clients (OAuth 2.1 §7.5.2)
        if not code_verifier:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "invalid_request",
                    "error_description": "code_verifier required for public clients",
                },
            )

    # Get and consume authorization code
    code_data = await get_oauth_storage().get_authorization_code(code)
    if not code_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_grant",
                "error_description": "Invalid or expired authorization code",
            },
        )

    # Validate client_id matches
    if code_data["client_id"] != final_client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_grant",
                "error_description": "Authorization code was issued to a different client",
            },
        )

    # Validate redirect_uri if provided
    if redirect_uri and code_data["redirect_uri"] != redirect_uri:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_grant",
                "error_description": "redirect_uri does not match the one used in authorization request",
            },
        )

    # PKCE verification
    stored_challenge = code_data.get("code_challenge")
    if stored_challenge:
        if not code_verifier:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "invalid_grant",
                    "error_description": "code_verifier required for PKCE",
                },
            )

        computed = (
            base64.urlsafe_b64encode(
                hashlib.sha256(code_verifier.encode("ascii")).digest()
            )
            .rstrip(b"=")
            .decode("ascii")
        )
        if computed != stored_challenge:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "invalid_grant",
                    "error_description": "PKCE code_verifier does not match code_challenge",
                },
            )

    # Create access token
    access_token, expires_in = create_access_token(final_client_id, code_data["scope"])

    # Store access token for validation
    await get_oauth_storage().store_access_token(
        token=access_token,
        client_id=final_client_id,
        scope=code_data["scope"],
        expires_in=expires_in,
    )

    # Issue a refresh token only when the client explicitly requested it
    # via the "offline_access" scope (OIDC convention / MCP SEP-2207).
    # Clients that don't opt in keep their current single-token behavior,
    # so this change is non-breaking.
    refresh_token: Optional[str] = None
    if _scope_has_offline_access(code_data["scope"]):
        refresh_token, _ = await create_refresh_token(
            client_id=final_client_id,
            scope=code_data["scope"],
        )
        logger.info("Access + refresh tokens issued (offline_access requested)")
    else:
        logger.info("Access token issued")

    return TokenResponse(
        access_token=access_token,
        token_type="Bearer",
        expires_in=expires_in,
        refresh_token=refresh_token,
        scope=code_data["scope"],
    )


async def _authenticate_refresh_client(
    client: RegisteredClient,
    bound_client_id: str,
    final_client_id: Optional[str],
    final_client_secret: Optional[str],
) -> None:
    """Enforce client identification and authentication on a refresh."""
    # Public clients still MUST send client_id (RFC 6749 §6) — the token's
    # internal binding is not a substitute for the explicit parameter.
    if client.token_endpoint_auth_method == "none" and not final_client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_request",
                "error_description": "Missing required parameter: client_id",
            },
        )

    # Confidential clients must authenticate on refresh (RFC 6749 §6).
    # Public clients (token_endpoint_auth_method=none) do not present a secret;
    # the rotated refresh token itself is the binding.
    if client.token_endpoint_auth_method != "none":
        if not final_client_secret:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "error": "invalid_client",
                    "error_description": "Client authentication required",
                },
            )
        if not await get_oauth_storage().authenticate_client(
            bound_client_id, final_client_secret
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "error": "invalid_client",
                    "error_description": "Client authentication failed",
                },
            )


def _refreshed_scope(
    requested_scope: Optional[str],
    original_scope: Optional[str],
    client: RegisteredClient,
) -> Optional[str]:
    """Resolve the scope of a refreshed token.

    RFC 6749 §6: scope may be narrowed on refresh but not broadened. A refresh
    token issued before admin was restricted must not keep minting it.
    """
    if requested_scope is not None and not _is_scope_subset(requested_scope, original_scope):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_scope",
                "error_description": "Requested scope exceeds originally granted scope",
            },
        )
    scope = requested_scope or original_scope
    granted = _grantable_scope(scope, client)
    if granted != scope and not _without(granted, OFFLINE_ACCESS_SCOPE):
        # admin was all the data access there was. Passing the empty rest on
        # would let create_access_token() default it to the global
        # "read write", wider than what was granted.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_scope",
                "error_description": "The admin scope is no longer granted to this client",
            },
        )
    return granted


async def _handle_refresh_token_grant(
    final_client_id: Optional[str],
    final_client_secret: Optional[str],
    refresh_token_value: Optional[str],
    requested_scope: Optional[str],
) -> TokenResponse:
    """
    Handle the OAuth 2.1 refresh_token grant (RFC 6749 §6).

    Implements the OAuth 2.1 rotation requirement (§4.3.1): every successful
    refresh issues a new refresh token AND revokes the one that was presented,
    so a stolen refresh token is single-use.
    """
    if not refresh_token_value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_request",
                "error_description": "Missing required parameter: refresh_token",
            },
        )

    token_data = await get_oauth_storage().get_refresh_token(refresh_token_value)
    if not token_data:
        # Unknown, expired, or already-rotated token. If it was previously
        # rotated (revoked), this looks like a replay — the legitimate client
        # would not present a stale token. We cannot distinguish replay from
        # the attacker case, so per OAuth 2.1 §4.3.1 we also revoke every
        # other live token in the same rotation chain (compromise mitigation).
        # `revoke_refresh_token_chain` is a no-op (returns 0) for genuinely
        # unknown tokens, so this is safe to call unconditionally here.
        chain_revoked = await get_oauth_storage().revoke_refresh_token_chain(
            refresh_token_value
        )
        if chain_revoked:
            logger.warning(
                "Refresh token replay detected; revoked %d additional "
                "token(s) in the rotation chain",
                chain_revoked,
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_grant",
                "error_description": "Refresh token is invalid, expired, or revoked",
            },
        )

    bound_client_id = token_data["client_id"]

    # If the client authenticated, it must match the refresh token binding.
    # Public clients may omit client_id in the request body (the token itself
    # is the binding), but any supplied client_id must match.
    if final_client_id and final_client_id != bound_client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_grant",
                "error_description": "Refresh token was issued to a different client",
            },
        )

    client = await get_oauth_storage().get_client(bound_client_id)
    if not client:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_grant",
                "error_description": "Client no longer registered",
            },
        )

    await _authenticate_refresh_client(
        client, bound_client_id, final_client_id, final_client_secret
    )
    granted_scope = _refreshed_scope(requested_scope, token_data["scope"], client)

    # Rotation: atomically revoke the presented refresh token. If we lose the
    # race (another refresh already consumed it), fail the grant — defense in
    # depth on top of the unknown-token check above.
    if not await get_oauth_storage().revoke_refresh_token(refresh_token_value):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_grant",
                "error_description": "Refresh token is invalid, expired, or revoked",
            },
        )

    access_token, expires_in = create_access_token(bound_client_id, granted_scope)
    await get_oauth_storage().store_access_token(
        token=access_token,
        client_id=bound_client_id,
        scope=granted_scope,
        expires_in=expires_in,
    )

    new_refresh_token, _ = await create_refresh_token(
        client_id=bound_client_id,
        scope=granted_scope,
        parent_token=refresh_token_value,
    )

    logger.info("Access + refresh tokens issued via refresh_token grant (rotated)")
    return TokenResponse(
        access_token=access_token,
        token_type="Bearer",
        expires_in=expires_in,
        refresh_token=new_refresh_token,
        scope=granted_scope,
    )


def _client_credentials_scope(
    requested_scope: Optional[str], ceiling: Optional[str] = None
) -> str:
    """Resolve the scope granted to a client_credentials token.

    ``ceiling`` is the client's registered ``allowed_scope`` (``None`` = no
    ceiling, the historical behaviour). A requested scope is honored verbatim
    (so a client can self-restrict to a store-scoped grant such as
    ``read:projectA``), except that ``admin`` may never be obtained this way,
    and a request that exceeds the client's ceiling is refused — both as HTTP
    400 invalid_scope. With no requested scope the grant defaults to the
    client's ceiling when it has one, else the historical ``read write``.
    """
    if requested_scope and requested_scope.strip():
        requested = requested_scope.strip()
        if "admin" in requested.split():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "invalid_scope",
                    "error_description": "The client_credentials grant cannot issue 'admin' scope",
                },
            )
        if not scope_within_ceiling(requested, ceiling):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "invalid_scope",
                    "error_description": (
                        "Requested scope exceeds the scope this client is "
                        f"registered for ('{ceiling}')"
                    ),
                },
            )
        return requested
    # No requested scope: default to the client's ceiling (else the historical
    # "read write"), but client_credentials must never confer admin even when
    # the registered ceiling includes it — same rule as the requested branch.
    default = ceiling if ceiling else "read write"
    granted = " ".join(t for t in default.split() if t != "admin")
    if not granted:
        # Ceiling was admin-only: there is nothing non-admin to grant, and an
        # empty string would be expanded back to the global "read write" default
        # by create_access_token(). Refuse rather than silently grant everything.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_scope",
                "error_description": "This client has no client_credentials-issuable (non-admin) scope",
            },
        )
    return granted


async def _handle_client_credentials_grant(
    final_client_id: str, final_client_secret: str, requested_scope: Optional[str] = None
) -> TokenResponse:
    """Handle OAuth client_credentials grant type.

    When the client requests a ``scope``, it is honored so machine-to-machine
    clients can mint store-restricted tokens (e.g. ``read:projectA``). The
    request may only narrow, never escalate: ``admin`` cannot be obtained this
    way. With no requested scope the grant keeps its historical default of
    ``read write``.
    """
    # Registration gates the grant, not just at registration time. Refusing
    # only in /oauth/register would leave every client that registered while
    # DCR was open still holding the grant, and those are exactly the
    # self-service registrations GHSA-6mvm-q4j3-27qg describes. If this
    # deployment cannot say who registered a client, it cannot honour a
    # machine-to-machine grant from one.
    from ...config import DCR_REGISTRATION_KEY
    if not DCR_REGISTRATION_KEY:
        logger.warning(
            "Refused client_credentials: registration is open, so a registered "
            "client is not evidence of owner consent"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "unauthorized_client",
                "error_description": (
                    "The client_credentials grant is disabled while Dynamic Client "
                    "Registration is open. Set MCP_DCR_REGISTRATION_KEY, or use "
                    "authorization_code with PKCE."
                ),
            },
        )

    if not final_client_id or not final_client_secret:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_request",
                "error_description": "Missing required parameters: client_id and client_secret",
            },
        )

    # Authenticate client
    if not await get_oauth_storage().authenticate_client(
        final_client_id, final_client_secret
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "invalid_client",
                "error_description": "Client authentication failed",
            },
        )

    # Authentication is not authorization. Two properties of the registration
    # have to hold before this grant may issue a token, and both are read off the
    # stored client rather than taken from the request — the same rule the
    # authorization_code grant already follows.
    client = await get_oauth_storage().get_client(final_client_id)
    if not client:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "invalid_client",
                "error_description": "Client authentication failed",
            },
        )

    # A public client authenticates with PKCE, never with a secret. Accepting one
    # here let a client registered as public act as a confidential one.
    if client.token_endpoint_auth_method == "none":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "invalid_client",
                "error_description": "Client authentication failed",
            },
        )

    # RFC 6749 section 4.4: the grant must be one the client registered for.
    if "client_credentials" not in client.grant_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "unauthorized_client",
                "error_description": "Client is not authorized to use this grant type",
            },
        )

    # Resolve the granted scope (default "read write" or the client's ceiling; a
    # request may narrow it, including to store-scoped grants, but never escalate
    # to "admin" nor beyond the client's registered ceiling).
    granted_scope = _client_credentials_scope(requested_scope, client.allowed_scope)

    # Create access token
    access_token, expires_in = create_access_token(final_client_id, granted_scope)

    # Store access token
    await get_oauth_storage().store_access_token(
        token=access_token,
        client_id=final_client_id,
        scope=granted_scope,
        expires_in=expires_in,
    )

    logger.info("Client credentials token issued")
    return TokenResponse(
        access_token=access_token,
        token_type="Bearer",
        expires_in=expires_in,
        scope=granted_scope,
    )


@router.post("/token", response_model=TokenResponse, dependencies=[Depends(auth_rate_limit)])
async def token(
    request: Request,
    grant_type: str = Form(..., description="OAuth grant type"),
    code: Optional[str] = Form(None, description="Authorization code"),
    redirect_uri: Optional[str] = Form(None, description="Redirection URI"),
    client_id: Optional[str] = Form(None, description="OAuth client identifier"),
    client_secret: Optional[str] = Form(None, description="OAuth client secret"),
    code_verifier: Optional[str] = Form(None, description="PKCE code verifier"),
    refresh_token: Optional[str] = Form(None, description="Refresh token (refresh_token grant)"),
    scope: Optional[str] = Form(None, description="Requested scope on refresh"),
):
    """
    OAuth 2.1 Token endpoint.

    Exchanges authorization codes for access tokens and rotates refresh
    tokens. Supports authorization_code, client_credentials, and
    refresh_token grant types. Accepts both client_secret_post (form data)
    and client_secret_basic (HTTP Basic auth).
    """
    # Bound opaque inputs so a caller cannot force unbounded parsing/hashing.
    reject_if_too_long(code, MAX_AUTHORIZATION_CODE_LEN, "code")
    reject_if_too_long(client_secret, MAX_CLIENT_SECRET_LEN, "client_secret")
    reject_if_too_long(refresh_token, MAX_REFRESH_TOKEN_LEN, "refresh_token")
    reject_if_too_long(scope, MAX_SCOPE_LEN, "scope")

    # Extract client credentials from either HTTP Basic auth or form data
    auth_header = request.headers.get("authorization")
    basic_client_id, basic_client_secret = parse_basic_auth(auth_header)

    # Use Basic auth credentials if available, otherwise fall back to form data
    final_client_id = basic_client_id or client_id
    final_client_secret = basic_client_secret or client_secret

    logger.info("Token request received")

    try:
        if grant_type == "authorization_code":
            return await _handle_authorization_code_grant(
                final_client_id, final_client_secret, code, redirect_uri, code_verifier
            )
        elif grant_type == "client_credentials":
            return await _handle_client_credentials_grant(
                final_client_id, final_client_secret, scope
            )
        elif grant_type == "refresh_token":
            return await _handle_refresh_token_grant(
                final_client_id, final_client_secret, refresh_token, scope
            )

        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "unsupported_grant_type",
                    "error_description": f"Grant type '{grant_type}' is not supported",
                },
            )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception:
        logger.error("Token endpoint error", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "server_error",
                "error_description": "Internal server error",
            },
        )
