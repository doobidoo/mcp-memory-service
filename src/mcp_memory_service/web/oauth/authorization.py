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
from urllib.parse import urlencode
from fastapi import APIRouter, Depends, HTTPException, status, Form, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
import jwt

from ...config import (
    OAUTH_ISSUER,
    OAUTH_ACCESS_TOKEN_EXPIRE_MINUTES,
    OAUTH_AUTHORIZATION_CODE_EXPIRE_MINUTES,
    OAUTH_REFRESH_TOKEN_EXPIRE_DAYS,
    API_KEY,
    get_jwt_algorithm,
    get_jwt_signing_key,
)
from .models import TokenResponse
from .storage import get_oauth_storage
from .rate_limit import auth_rate_limit
from .redirect import (
    build_redirect_url as _build_redirect_url,
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


def _sanitize_log_value(value: object) -> str:
    """Sanitize a user-provided value for safe inclusion in log messages."""
    return str(value).replace("\n", "\\n").replace("\r", "\\r").replace("\x1b", "\\x1b")


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


def _build_authorize_page(query_string: str, error: Optional[str] = None) -> str:
    """Build the HTML authorization/login page.

    ``query_string`` MUST already be HTML-attribute-safe (see
    ``_authorize_form_query``). ``error`` is HTML-escaped here defensively.
    """
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
  <p>Enter your API key to authorize this connection.</p>
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
    if redirect_uri:
        await validate_redirect_uri(client_id, redirect_uri)

    if response_type != "code":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "unsupported_response_type",
                "error_description": "Only 'code' response type is supported",
            },
        )

    await _validate_pkce_authorize_params(client_id, code_challenge, code_challenge_method)

    # Show login form — re-serialize the validated params (never reflect the raw query)
    form_query = _authorize_form_query(
        response_type, client_id, redirect_uri, scope, state,
        code_challenge, code_challenge_method,
    )
    return HTMLResponse(_build_authorize_page(form_query))


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

    form_query = _authorize_form_query(
        response_type, client_id, redirect_uri, scope, state,
        code_challenge, code_challenge_method,
    )

    # Validate API key
    if not API_KEY or not secrets.compare_digest(api_key.encode(), API_KEY.encode()):
        logger.warning("Authorization denied: invalid API key")
        return HTMLResponse(
            _build_authorize_page(
                form_query, error="Invalid API key. Please try again."
            ),
            status_code=403,
        )

    try:
        # Validate redirect_uri against the registered client once and keep the
        # validated value for both success and error redirects.
        validated_redirect_uri = await validate_redirect_uri(client_id, redirect_uri)

        # Generate and store authorization code
        auth_code = get_oauth_storage().generate_authorization_code()
        await get_oauth_storage().store_authorization_code(
            code=auth_code,
            client_id=client_id,
            redirect_uri=validated_redirect_uri,
            scope=scope,
            expires_in=OAUTH_AUTHORIZATION_CODE_EXPIRE_MINUTES * 60,
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method,
        )

        # Redirect with authorization code
        redirect_params = {"code": auth_code}
        if state:
            redirect_params["state"] = state

        redirect_url = _build_redirect_url(validated_redirect_uri, redirect_params)
        logger.info("Authorization granted, redirecting to callback")
        # Use HTML meta-refresh + JS redirect for maximum popup compatibility.
        # Some OAuth clients (Claude.ai) use popups where HTTP 302 from a
        # form POST can be unreliable across cross-origin boundaries.

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

    except HTTPException:
        raise
    except Exception:
        logger.error("Authorization error occurred", exc_info=True)
        error_params = {
            "error": "server_error",
            "error_description": "Internal server error",
        }
        if state:
            error_params["state"] = state
        if validated_redirect_uri:
            return RedirectResponse(
                url=_build_redirect_url(validated_redirect_uri, error_params)
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_params
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

    original_scope = token_data["scope"]

    # RFC 6749 §6: scope may be narrowed on refresh but not broadened.
    if requested_scope is not None and not _is_scope_subset(requested_scope, original_scope):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_scope",
                "error_description": "Requested scope exceeds originally granted scope",
            },
        )
    granted_scope = requested_scope if requested_scope else original_scope

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


async def _handle_client_credentials_grant(
    final_client_id: str, final_client_secret: str
) -> TokenResponse:
    """Handle OAuth client_credentials grant type."""
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

    # Create access token
    access_token, expires_in = create_access_token(final_client_id, "read write")

    # Store access token
    await get_oauth_storage().store_access_token(
        token=access_token,
        client_id=final_client_id,
        scope="read write",
        expires_in=expires_in,
    )

    logger.info("Client credentials token issued")
    return TokenResponse(
        access_token=access_token,
        token_type="Bearer",
        expires_in=expires_in,
        scope="read write",
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
                final_client_id, final_client_secret
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
