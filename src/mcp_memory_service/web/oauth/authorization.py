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
import logging
import base64
import secrets
from typing import Optional, Tuple
from urllib.parse import urlencode, urlparse, ParseResult
from fastapi import APIRouter, HTTPException, status, Form, Query, Request
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

OFFLINE_ACCESS_SCOPE = "offline_access"

logger = logging.getLogger(__name__)

router = APIRouter()

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _sanitize_log_value(value: object) -> str:
    """Sanitize a user-provided value for safe inclusion in log messages."""
    return str(value).replace("\n", "\\n").replace("\r", "\\r").replace("\x1b", "\\x1b")


def _sanitize_state(state: str) -> str:
    """Sanitize the OAuth state parameter to prevent log injection and open redirect abuse."""
    # Allow only alphanumeric, hyphen, underscore, and dot characters (RFC 6749 opaque value)
    import re as _re

    return _re.sub(r"[^A-Za-z0-9\-_.]", "", state)[:128]


def _is_loopback_http_redirect(parsed: ParseResult) -> bool:
    """Return True when the parsed URI is an HTTP loopback redirect."""
    return (
        parsed.scheme.lower() == "http"
        and (parsed.hostname or "").lower() in _LOOPBACK_HOSTS
    )


def _loopback_redirect_matches(registered_uri: str, requested_uri: str) -> bool:
    """
    Match native-app loopback redirects while allowing runtime-assigned ports.

    RFC 8252 recommends loopback redirects for native apps and requires OAuth
    servers to tolerate ephemeral localhost ports. Keep scheme/path strict while
    allowing host aliases inside the loopback set and any runtime port.
    """
    registered = urlparse(registered_uri)
    requested = urlparse(requested_uri)

    if not (
        _is_loopback_http_redirect(registered) and _is_loopback_http_redirect(requested)
    ):
        return False

    return (
        registered.path == requested.path
        and registered.params == requested.params
        and registered.query == requested.query
        and registered.fragment == requested.fragment
    )


_DANGEROUS_REDIRECT_SCHEMES = frozenset({
    "javascript",
    "data",
    "vbscript",
    "file",
    "about",
    "blob",
})


def _build_redirect_url(redirect_uri: str, params: dict[str, str]) -> str:
    """Build a redirect URL from a previously validated redirect URI.

    Callers must pass a URI that has already been allowlisted by
    ``validate_redirect_uri``. This function adds a belt-and-braces check
    that rejects browser-executable or script-capable schemes (``javascript:``,
    ``data:``, ``vbscript:``, ``file:``, ``about:``, ``blob:``), even if one
    somehow slipped past the allowlist.

    Per RFC 8252 §7.1, native apps may legitimately register custom URI
    schemes (e.g. ``myapp://callback``), so we **denylist** dangerous schemes
    rather than allowlisting http(s) only.
    """
    scheme = urlparse(redirect_uri).scheme.lower()
    if scheme in _DANGEROUS_REDIRECT_SCHEMES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_redirect_uri",
                "error_description": "redirect_uri uses a disallowed scheme",
            },
        )
    return f"{redirect_uri}?{urlencode(params)}"


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
    """Validate redirect URI against registered client."""
    client = await get_oauth_storage().get_client(client_id)
    if not client:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_client",
                "error_description": "Invalid client_id",
            },
        )

    # If no redirect_uri provided, use the first registered one
    if not redirect_uri:
        if not client.redirect_uris:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "invalid_request",
                    "error_description": "redirect_uri is required when client has no registered redirect URIs",
                },
            )
        return client.redirect_uris[0]

    # Validate that the redirect_uri is registered; return the stored (trusted) value
    for registered_uri in client.redirect_uris:
        if registered_uri == redirect_uri:
            return registered_uri  # Return the stored value, not the user-supplied one

    # Native-app loopback redirects use runtime-assigned ports. After validating
    # that the request matches a registered loopback callback path, preserve the
    # runtime URI so the browser lands on the port the client actually opened.
    for registered_uri in client.redirect_uris:
        if _loopback_redirect_matches(registered_uri, redirect_uri):
            return redirect_uri

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={
            "error": "invalid_redirect_uri",
            "error_description": "redirect_uri not registered for this client",
        },
    )


def _build_authorize_page(query_string: str, error: Optional[str] = None) -> str:
    """Build the HTML authorization/login page."""
    error_html = ""
    if error:
        error_html = f'<div style="color:#ef4444;background:#fef2f2;border:1px solid #fecaca;padding:12px;border-radius:8px;margin-bottom:16px;font-size:14px;">{error}</div>'

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
  <form method="POST" action="/oauth/authorize?{query_string}">
    <label for="api_key">API Key</label>
    <input type="password" id="api_key" name="api_key" required autofocus autocomplete="current-password">
    <button type="submit">Authorize</button>
  </form>
</div>
</body>
</html>"""


@router.get("/authorize")
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

    # Show login form — pass all query params through so the POST can use them
    return HTMLResponse(_build_authorize_page(str(request.url.query)))


@router.post("/authorize")
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

    # Validate API key
    if not API_KEY or not secrets.compare_digest(api_key.encode(), API_KEY.encode()):
        logger.warning("Authorization denied: invalid API key")
        return HTMLResponse(
            _build_authorize_page(
                str(request.url.query), error="Invalid API key. Please try again."
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
            redirect_params["state"] = _sanitize_state(state)

        redirect_url = _build_redirect_url(validated_redirect_uri, redirect_params)
        logger.info(f"Authorization granted, redirecting to callback")
        # Use HTML meta-refresh + JS redirect for maximum popup compatibility.
        # Some OAuth clients (Claude.ai) use popups where HTTP 302 from a
        # form POST can be unreliable across cross-origin boundaries.
        import json

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
            error_params["state"] = _sanitize_state(state)
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
        import hashlib

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


@router.post("/token", response_model=TokenResponse)
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
