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

import time
import logging
import base64
import secrets
from typing import Optional, Tuple
from urllib.parse import urlencode
from fastapi import APIRouter, HTTPException, status, Form, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
import jwt

from ...config import (
    OAUTH_ISSUER,
    OAUTH_ACCESS_TOKEN_EXPIRE_MINUTES,
    OAUTH_AUTHORIZATION_CODE_EXPIRE_MINUTES,
    API_KEY,
    get_jwt_algorithm,
    get_jwt_signing_key
)
from .models import TokenResponse
from .storage import get_oauth_storage

logger = logging.getLogger(__name__)

router = APIRouter()


def _sanitize_log_value(value: object) -> str:
    """Sanitize a user-provided value for safe inclusion in log messages."""
    return str(value).replace("\n", "\\n").replace("\r", "\\r").replace("\x1b", "\\x1b")


def _sanitize_state(state: str) -> str:
    """Sanitize the OAuth state parameter to prevent log injection and open redirect abuse."""
    # Allow only alphanumeric, hyphen, underscore, and dot characters (RFC 6749 opaque value)
    import re as _re
    return _re.sub(r'[^A-Za-z0-9\-_.]', '', state)[:128]


def parse_basic_auth(authorization_header: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse HTTP Basic authentication header.

    Returns:
        Tuple of (client_id, client_secret) or (None, None) if not valid
    """
    if not authorization_header:
        return None, None

    try:
        # Check if it's Basic authentication
        if not authorization_header.startswith('Basic '):
            return None, None

        # Extract and decode the credentials
        encoded_credentials = authorization_header[6:]  # Remove 'Basic ' prefix
        decoded_credentials = base64.b64decode(encoded_credentials).decode('utf-8')

        # Split username:password
        if ':' not in decoded_credentials:
            return None, None

        client_id, client_secret = decoded_credentials.split(':', 1)
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
        "scope": scope or "read write"
    }

    algorithm = get_jwt_algorithm()
    signing_key = get_jwt_signing_key()

    logger.debug("Creating JWT token")
    token = jwt.encode(payload, signing_key, algorithm=algorithm)
    return token, expires_in


async def validate_redirect_uri(client_id: str, redirect_uri: Optional[str]) -> str:
    """Validate redirect URI against registered client."""
    client = await get_oauth_storage().get_client(client_id)
    if not client:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_client",
                "error_description": "Invalid client_id"
            }
        )

    # If no redirect_uri provided, use the first registered one
    if not redirect_uri:
        if not client.redirect_uris:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "invalid_request",
                    "error_description": "redirect_uri is required when client has no registered redirect URIs"
                }
            )
        return client.redirect_uris[0]

    # Validate that the redirect_uri is registered; return the stored (trusted) value
    for registered_uri in client.redirect_uris:
        if registered_uri == redirect_uri:
            return registered_uri  # Return the stored value, not the user-supplied one
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={
            "error": "invalid_redirect_uri",
            "error_description": "redirect_uri not registered for this client"
        }
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
    code_challenge_method: Optional[str] = Query(None, description="PKCE code challenge method (S256)")
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
            detail={"error": "unsupported_response_type", "error_description": "Only 'code' response type is supported"}
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
    code_challenge_method: Optional[str] = Query(None, description="PKCE code challenge method (S256)"),
    api_key: str = Form(..., description="API key for authorization")
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
            _build_authorize_page(str(request.url.query), error="Invalid API key. Please try again."),
            status_code=403
        )

    # Validate redirect_uri against registered client
    validated_redirect_uri: Optional[str] = None
    if redirect_uri:
        try:
            validated_redirect_uri = await validate_redirect_uri(client_id, redirect_uri)
        except HTTPException:
            raise

    try:
        safe_redirect_uri = validated_redirect_uri or await validate_redirect_uri(client_id, redirect_uri)

        # Generate and store authorization code
        auth_code = get_oauth_storage().generate_authorization_code()
        await get_oauth_storage().store_authorization_code(
            code=auth_code,
            client_id=client_id,
            redirect_uri=safe_redirect_uri,
            scope=scope,
            expires_in=OAUTH_AUTHORIZATION_CODE_EXPIRE_MINUTES * 60,
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method
        )

        # Redirect with authorization code
        redirect_params = {"code": auth_code}
        if state:
            redirect_params["state"] = _sanitize_state(state)

        redirect_url = f"{safe_redirect_uri}?{urlencode(redirect_params)}"
        logger.info(f"Authorization granted, redirecting to callback")
        # Use HTML meta-refresh + JS redirect for maximum popup compatibility.
        # Some OAuth clients (Claude.ai) use popups where HTTP 302 from a
        # form POST can be unreliable across cross-origin boundaries.
        return HTMLResponse(f"""<!DOCTYPE html>
<html><head>
<meta http-equiv="refresh" content="0;url={redirect_url}">
<script>window.location.href = {__import__('json').dumps(redirect_url)};</script>
</head><body>Redirecting...</body></html>""")

    except HTTPException:
        raise
    except Exception:
        logger.error("Authorization error occurred")
        error_params = {"error": "server_error", "error_description": "Internal server error"}
        if state:
            error_params["state"] = _sanitize_state(state)
        if validated_redirect_uri:
            return RedirectResponse(url=f"{validated_redirect_uri}?{urlencode(error_params)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_params)


async def _handle_authorization_code_grant(
    final_client_id: str,
    final_client_secret: str,
    code: Optional[str],
    redirect_uri: Optional[str],
    code_verifier: Optional[str] = None
) -> TokenResponse:
    """Handle OAuth authorization_code grant type."""
    if not code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_request",
                "error_description": "Missing required parameter: code"
            }
        )

    if not final_client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_request",
                "error_description": "Missing required parameter: client_id"
            }
        )

    # Authenticate client
    if not await get_oauth_storage().authenticate_client(final_client_id, final_client_secret or ""):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "invalid_client",
                "error_description": "Client authentication failed"
            }
        )

    # Get and consume authorization code
    code_data = await get_oauth_storage().get_authorization_code(code)
    if not code_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_grant",
                "error_description": "Invalid or expired authorization code"
            }
        )

    # Validate client_id matches
    if code_data["client_id"] != final_client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_grant",
                "error_description": "Authorization code was issued to a different client"
            }
        )

    # Validate redirect_uri if provided
    if redirect_uri and code_data["redirect_uri"] != redirect_uri:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_grant",
                "error_description": "redirect_uri does not match the one used in authorization request"
            }
        )

    # PKCE verification
    stored_challenge = code_data.get("code_challenge")
    if stored_challenge:
        if not code_verifier:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "invalid_grant",
                    "error_description": "code_verifier required for PKCE"
                }
            )
        import hashlib
        computed = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode("ascii")).digest()
        ).rstrip(b"=").decode("ascii")
        if computed != stored_challenge:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "invalid_grant",
                    "error_description": "PKCE code_verifier does not match code_challenge"
                }
            )

    # Create access token
    access_token, expires_in = create_access_token(final_client_id, code_data["scope"])

    # Store access token for validation
    await get_oauth_storage().store_access_token(
        token=access_token,
        client_id=final_client_id,
        scope=code_data["scope"],
        expires_in=expires_in
    )

    logger.info("Access token issued")
    return TokenResponse(
        access_token=access_token,
        token_type="Bearer",
        expires_in=expires_in,
        scope=code_data["scope"]
    )

async def _handle_client_credentials_grant(
    final_client_id: str,
    final_client_secret: str
) -> TokenResponse:
    """Handle OAuth client_credentials grant type."""
    if not final_client_id or not final_client_secret:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_request",
                "error_description": "Missing required parameters: client_id and client_secret"
            }
        )

    # Authenticate client
    if not await get_oauth_storage().authenticate_client(final_client_id, final_client_secret):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "invalid_client",
                "error_description": "Client authentication failed"
            }
        )

    # Create access token
    access_token, expires_in = create_access_token(final_client_id, "read write")

    # Store access token
    await get_oauth_storage().store_access_token(
        token=access_token,
        client_id=final_client_id,
        scope="read write",
        expires_in=expires_in
    )

    logger.info("Client credentials token issued")
    return TokenResponse(
        access_token=access_token,
        token_type="Bearer",
        expires_in=expires_in,
        scope="read write"
    )

@router.post("/token", response_model=TokenResponse)
async def token(
    request: Request,
    grant_type: str = Form(..., description="OAuth grant type"),
    code: Optional[str] = Form(None, description="Authorization code"),
    redirect_uri: Optional[str] = Form(None, description="Redirection URI"),
    client_id: Optional[str] = Form(None, description="OAuth client identifier"),
    client_secret: Optional[str] = Form(None, description="OAuth client secret"),
    code_verifier: Optional[str] = Form(None, description="PKCE code verifier")
):
    """
    OAuth 2.1 Token endpoint.

    Exchanges authorization codes for access tokens.
    Supports both authorization_code and client_credentials grant types.
    Supports both client_secret_post (form data) and client_secret_basic (HTTP Basic auth).
    """
    # Extract client credentials from either HTTP Basic auth or form data
    auth_header = request.headers.get('authorization')
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

        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "unsupported_grant_type",
                    "error_description": f"Grant type '{grant_type}' is not supported"
                }
            )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception:
        logger.error("Token endpoint error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "server_error",
                "error_description": "Internal server error"
            }
        )
