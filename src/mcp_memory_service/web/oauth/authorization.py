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
from typing import Optional, Tuple
from urllib.parse import urlencode
from fastapi import APIRouter, HTTPException, status, Form, Query, Request
from fastapi.responses import RedirectResponse
from jose import jwt

from ...config import (
    OAUTH_ISSUER,
    OAUTH_ACCESS_TOKEN_EXPIRE_MINUTES,
    OAUTH_AUTHORIZATION_CODE_EXPIRE_MINUTES,
    get_jwt_algorithm,
    get_jwt_signing_key
)
from .models import TokenResponse
from .storage import oauth_storage

logger = logging.getLogger(__name__)

router = APIRouter()


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

    except Exception as e:
        logger.debug(f"Failed to parse Basic auth header: {e}")
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

    logger.debug(f"Creating JWT token with algorithm: {algorithm}")
    token = jwt.encode(payload, signing_key, algorithm=algorithm)
    return token, expires_in


async def validate_redirect_uri(client_id: str, redirect_uri: Optional[str]) -> str:
    """Validate redirect URI against registered client."""
    client = await oauth_storage.get_client(client_id)
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

    # Validate that the redirect_uri is registered
    if redirect_uri not in client.redirect_uris:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_redirect_uri",
                "error_description": "redirect_uri not registered for this client"
            }
        )

    return redirect_uri


@router.get("/authorize")
async def authorize(
    response_type: str = Query(..., description="OAuth response type"),
    client_id: str = Query(..., description="OAuth client identifier"),
    redirect_uri: Optional[str] = Query(None, description="Redirection URI"),
    scope: Optional[str] = Query(None, description="Requested scope"),
    state: Optional[str] = Query(None, description="Opaque value for CSRF protection")
):
    """
    OAuth 2.1 Authorization endpoint.

    Implements the authorization code flow. For MVP, this auto-approves
    all requests without user interaction.
    """
    logger.info(f"Authorization request: client_id={client_id}, response_type={response_type}")

    try:
        # Validate response_type
        if response_type != "code":
            error_params = {
                "error": "unsupported_response_type",
                "error_description": "Only 'code' response type is supported"
            }
            if state:
                error_params["state"] = state

            # If we have a redirect_uri, redirect with error
            if redirect_uri:
                error_url = f"{redirect_uri}?{urlencode(error_params)}"
                return RedirectResponse(url=error_url)
            else:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_params)

        # Validate client and redirect_uri
        validated_redirect_uri = await validate_redirect_uri(client_id, redirect_uri)

        # Generate authorization code
        auth_code = oauth_storage.generate_authorization_code()

        # Store authorization code
        await oauth_storage.store_authorization_code(
            code=auth_code,
            client_id=client_id,
            redirect_uri=validated_redirect_uri,
            scope=scope,
            expires_in=OAUTH_AUTHORIZATION_CODE_EXPIRE_MINUTES * 60
        )

        # Build redirect URL with authorization code
        redirect_params = {"code": auth_code}
        if state:
            redirect_params["state"] = state

        redirect_url = f"{validated_redirect_uri}?{urlencode(redirect_params)}"

        logger.info(f"Authorization granted for client_id={client_id}")
        return RedirectResponse(url=redirect_url)

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Authorization error: {e}")

        error_params = {
            "error": "server_error",
            "error_description": "Internal server error"
        }
        if state:
            error_params["state"] = state

        if redirect_uri:
            error_url = f"{redirect_uri}?{urlencode(error_params)}"
            return RedirectResponse(url=error_url)
        else:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_params)


@router.post("/token", response_model=TokenResponse)
async def token(
    request: Request,
    grant_type: str = Form(..., description="OAuth grant type"),
    code: Optional[str] = Form(None, description="Authorization code"),
    redirect_uri: Optional[str] = Form(None, description="Redirection URI"),
    client_id: Optional[str] = Form(None, description="OAuth client identifier"),
    client_secret: Optional[str] = Form(None, description="OAuth client secret")
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

    auth_method = "client_secret_basic" if basic_client_id else "client_secret_post"
    logger.info(f"Token request: grant_type={grant_type}, client_id={final_client_id}, auth_method={auth_method}")

    try:
        if grant_type == "authorization_code":
            # Validate required parameters
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
            if not await oauth_storage.authenticate_client(final_client_id, final_client_secret or ""):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail={
                        "error": "invalid_client",
                        "error_description": "Client authentication failed"
                    }
                )

            # Get and consume authorization code
            code_data = await oauth_storage.get_authorization_code(code)
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

            # Create access token
            access_token, expires_in = create_access_token(final_client_id, code_data["scope"])

            # Store access token for validation
            await oauth_storage.store_access_token(
                token=access_token,
                client_id=final_client_id,
                scope=code_data["scope"],
                expires_in=expires_in
            )

            logger.info(f"Access token issued for client_id={final_client_id}")
            return TokenResponse(
                access_token=access_token,
                token_type="Bearer",
                expires_in=expires_in,
                scope=code_data["scope"]
            )

        elif grant_type == "client_credentials":
            # Client credentials flow for server-to-server authentication
            if not final_client_id or not final_client_secret:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "error": "invalid_request",
                        "error_description": "Missing required parameters: client_id and client_secret"
                    }
                )

            # Authenticate client
            if not await oauth_storage.authenticate_client(final_client_id, final_client_secret):
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
            await oauth_storage.store_access_token(
                token=access_token,
                client_id=final_client_id,
                scope="read write",
                expires_in=expires_in
            )

            logger.info(f"Client credentials token issued for client_id={final_client_id}")
            return TokenResponse(
                access_token=access_token,
                token_type="Bearer",
                expires_in=expires_in,
                scope="read write"
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
    except Exception as e:
        logger.error(f"Token endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "server_error",
                "error_description": "Internal server error"
            }
        )