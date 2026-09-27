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
OAuth 2.1 Dynamic Client Registration implementation for MCP Memory Service.

Implements RFC 7591 - OAuth 2.0 Dynamic Client Registration Protocol.
"""

import secrets
import time
import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import ValidationError

from .models import (
    ClientRegistrationRequest,
    ClientRegistrationResponse,
    RegisteredClient
)
from .storage import get_oauth_storage
from .rate_limit import auth_rate_limit
from .redirect import validate_registration_redirect_uris
from mcp_memory_service.config import DCR_REGISTRATION_KEY

logger = logging.getLogger(__name__)

router = APIRouter()


def _sanitize_log_value(value: object) -> str:
    """Sanitize a user-provided value for safe inclusion in log messages."""
    return str(value).replace("\n", "\\n").replace("\r", "\\r").replace("\x1b", "\\x1b")


# Redirect-URI scheme validation lives in one place (oauth/redirect.py). This
# thin alias preserves the historical import path used by callers and tests.
validate_redirect_uris = validate_registration_redirect_uris


def validate_grant_types(grant_types: List[str]) -> None:
    """Validate that requested grant types are supported."""
    supported_grant_types = {"authorization_code", "client_credentials", "refresh_token"}

    for grant_type in grant_types:
        if grant_type not in supported_grant_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "invalid_client_metadata",
                    "error_description": f"Unsupported grant type: {grant_type}. Supported: {list(supported_grant_types)}"
                }
            )


def validate_response_types(response_types: List[str]) -> None:
    """Validate that requested response types are supported."""
    supported_response_types = {"code"}

    for response_type in response_types:
        if response_type not in supported_response_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "invalid_client_metadata",
                    "error_description": f"Unsupported response type: {response_type}. Supported: {list(supported_response_types)}"
                }
            )


def _validate_registration_key(raw_request: Request) -> None:
    """
    FastAPI dependency that enforces the optional DCR registration key.

    When MCP_DCR_REGISTRATION_KEY is set, the /oauth/register endpoint
    requires Authorization: Bearer <key>. When unset, DCR remains open
    (backward compatible with RFC 7591 default behavior).

    Raises:
        HTTPException 401: Header missing or not Bearer scheme (RFC 7235).
        HTTPException 401: Provided key is invalid (RFC 6750 §3.1).
    """
    if not DCR_REGISTRATION_KEY:
        return

    auth_header = raw_request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "invalid_token",
                "error_description": "Registration key required. Provide Authorization: Bearer <registration-key>"
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    provided_key = auth_header[len("Bearer "):]
    if not secrets.compare_digest(provided_key, DCR_REGISTRATION_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "invalid_token",
                "error_description": "Invalid registration key"
            },
            headers={"WWW-Authenticate": 'Bearer error="invalid_token"'},
        )


def _refuse_client_credentials_while_open(grant_types) -> None:
    """Reject a self-service registration that asks for client_credentials.

    With MCP_DCR_REGISTRATION_KEY unset this endpoint is open by design, for
    RFC 7591 and Claude.ai Remote MCP. That is defensible for
    authorization_code, which puts a human in the loop before any token
    exists. client_credentials has no such step: a caller could register
    itself as confidential, exchange its own client_id and client_secret, and
    hold a read-write token without the owner ever being consulted
    (GHSA-6mvm-q4j3-27qg). The grant is machine-to-machine, so the
    registration behind it has to be gated too.

    The token endpoint refuses the same combination independently. That is
    not redundant: this check only covers registrations made from now on,
    while the advisory is about clients that already registered.
    """
    if not grant_types or "client_credentials" not in grant_types:
        return
    if DCR_REGISTRATION_KEY:
        return
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={
            "error": "invalid_client_metadata",
            "error_description": (
                "The client_credentials grant requires gated registration. "
                "Set MCP_DCR_REGISTRATION_KEY and register with it, or use "
                "authorization_code with PKCE."
            ),
        },
    )


@router.post(
    "/register",
    response_model=ClientRegistrationResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(auth_rate_limit)],
)
async def register_client(
    request: ClientRegistrationRequest,
    _: None = Depends(_validate_registration_key),
) -> ClientRegistrationResponse:
    """
    OAuth 2.1 Dynamic Client Registration endpoint.

    Implements RFC 7591 - OAuth 2.0 Dynamic Client Registration Protocol.
    Allows clients to register dynamically with the authorization server.

    When MCP_DCR_REGISTRATION_KEY env var is set, requests must include
    Authorization: Bearer <key> header. When unset, registration is open.
    """
    logger.info("OAuth client registration request received")

    try:
        # Validate client metadata
        if request.redirect_uris:
            validate_redirect_uris([str(uri) for uri in request.redirect_uris])

        if request.grant_types:
            validate_grant_types(request.grant_types)

        if request.response_types:
            validate_response_types(request.response_types)

        _refuse_client_credentials_while_open(request.grant_types)

        # Prepare default values
        grant_types = request.grant_types or ["authorization_code"]
        response_types = request.response_types or ["code"]
        token_endpoint_auth_method = request.token_endpoint_auth_method or "client_secret_basic"

        # Generate client credentials. A public client (RFC 7591: auth method
        # "none", e.g. the PKCE flow claude.ai uses) must not be issued a secret
        # at all — handing one out let a caller turn a registration into client
        # authentication. The empty string is stored rather than None so the model
        # and the SQLite row stay unchanged; verify_client_secret() already
        # refuses an empty stored secret, so such a client cannot authenticate
        # with one.
        client_id = get_oauth_storage().generate_client_id()
        is_public_client = token_endpoint_auth_method == "none"
        client_secret = "" if is_public_client else get_oauth_storage().generate_client_secret()

        # Create registered client
        registered_client = RegisteredClient(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uris=[str(uri) for uri in request.redirect_uris] if request.redirect_uris else [],
            grant_types=grant_types,
            response_types=response_types,
            token_endpoint_auth_method=token_endpoint_auth_method,
            client_name=request.client_name,
            created_at=time.time()
        )

        # Store the client
        await get_oauth_storage().store_client(registered_client)

        # Create response. client_secret is omitted entirely for a public client.
        response = ClientRegistrationResponse(
            client_id=client_id,
            client_secret=client_secret or None,
            redirect_uris=registered_client.redirect_uris,
            grant_types=grant_types,
            response_types=response_types,
            token_endpoint_auth_method=token_endpoint_auth_method,
            client_name=request.client_name
        )

        logger.info(f"OAuth client registered successfully: client_id={_sanitize_log_value(client_id)}, name={_sanitize_log_value(request.client_name)}")
        return response

    except ValidationError as e:
        logger.warning(f"OAuth client registration validation error: {_sanitize_log_value(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_client_metadata",
                "error_description": f"Invalid client metadata: {str(e)}"
            }
        )
    except HTTPException:
        # Re-raise HTTP exceptions (validation errors)
        raise
    except Exception as e:
        logger.error(f"OAuth client registration error: {_sanitize_log_value(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "server_error",
                "error_description": "Internal server error during client registration"
            }
        )


@router.get("/clients/{client_id}")
async def get_client_info(client_id: str) -> ClientRegistrationResponse:
    """
    Get information about a registered client.

    Note: This is an extension endpoint, not part of RFC 7591.
    Useful for debugging and client management.
    """
    logger.info(f"Client info request for client_id={_sanitize_log_value(client_id)}")

    client = await get_oauth_storage().get_client(client_id)
    if not client:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "invalid_client",
                "error_description": "Client not found"
            }
        )

    # Return client information (without secret for security)
    return ClientRegistrationResponse(
        client_id=client.client_id,
        client_secret="[REDACTED]",  # Don't expose the secret
        redirect_uris=client.redirect_uris,
        grant_types=client.grant_types,
        response_types=client.response_types,
        token_endpoint_auth_method=client.token_endpoint_auth_method,
        client_name=client.client_name
    )