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

import time
import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import ValidationError

from .models import (
    ClientRegistrationRequest,
    ClientRegistrationResponse,
    RegisteredClient
)
from .storage import oauth_storage

logger = logging.getLogger(__name__)

router = APIRouter()


def validate_redirect_uris(redirect_uris: Optional[List[str]]) -> None:
    """Validate redirect URIs according to OAuth 2.1 security requirements."""
    if not redirect_uris:
        return

    for uri in redirect_uris:
        uri_str = str(uri)

        # OAuth 2.1 security requirements
        if uri_str.startswith("http://"):
            # Only allow localhost for http in development
            if not ("localhost" in uri_str or "127.0.0.1" in uri_str):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "error": "invalid_redirect_uri",
                        "error_description": "HTTP redirect URIs must use localhost or 127.0.0.1"
                    }
                )

        # Check for custom schemes (allowed for native apps)
        if not (uri_str.startswith("https://") or uri_str.startswith("http://") or "://" in uri_str):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "invalid_redirect_uri",
                    "error_description": f"Invalid redirect URI format: {uri_str}"
                }
            )


def validate_grant_types(grant_types: List[str]) -> None:
    """Validate that requested grant types are supported."""
    supported_grant_types = {"authorization_code", "client_credentials"}

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


@router.post("/register", response_model=ClientRegistrationResponse, status_code=status.HTTP_201_CREATED)
async def register_client(request: ClientRegistrationRequest) -> ClientRegistrationResponse:
    """
    OAuth 2.1 Dynamic Client Registration endpoint.

    Implements RFC 7591 - OAuth 2.0 Dynamic Client Registration Protocol.
    Allows clients to register dynamically with the authorization server.
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

        # Generate client credentials
        client_id = oauth_storage.generate_client_id()
        client_secret = oauth_storage.generate_client_secret()

        # Prepare default values
        grant_types = request.grant_types or ["authorization_code"]
        response_types = request.response_types or ["code"]
        token_endpoint_auth_method = request.token_endpoint_auth_method or "client_secret_basic"

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
        await oauth_storage.store_client(registered_client)

        # Create response
        response = ClientRegistrationResponse(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uris=registered_client.redirect_uris,
            grant_types=grant_types,
            response_types=response_types,
            token_endpoint_auth_method=token_endpoint_auth_method,
            client_name=request.client_name
        )

        logger.info(f"OAuth client registered successfully: client_id={client_id}, name={request.client_name}")
        return response

    except ValidationError as e:
        logger.warning(f"OAuth client registration validation error: {e}")
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
        logger.error(f"OAuth client registration error: {e}")
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
    logger.info(f"Client info request for client_id={client_id}")

    client = await oauth_storage.get_client(client_id)
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