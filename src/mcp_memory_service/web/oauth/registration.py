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
from urllib.parse import urlparse, ParseResult
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
    """
    Validate redirect URIs according to OAuth 2.1 security requirements.

    Uses proper URL parsing to prevent bypass attacks and validates schemes
    against a secure whitelist to prevent dangerous scheme injection.
    """
    if not redirect_uris:
        return

    # Allowed schemes - whitelist approach for security
    ALLOWED_SCHEMES = {
        'https',    # HTTPS (preferred)
        'http',     # HTTP (localhost only)
        # Native app custom schemes (common patterns)
        'com.example.app',  # Reverse domain notation
        'myapp',           # Simple custom scheme
        # Add more custom schemes as needed, but NEVER allow:
        # javascript:, data:, file:, vbscript:, about:, chrome:, etc.
    }

    # Dangerous schemes that must be blocked
    DANGEROUS_SCHEMES = {
        'javascript', 'data', 'file', 'vbscript', 'about', 'chrome',
        'chrome-extension', 'moz-extension', 'ms-appx', 'blob'
    }

    for uri in redirect_uris:
        uri_str = str(uri).strip()

        if not uri_str:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "invalid_redirect_uri",
                    "error_description": "Empty redirect URI not allowed"
                }
            )

        try:
            # Parse URL using proper URL parser to prevent bypass attacks
            parsed: ParseResult = urlparse(uri_str)

            if not parsed.scheme:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "error": "invalid_redirect_uri",
                        "error_description": f"Missing scheme in redirect URI: {uri_str}"
                    }
                )

            # Check for dangerous schemes first (security)
            if parsed.scheme.lower() in DANGEROUS_SCHEMES:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "error": "invalid_redirect_uri",
                        "error_description": f"Dangerous scheme '{parsed.scheme}' not allowed in redirect URI"
                    }
                )

            # For HTTP scheme, enforce strict localhost validation
            if parsed.scheme.lower() == 'http':
                if not parsed.netloc:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail={
                            "error": "invalid_redirect_uri",
                            "error_description": f"HTTP URI missing host: {uri_str}"
                        }
                    )

                # Extract hostname from netloc (handles port numbers correctly)
                hostname = parsed.hostname
                if not hostname:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail={
                            "error": "invalid_redirect_uri",
                            "error_description": f"Cannot extract hostname from HTTP URI: {uri_str}"
                        }
                    )

                # Strict localhost validation - only allow exact matches
                if hostname.lower() not in ('localhost', '127.0.0.1', '::1'):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail={
                            "error": "invalid_redirect_uri",
                            "error_description": f"HTTP redirect URIs must use localhost, 127.0.0.1, or ::1. Got: {hostname}"
                        }
                    )

            # For HTTPS, allow any valid hostname (production requirement)
            elif parsed.scheme.lower() == 'https':
                if not parsed.netloc:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail={
                            "error": "invalid_redirect_uri",
                            "error_description": f"HTTPS URI missing host: {uri_str}"
                        }
                    )

            # For custom schemes (native apps), validate they're in allowed list
            elif parsed.scheme.lower() not in [s.lower() for s in ALLOWED_SCHEMES]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "error": "invalid_redirect_uri",
                        "error_description": f"Unsupported scheme '{parsed.scheme}'. Allowed: {', '.join(sorted(ALLOWED_SCHEMES))}"
                    }
                )

        except ValueError as e:
            # URL parsing failed
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "invalid_redirect_uri",
                    "error_description": f"Invalid URL format: {uri_str}. Error: {str(e)}"
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