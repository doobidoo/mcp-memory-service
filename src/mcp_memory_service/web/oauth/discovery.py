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
OAuth 2.1 Discovery endpoints for MCP Memory Service.

Implements .well-known endpoints required for OAuth 2.1 Dynamic Client Registration.
"""

import logging
from fastapi import APIRouter
from ...config import OAUTH_ISSUER, get_jwt_algorithm
from .models import OAuthServerMetadata

logger = logging.getLogger(__name__)

router = APIRouter()



@router.get("/.well-known/oauth-authorization-server/mcp")
async def oauth_authorization_server_metadata() -> OAuthServerMetadata:
    """
    OAuth 2.1 Authorization Server Metadata endpoint.

    Returns metadata about the OAuth 2.1 authorization server as specified
    in RFC 8414. This endpoint is required for OAuth 2.1 Dynamic Client Registration.
    """
    logger.info("OAuth authorization server metadata requested")

    # Use OAUTH_ISSUER consistently for both issuer field and endpoint URLs
    # This ensures URL consistency across discovery and JWT token validation
    algorithm = get_jwt_algorithm()
    metadata = OAuthServerMetadata(
        issuer=OAUTH_ISSUER,
        authorization_endpoint=f"{OAUTH_ISSUER}/oauth/authorize",
        token_endpoint=f"{OAUTH_ISSUER}/oauth/token",
        registration_endpoint=f"{OAUTH_ISSUER}/oauth/register",
        grant_types_supported=["authorization_code", "client_credentials"],
        response_types_supported=["code"],
        token_endpoint_auth_methods_supported=["client_secret_basic", "client_secret_post"],
        scopes_supported=["read", "write", "admin"],
        id_token_signing_alg_values_supported=[algorithm]
    )

    logger.debug(f"Returning OAuth metadata: issuer={metadata.issuer}")
    return metadata


@router.get("/.well-known/openid-configuration/mcp")
async def openid_configuration() -> OAuthServerMetadata:
    """
    OpenID Connect Discovery endpoint.

    Some OAuth 2.1 clients may also check this endpoint for compatibility.
    For now, we return the same metadata as the OAuth authorization server.
    """
    logger.info("OpenID Connect configuration requested")

    # Return the same metadata as OAuth authorization server for compatibility
    return await oauth_authorization_server_metadata()


@router.get("/.well-known/oauth-authorization-server")
async def oauth_authorization_server_metadata_generic() -> OAuthServerMetadata:
    """
    Generic OAuth 2.1 Authorization Server Metadata endpoint.

    Fallback endpoint for clients that don't append the /mcp suffix.
    """
    logger.info("Generic OAuth authorization server metadata requested")
    return await oauth_authorization_server_metadata()


@router.get("/.well-known/openid-configuration")
async def openid_configuration_generic() -> OAuthServerMetadata:
    """
    Generic OpenID Connect Discovery endpoint.

    Fallback endpoint for clients that don't append the /mcp suffix.
    """
    logger.info("Generic OpenID Connect configuration requested")
    return await oauth_authorization_server_metadata()