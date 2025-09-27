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
OAuth 2.1 authentication middleware for MCP Memory Service.

Provides Bearer token validation with fallback to API key authentication.
"""

import logging
from typing import Optional, Dict, Any
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt

from ...config import OAUTH_SECRET_KEY, OAUTH_ISSUER, API_KEY, ALLOW_ANONYMOUS_ACCESS
from .storage import oauth_storage

logger = logging.getLogger(__name__)

# Optional Bearer token security scheme
bearer_scheme = HTTPBearer(auto_error=False)


class AuthenticationResult:
    """Result of authentication attempt."""

    def __init__(
        self,
        authenticated: bool,
        client_id: Optional[str] = None,
        scope: Optional[str] = None,
        auth_method: Optional[str] = None,
        error: Optional[str] = None
    ):
        self.authenticated = authenticated
        self.client_id = client_id
        self.scope = scope
        self.auth_method = auth_method  # "oauth", "api_key", or "none"
        self.error = error

    def has_scope(self, required_scope: str) -> bool:
        """Check if the authenticated user has the required scope."""
        if not self.authenticated or not self.scope:
            return False

        # Split scopes and check if required scope is present
        scopes = self.scope.split()
        return required_scope in scopes

    def require_scope(self, required_scope: str) -> None:
        """Raise an exception if the required scope is not present."""
        if not self.has_scope(required_scope):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "insufficient_scope",
                    "error_description": f"Required scope '{required_scope}' not granted"
                }
            )


def validate_jwt_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Validate a JWT access token.

    Returns:
        JWT payload if valid, None if invalid
    """
    try:
        payload = jwt.decode(
            token,
            OAUTH_SECRET_KEY,
            algorithms=["HS256"],
            issuer=OAUTH_ISSUER,
            audience="mcp-memory-service"
        )
        return payload
    except JWTError as e:
        logger.debug(f"JWT validation failed: {e}")
        return None


async def authenticate_bearer_token(token: str) -> AuthenticationResult:
    """
    Authenticate using OAuth Bearer token.

    Returns:
        AuthenticationResult with authentication status and details
    """
    # First, try JWT validation
    jwt_payload = validate_jwt_token(token)
    if jwt_payload:
        client_id = jwt_payload.get("sub")
        scope = jwt_payload.get("scope", "")

        logger.debug(f"JWT authentication successful: client_id={client_id}, scope={scope}")
        return AuthenticationResult(
            authenticated=True,
            client_id=client_id,
            scope=scope,
            auth_method="oauth"
        )

    # Fallback: check if token is stored in OAuth storage
    token_data = await oauth_storage.get_access_token(token)
    if token_data:
        logger.debug(f"OAuth storage authentication successful: client_id={token_data['client_id']}")
        return AuthenticationResult(
            authenticated=True,
            client_id=token_data["client_id"],
            scope=token_data.get("scope", ""),
            auth_method="oauth"
        )

    logger.debug("Bearer token authentication failed")
    return AuthenticationResult(
        authenticated=False,
        auth_method="oauth",
        error="invalid_token"
    )


def authenticate_api_key(api_key: str) -> AuthenticationResult:
    """
    Authenticate using legacy API key.

    Returns:
        AuthenticationResult with authentication status
    """
    if API_KEY and api_key == API_KEY:
        logger.debug("API key authentication successful")
        return AuthenticationResult(
            authenticated=True,
            client_id="api_key_client",
            scope="read write admin",  # API key gets full access
            auth_method="api_key"
        )

    logger.debug("API key authentication failed")
    return AuthenticationResult(
        authenticated=False,
        auth_method="api_key",
        error="invalid_api_key"
    )


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)
) -> AuthenticationResult:
    """
    Get current authenticated user with fallback authentication methods.

    Tries in order:
    1. OAuth Bearer token (JWT or stored token)
    2. Legacy API key authentication
    3. Anonymous access (if no authentication configured)

    Returns:
        AuthenticationResult with authentication details
    """
    # Try OAuth Bearer token authentication first
    if credentials and credentials.scheme.lower() == "bearer":
        auth_result = await authenticate_bearer_token(credentials.credentials)
        if auth_result.authenticated:
            return auth_result

        # Bearer token provided but invalid - try API key fallback before giving up
        if API_KEY:
            # Some clients might send API key as Bearer token
            api_key_result = authenticate_api_key(credentials.credentials)
            if api_key_result.authenticated:
                return api_key_result

        # Both OAuth and API key failed - raise exception
        logger.warning("Invalid Bearer token provided and API key fallback failed")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "invalid_token",
                "error_description": "The access token provided is expired, revoked, malformed, or invalid"
            },
            headers={"WWW-Authenticate": "Bearer"}
        )

    # Allow anonymous access only if explicitly enabled
    if not API_KEY and ALLOW_ANONYMOUS_ACCESS:
        logger.debug("Anonymous access explicitly enabled, granting read-only access")
        return AuthenticationResult(
            authenticated=True,
            client_id="anonymous",
            scope="read",  # Anonymous users get read-only access for security
            auth_method="none"
        )

    # No credentials provided and anonymous access not allowed
    if not API_KEY:
        logger.debug("No authentication configured and anonymous access disabled")
        error_msg = "Authentication is required. Set MCP_ALLOW_ANONYMOUS_ACCESS=true to enable anonymous access."
    else:
        logger.debug("No valid authentication provided")
        error_msg = "Authorization is required to access this resource"

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={
            "error": "authorization_required",
            "error_description": error_msg
        },
        headers={"WWW-Authenticate": "Bearer"}
    )


# Convenience dependency for requiring specific scopes
def require_scope(scope: str):
    """
    Create a dependency that requires a specific OAuth scope.

    Usage:
        @app.get("/admin", dependencies=[Depends(require_scope("admin"))])
    """
    async def scope_dependency(user: AuthenticationResult = Depends(get_current_user)):
        user.require_scope(scope)
        return user

    return scope_dependency


# Convenience dependencies for common access patterns
async def require_read_access(user: AuthenticationResult = Depends(get_current_user)) -> AuthenticationResult:
    """Require read access to the resource."""
    user.require_scope("read")
    return user


async def require_write_access(user: AuthenticationResult = Depends(get_current_user)) -> AuthenticationResult:
    """Require write access to the resource."""
    user.require_scope("write")
    return user


async def require_admin_access(user: AuthenticationResult = Depends(get_current_user)) -> AuthenticationResult:
    """Require admin access to the resource."""
    user.require_scope("admin")
    return user


# Optional authentication (for endpoints that work with or without auth)
async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)
) -> Optional[AuthenticationResult]:
    """
    Get current user but don't require authentication.

    Returns:
        AuthenticationResult if authenticated, None if not
    """
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None