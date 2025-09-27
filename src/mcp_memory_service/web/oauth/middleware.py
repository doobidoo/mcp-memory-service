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
from jose import JWTError, jwt, ExpiredSignatureError
from jose.jwt import JWTClaimsError

from ...config import (
    OAUTH_ISSUER,
    API_KEY,
    ALLOW_ANONYMOUS_ACCESS,
    OAUTH_ENABLED,
    get_jwt_algorithm,
    get_jwt_verification_key
)
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
    Validate a JWT access token with comprehensive error handling.

    Supports both RS256 and HS256 algorithms based on available keys.
    Provides detailed error logging for debugging purposes.

    Returns:
        JWT payload if valid, None if invalid
    """
    # Input validation
    if not token or not isinstance(token, str):
        logger.debug("Invalid token: empty or non-string token provided")
        return None

    # Basic token format validation
    token = token.strip()
    if not token:
        logger.debug("Invalid token: empty token after stripping")
        return None

    # JWT tokens should have 3 parts separated by dots
    parts = token.split('.')
    if len(parts) != 3:
        logger.debug(f"Invalid token format: expected 3 parts, got {len(parts)}")
        return None

    try:
        algorithm = get_jwt_algorithm()
        verification_key = get_jwt_verification_key()

        logger.debug(f"Validating JWT token with algorithm: {algorithm}")
        payload = jwt.decode(
            token,
            verification_key,
            algorithms=[algorithm],
            issuer=OAUTH_ISSUER,
            audience="mcp-memory-service"
        )

        # Additional payload validation
        required_claims = ['sub', 'iss', 'aud', 'exp', 'iat']
        missing_claims = [claim for claim in required_claims if claim not in payload]
        if missing_claims:
            logger.warning(f"JWT token missing required claims: {missing_claims}")
            return None

        logger.debug(f"JWT validation successful for subject: {payload.get('sub')}")
        return payload

    except ExpiredSignatureError:
        logger.debug("JWT validation failed: token has expired")
        return None
    except JWTClaimsError as e:
        logger.debug(f"JWT validation failed: invalid claims - {e}")
        return None
    except ValueError as e:
        logger.debug(f"JWT validation failed: configuration error - {e}")
        return None
    except JWTError as e:
        # Catch-all for other JWT-related errors
        error_type = type(e).__name__
        logger.debug(f"JWT validation failed: {error_type} - {e}")
        return None
    except Exception as e:
        # Unexpected errors should be logged but not crash the system
        error_type = type(e).__name__
        logger.error(f"Unexpected error during JWT validation: {error_type} - {e}")
        return None


async def authenticate_bearer_token(token: str) -> AuthenticationResult:
    """
    Authenticate using OAuth Bearer token with comprehensive error handling.

    Returns:
        AuthenticationResult with authentication status and details
    """
    # Input validation
    if not token or not isinstance(token, str):
        logger.debug("Bearer token authentication failed: invalid token input")
        return AuthenticationResult(
            authenticated=False,
            auth_method="oauth",
            error="invalid_token"
        )

    token = token.strip()
    if not token:
        logger.debug("Bearer token authentication failed: empty token")
        return AuthenticationResult(
            authenticated=False,
            auth_method="oauth",
            error="invalid_token"
        )

    try:
        # First, try JWT validation
        jwt_payload = validate_jwt_token(token)
        if jwt_payload:
            client_id = jwt_payload.get("sub")
            scope = jwt_payload.get("scope", "")

            # Validate client_id is present
            if not client_id:
                logger.warning("JWT authentication failed: missing client_id in token payload")
                return AuthenticationResult(
                    authenticated=False,
                    auth_method="oauth",
                    error="invalid_token"
                )

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
            client_id = token_data.get("client_id")
            if not client_id:
                logger.warning("OAuth storage authentication failed: missing client_id in stored token")
                return AuthenticationResult(
                    authenticated=False,
                    auth_method="oauth",
                    error="invalid_token"
                )

            logger.debug(f"OAuth storage authentication successful: client_id={client_id}")
            return AuthenticationResult(
                authenticated=True,
                client_id=client_id,
                scope=token_data.get("scope", ""),
                auth_method="oauth"
            )

    except Exception as e:
        # Catch any unexpected errors during authentication
        error_type = type(e).__name__
        logger.error(f"Unexpected error during bearer token authentication: {error_type} - {e}")
        return AuthenticationResult(
            authenticated=False,
            auth_method="oauth",
            error="server_error"
        )

    logger.debug("Bearer token authentication failed: token not found or invalid")
    return AuthenticationResult(
        authenticated=False,
        auth_method="oauth",
        error="invalid_token"
    )


def authenticate_api_key(api_key: str) -> AuthenticationResult:
    """
    Authenticate using legacy API key with enhanced validation.

    Returns:
        AuthenticationResult with authentication status
    """
    # Input validation
    if not api_key or not isinstance(api_key, str):
        logger.debug("API key authentication failed: invalid input")
        return AuthenticationResult(
            authenticated=False,
            auth_method="api_key",
            error="invalid_api_key"
        )

    api_key = api_key.strip()
    if not api_key:
        logger.debug("API key authentication failed: empty key")
        return AuthenticationResult(
            authenticated=False,
            auth_method="api_key",
            error="invalid_api_key"
        )

    # Check if API key is configured
    if not API_KEY:
        logger.debug("API key authentication failed: no API key configured")
        return AuthenticationResult(
            authenticated=False,
            auth_method="api_key",
            error="api_key_not_configured"
        )

    # Validate API key
    if api_key == API_KEY:
        logger.debug("API key authentication successful")
        return AuthenticationResult(
            authenticated=True,
            client_id="api_key_client",
            scope="read write admin",  # API key gets full access
            auth_method="api_key"
        )

    logger.debug("API key authentication failed: key mismatch")
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
    1. OAuth Bearer token (JWT or stored token) - only if OAuth is enabled
    2. Legacy API key authentication
    3. Anonymous access (if explicitly enabled)

    Returns:
        AuthenticationResult with authentication details
    """
    # Try OAuth Bearer token authentication first (only if OAuth is enabled)
    if credentials and credentials.scheme.lower() == "bearer":
        # OAuth Bearer token validation only if OAuth is enabled
        if OAUTH_ENABLED:
            auth_result = await authenticate_bearer_token(credentials.credentials)
            if auth_result.authenticated:
                return auth_result

            # OAuth token provided but invalid - log the attempt
            logger.debug(f"OAuth Bearer token validation failed for enabled OAuth system")

        # Try API key authentication as fallback (works regardless of OAuth state)
        if API_KEY:
            # Some clients might send API key as Bearer token
            api_key_result = authenticate_api_key(credentials.credentials)
            if api_key_result.authenticated:
                return api_key_result

        # Determine appropriate error message based on OAuth state
        if OAUTH_ENABLED:
            error_msg = "The access token provided is expired, revoked, malformed, or invalid"
            logger.warning("Invalid Bearer token provided and API key fallback failed")
        else:
            error_msg = "OAuth is disabled. Use API key authentication or enable anonymous access."
            logger.debug("Bearer token provided but OAuth is disabled, API key fallback failed")

        # All Bearer token authentication methods failed
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "invalid_token",
                "error_description": error_msg
            },
            headers={"WWW-Authenticate": "Bearer"}
        )

    # Allow anonymous access only if explicitly enabled
    if ALLOW_ANONYMOUS_ACCESS:
        logger.debug("Anonymous access explicitly enabled, granting read-only access")
        return AuthenticationResult(
            authenticated=True,
            client_id="anonymous",
            scope="read",  # Anonymous users get read-only access for security
            auth_method="none"
        )

    # No credentials provided and anonymous access not allowed
    if API_KEY or OAUTH_ENABLED:
        logger.debug("No valid authentication provided")
        if OAUTH_ENABLED and API_KEY:
            error_msg = "Authorization required. Provide valid OAuth Bearer token or API key."
        elif OAUTH_ENABLED:
            error_msg = "Authorization required. Provide valid OAuth Bearer token."
        else:
            error_msg = "Authorization required. Provide valid API key."
    else:
        logger.debug("No authentication configured and anonymous access disabled")
        error_msg = "Authentication is required. Set MCP_ALLOW_ANONYMOUS_ACCESS=true to enable anonymous access."

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