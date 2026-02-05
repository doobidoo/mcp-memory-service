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
Authentication middleware for MCP Memory Service.

Provides API key authentication with optional anonymous access.
OAuth 2.1 support removed during security remediation (CVE in python-jose).
"""

import logging
import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ...config import (
    ALLOW_ANONYMOUS_ACCESS,
    API_KEY,
)

logger = logging.getLogger(__name__)

# Optional Bearer token security scheme (for API key passed as Bearer)
bearer_scheme = HTTPBearer(auto_error=False)


class AuthenticationResult:
    """Result of authentication attempt."""

    def __init__(
        self,
        authenticated: bool,
        client_id: str | None = None,
        scope: str | None = None,
        auth_method: str | None = None,
        error: str | None = None,
    ):
        self.authenticated = authenticated
        self.client_id = client_id
        self.scope = scope
        self.auth_method = auth_method  # "api_key" or "none"
        self.error = error

    def has_scope(self, required_scope: str) -> bool:
        """Check if the authenticated user has the required scope."""
        if not self.authenticated or not self.scope:
            return False
        scopes = self.scope.split()
        return required_scope in scopes

    def require_scope(self, required_scope: str) -> None:
        """Raise an exception if the required scope is not present."""
        if not self.has_scope(required_scope):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"error": "insufficient_scope", "error_description": f"Required scope '{required_scope}' not granted"},
            )


def authenticate_api_key(api_key: str | None) -> AuthenticationResult:
    """
    Validate API key authentication using constant-time comparison.

    Returns:
        AuthenticationResult with authentication status
    """
    if not api_key or not isinstance(api_key, str):
        logger.debug("API key authentication failed: invalid input")
        return AuthenticationResult(authenticated=False, auth_method="api_key", error="invalid_api_key")

    api_key = api_key.strip()
    if not api_key:
        logger.debug("API key authentication failed: empty key")
        return AuthenticationResult(authenticated=False, auth_method="api_key", error="invalid_api_key")

    # Check if API key is configured
    if not API_KEY:
        logger.debug("API key authentication failed: no API key configured")
        return AuthenticationResult(authenticated=False, auth_method="api_key", error="api_key_not_configured")

    # Validate API key using constant-time comparison (CWE-208 fix)
    if secrets.compare_digest(api_key, API_KEY):
        logger.debug("API key authentication successful")
        return AuthenticationResult(
            authenticated=True,
            client_id="api_key_client",
            scope="read write admin",  # API key gets full access
            auth_method="api_key",
        )

    logger.debug("API key authentication failed: key mismatch")
    return AuthenticationResult(authenticated=False, auth_method="api_key", error="invalid_api_key")


async def get_current_user(credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme)) -> AuthenticationResult:
    """
    Get current authenticated user.

    Tries in order:
    1. API key authentication (via Bearer token or header)
    2. Anonymous access (if explicitly enabled)

    Returns:
        AuthenticationResult with authentication details
    """
    # Try Bearer token (could be API key)
    if credentials and credentials.scheme.lower() == "bearer":
        if API_KEY:
            api_key_result = authenticate_api_key(credentials.credentials)
            if api_key_result.authenticated:
                return api_key_result

        # Bearer token provided but not a valid API key
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "invalid_token", "error_description": "Use API key authentication or enable anonymous access."},
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Allow anonymous access only if explicitly enabled
    if ALLOW_ANONYMOUS_ACCESS:
        logger.debug("Anonymous access explicitly enabled, granting read-only access")
        return AuthenticationResult(
            authenticated=True,
            client_id="anonymous",
            scope="read",  # Read-only: anonymous users cannot modify data (CWE-269 fix)
            auth_method="none",
        )

    # No credentials provided and anonymous access not allowed
    if API_KEY:
        error_msg = "Authorization required. Provide valid API key."
    else:
        error_msg = "Authentication is required. Set MCP_ALLOW_ANONYMOUS_ACCESS=true to enable anonymous access."

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"error": "authorization_required", "error_description": error_msg},
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_scope(scope: str):
    """
    Create a dependency that requires a specific scope.

    Usage:
        @app.get("/admin", dependencies=[Depends(require_scope("admin"))])
    """

    async def scope_dependency(user: AuthenticationResult = Depends(get_current_user)):
        user.require_scope(scope)
        return user

    return scope_dependency


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
