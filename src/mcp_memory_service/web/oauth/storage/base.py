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
Abstract base class for OAuth 2.1 storage backends.

Defines the interface that all OAuth storage implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
from ..models import RegisteredClient


class OAuthStorage(ABC):
    """
    Abstract base class for OAuth 2.1 storage backends.

    Provides interface for storing and retrieving OAuth clients, authorization codes,
    and access tokens. Concrete implementations can use different storage backends
    (in-memory, SQLite, etc.).
    """

    @abstractmethod
    async def store_client(self, client: RegisteredClient) -> None:
        """
        Store a registered OAuth client.

        Args:
            client: RegisteredClient instance containing client metadata

        Raises:
            Exception: If storage operation fails
        """
        pass

    @abstractmethod
    async def get_client(self, client_id: str) -> Optional[RegisteredClient]:
        """
        Retrieve a registered OAuth client by client ID.

        Args:
            client_id: OAuth client identifier

        Returns:
            RegisteredClient instance if found, None otherwise

        Raises:
            Exception: If retrieval operation fails
        """
        pass

    @abstractmethod
    async def authenticate_client(self, client_id: str, client_secret: str) -> bool:
        """
        Authenticate a client using client_id and client_secret.

        Args:
            client_id: OAuth client identifier
            client_secret: OAuth client secret

        Returns:
            True if authentication succeeds, False otherwise

        Raises:
            Exception: If authentication operation fails
        """
        pass

    @abstractmethod
    async def store_authorization_code(
        self,
        code: str,
        client_id: str,
        redirect_uri: Optional[str] = None,
        scope: Optional[str] = None,
        expires_in: Optional[int] = None
    ) -> None:
        """
        Store an authorization code for the authorization code flow.

        Args:
            code: Authorization code value
            client_id: OAuth client identifier
            redirect_uri: Redirect URI associated with this authorization
            scope: Space-separated list of granted scopes
            expires_in: Expiration time in seconds (None uses default)

        Raises:
            Exception: If storage operation fails
        """
        pass

    @abstractmethod
    async def get_authorization_code(self, code: str) -> Optional[Dict]:
        """
        Retrieve and consume an authorization code (one-time use).

        Authorization codes are single-use tokens. This method should:
        1. Retrieve the code data
        2. Validate it hasn't expired
        3. Remove it from storage (consume)

        Args:
            code: Authorization code value

        Returns:
            Dict with keys: client_id, redirect_uri, scope, expires_at
            None if code not found or expired

        Raises:
            Exception: If retrieval operation fails
        """
        pass

    @abstractmethod
    async def store_access_token(
        self,
        token: str,
        client_id: str,
        scope: Optional[str] = None,
        expires_in: Optional[int] = None
    ) -> None:
        """
        Store an access token.

        Args:
            token: Access token value
            client_id: OAuth client identifier
            scope: Space-separated list of granted scopes
            expires_in: Expiration time in seconds (None uses default)

        Raises:
            Exception: If storage operation fails
        """
        pass

    @abstractmethod
    async def get_access_token(self, token: str) -> Optional[Dict]:
        """
        Retrieve access token information if valid.

        Should validate token hasn't expired and clean up expired tokens.

        Args:
            token: Access token value

        Returns:
            Dict with keys: client_id, scope, expires_at
            None if token not found or expired

        Raises:
            Exception: If retrieval operation fails
        """
        pass

    @abstractmethod
    async def revoke_access_token(self, token: str) -> bool:
        """
        Revoke an access token (remove from storage).

        Args:
            token: Access token value to revoke

        Returns:
            True if token was revoked, False if token not found

        Raises:
            Exception: If revocation operation fails
        """
        pass

    @abstractmethod
    async def cleanup_expired(self) -> Dict[str, int]:
        """
        Clean up expired authorization codes and access tokens.

        This method should be called periodically to prevent storage bloat.

        Returns:
            Dict with keys: expired_codes_cleaned, expired_tokens_cleaned

        Raises:
            Exception: If cleanup operation fails
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        Clean up storage resources (connections, file handles, etc.).

        Should be called when shutting down the OAuth server.

        Raises:
            Exception: If cleanup operation fails
        """
        pass

    # Token generation methods (not storage-specific, but part of interface)
    def generate_client_id(self) -> str:
        """
        Generate a unique client ID.

        Returns:
            Unique client identifier string
        """
        import secrets
        return f"mcp_client_{secrets.token_urlsafe(16)}"

    def generate_client_secret(self) -> str:
        """
        Generate a secure client secret.

        Returns:
            Secure random client secret string
        """
        import secrets
        return secrets.token_urlsafe(32)

    def generate_authorization_code(self) -> str:
        """
        Generate a secure authorization code.

        Returns:
            Secure random authorization code string
        """
        import secrets
        return secrets.token_urlsafe(32)

    def generate_access_token(self) -> str:
        """
        Generate a secure access token.

        Returns:
            Secure random access token string
        """
        import secrets
        return secrets.token_urlsafe(32)

    # Optional statistics method
    async def get_stats(self) -> Dict:
        """
        Get storage statistics (optional, implementation-specific).

        Returns:
            Dict with backend-specific statistics
        """
        return {}
