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
OAuth 2.1 in-memory storage for MCP Memory Service.

Provides simple in-memory storage for OAuth clients and authorization codes.
This is an MVP implementation - production deployments should use persistent storage.
"""

import time
import secrets
import asyncio
from typing import Dict, Optional
from .models import RegisteredClient
from ...config import OAUTH_ACCESS_TOKEN_EXPIRE_MINUTES, OAUTH_AUTHORIZATION_CODE_EXPIRE_MINUTES


class OAuthStorage:
    """In-memory storage for OAuth 2.1 clients and authorization codes."""

    def __init__(self):
        # Registered OAuth clients
        self._clients: Dict[str, RegisteredClient] = {}

        # Active authorization codes (code -> client_id, expires_at, redirect_uri, scope)
        self._authorization_codes: Dict[str, Dict] = {}

        # Active access tokens (token -> client_id, expires_at, scope)
        self._access_tokens: Dict[str, Dict] = {}

        # Thread safety lock for concurrent access
        self._lock = asyncio.Lock()

    async def store_client(self, client: RegisteredClient) -> None:
        """Store a registered OAuth client."""
        async with self._lock:
            self._clients[client.client_id] = client

    async def get_client(self, client_id: str) -> Optional[RegisteredClient]:
        """Get a registered OAuth client by ID."""
        async with self._lock:
            return self._clients.get(client_id)

    async def authenticate_client(self, client_id: str, client_secret: str) -> bool:
        """Authenticate a client using client_id and client_secret."""
        client = await self.get_client(client_id)
        if not client:
            return False
        return client.client_secret == client_secret

    async def store_authorization_code(
        self,
        code: str,
        client_id: str,
        redirect_uri: Optional[str] = None,
        scope: Optional[str] = None,
        expires_in: Optional[int] = None
    ) -> None:
        """Store an authorization code."""
        if expires_in is None:
            expires_in = OAUTH_AUTHORIZATION_CODE_EXPIRE_MINUTES * 60
        async with self._lock:
            self._authorization_codes[code] = {
                "client_id": client_id,
                "redirect_uri": redirect_uri,
                "scope": scope,
                "expires_at": time.time() + expires_in
            }

    async def get_authorization_code(self, code: str) -> Optional[Dict]:
        """Get and consume an authorization code (one-time use)."""
        async with self._lock:
            code_data = self._authorization_codes.pop(code, None)

            # Check if code exists and hasn't expired
            if code_data and code_data["expires_at"] > time.time():
                return code_data
            return None

    async def store_access_token(
        self,
        token: str,
        client_id: str,
        scope: Optional[str] = None,
        expires_in: Optional[int] = None
    ) -> None:
        """Store an access token."""
        if expires_in is None:
            expires_in = OAUTH_ACCESS_TOKEN_EXPIRE_MINUTES * 60
        async with self._lock:
            self._access_tokens[token] = {
                "client_id": client_id,
                "scope": scope,
                "expires_at": time.time() + expires_in
            }

    async def get_access_token(self, token: str) -> Optional[Dict]:
        """Get access token information if valid."""
        async with self._lock:
            token_data = self._access_tokens.get(token)

            # Check if token exists and hasn't expired
            if token_data and token_data["expires_at"] > time.time():
                return token_data

            # Clean up expired token
            if token_data:
                self._access_tokens.pop(token, None)

            return None

    async def cleanup_expired(self) -> Dict[str, int]:
        """Clean up expired authorization codes and access tokens."""
        async with self._lock:
            current_time = time.time()

            # Clean up expired authorization codes
            expired_codes = [
                code for code, data in self._authorization_codes.items()
                if data["expires_at"] <= current_time
            ]
            for code in expired_codes:
                self._authorization_codes.pop(code, None)

            # Clean up expired access tokens
            expired_tokens = [
                token for token, data in self._access_tokens.items()
                if data["expires_at"] <= current_time
            ]
            for token in expired_tokens:
                self._access_tokens.pop(token, None)

            return {
                "expired_codes_cleaned": len(expired_codes),
                "expired_tokens_cleaned": len(expired_tokens)
            }

    def generate_client_id(self) -> str:
        """Generate a unique client ID."""
        return f"mcp_client_{secrets.token_urlsafe(16)}"

    def generate_client_secret(self) -> str:
        """Generate a secure client secret."""
        return secrets.token_urlsafe(32)

    def generate_authorization_code(self) -> str:
        """Generate a secure authorization code."""
        return secrets.token_urlsafe(32)

    def generate_access_token(self) -> str:
        """Generate a secure access token."""
        return secrets.token_urlsafe(32)

    # Statistics and management methods
    async def get_stats(self) -> Dict:
        """Get storage statistics."""
        async with self._lock:
            return {
                "registered_clients": len(self._clients),
                "active_authorization_codes": len(self._authorization_codes),
                "active_access_tokens": len(self._access_tokens)
            }


# Global OAuth storage instance
oauth_storage = OAuthStorage()