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
SQLite OAuth 2.1 storage implementation.

Provides persistent storage for OAuth clients, authorization codes,
and access tokens using SQLite. This is suitable for production deployments
with multiple workers and persistence requirements.

Features:
- WAL mode for multi-process safety
- Atomic one-time code consumption (prevents replay attacks)
- Automatic schema initialization
- Efficient indexing for fast lookups
- JSON serialization for arrays and metadata
"""

import json
import time
import asyncio
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import aiosqlite

from .base import OAuthStorage
from ..models import RegisteredClient
from ....config import OAUTH_ACCESS_TOKEN_EXPIRE_MINUTES, OAUTH_AUTHORIZATION_CODE_EXPIRE_MINUTES

logger = logging.getLogger(__name__)


class SQLiteOAuthStorage(OAuthStorage):
    """
    SQLite-based storage for OAuth 2.1 clients and tokens.

    Provides persistent storage with multi-process safety using WAL mode.
    Suitable for:
    - Production deployments with multiple workers
    - Persistent storage requirements
    - High concurrency scenarios

    For development and single-instance deployments, MemoryOAuthStorage
    may be simpler.
    """

    def __init__(self, db_path: str):
        """
        Initialize SQLite storage backend.

        Args:
            db_path: Path to SQLite database file (will be created if doesn't exist)
        """
        self.db_path = db_path
        self._connection: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock()
        self._initialized = False

        # Ensure parent directory exists
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            Path(db_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized SQLite OAuth storage at {db_path}")

    async def _get_connection(self) -> aiosqlite.Connection:
        """
        Get or create database connection.

        Returns:
            aiosqlite.Connection instance

        Raises:
            Exception: If connection fails
        """
        if self._connection is None:
            self._connection = await aiosqlite.connect(self.db_path)
            self._connection.row_factory = aiosqlite.Row
            if not self._initialized:
                await self._init_db()
                self._initialized = True
        return self._connection

    async def _execute(self, query: str, params: tuple = ()):
        """
        Execute a SQL query.

        Args:
            query: SQL query string
            params: Query parameters tuple

        Returns:
            Cursor after execution

        Raises:
            Exception: If query execution fails
        """
        conn = await self._get_connection()
        return await conn.execute(query, params)

    async def _commit(self):
        """
        Commit pending transactions.

        Raises:
            Exception: If commit fails
        """
        if self._connection:
            await self._connection.commit()

    async def _init_db(self):
        """
        Initialize database schema.

        Creates tables and indexes if they don't exist.
        Enables WAL mode for multi-process safety.

        Raises:
            Exception: If schema initialization fails
        """
        # Enable WAL mode for multi-process safety
        await self._execute("PRAGMA journal_mode=WAL")
        await self._execute("PRAGMA synchronous=NORMAL")
        await self._execute("PRAGMA busy_timeout=5000")

        # Create oauth_clients table
        await self._execute("""
            CREATE TABLE IF NOT EXISTS oauth_clients (
                client_id TEXT PRIMARY KEY,
                client_secret TEXT NOT NULL,
                client_name TEXT,
                redirect_uris TEXT NOT NULL,
                response_types TEXT NOT NULL,
                grant_types TEXT NOT NULL,
                scope TEXT,
                token_endpoint_auth_method TEXT,
                created_at REAL NOT NULL,
                metadata TEXT
            )
        """)

        # Create oauth_authorization_codes table
        await self._execute("""
            CREATE TABLE IF NOT EXISTS oauth_authorization_codes (
                code TEXT PRIMARY KEY,
                client_id TEXT NOT NULL,
                redirect_uri TEXT,
                scope TEXT,
                expires_at REAL NOT NULL,
                consumed INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                FOREIGN KEY (client_id) REFERENCES oauth_clients(client_id) ON DELETE CASCADE
            )
        """)

        # Create indexes for authorization codes
        await self._execute(
            "CREATE INDEX IF NOT EXISTS idx_auth_codes_expires ON oauth_authorization_codes(expires_at)"
        )
        await self._execute(
            "CREATE INDEX IF NOT EXISTS idx_auth_codes_consumed ON oauth_authorization_codes(consumed)"
        )

        # Create oauth_access_tokens table
        await self._execute("""
            CREATE TABLE IF NOT EXISTS oauth_access_tokens (
                token TEXT PRIMARY KEY,
                client_id TEXT NOT NULL,
                scope TEXT,
                expires_at REAL NOT NULL,
                created_at REAL NOT NULL,
                FOREIGN KEY (client_id) REFERENCES oauth_clients(client_id) ON DELETE CASCADE
            )
        """)

        # Create indexes for access tokens
        await self._execute(
            "CREATE INDEX IF NOT EXISTS idx_access_tokens_expires ON oauth_access_tokens(expires_at)"
        )
        await self._execute(
            "CREATE INDEX IF NOT EXISTS idx_access_tokens_client ON oauth_access_tokens(client_id)"
        )

        await self._commit()
        logger.info("SQLite OAuth schema initialized")

    async def store_client(self, client: RegisteredClient) -> None:
        """
        Store a registered OAuth client.

        Args:
            client: RegisteredClient instance containing client metadata

        Raises:
            Exception: If storage operation fails
        """
        async with self._lock:
            # Serialize list fields to JSON
            redirect_uris_json = json.dumps(client.redirect_uris)
            response_types_json = json.dumps(client.response_types)
            grant_types_json = json.dumps(client.grant_types)

            await self._execute(
                """
                INSERT OR REPLACE INTO oauth_clients
                (client_id, client_secret, client_name, redirect_uris, response_types,
                 grant_types, scope, token_endpoint_auth_method, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    client.client_id,
                    client.client_secret,
                    client.client_name,
                    redirect_uris_json,
                    response_types_json,
                    grant_types_json,
                    None,  # scope field (currently unused)
                    client.token_endpoint_auth_method,
                    client.created_at,
                    None  # metadata (currently unused)
                )
            )
            await self._commit()
            logger.debug(f"Stored OAuth client: {client.client_id}")

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
        async with self._lock:
            cursor = await self._execute(
                "SELECT * FROM oauth_clients WHERE client_id = ?",
                (client_id,)
            )
            row = await cursor.fetchone()

            if not row:
                return None

            # Deserialize JSON fields
            redirect_uris = json.loads(row["redirect_uris"])
            response_types = json.loads(row["response_types"])
            grant_types = json.loads(row["grant_types"])

            return RegisteredClient(
                client_id=row["client_id"],
                client_secret=row["client_secret"],
                client_name=row["client_name"],
                redirect_uris=redirect_uris,
                response_types=response_types,
                grant_types=grant_types,
                token_endpoint_auth_method=row["token_endpoint_auth_method"],
                created_at=row["created_at"]
            )

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
        if expires_in is None:
            expires_in = OAUTH_AUTHORIZATION_CODE_EXPIRE_MINUTES * 60

        async with self._lock:
            now = time.time()
            await self._execute(
                """
                INSERT INTO oauth_authorization_codes
                (code, client_id, redirect_uri, scope, expires_at, consumed, created_at)
                VALUES (?, ?, ?, ?, ?, 0, ?)
                """,
                (code, client_id, redirect_uri, scope, now + expires_in, now)
            )
            await self._commit()
            logger.debug(f"Stored authorization code for client: {client_id}")

    async def get_authorization_code(self, code: str) -> Optional[Dict]:
        """
        Retrieve and consume an authorization code (one-time use).

        This method implements atomic one-time code consumption to prevent
        replay attacks. The code is marked as consumed using an atomic
        UPDATE WHERE consumed=0 operation.

        Args:
            code: Authorization code value

        Returns:
            Dict with keys: client_id, redirect_uri, scope, expires_at
            None if code not found, expired, or already consumed

        Raises:
            Exception: If retrieval operation fails
        """
        async with self._lock:
            now = time.time()

            # Check if code exists, not consumed, and not expired
            cursor = await self._execute(
                """
                SELECT * FROM oauth_authorization_codes
                WHERE code = ? AND consumed = 0 AND expires_at > ?
                """,
                (code, now)
            )
            row = await cursor.fetchone()

            if not row:
                return None

            # Atomically mark as consumed (prevents replay attacks)
            cursor = await self._execute(
                """
                UPDATE oauth_authorization_codes
                SET consumed = 1
                WHERE code = ? AND consumed = 0
                """,
                (code,)
            )

            # Check if update succeeded (race condition protection)
            if cursor.rowcount == 0:
                logger.warning(f"Authorization code already consumed (race condition): {code}")
                return None

            await self._commit()
            logger.debug(f"Consumed authorization code for client: {row['client_id']}")

            return {
                "client_id": row["client_id"],
                "redirect_uri": row["redirect_uri"],
                "scope": row["scope"],
                "expires_at": row["expires_at"]
            }

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
        if expires_in is None:
            expires_in = OAUTH_ACCESS_TOKEN_EXPIRE_MINUTES * 60

        async with self._lock:
            now = time.time()
            await self._execute(
                """
                INSERT INTO oauth_access_tokens
                (token, client_id, scope, expires_at, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (token, client_id, scope, now + expires_in, now)
            )
            await self._commit()
            logger.debug(f"Stored access token for client: {client_id}")

    async def get_access_token(self, token: str) -> Optional[Dict]:
        """
        Retrieve access token information if valid.

        Args:
            token: Access token value

        Returns:
            Dict with keys: client_id, scope, expires_at
            None if token not found or expired

        Raises:
            Exception: If retrieval operation fails
        """
        async with self._lock:
            now = time.time()

            cursor = await self._execute(
                """
                SELECT * FROM oauth_access_tokens
                WHERE token = ?
                """,
                (token,)
            )
            row = await cursor.fetchone()

            if not row:
                return None

            # Check if token is expired
            if row["expires_at"] <= now:
                # Clean up expired token
                await self._execute(
                    "DELETE FROM oauth_access_tokens WHERE token = ?",
                    (token,)
                )
                await self._commit()
                return None

            return {
                "client_id": row["client_id"],
                "scope": row["scope"],
                "expires_at": row["expires_at"]
            }

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
        async with self._lock:
            cursor = await self._execute(
                "DELETE FROM oauth_access_tokens WHERE token = ?",
                (token,)
            )
            await self._commit()
            revoked = cursor.rowcount > 0
            if revoked:
                logger.debug(f"Revoked access token")
            return revoked

    async def cleanup_expired(self) -> Dict[str, int]:
        """
        Clean up expired authorization codes and access tokens.

        This method should be called periodically to prevent storage bloat.

        Returns:
            Dict with keys: expired_codes_cleaned, expired_tokens_cleaned

        Raises:
            Exception: If cleanup operation fails
        """
        async with self._lock:
            now = time.time()

            # Clean up expired authorization codes
            cursor1 = await self._execute(
                "DELETE FROM oauth_authorization_codes WHERE expires_at < ?",
                (now,)
            )
            codes_cleaned = cursor1.rowcount

            # Clean up expired access tokens
            cursor2 = await self._execute(
                "DELETE FROM oauth_access_tokens WHERE expires_at < ?",
                (now,)
            )
            tokens_cleaned = cursor2.rowcount

            await self._commit()

            if codes_cleaned > 0 or tokens_cleaned > 0:
                logger.info(
                    f"Cleaned up {codes_cleaned} expired authorization codes "
                    f"and {tokens_cleaned} expired access tokens"
                )

            return {
                "expired_codes_cleaned": codes_cleaned,
                "expired_tokens_cleaned": tokens_cleaned
            }

    async def close(self) -> None:
        """
        Clean up storage resources (database connection).

        Should be called when shutting down the OAuth server.

        Raises:
            Exception: If cleanup operation fails
        """
        if self._connection:
            await self._connection.close()
            self._connection = None
            self._initialized = False
            logger.info("Closed SQLite OAuth storage connection")

    async def get_stats(self) -> Dict:
        """
        Get storage statistics.

        Returns:
            Dict with counts of registered clients, active codes, and active tokens
        """
        async with self._lock:
            # Count registered clients
            cursor1 = await self._execute("SELECT COUNT(*) FROM oauth_clients")
            row1 = await cursor1.fetchone()
            client_count = row1[0] if row1 else 0

            # Count active (non-consumed, non-expired) authorization codes
            cursor2 = await self._execute(
                "SELECT COUNT(*) FROM oauth_authorization_codes WHERE consumed = 0 AND expires_at > ?",
                (time.time(),)
            )
            row2 = await cursor2.fetchone()
            code_count = row2[0] if row2 else 0

            # Count active (non-expired) access tokens
            cursor3 = await self._execute(
                "SELECT COUNT(*) FROM oauth_access_tokens WHERE expires_at > ?",
                (time.time(),)
            )
            row3 = await cursor3.fetchone()
            token_count = row3[0] if row3 else 0

            return {
                "backend": "sqlite",
                "db_path": self.db_path,
                "registered_clients": client_count,
                "active_authorization_codes": code_count,
                "active_access_tokens": token_count
            }
