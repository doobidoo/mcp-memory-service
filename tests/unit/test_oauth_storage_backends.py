#!/usr/bin/env python3
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
Unit tests for OAuth storage backends (memory and SQLite).

Tests all storage backends using parametrized fixtures to ensure consistent
behavior across implementations. Includes security-critical tests for:
- One-time authorization code consumption (prevents replay attacks)
- Token expiration handling
- Client authentication
- Concurrent access safety (SQLite)
"""

import asyncio
import os
import tempfile
import time

import pytest

from mcp_memory_service.web.oauth.models import RegisteredClient
from mcp_memory_service.web.oauth.storage import (
    MemoryOAuthStorage,
    create_oauth_storage,
)


# Parametrize over all backends
@pytest.fixture(params=["memory", "sqlite"])
async def storage(request):
    """
    Create storage backend for testing (parametrized).

    Yields storage instance configured for the requested backend type.
    Automatically handles cleanup (close connection, delete temp files).

    Args:
        request: Pytest request fixture containing backend type parameter

    Yields:
        OAuthStorage instance (MemoryOAuthStorage or SQLiteOAuthStorage)
    """
    backend_type = request.param
    db_path = None

    if backend_type == "sqlite":
        # Use temp file for SQLite tests
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            db_path = tmp.name
        storage = create_oauth_storage("sqlite", db_path=db_path)
    else:
        storage = create_oauth_storage("memory")

    yield storage

    # Cleanup
    await storage.close()
    if backend_type == "sqlite" and db_path and os.path.exists(db_path):
        os.unlink(db_path)


class TestOAuthStorageBackends:
    """Parametrized tests for all OAuth storage backends."""

    @pytest.mark.asyncio
    async def test_store_and_get_client(self, storage):
        """Test basic client registration and retrieval."""
        # Create test client
        client = RegisteredClient(
            client_id="test_client_1",
            client_secret="secret_123",
            client_name="Test Client",
            redirect_uris=["http://localhost/callback"],
            grant_types=["authorization_code"],
            response_types=["code"],
            token_endpoint_auth_method="client_secret_basic",
            created_at=time.time(),
        )

        # Store client
        await storage.store_client(client)

        # Retrieve client
        retrieved = await storage.get_client("test_client_1")

        # Verify
        assert retrieved is not None
        assert retrieved.client_id == "test_client_1"
        assert retrieved.client_secret == "secret_123"
        assert retrieved.client_name == "Test Client"
        assert retrieved.redirect_uris == ["http://localhost/callback"]
        assert retrieved.grant_types == ["authorization_code"]
        assert retrieved.response_types == ["code"]

    @pytest.mark.asyncio
    async def test_authenticate_client(self, storage):
        """Test client authentication with valid and invalid secrets."""
        # Register client
        client = RegisteredClient(
            client_id="auth_client",
            client_secret="correct_secret",
            client_name="Auth Test",
            redirect_uris=["http://localhost/callback"],
            created_at=time.time(),
        )
        await storage.store_client(client)

        # Valid authentication
        result = await storage.authenticate_client("auth_client", "correct_secret")
        assert result is True

        # Invalid secret
        result = await storage.authenticate_client("auth_client", "wrong_secret")
        assert result is False

        # Non-existent client
        result = await storage.authenticate_client("nonexistent", "any_secret")
        assert result is False

    @pytest.mark.asyncio
    async def test_store_and_get_authorization_code(self, storage):
        """Test authorization code lifecycle (store, retrieve, consume)."""
        # Store authorization code
        await storage.store_authorization_code(
            code="test_code_123",
            client_id="client_1",
            redirect_uri="http://localhost/callback",
            scope="read write",
            expires_in=300,  # 5 minutes
        )

        # Retrieve code (should succeed and consume)
        code_data = await storage.get_authorization_code("test_code_123")

        assert code_data is not None
        assert code_data["client_id"] == "client_1"
        assert code_data["redirect_uri"] == "http://localhost/callback"
        assert code_data["scope"] == "read write"
        assert code_data["expires_at"] > time.time()

    @pytest.mark.asyncio
    async def test_authorization_code_expiration(self, storage):
        """Test that expired authorization codes return None."""
        # Store code with 1 second expiration
        await storage.store_authorization_code(
            code="expired_code", client_id="client_1", expires_in=1
        )

        # Wait for expiration
        await asyncio.sleep(1.1)

        # Try to retrieve expired code
        code_data = await storage.get_authorization_code("expired_code")
        assert code_data is None

    @pytest.mark.asyncio
    async def test_authorization_code_one_time_use(self, storage):
        """
        Test that authorization codes can only be used once (security critical).

        This test prevents replay attacks by ensuring authorization codes are
        consumed on first use and cannot be reused.
        """
        # Store code
        await storage.store_authorization_code(
            code="one_time_code", client_id="client_1", expires_in=300
        )

        # First use: should succeed
        result1 = await storage.get_authorization_code("one_time_code")
        assert result1 is not None
        assert result1["client_id"] == "client_1"

        # Second use: should fail (code consumed)
        result2 = await storage.get_authorization_code("one_time_code")
        assert result2 is None

    @pytest.mark.asyncio
    async def test_store_and_get_access_token(self, storage):
        """Test access token lifecycle (store and retrieve)."""
        # Store access token
        await storage.store_access_token(
            token="access_token_123",
            client_id="client_1",
            scope="read write",
            expires_in=3600,  # 1 hour
        )

        # Retrieve token
        token_data = await storage.get_access_token("access_token_123")

        assert token_data is not None
        assert token_data["client_id"] == "client_1"
        assert token_data["scope"] == "read write"
        assert token_data["expires_at"] > time.time()

    @pytest.mark.asyncio
    async def test_access_token_expiration(self, storage):
        """Test that expired access tokens return None."""
        # Store token with 1 second expiration
        await storage.store_access_token(
            token="expired_token", client_id="client_1", expires_in=1
        )

        # Wait for expiration
        await asyncio.sleep(1.1)

        # Try to retrieve expired token
        token_data = await storage.get_access_token("expired_token")
        assert token_data is None

    @pytest.mark.asyncio
    async def test_revoke_access_token(self, storage):
        """Test token revocation."""
        # Store token
        await storage.store_access_token(
            token="revoke_token", client_id="client_1", expires_in=3600
        )

        # Verify token exists
        token_data = await storage.get_access_token("revoke_token")
        assert token_data is not None

        # Revoke token
        revoked = await storage.revoke_access_token("revoke_token")
        assert revoked is True

        # Verify token no longer exists
        token_data = await storage.get_access_token("revoke_token")
        assert token_data is None

        # Revoke non-existent token
        revoked = await storage.revoke_access_token("nonexistent_token")
        assert revoked is False

    @pytest.mark.asyncio
    async def test_cleanup_expired(self, storage):
        """Test cleanup removes only expired items."""
        # Store valid code and token
        await storage.store_authorization_code(
            code="valid_code", client_id="client_1", expires_in=3600
        )
        await storage.store_access_token(
            token="valid_token", client_id="client_1", expires_in=3600
        )

        # Store expired code and token
        await storage.store_authorization_code(
            code="expired_code", client_id="client_1", expires_in=1
        )
        await storage.store_access_token(
            token="expired_token", client_id="client_1", expires_in=1
        )

        # Wait for expiration
        await asyncio.sleep(1.1)

        # Run cleanup
        result = await storage.cleanup_expired()

        # Verify cleanup results
        assert "expired_codes_cleaned" in result
        assert "expired_tokens_cleaned" in result
        assert result["expired_codes_cleaned"] >= 1
        assert result["expired_tokens_cleaned"] >= 1

        # Verify valid items still exist
        valid_code = await storage.get_authorization_code("valid_code")
        assert valid_code is not None

        valid_token = await storage.get_access_token("valid_token")
        assert valid_token is not None

        # Verify expired items were cleaned
        expired_code = await storage.get_authorization_code("expired_code")
        assert expired_code is None

        expired_token = await storage.get_access_token("expired_token")
        assert expired_token is None

    @pytest.mark.asyncio
    async def test_get_client_not_found(self, storage):
        """Test that non-existent client returns None."""
        result = await storage.get_client("nonexistent_client")
        assert result is None

    @pytest.mark.asyncio
    async def test_client_with_metadata(self, storage):
        """Test complex RegisteredClient with all fields."""
        # Create client with all optional fields
        client = RegisteredClient(
            client_id="complex_client",
            client_secret="secret_xyz",
            client_name="Complex Test Client",
            redirect_uris=[
                "http://localhost/callback",
                "https://app.example.com/oauth/callback",
            ],
            grant_types=["authorization_code", "client_credentials"],
            response_types=["code"],
            token_endpoint_auth_method="client_secret_post",
            created_at=time.time(),
        )

        # Store and retrieve
        await storage.store_client(client)
        retrieved = await storage.get_client("complex_client")

        # Verify all fields preserved
        assert retrieved is not None
        assert retrieved.client_id == "complex_client"
        assert retrieved.client_secret == "secret_xyz"
        assert retrieved.client_name == "Complex Test Client"
        assert len(retrieved.redirect_uris) == 2
        assert "http://localhost/callback" in retrieved.redirect_uris
        assert "https://app.example.com/oauth/callback" in retrieved.redirect_uris
        assert set(retrieved.grant_types) == {
            "authorization_code",
            "client_credentials",
        }
        assert retrieved.response_types == ["code"]
        assert retrieved.token_endpoint_auth_method == "client_secret_post"

    @pytest.mark.asyncio
    async def test_concurrent_code_consumption(self, storage):
        """
        Test that concurrent code consumption is handled safely (SQLite only).

        This test ensures the SQLite backend's atomic UPDATE WHERE consumed=0
        operation prevents race conditions when multiple workers try to consume
        the same authorization code simultaneously.
        """
        # Skip for memory backend (doesn't need race condition tests)
        if isinstance(storage, MemoryOAuthStorage):
            pytest.skip("Memory backend doesn't need race condition tests")

        # Store code
        await storage.store_authorization_code(
            code="race_code", client_id="client_1", expires_in=300
        )

        # Simulate concurrent consumption attempts (10 workers racing)
        tasks = [storage.get_authorization_code("race_code") for _ in range(10)]
        results = await asyncio.gather(*tasks)

        # Exactly one should succeed, rest should get None
        successful = [r for r in results if r is not None]
        failed = [r for r in results if r is None]

        assert len(successful) == 1, f"Expected 1 success, got {len(successful)}"
        assert len(failed) == 9, f"Expected 9 failures, got {len(failed)}"
        assert successful[0]["client_id"] == "client_1"


class TestSQLiteSpecific:
    """SQLite-specific tests (persistence, multi-process)."""

    @pytest.mark.asyncio
    async def test_persistence_across_connections(self):
        """Test that data persists when connection is closed and reopened."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            db_path = tmp.name

        try:
            # Store client in first connection
            storage1 = create_oauth_storage("sqlite", db_path=db_path)
            client = RegisteredClient(
                client_id="persist_test",
                client_secret="secret123",
                client_name="Persistence Test",
                redirect_uris=["http://localhost/callback"],
                created_at=time.time(),
            )
            await storage1.store_client(client)

            # Store authorization code and access token
            await storage1.store_authorization_code(
                code="persist_code", client_id="persist_test", expires_in=3600
            )
            await storage1.store_access_token(
                token="persist_token", client_id="persist_test", expires_in=3600
            )

            await storage1.close()

            # Retrieve in second connection
            storage2 = create_oauth_storage("sqlite", db_path=db_path)
            retrieved_client = await storage2.get_client("persist_test")
            assert retrieved_client is not None
            assert retrieved_client.client_name == "Persistence Test"

            # Verify code persisted (but gets consumed on retrieval)
            retrieved_code = await storage2.get_authorization_code("persist_code")
            assert retrieved_code is not None
            assert retrieved_code["client_id"] == "persist_test"

            # Verify token persisted
            retrieved_token = await storage2.get_access_token("persist_token")
            assert retrieved_token is not None
            assert retrieved_token["client_id"] == "persist_test"

            # Cleanup
            await storage2.close()

        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    @pytest.mark.asyncio
    async def test_wal_mode_enabled(self):
        """Test that WAL mode is enabled for multi-process safety."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            db_path = tmp.name

        try:
            storage = create_oauth_storage("sqlite", db_path=db_path)

            # Force initialization by storing a client
            client = RegisteredClient(
                client_id="wal_test",
                client_secret="secret",
                redirect_uris=["http://localhost/callback"],
                created_at=time.time(),
            )
            await storage.store_client(client)

            # Check WAL mode is enabled
            cursor = await storage._execute("PRAGMA journal_mode")
            row = await cursor.fetchone()
            journal_mode = row[0].lower() if row else None

            assert journal_mode == "wal", f"Expected WAL mode, got {journal_mode}"

            await storage.close()

        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)
                # Clean up WAL files
                for suffix in ["-wal", "-shm"]:
                    wal_file = db_path + suffix
                    if os.path.exists(wal_file):
                        os.unlink(wal_file)


@pytest.mark.benchmark
class TestPerformance:
    """Performance benchmarks for token operations."""

    @pytest.mark.asyncio
    async def test_token_operations_performance(self, storage):
        """
        Test that token operations complete within <10ms requirement.

        Note: First operation may be slower due to connection initialization.
        We warm up the connection first, then measure actual performance.
        """
        # Warm up: ensure connection is initialized
        await storage.store_access_token("warmup_token", "client_1", expires_in=3600)
        await storage.get_access_token("warmup_token")

        # Actual performance test: Store token
        start = time.perf_counter()
        await storage.store_access_token(
            "perf_token", "client_1", expires_in=3600
        )
        store_time = (time.perf_counter() - start) * 1000

        # Get token
        start = time.perf_counter()
        result = await storage.get_access_token("perf_token")
        get_time = (time.perf_counter() - start) * 1000

        assert result is not None
        assert store_time < 10, f"Store took {store_time:.2f}ms (>10ms)"
        assert get_time < 10, f"Get took {get_time:.2f}ms (>10ms)"

    @pytest.mark.asyncio
    async def test_client_operations_performance(self, storage):
        """
        Test that client operations complete within <10ms requirement.

        Note: First operation may be slower due to connection initialization.
        We warm up the connection first, then measure actual performance.
        """
        # Warm up: ensure connection is initialized
        warmup_client = RegisteredClient(
            client_id="warmup_client",
            client_secret="warmup_secret",
            redirect_uris=["http://localhost/callback"],
            created_at=time.time(),
        )
        await storage.store_client(warmup_client)
        await storage.get_client("warmup_client")

        # Actual performance test
        client = RegisteredClient(
            client_id="perf_client",
            client_secret="secret",
            redirect_uris=["http://localhost/callback"],
            created_at=time.time(),
        )

        # Store client
        start = time.perf_counter()
        await storage.store_client(client)
        store_time = (time.perf_counter() - start) * 1000

        # Get client
        start = time.perf_counter()
        result = await storage.get_client("perf_client")
        get_time = (time.perf_counter() - start) * 1000

        # Authenticate client
        start = time.perf_counter()
        auth_result = await storage.authenticate_client("perf_client", "secret")
        auth_time = (time.perf_counter() - start) * 1000

        assert result is not None
        assert auth_result is True
        assert store_time < 10, f"Store took {store_time:.2f}ms (>10ms)"
        assert get_time < 10, f"Get took {get_time:.2f}ms (>10ms)"
        assert auth_time < 10, f"Auth took {auth_time:.2f}ms (>10ms)"
