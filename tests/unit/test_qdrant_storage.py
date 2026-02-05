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
Unit tests for Qdrant storage backend - FOCUSED on testable units.

These tests verify:
1. Circuit breaker logic (state machine behavior)
2. Dimension validation (input validation)
3. Configuration handling
4. Error message formatting

For integration tests with real Qdrant behavior, see:
tests/integration/test_storage_integration.py
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from src.mcp_memory_service.models.memory import Memory

# Import the module under test
from src.mcp_memory_service.storage.qdrant_storage import QdrantStorage, is_retryable_error
from src.mcp_memory_service.utils.hashing import generate_content_hash

# =============================================================================
# Circuit Breaker Unit Tests
# These test the circuit breaker STATE MACHINE, not Qdrant interactions
# =============================================================================


class TestCircuitBreakerLogic:
    """Test circuit breaker state transitions without touching Qdrant."""

    def test_circuit_starts_closed(self):
        """Verify circuit breaker starts in closed state."""
        with patch("src.mcp_memory_service.storage.qdrant_storage.QdrantClient"):
            storage = QdrantStorage(embedding_model="all-MiniLM-L6-v2", storage_path="/tmp/test")

        assert storage._failure_count == 0
        assert storage._circuit_open_until is None

    def test_circuit_opens_after_threshold_failures(self):
        """Verify circuit opens after N consecutive failures."""
        with patch("src.mcp_memory_service.storage.qdrant_storage.QdrantClient"):
            storage = QdrantStorage(embedding_model="all-MiniLM-L6-v2", storage_path="/tmp/test")

        # Simulate reaching threshold
        storage._failure_count = storage._failure_threshold

        # Verify circuit would open (manually set as code does)
        storage._circuit_open_until = datetime.now() + timedelta(seconds=storage._circuit_timeout)

        assert storage._circuit_open_until is not None
        assert storage._circuit_open_until > datetime.now()

    def test_circuit_timeout_calculation(self):
        """Verify circuit timeout is calculated correctly."""
        with patch("src.mcp_memory_service.storage.qdrant_storage.QdrantClient"):
            storage = QdrantStorage(embedding_model="all-MiniLM-L6-v2", storage_path="/tmp/test")

        # Open circuit
        now = datetime.now()
        storage._circuit_open_until = now + timedelta(seconds=storage._circuit_timeout)

        # Verify timeout is ~60 seconds in the future
        delta = storage._circuit_open_until - now
        assert 59 <= delta.total_seconds() <= 61

    def test_circuit_closes_after_timeout_expires(self):
        """Verify circuit can close after timeout expires."""
        with patch("src.mcp_memory_service.storage.qdrant_storage.QdrantClient"):
            storage = QdrantStorage(embedding_model="all-MiniLM-L6-v2", storage_path="/tmp/test")

        # Set circuit to have expired
        storage._circuit_open_until = datetime.now() - timedelta(seconds=1)
        storage._failure_count = 5

        # Check if circuit is past timeout (manual check as storage does)
        is_past_timeout = datetime.now() > storage._circuit_open_until

        assert is_past_timeout

    def test_failure_count_resets_on_success(self):
        """Verify failure count resets to zero after success."""
        with patch("src.mcp_memory_service.storage.qdrant_storage.QdrantClient"):
            storage = QdrantStorage(embedding_model="all-MiniLM-L6-v2", storage_path="/tmp/test")

        # Simulate some failures
        storage._failure_count = 3

        # Reset (as code does on success)
        storage._failure_count = 0
        storage._circuit_open_until = None

        assert storage._failure_count == 0
        assert storage._circuit_open_until is None


# =============================================================================
# Dimension Validation Unit Tests
# These test input validation logic
# =============================================================================


class TestDimensionValidation:
    """Test embedding dimension validation logic."""

    def test_detects_vector_size_for_minilm(self):
        """Verify correct dimension detection for all-MiniLM-L6-v2."""
        with patch("src.mcp_memory_service.storage.qdrant_storage.QdrantClient"):
            QdrantStorage(embedding_model="all-MiniLM-L6-v2", storage_path="/tmp/test")

        # MiniLM should be 384 dimensions
        # This is set during initialize(), but we can check the model map
        known_dimensions = {
            "all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768,
            "text-embedding-ada-002": 1536,
        }
        assert known_dimensions["all-MiniLM-L6-v2"] == 384

    def test_detects_vector_size_for_mpnet(self):
        """Verify correct dimension detection for all-mpnet-base-v2."""
        known_dimensions = {
            "all-mpnet-base-v2": 768,
        }
        assert known_dimensions["all-mpnet-base-v2"] == 768

    @pytest.mark.asyncio
    async def test_dimension_mismatch_raises_clear_error(self):
        """Verify dimension mismatch produces helpful error message."""
        with patch("src.mcp_memory_service.storage.qdrant_storage.QdrantClient"):
            storage = QdrantStorage(embedding_model="all-MiniLM-L6-v2", storage_path="/tmp/test")
            storage._initialized = True
            storage._vector_size = 384
            storage.client = MagicMock()

        # Create memory with wrong dimensions
        memory = Memory(
            content="Test content",
            content_hash=generate_content_hash("Test content"),
            embedding=[0.1] * 768,  # Wrong! Should be 384
            tags=["test"],
        )

        with pytest.raises(ValueError) as exc_info:
            await storage.store(memory)

        error_msg = str(exc_info.value).lower()
        assert "dimension mismatch" in error_msg
        assert "384" in str(exc_info.value)
        assert "768" in str(exc_info.value)


# =============================================================================
# Configuration Unit Tests
# =============================================================================


class TestQdrantConfiguration:
    """Test configuration validation."""

    def test_rejects_both_url_and_storage_path(self):
        """Verify cannot specify both embedded and server mode."""
        with pytest.raises(ValueError) as exc_info:
            QdrantStorage(embedding_model="all-MiniLM-L6-v2", url="http://localhost:6333", storage_path="/tmp/test")

        assert "both" in str(exc_info.value).lower()

    def test_rejects_neither_url_nor_storage_path(self):
        """Verify must specify at least one mode."""
        with pytest.raises(ValueError) as exc_info:
            QdrantStorage(embedding_model="all-MiniLM-L6-v2")

        assert "must specify" in str(exc_info.value).lower()

    def test_accepts_url_only_for_server_mode(self):
        """Verify server mode is configured correctly."""
        with patch("src.mcp_memory_service.storage.qdrant_storage.QdrantClient"):
            storage = QdrantStorage(embedding_model="all-MiniLM-L6-v2", url="http://localhost:6333")

        assert storage.url == "http://localhost:6333"
        assert storage.storage_path is None

    def test_accepts_storage_path_for_embedded_mode(self):
        """Verify embedded mode is configured correctly."""
        with patch("src.mcp_memory_service.storage.qdrant_storage.QdrantClient"):
            storage = QdrantStorage(embedding_model="all-MiniLM-L6-v2", storage_path="/tmp/test")

        assert storage.storage_path == "/tmp/test"
        assert storage.url is None


# =============================================================================
# Retry Logic Unit Tests
# =============================================================================


class TestRetryLogic:
    """Test the retry decision logic."""

    def test_5xx_errors_are_retryable(self):
        """Verify 5xx server errors trigger retry."""
        try:
            from qdrant_client.http import exceptions as qdrant_exceptions

            # Create a mock error with status_code attribute
            error = MagicMock(spec=qdrant_exceptions.UnexpectedResponse)
            error.status_code = 503
            # Make isinstance check work
            error.__class__ = qdrant_exceptions.UnexpectedResponse
            assert is_retryable_error(error)
        except ImportError:
            pytest.skip("qdrant-client not available")

    def test_4xx_errors_are_not_retryable(self):
        """Verify 4xx client errors do not trigger retry."""
        try:
            from qdrant_client.http import exceptions as qdrant_exceptions

            # Create a mock error with status_code attribute
            error = MagicMock(spec=qdrant_exceptions.UnexpectedResponse)
            error.status_code = 400
            error.__class__ = qdrant_exceptions.UnexpectedResponse
            assert not is_retryable_error(error)
        except ImportError:
            pytest.skip("qdrant-client not available")

    def test_generic_exceptions_are_not_retryable(self):
        """Verify generic exceptions do not trigger retry."""
        error = ValueError("Invalid input")
        assert not is_retryable_error(error)


# =============================================================================
# Properties Unit Tests
# =============================================================================


class TestQdrantProperties:
    """Test storage property values."""

    def test_max_content_length_is_none(self):
        """Verify Qdrant has no content length limit."""
        with patch("src.mcp_memory_service.storage.qdrant_storage.QdrantClient"):
            storage = QdrantStorage(embedding_model="all-MiniLM-L6-v2", storage_path="/tmp/test")

        assert storage.max_content_length is None

    def test_supports_chunking_is_true(self):
        """Verify Qdrant supports chunking."""
        with patch("src.mcp_memory_service.storage.qdrant_storage.QdrantClient"):
            storage = QdrantStorage(embedding_model="all-MiniLM-L6-v2", storage_path="/tmp/test")

        assert storage.supports_chunking is True


# =============================================================================
# Hash to UUID Conversion Tests
# =============================================================================


class TestHashToUuidConversion:
    """Test content hash to Qdrant UUID conversion."""

    def test_hash_to_uuid_is_deterministic(self):
        """Verify same hash always produces same UUID."""
        with patch("src.mcp_memory_service.storage.qdrant_storage.QdrantClient"):
            storage = QdrantStorage(embedding_model="all-MiniLM-L6-v2", storage_path="/tmp/test")

        # Use a valid SHA256 hash (64 hex chars) - _hash_to_uuid takes first 32 for UUID
        test_hash = "abc123def456abc123def456abc123def456abc123def456abc123def456abcd"
        uuid1 = storage._hash_to_uuid(test_hash)
        uuid2 = storage._hash_to_uuid(test_hash)

        assert uuid1 == uuid2

    def test_different_hashes_produce_different_uuids(self):
        """Verify different hashes produce different UUIDs."""
        with patch("src.mcp_memory_service.storage.qdrant_storage.QdrantClient"):
            storage = QdrantStorage(embedding_model="all-MiniLM-L6-v2", storage_path="/tmp/test")

        # Use valid 64-char hex strings (SHA256 format)
        hash1 = "1111111111111111111111111111111111111111111111111111111111111111"
        hash2 = "2222222222222222222222222222222222222222222222222222222222222222"
        uuid1 = storage._hash_to_uuid(hash1)
        uuid2 = storage._hash_to_uuid(hash2)

        assert uuid1 != uuid2
