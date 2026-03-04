"""Tests for content hash consistency (Issue #522).

Verifies that generate_content_hash produces identical hashes
for identical content regardless of how it's called.
"""
import pytest
from mcp_memory_service.utils.hashing import generate_content_hash


@pytest.mark.unit
class TestContentHashConsistency:
    """Verify content-only hashing policy."""

    def test_same_content_same_hash(self):
        """Identical content must always produce the same hash."""
        content = "Test memory about Python programming"
        hash1 = generate_content_hash(content)
        hash2 = generate_content_hash(content)
        assert hash1 == hash2

    def test_different_content_different_hash(self):
        """Different content must produce different hashes."""
        hash1 = generate_content_hash("Memory about cats")
        hash2 = generate_content_hash("Memory about dogs")
        assert hash1 != hash2

    def test_no_metadata_parameter(self):
        """generate_content_hash should reject unexpected arguments (Issue #522)."""
        with pytest.raises(TypeError):
            generate_content_hash("content", {"key": "value"})

    def test_hash_is_sha256_hex(self):
        """Hash should be a valid SHA-256 hex digest."""
        result = generate_content_hash("test")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_whitespace_normalization(self):
        """Leading/trailing whitespace should not affect hash."""
        hash1 = generate_content_hash("  hello world  ")
        hash2 = generate_content_hash("hello world")
        assert hash1 == hash2

    def test_case_normalization(self):
        """Hash should be case-insensitive."""
        hash1 = generate_content_hash("Hello World")
        hash2 = generate_content_hash("hello world")
        assert hash1 == hash2

    def test_internal_whitespace_preserved(self):
        """Internal whitespace differences should produce different hashes."""
        hash1 = generate_content_hash("hello  world")
        hash2 = generate_content_hash("hello world")
        assert hash1 != hash2
