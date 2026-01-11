"""
Tests for response size limiter functionality.

Tests the response_limiter module which prevents context window overflow
by truncating large memory retrieval responses at memory boundaries.
"""

import pytest
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

from mcp_memory_service.server.utils.response_limiter import (
    truncate_memories,
    format_truncated_response,
    apply_response_limit,
    safe_retrieve_response,
    DEFAULT_MAX_CHARS,
    MEMORY_OVERHEAD_CHARS,
)


# ============================================
# Test Fixtures
# ============================================


@pytest.fixture
def sample_memory() -> Dict[str, Any]:
    """Single sample memory for testing."""
    return {
        "content": "This is test content for memory testing.",
        "content_hash": "abc123def456",
        "created_at": "2026-01-10T12:00:00Z",
        "tags": ["test", "sample"],
        "relevance_score": 0.85,
    }


@pytest.fixture
def large_memories() -> List[Dict[str, Any]]:
    """List of memories that exceeds typical limits."""
    return [
        {
            "content": f"Memory content {i} " + "x" * 5000,
            "content_hash": f"hash_{i}",
            "created_at": f"2026-01-10T{i:02d}:00:00Z",
            "tags": [f"tag_{i}"],
            "relevance_score": 0.9 - (i * 0.1),
        }
        for i in range(20)
    ]


@pytest.fixture
def small_memories() -> List[Dict[str, Any]]:
    """List of small memories that fit within limits."""
    return [
        {
            "content": f"Small content {i}",
            "content_hash": f"hash_{i}",
        }
        for i in range(5)
    ]


# ============================================
# truncate_memories() Tests
# ============================================


class TestTruncateMemories:
    """Tests for truncate_memories function."""

    def test_empty_list_returns_empty(self):
        """Empty input should return empty list with zero metadata."""
        result, meta = truncate_memories([])

        assert result == []
        assert meta["total_results"] == 0
        assert meta["shown_results"] == 0
        assert meta["truncated"] is False
        assert meta["omitted_count"] == 0

    def test_no_limit_returns_all(self, small_memories):
        """With no limit (0), all memories should be returned."""
        result, meta = truncate_memories(small_memories, max_chars=0)

        assert len(result) == len(small_memories)
        assert meta["truncated"] is False
        assert meta["omitted_count"] == 0

    def test_under_limit_returns_all(self, small_memories):
        """Memories under the limit should all be returned."""
        result, meta = truncate_memories(small_memories, max_chars=100000)

        assert len(result) == len(small_memories)
        assert meta["truncated"] is False

    def test_over_limit_truncates(self, large_memories):
        """Memories exceeding limit should be truncated."""
        result, meta = truncate_memories(large_memories, max_chars=10000)

        assert len(result) < len(large_memories)
        assert meta["truncated"] is True
        assert meta["omitted_count"] > 0
        assert meta["total_results"] == len(large_memories)

    def test_always_returns_at_least_one(self, large_memories):
        """Even with tiny limit, at least one memory should be returned."""
        result, meta = truncate_memories(large_memories, max_chars=1)

        assert len(result) >= 1
        assert meta["shown_results"] >= 1

    def test_truncates_at_memory_boundaries(self, large_memories):
        """Truncation should occur at memory boundaries, not mid-content."""
        result, meta = truncate_memories(large_memories, max_chars=10000)

        # Each returned memory should have complete content
        for memory in result:
            assert "content" in memory
            # Content should not be cut off (original content intact)
            original = next(
                m for m in large_memories 
                if m["content_hash"] == memory["content_hash"]
            )
            assert memory["content"] == original["content"]

    def test_metadata_accuracy(self, large_memories):
        """Metadata should accurately reflect truncation state."""
        result, meta = truncate_memories(large_memories, max_chars=10000)

        assert meta["total_results"] == len(large_memories)
        assert meta["shown_results"] == len(result)
        assert meta["omitted_count"] == meta["total_results"] - meta["shown_results"]
        # shown_chars may exceed max_chars when at least one memory is returned
        assert meta["shown_results"] == 1 or meta["shown_chars"] <= 10000

    def test_preserves_memory_order(self, large_memories):
        """Truncation should preserve the original order of memories."""
        result, meta = truncate_memories(large_memories, max_chars=20000)

        # First few memories should match original order
        for i, memory in enumerate(result):
            assert memory["content_hash"] == large_memories[i]["content_hash"]


# ============================================
# format_truncated_response() Tests
# ============================================


class TestFormatTruncatedResponse:
    """Tests for format_truncated_response function."""

    def test_formats_single_memory(self, sample_memory):
        """Single memory should be formatted correctly."""
        meta = {"truncated": False, "total_results": 1, "shown_results": 1}
        result = format_truncated_response([sample_memory], meta)

        assert "=== Memory 1 ===" in result
        assert sample_memory["content"] in result
        assert sample_memory["content_hash"] in result

    def test_includes_timestamp_when_present(self, sample_memory):
        """Timestamp should be included when available."""
        meta = {"truncated": False}
        result = format_truncated_response([sample_memory], meta)

        assert "Timestamp:" in result
        assert sample_memory["created_at"] in result

    def test_includes_tags_when_present(self, sample_memory):
        """Tags should be included when available."""
        meta = {"truncated": False}
        result = format_truncated_response([sample_memory], meta)

        assert "Tags:" in result
        assert "test" in result
        assert "sample" in result

    def test_includes_relevance_score(self, sample_memory):
        """Relevance score should be formatted when present."""
        meta = {"truncated": False}
        result = format_truncated_response([sample_memory], meta)

        assert "Relevance Score:" in result
        assert "0.85" in result

    def test_truncation_warning_header(self, small_memories):
        """Truncation warning should appear when truncated."""
        meta = {
            "truncated": True,
            "total_results": 10,
            "shown_results": 5,
            "total_chars": 10000,
            "shown_chars": 5000,
            "omitted_count": 5,
        }
        result = format_truncated_response(small_memories, meta)

        assert "[!] RESPONSE TRUNCATED" in result
        assert "5 of 10" in result
        assert "omitted" in result.lower()

    def test_no_warning_when_not_truncated(self, small_memories):
        """No truncation warning when not truncated."""
        meta = {"truncated": False}
        result = format_truncated_response(small_memories, meta)

        assert "TRUNCATED" not in result

    def test_handles_missing_fields_gracefully(self):
        """Should handle memories with missing optional fields."""
        minimal_memory = {"content": "minimal", "content_hash": "hash1"}
        meta = {"truncated": False}

        result = format_truncated_response([minimal_memory], meta)

        assert "minimal" in result
        assert "hash1" in result
        # Should not raise KeyError

    def test_handles_string_tags(self):
        """Should handle tags as string instead of list."""
        memory = {
            "content": "test",
            "content_hash": "hash1",
            "tags": "single-tag",
        }
        meta = {"truncated": False}

        result = format_truncated_response([memory], meta)

        assert "Tags: single-tag" in result


# ============================================
# apply_response_limit() Tests
# ============================================


class TestApplyResponseLimit:
    """Tests for apply_response_limit convenience function."""

    def test_combines_truncate_and_format(self, small_memories):
        """Should truncate and format in one call."""
        result = apply_response_limit(small_memories, max_chars=100000)

        assert isinstance(result, str)
        assert "=== Memory 1 ===" in result

    def test_respects_max_chars(self, large_memories):
        """Should respect the max_chars limit."""
        result = apply_response_limit(large_memories, max_chars=5000)

        # Result should be roughly within limit
        assert len(result) < 10000  # Some buffer for formatting

    def test_uses_env_default_when_zero(self, small_memories):
        """Should use environment default when max_chars is 0.
        
        Note: DEFAULT_MAX_CHARS is read at module import time, so this test
        verifies the function logic path rather than dynamic env var reading.
        """
        result = apply_response_limit(small_memories, max_chars=0)
        assert isinstance(result, str)


# ============================================
# safe_retrieve_response() Tests
# ============================================


class TestSafeRetrieveResponse:
    """Tests for safe_retrieve_response convenience function."""

    def test_default_limit_is_40000(self, small_memories):
        """Default limit should be 40000 characters."""
        result = safe_retrieve_response(small_memories)

        assert isinstance(result, str)
        # Small memories should all fit
        assert "=== Memory 1 ===" in result

    def test_custom_limit_respected(self, large_memories):
        """Custom limit should be respected."""
        result = safe_retrieve_response(large_memories, max_chars=5000)

        # Should be truncated
        assert "TRUNCATED" in result


# ============================================
# Integration Tests
# ============================================


class TestResponseLimiterIntegration:
    """Integration tests for full response limiting workflow."""

    def test_full_workflow_small_response(self, small_memories):
        """Full workflow with small response (no truncation)."""
        # Truncate
        truncated, meta = truncate_memories(small_memories, max_chars=100000)
        
        # Format
        response = format_truncated_response(truncated, meta)

        # Verify
        assert len(truncated) == len(small_memories)
        assert "TRUNCATED" not in response
        for i, memory in enumerate(small_memories, 1):
            assert f"=== Memory {i} ===" in response

    def test_full_workflow_large_response(self, large_memories):
        """Full workflow with large response (requires truncation)."""
        max_chars = 10000
        
        # Truncate
        truncated, meta = truncate_memories(large_memories, max_chars=max_chars)
        
        # Format
        response = format_truncated_response(truncated, meta)

        # Verify truncation occurred
        assert len(truncated) < len(large_memories)
        assert meta["truncated"] is True
        assert "TRUNCATED" in response
        
        # Verify guidance for user
        assert "specific queries" in response.lower() or "hash-based" in response.lower()

    def test_response_size_within_limit(self, large_memories):
        """Formatted response should be approximately within limit."""
        max_chars = 20000
        
        truncated, meta = truncate_memories(large_memories, max_chars=max_chars)
        response = format_truncated_response(truncated, meta)

        # Response should be in reasonable range of limit
        # (exact limit is on input content, not formatted output)
        assert len(response) < max_chars * 2  # Allow for formatting overhead


# ============================================
# Edge Case Tests
# ============================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_large_memory(self):
        """Single memory larger than limit should still be returned."""
        huge_memory = {
            "content": "x" * 100000,
            "content_hash": "huge_hash",
        }
        
        result, meta = truncate_memories([huge_memory], max_chars=1000)

        assert len(result) == 1
        assert meta["shown_results"] == 1

    def test_empty_content_memories(self):
        """Memories with empty content should be handled."""
        memories = [
            {"content": "", "content_hash": "empty1"},
            {"content": "", "content_hash": "empty2"},
        ]
        
        result, meta = truncate_memories(memories, max_chars=1000)
        response = format_truncated_response(result, meta)

        assert len(result) == 2
        assert "empty1" in response

    def test_none_content_handling(self):
        """Memories with None content should be handled gracefully."""
        memories = [
            {"content": None, "content_hash": "none1"},
        ]
        
        result, meta = truncate_memories(memories, max_chars=1000)
        
        # Should not raise
        assert len(result) == 1

    def test_unicode_content(self):
        """Unicode content should be handled correctly."""
        memories = [
            {"content": "日本語テスト 🎉 émojis", "content_hash": "unicode1"},
        ]
        
        result, meta = truncate_memories(memories, max_chars=1000)
        response = format_truncated_response(result, meta)

        assert "日本語" in response
        assert "🎉" in response

    def test_special_characters_in_content(self):
        """Special characters should not break formatting."""
        memories = [
            {
                "content": "Test with <html> & \"quotes\" and 'apostrophes'",
                "content_hash": "special1",
            },
        ]
        
        result, meta = truncate_memories(memories, max_chars=1000)
        response = format_truncated_response(result, meta)

        assert "<html>" in response
        assert "&" in response
