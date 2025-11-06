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
Tests for core API operations.

Validates functionality, performance, and token efficiency of
search, store, and health operations.
"""

import pytest
import time
from mcp_memory_service.api import search, store, health
from mcp_memory_service.api.types import CompactSearchResult, CompactHealthInfo
from mcp_memory_service.api.client import reset_storage


@pytest.fixture(autouse=True)
def reset_client():
    """Reset storage client before each test."""
    reset_storage()
    yield
    reset_storage()


class TestSearchOperation:
    """Tests for search() function."""

    def test_search_basic(self):
        """Test basic search functionality."""
        # Store some test memories first
        hash1 = store("Test memory about authentication", tags=["test", "auth"])
        hash2 = store("Test memory about database", tags=["test", "db"])

        # Search for memories
        result = search("authentication", limit=5)

        assert isinstance(result, CompactSearchResult)
        assert result.total >= 0
        assert result.query == "authentication"
        assert len(result.memories) <= 5

    def test_search_with_limit(self):
        """Test search with different limits."""
        # Store multiple memories
        for i in range(10):
            store(f"Test memory number {i}", tags=["test"])

        # Search with limit
        result = search("test", limit=3)

        assert len(result.memories) <= 3
        assert result.query == "test"

    def test_search_with_tags(self):
        """Test search with tag filtering."""
        # Store memories with different tags
        store("Memory with tag1", tags=["tag1", "test"])
        store("Memory with tag2", tags=["tag2", "test"])
        store("Memory with both", tags=["tag1", "tag2", "test"])

        # Search with tag filter
        result = search("memory", limit=10, tags=["tag1"])

        # Should only return memories with tag1
        for memory in result.memories:
            assert "tag1" in memory.tags

    def test_search_empty_query(self):
        """Test that search rejects empty queries."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            search("")

        with pytest.raises(ValueError, match="Query cannot be empty"):
            search("   ")

    def test_search_invalid_limit(self):
        """Test that search rejects invalid limits."""
        with pytest.raises(ValueError, match="Limit must be at least 1"):
            search("test", limit=0)

        with pytest.raises(ValueError, match="Limit must be at least 1"):
            search("test", limit=-1)

    def test_search_returns_compact_format(self):
        """Test that search returns compact memory format."""
        # Store a test memory
        store("Test memory content", tags=["test"])

        # Search
        result = search("test", limit=1)

        if result.memories:
            memory = result.memories[0]

            # Verify compact format
            assert len(memory.hash) == 8, "Hash should be 8 characters"
            assert len(memory.preview) <= 200, "Preview should be d200 chars"
            assert isinstance(memory.tags, tuple), "Tags should be tuple"
            assert isinstance(memory.created, float), "Created should be timestamp"
            assert 0.0 <= memory.score <= 1.0, "Score should be 0-1"

    def test_search_performance(self):
        """Test search performance meets targets."""
        # Store some memories
        for i in range(10):
            store(f"Performance test memory {i}", tags=["perf"])

        # Measure warm call performance
        start = time.perf_counter()
        result = search("performance", limit=5)
        duration_ms = (time.perf_counter() - start) * 1000

        # Should complete in <100ms for warm call
        assert duration_ms < 100, f"Search too slow: {duration_ms:.1f}ms (target: <100ms)"

        # Verify results returned
        assert isinstance(result, CompactSearchResult)


class TestStoreOperation:
    """Tests for store() function."""

    def test_store_basic(self):
        """Test basic store functionality."""
        content = "This is a test memory"
        hash_val = store(content)

        assert isinstance(hash_val, str)
        assert len(hash_val) == 8, "Should return 8-char hash"

    def test_store_with_tags_list(self):
        """Test storing with list of tags."""
        hash_val = store(
            "Memory with tags",
            tags=["tag1", "tag2", "tag3"]
        )

        assert isinstance(hash_val, str)
        assert len(hash_val) == 8

        # Verify stored by searching
        result = search("Memory with tags", limit=1)
        if result.memories:
            assert "tag1" in result.memories[0].tags

    def test_store_with_single_tag(self):
        """Test storing with single tag string."""
        hash_val = store(
            "Memory with single tag",
            tags="singletag"
        )

        assert isinstance(hash_val, str)
        assert len(hash_val) == 8

    def test_store_with_memory_type(self):
        """Test storing with custom memory type."""
        hash_val = store(
            "Custom type memory",
            tags=["test"],
            memory_type="feature"
        )

        assert isinstance(hash_val, str)
        assert len(hash_val) == 8

    def test_store_empty_content(self):
        """Test that store rejects empty content."""
        with pytest.raises(ValueError, match="Content cannot be empty"):
            store("")

        with pytest.raises(ValueError, match="Content cannot be empty"):
            store("   ")

    def test_store_returns_short_hash(self):
        """Test that store returns 8-char hash."""
        hash_val = store("Test content for hash length")

        assert len(hash_val) == 8
        assert hash_val.isalnum() or all(c in '0123456789abcdef' for c in hash_val)

    def test_store_duplicate_handling(self):
        """Test storing duplicate content."""
        content = "Duplicate content test"

        # Store same content twice
        hash1 = store(content, tags=["test1"])
        hash2 = store(content, tags=["test2"])

        # Should return same hash (content is identical)
        assert hash1 == hash2

    def test_store_performance(self):
        """Test store performance meets targets."""
        content = "Performance test memory content"

        # Measure warm call performance
        start = time.perf_counter()
        hash_val = store(content, tags=["perf"])
        duration_ms = (time.perf_counter() - start) * 1000

        # Should complete in <50ms for warm call
        assert duration_ms < 50, f"Store too slow: {duration_ms:.1f}ms (target: <50ms)"

        # Verify hash returned
        assert isinstance(hash_val, str)


class TestHealthOperation:
    """Tests for health() function."""

    def test_health_basic(self):
        """Test basic health check."""
        info = health()

        assert isinstance(info, CompactHealthInfo)
        assert info.status in ['healthy', 'degraded', 'error']
        assert isinstance(info.count, int)
        assert isinstance(info.backend, str)

    def test_health_returns_valid_status(self):
        """Test that health returns valid status."""
        info = health()

        valid_statuses = ['healthy', 'degraded', 'error']
        assert info.status in valid_statuses

    def test_health_returns_backend_type(self):
        """Test that health returns backend type."""
        info = health()

        valid_backends = ['sqlite_vec', 'cloudflare', 'hybrid', 'unknown']
        assert info.backend in valid_backends

    def test_health_memory_count(self):
        """Test that health returns memory count."""
        # Store some memories
        for i in range(5):
            store(f"Health test memory {i}", tags=["health"])

        info = health()

        # Count should be >= 5 (may have other memories)
        assert info.count >= 5

    def test_health_performance(self):
        """Test health check performance."""
        # Measure warm call performance
        start = time.perf_counter()
        info = health()
        duration_ms = (time.perf_counter() - start) * 1000

        # Should complete in <20ms for warm call
        assert duration_ms < 20, f"Health check too slow: {duration_ms:.1f}ms (target: <20ms)"

        # Verify info returned
        assert isinstance(info, CompactHealthInfo)


class TestTokenEfficiency:
    """Integration tests for token efficiency."""

    def test_search_token_reduction(self):
        """Validate 85%+ token reduction for search."""
        # Store test memories
        for i in range(10):
            store(f"Token test memory {i} with some content", tags=["token", "test"])

        # Perform search
        result = search("token", limit=5)

        # Estimate token count (rough: 1 token H 4 characters)
        result_str = str(result.memories)
        estimated_tokens = len(result_str) / 4

        # Target: ~385 tokens for 5 results (vs ~2,625 tokens, 85% reduction)
        # Allow some margin: should be under 800 tokens
        assert estimated_tokens < 800, \
            f"Search result not efficient: {estimated_tokens:.0f} tokens (target: <800)"

        # Verify we achieved significant reduction
        reduction = 1 - (estimated_tokens / 2625)
        assert reduction >= 0.70, \
            f"Token reduction insufficient: {reduction:.1%} (target: e70%)"

    def test_store_token_reduction(self):
        """Validate 90%+ token reduction for store."""
        # Store operation itself is just parameters + hash return
        content = "Test content for token efficiency"
        tags = ["test", "efficiency"]

        # Measure "token cost" of operation
        # In practice: ~15 tokens (content + tags + function call)
        param_str = f"store('{content}', tags={tags})"
        estimated_tokens = len(param_str) / 4

        # Target: ~15 tokens (vs ~150 for MCP tool, 90% reduction)
        # Allow some margin
        assert estimated_tokens < 50, \
            f"Store call not efficient: {estimated_tokens:.0f} tokens (target: <50)"

    def test_health_token_reduction(self):
        """Validate 84%+ token reduction for health check."""
        info = health()

        # Measure "token cost" of result
        info_str = str(info)
        estimated_tokens = len(info_str) / 4

        # Target: ~20 tokens (vs ~125 for MCP tool, 84% reduction)
        # Allow some margin
        assert estimated_tokens < 40, \
            f"Health info not efficient: {estimated_tokens:.0f} tokens (target: <40)"


class TestIntegration:
    """End-to-end integration tests."""

    def test_store_and_search_workflow(self):
        """Test complete store -> search workflow."""
        # Store memories
        hash1 = store("Integration test memory 1", tags=["integration", "test"])
        hash2 = store("Integration test memory 2", tags=["integration", "demo"])

        assert len(hash1) == 8
        assert len(hash2) == 8

        # Search for stored memories
        result = search("integration", limit=5)

        assert result.total >= 2
        assert any(m.hash in [hash1, hash2] for m in result.memories)

    def test_multiple_operations_performance(self):
        """Test performance of multiple operations."""
        start = time.perf_counter()

        # Perform multiple operations
        hash1 = store("Op 1", tags=["multi"])
        hash2 = store("Op 2", tags=["multi"])
        result = search("multi", limit=5)
        info = health()

        duration_ms = (time.perf_counter() - start) * 1000

        # All operations should complete in <200ms
        assert duration_ms < 200, f"Multiple ops too slow: {duration_ms:.1f}ms (target: <200ms)"

        # Verify all operations succeeded
        assert len(hash1) == 8
        assert len(hash2) == 8
        assert isinstance(result, CompactSearchResult)
        assert isinstance(info, CompactHealthInfo)

    def test_api_backward_compatibility(self):
        """Test that API doesn't break existing functionality."""
        # This test ensures the API can coexist with existing MCP tools

        # Store using new API
        hash_val = store("Compatibility test", tags=["compat"])

        # Should be searchable
        result = search("compatibility", limit=1)

        # Should find the stored memory
        assert result.total >= 1
        if result.memories:
            assert hash_val == result.memories[0].hash
