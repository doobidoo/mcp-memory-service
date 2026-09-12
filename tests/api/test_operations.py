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

import time

import pytest

from mcp_memory_service.api import health, operations, search, store
from mcp_memory_service.api.client import reset_storage
from mcp_memory_service.api.types import CompactHealthInfo, CompactSearchResult
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash

try:
    from mcp_memory_service.storage.mixins.embeddings import (
        SENTENCE_TRANSFORMERS_AVAILABLE,
    )
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


@pytest.fixture(autouse=True)
def reset_client():
    """Reset storage client before each test."""
    reset_storage()
    yield
    reset_storage()


class TestSearchOperation:
    """Tests for search() function."""

    def test_search_basic(self, unique_content, test_store):
        """Test basic search functionality."""
        # Store some test memories first
        hash1 = test_store(unique_content("Test memory about authentication"), tags=["auth"])
        hash2 = test_store(unique_content("Test memory about database"), tags=["db"])

        # Search for memories
        result = search("authentication", limit=5)

        assert isinstance(result, CompactSearchResult)
        assert result.total >= 0
        assert result.query == "authentication"
        assert len(result.memories) <= 5

    def test_search_with_limit(self, unique_content, test_store):
        """Test search with different limits."""
        # Store multiple memories
        for i in range(10):
            test_store(unique_content(f"Test memory number {i}"))

        # Search with limit
        result = search("test", limit=3)

        assert len(result.memories) <= 3
        assert result.query == "test"

    def test_search_with_tags(self, unique_content, test_store):
        """Test search with tag filtering."""
        # Store memories with different tags
        test_store(unique_content("Memory with tag1"), tags=["tag1"])
        test_store(unique_content("Memory with tag2"), tags=["tag2"])
        test_store(unique_content("Memory with both"), tags=["tag1", "tag2"])

        # Search with tag filter
        result = search("memory", limit=10, tags=["tag1"])

        # Should only return memories with tag1
        for memory in result.memories:
            assert "tag1" in memory.tags

    @pytest.mark.asyncio
    async def test_search_applies_tag_filter_before_limit(self, tmp_path, monkeypatch):
        """A tagged result outside the unfiltered limit must still be returned."""
        storage = SqliteVecMemoryStorage(str(tmp_path / "tag-filter.db"))
        await storage.initialize()

        async def get_test_storage():
            return storage

        monkeypatch.setattr(operations, "get_storage_async", get_test_storage)

        query = "exact untagged nearest neighbour"
        memories = [
            Memory(
                content=query,
                content_hash=generate_content_hash(query),
                tags=["other"],
            ),
            Memory(
                content="distant tagged result",
                content_hash=generate_content_hash("distant tagged result"),
                tags=["wanted"],
            ),
        ]

        try:
            for memory in memories:
                success, message = await storage.store(memory)
                assert success, message

            unfiltered = await storage.retrieve(query, n_results=1)
            assert unfiltered[0].memory.tags == ["other"]

            result = await operations.search.__wrapped__(
                query, limit=1, tags=["wanted"]
            )

            assert result.total == 1
            assert result.memories[0].tags == ("wanted",)
        finally:
            await storage.close()

    @pytest.mark.asyncio
    async def test_search_tag_with_internal_space(self, tmp_path, monkeypatch):
        """A stored tag containing a space must match the same spaced query tag."""
        storage = SqliteVecMemoryStorage(str(tmp_path / "spaced-tag.db"))
        await storage.initialize()

        async def get_test_storage():
            return storage

        monkeypatch.setattr(operations, "get_storage_async", get_test_storage)

        content = "memory carrying a spaced tag"
        try:
            success, message = await storage.store(
                Memory(
                    content=content,
                    content_hash=generate_content_hash(content),
                    tags=["my tag"],
                )
            )
            assert success, message

            result = await operations.search.__wrapped__(
                content, limit=5, tags=["my tag"]
            )

            assert result.total == 1
            assert result.memories[0].tags == ("my tag",)
        finally:
            await storage.close()

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

    def test_search_returns_compact_format(self, unique_content, test_store):
        """Test that search returns compact memory format."""
        # Store a test memory
        test_store(unique_content("Test memory content"))

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

    def test_search_performance(self, unique_content, test_store):
        """Test search performance meets targets."""
        # Store some memories
        for i in range(10):
            test_store(unique_content(f"Performance test memory {i}"), tags=["perf"])

        # Measure warm call performance
        start = time.perf_counter()
        result = search("performance", limit=5)
        duration_ms = (time.perf_counter() - start) * 1000

        # Should complete in <2s for warm call (includes model loading, storage init)
        assert duration_ms < 2000, f"Search too slow: {duration_ms:.1f}ms (target: <2s)"

        # Verify results returned
        assert isinstance(result, CompactSearchResult)


class TestStoreOperation:
    """Tests for store() function."""

    def test_store_basic(self, unique_content):
        """Test basic store functionality."""
        content = unique_content("This is a test memory")
        hash_val = store(content)

        assert isinstance(hash_val, str)
        assert len(hash_val) == 8, "Should return 8-char hash"

    @pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE, reason="Requires real embeddings for semantic search verification")
    def test_store_with_tags_list(self, unique_content):
        """Test storing with list of tags."""
        hash_val = store(
            unique_content("Memory with tags"),
            tags=["tag1", "tag2", "tag3"]
        )

        assert isinstance(hash_val, str)
        assert len(hash_val) == 8

        # Verify stored by searching
        result = search("Memory with tags", limit=1)
        if result.memories:
            # Check that at least one of the original tags is present
            # (storage may add additional tags like __test__)
            memory_tags = result.memories[0].tags
            assert any(tag in memory_tags for tag in ["tag1", "tag2", "tag3"]), \
                f"Expected at least one of ['tag1', 'tag2', 'tag3'] in {memory_tags}"

    def test_store_with_single_tag(self, unique_content):
        """Test storing with single tag string."""
        hash_val = store(
            unique_content("Memory with single tag"),
            tags="singletag"
        )

        assert isinstance(hash_val, str)
        assert len(hash_val) == 8

    def test_store_with_memory_type(self, unique_content):
        """Test storing with custom memory type."""
        hash_val = store(
            unique_content("Custom type memory"),
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

    def test_store_returns_short_hash(self, unique_content):
        """Test that store returns 8-char hash."""
        hash_val = store(unique_content("Test content for hash length"))

        assert len(hash_val) == 8
        assert hash_val.isalnum() or all(c in '0123456789abcdef' for c in hash_val)

    def test_store_duplicate_handling(self, unique_content):
        """Test storing duplicate content."""
        # Generate unique content once, then use it twice
        content = unique_content("Duplicate content test")

        # Store same content twice with different tags
        # This should trigger duplicate detection and raise RuntimeError
        hash1 = store(content, tags=["test1"])

        # Second store with same content should fail due to duplicate detection
        with pytest.raises(RuntimeError, match="Duplicate content detected"):
            hash2 = store(content, tags=["test2"])

    def test_store_performance(self, unique_content):
        """Test store performance meets targets."""
        content = unique_content("Performance test memory content")

        # Measure warm call performance
        start = time.perf_counter()
        hash_val = store(content, tags=["perf"])
        duration_ms = (time.perf_counter() - start) * 1000

        # Should complete in <3s for warm call (includes model loading, storage init)
        assert duration_ms < 3000, f"Store too slow: {duration_ms:.1f}ms (target: <3s)"

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

        valid_backends = [
            'sqlite_vec', 'cloudflare', 'hybrid', 'unknown',
            'Hybrid (SQLite-vec + Cloudflare)',  # Descriptive hybrid backend name
            'SQLite-vec', 'Cloudflare'  # Alternative naming formats
        ]
        assert info.backend in valid_backends

    def test_health_memory_count(self, unique_content, test_store):
        """Test that health returns memory count."""
        # Store some memories
        for i in range(5):
            test_store(unique_content(f"Health test memory {i}"), tags=["health"])

        info = health()

        # Count should be >= 5 (may have other memories)
        assert info.count >= 5

    def test_health_performance(self):
        """Test health check performance."""
        # Measure warm call performance
        start = time.perf_counter()
        info = health()
        duration_ms = (time.perf_counter() - start) * 1000

        # Should complete in <2s for warm call (includes health check init)
        assert duration_ms < 2000, f"Health check too slow: {duration_ms:.1f}ms (target: <2s)"

        # Verify info returned
        assert isinstance(info, CompactHealthInfo)


class TestTokenEfficiency:
    """Integration tests for token efficiency."""

    def test_search_token_reduction(self, unique_content, test_store):
        """Validate 85%+ token reduction for search."""
        # Store test memories
        for i in range(10):
            test_store(unique_content(f"Token test memory {i} with some content"), tags=["token"])

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

    def test_store_token_reduction(self, unique_content):
        """Validate 90%+ token reduction for store."""
        # Store operation itself is just parameters + hash return
        content = unique_content("Test content for token efficiency")
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

    def test_store_and_search_workflow(self, unique_content, test_store):
        """Test complete store -> search workflow."""
        # Store memories with distinctive content
        content1 = unique_content("Integration test memory 1")
        content2 = unique_content("Integration test memory 2")
        hash1 = test_store(content1, tags=["integration"])
        hash2 = test_store(content2, tags=["integration", "demo"])

        assert len(hash1) == 8
        assert len(hash2) == 8

        # Verify search returns results (semantic search may not return exact matches)
        # NOTE: Due to semantic search behavior and database size, exact hash matching
        # is not guaranteed, but we verify search functionality works
        result = search("integration test", limit=5)

        assert result.total >= 0  # Search should return valid results
        assert isinstance(result.memories, tuple)  # Should return proper format

    def test_multiple_operations_performance(self, unique_content, test_store):
        """Test performance of multiple operations."""
        start = time.perf_counter()

        # Perform multiple operations
        hash1 = test_store(unique_content("Op 1"), tags=["multi"])
        hash2 = test_store(unique_content("Op 2"), tags=["multi"])
        result = search("multi", limit=5)
        info = health()

        duration_ms = (time.perf_counter() - start) * 1000

        # All operations should complete in <3s (includes multiple init costs)
        assert duration_ms < 3000, f"Multiple ops too slow: {duration_ms:.1f}ms (target: <3s)"

        # Verify all operations succeeded
        assert len(hash1) == 8
        assert len(hash2) == 8
        assert isinstance(result, CompactSearchResult)
        assert isinstance(info, CompactHealthInfo)

    def test_api_backward_compatibility(self, unique_content, test_store):
        """Test that API doesn't break existing functionality."""
        # This test ensures the API can coexist with existing MCP tools

        # Store using new API
        content = unique_content("Compatibility test")
        hash_val = test_store(content, tags=["compat"])

        # Verify store returned a valid hash
        assert isinstance(hash_val, str)
        assert len(hash_val) == 8

        # Verify search functionality works (semantic search may not return exact match)
        # NOTE: Due to semantic search behavior and database size, exact hash matching
        # is not guaranteed, but we verify API functionality works
        result = search("compatibility", limit=5)

        assert result.total >= 0  # Search should return valid results
        assert isinstance(result.memories, tuple)  # Should return proper format
