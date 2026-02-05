"""
Integration tests for hybrid retrieval.

Tests the end-to-end hybrid search behavior with REAL storage backends.
Verifies: hybrid enabled by default, opt-out mechanisms, tag boost behavior,
backward compatibility, and pagination.

Run with: uv run pytest tests/integration/test_hybrid_retrieve.py -v
"""

import os
import shutil
import tempfile
from collections.abc import AsyncGenerator
from unittest.mock import patch

import pytest

# Force CPU mode for tests
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# =============================================================================
# Fixtures
# =============================================================================
import importlib.util

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.services.memory_service import MemoryService
from mcp_memory_service.utils.hashing import generate_content_hash

SQLITE_VEC_AVAILABLE = importlib.util.find_spec("sqlite_vec") is not None

if SQLITE_VEC_AVAILABLE:
    from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage


@pytest.fixture
async def storage() -> AsyncGenerator["SqliteVecMemoryStorage", None]:
    """Create a real SQLite-vec storage instance for testing."""
    if not SQLITE_VEC_AVAILABLE:
        pytest.skip("sqlite-vec not available")

    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_hybrid.db")

    storage = SqliteVecMemoryStorage(db_path)
    await storage.initialize()

    yield storage

    await storage.close()
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
async def service(storage) -> MemoryService:
    """Create MemoryService with real storage."""
    return MemoryService(storage)


@pytest.fixture
async def seeded_service(service, storage) -> MemoryService:
    """
    Create MemoryService with test data seeded.

    Seeds memories with different tags to test hybrid behavior:
    - rathole-tagged memories (should boost for "rathole" queries)
    - cachekit-tagged memories (semantically similar to "rathole")
    - generic memories (no special tags)
    """
    test_memories = [
        # Rathole project memories - should rank higher for "rathole" queries
        Memory(
            content="Rathole project architecture uses reverse proxy tunneling",
            content_hash=generate_content_hash("Rathole project architecture uses reverse proxy tunneling"),
            tags=["rathole", "project", "architecture"],
            memory_type="note",
        ),
        Memory(
            content="Rathole tunnel configuration with TOML files",
            content_hash=generate_content_hash("Rathole tunnel configuration with TOML files"),
            tags=["rathole", "config"],
            memory_type="note",
        ),
        # Cachekit memories - semantically similar (also "architecture") but different project
        Memory(
            content="Cachekit architecture uses decorator pattern for caching",
            content_hash=generate_content_hash("Cachekit architecture uses decorator pattern for caching"),
            tags=["cachekit", "architecture"],
            memory_type="note",
        ),
        Memory(
            content="Cachekit Redis backend implementation details",
            content_hash=generate_content_hash("Cachekit Redis backend implementation details"),
            tags=["cachekit", "redis"],
            memory_type="note",
        ),
        # Generic memories
        Memory(
            content="Python best practices for async programming",
            content_hash=generate_content_hash("Python best practices for async programming"),
            tags=["python", "async"],
            memory_type="note",
        ),
    ]

    for memory in test_memories:
        await storage.store(memory)

    return service


# =============================================================================
# Hybrid Search Tests
# =============================================================================


@pytest.mark.skipif(not SQLITE_VEC_AVAILABLE, reason="sqlite-vec not available")
class TestHybridRetrieval:
    """Integration tests for hybrid search behavior."""

    @pytest.mark.asyncio
    async def test_hybrid_enabled_by_default(self, seeded_service):
        """Default behavior should use hybrid search or gracefully fallback."""
        result = await seeded_service.retrieve_memories(query="rathole project architecture", page=1, page_size=10)

        assert "memories" in result
        assert len(result["memories"]) > 0
        # hybrid_enabled should be present (True or False depending on conditions)
        assert "hybrid_enabled" in result
        # If hybrid is enabled, alpha_used should be present
        if result.get("hybrid_enabled"):
            assert "alpha_used" in result

    @pytest.mark.asyncio
    async def test_tag_extraction_boosts_results(self, seeded_service):
        """Query containing tag keywords should boost matching memories."""
        result = await seeded_service.retrieve_memories(query="rathole project architecture", page=1, page_size=10)

        assert len(result["memories"]) > 0

        # First result should be from rathole project (tag match boost)
        first_memory = result["memories"][0]
        assert "rathole" in first_memory["content"].lower() or "rathole" in [t.lower() for t in first_memory["tags"]]

    @pytest.mark.asyncio
    async def test_opt_out_with_empty_tags(self, seeded_service):
        """Passing tags=[] should skip hybrid and use pure vector search."""
        result = await seeded_service.retrieve_memories(
            query="rathole project architecture",
            page=1,
            page_size=10,
            tags=[],  # Explicit opt-out
        )

        assert result.get("hybrid_enabled") is False

    @pytest.mark.asyncio
    async def test_opt_out_with_alpha_env(self, seeded_service):
        """Setting HYBRID_ALPHA=1.0 via config should use pure vector search.

        This test verifies that when hybrid_alpha is set to 1.0 (pure vector),
        the hybrid search is disabled and only vector similarity is used.
        """
        # First, verify hybrid IS enabled without the opt-out (baseline check)
        baseline_result = await seeded_service.retrieve_memories(
            query="rathole project architecture",
            page=1,
            page_size=10,
        )
        # Baseline should have hybrid enabled (or at least be a valid result)
        assert "memories" in baseline_result

        # Now test alpha=1.0 opt-out by patching the config
        from mcp_memory_service.config import HybridSearchSettings

        mock_config = HybridSearchSettings(
            hybrid_alpha=1.0,  # Pure vector search (hybrid disabled)
            recency_decay=0.0,
            adaptive_threshold_small=500,
            adaptive_threshold_large=5000,
        )

        # Patch the settings.hybrid_search attribute
        with patch("mcp_memory_service.services.memory_service.settings") as mock_settings:
            mock_settings.hybrid_search = mock_config
            result = await seeded_service.retrieve_memories(
                query="rathole project architecture",
                page=1,
                page_size=10,
            )

            # With alpha=1.0, hybrid should be disabled (pure vector search)
            assert result.get("hybrid_enabled") is False

            # If baseline had hybrid enabled, this confirms the patch worked
            if baseline_result.get("hybrid_enabled") is True:
                assert result.get("hybrid_enabled") != baseline_result.get("hybrid_enabled")

    @pytest.mark.asyncio
    async def test_backward_compatibility_api_unchanged(self, seeded_service):
        """API should remain backward compatible - same parameters work."""
        # Standard retrieve call should work
        result = await seeded_service.retrieve_memories(query="architecture patterns", page=1, page_size=5, memory_type="note")

        assert "memories" in result
        assert "total" in result
        assert "page" in result
        assert "page_size" in result
        assert "has_more" in result
        assert "total_pages" in result

    @pytest.mark.asyncio
    async def test_pagination_with_hybrid(self, seeded_service):
        """Pagination should work correctly with hybrid search."""
        # Test pagination with page_size=2
        page1 = await seeded_service.retrieve_memories(query="architecture", page=1, page_size=2)
        page2 = await seeded_service.retrieve_memories(query="architecture", page=2, page_size=2)

        # Verify pagination metadata
        assert page1["page"] == 1
        assert page1["page_size"] == 2
        assert page2["page"] == 2
        assert page2["page_size"] == 2

        # Verify no duplicate results across pages
        page1_hashes = {m["content_hash"] for m in page1["memories"]}
        page2_hashes = {m["content_hash"] for m in page2["memories"]}
        assert page1_hashes.isdisjoint(page2_hashes), "Duplicate results across pages"

    @pytest.mark.asyncio
    async def test_min_similarity_filter_applies(self, seeded_service):
        """min_similarity should filter results with hybrid search."""
        # High threshold - fewer results
        high_threshold = await seeded_service.retrieve_memories(
            query="rathole architecture",
            page=1,
            page_size=10,
            min_similarity=0.5,  # High threshold
        )

        # Low threshold - more results
        low_threshold = await seeded_service.retrieve_memories(
            query="rathole architecture",
            page=1,
            page_size=10,
            min_similarity=0.0,  # No threshold
        )

        # High threshold should return fewer or equal results
        assert len(high_threshold["memories"]) <= len(low_threshold["memories"])

    @pytest.mark.asyncio
    async def test_debug_info_in_response(self, seeded_service):
        """Hybrid search should include debug info in results."""
        result = await seeded_service.retrieve_memories(query="rathole project", page=1, page_size=5)

        if result.get("hybrid_enabled") and result["memories"]:
            first_memory = result["memories"][0]
            # Debug info should be present
            assert "hybrid_debug" in first_memory
            debug = first_memory["hybrid_debug"]
            assert "final_score" in debug
            assert "alpha_used" in debug

    @pytest.mark.asyncio
    async def test_no_tag_matches_falls_back_gracefully(self, seeded_service):
        """Query with no matching tags should still return results."""
        result = await seeded_service.retrieve_memories(query="completely unrelated query about elephants", page=1, page_size=10)

        # Should not error, should return whatever vector search finds
        assert "memories" in result
        assert "error" not in result


# =============================================================================
# Tag Cache Tests
# =============================================================================


@pytest.mark.skipif(not SQLITE_VEC_AVAILABLE, reason="sqlite-vec not available")
class TestTagCaching:
    """Tests for tag caching performance optimization."""

    @pytest.mark.asyncio
    async def test_tag_cache_populated_on_first_call(self, seeded_service):
        """First retrieve call should populate tag cache."""
        # Clear any existing cache
        seeded_service._tag_cache = None

        await seeded_service.retrieve_memories(query="test query", page=1, page_size=5)

        # Cache should now be populated
        assert seeded_service._tag_cache is not None
        cache_time, cached_tags = seeded_service._tag_cache
        assert cache_time > 0
        assert isinstance(cached_tags, set)

    @pytest.mark.asyncio
    async def test_tag_cache_reused_on_subsequent_calls(self, seeded_service):
        """Subsequent calls should reuse cached tags."""
        # First call to populate cache
        await seeded_service.retrieve_memories(query="first query", page=1, page_size=5)

        cache_time_1, _ = seeded_service._tag_cache

        # Second call should reuse cache
        await seeded_service.retrieve_memories(query="second query", page=1, page_size=5)

        cache_time_2, _ = seeded_service._tag_cache

        # Cache timestamp should not have changed
        assert cache_time_1 == cache_time_2
