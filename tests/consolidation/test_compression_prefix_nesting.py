"""
Tests for compression metadata prefix nesting prevention — GitHub issue #543.

Verifies that SemanticCompressionEngine._aggregate_metadata() does not produce
exponentially-nested keys (common_common_common_...) when compression is applied
repeatedly to already-compressed memories.
"""

import pytest
from datetime import datetime

from mcp_memory_service.consolidation.compression import SemanticCompressionEngine
from mcp_memory_service.consolidation.base import MemoryCluster
from mcp_memory_service.models.memory import Memory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_memory(content_hash: str, content: str, metadata: dict = None) -> Memory:
    """Build a minimal Memory object with the given content and metadata."""
    base_time = datetime.now().timestamp()
    return Memory(
        content=content,
        content_hash=content_hash,
        tags=["test"],
        memory_type="observation",
        embedding=[0.1] * 320,
        metadata=metadata or {},
        created_at=base_time - 3600,
        created_at_iso=datetime.fromtimestamp(base_time - 3600).isoformat() + "Z",
    )


def _make_cluster(memory_hashes: list) -> MemoryCluster:
    """Build a minimal MemoryCluster for the given hashes."""
    return MemoryCluster(
        cluster_id="test_cluster",
        memory_hashes=memory_hashes,
        centroid_embedding=[0.1] * 320,
        coherence_score=0.8,
        created_at=datetime.now(),
        theme_keywords=["test", "compression"],
        metadata={},
    )


# ---------------------------------------------------------------------------
# _strip_compression_prefixes unit tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestStripCompressionPrefixes:
    """Unit tests for the _strip_compression_prefixes static helper."""

    def test_plain_key_unchanged(self):
        """Keys without prefixes are returned as-is."""
        assert SemanticCompressionEngine._strip_compression_prefixes("source_type") == "source_type"

    def test_single_common_prefix_stripped(self):
        """A single 'common_' prefix is removed."""
        assert SemanticCompressionEngine._strip_compression_prefixes("common_source_type") == "source_type"

    def test_single_varied_prefix_stripped(self):
        """A single 'varied_' prefix is removed."""
        assert SemanticCompressionEngine._strip_compression_prefixes("varied_source_type") == "source_type"

    def test_double_common_prefix_stripped(self):
        """Two 'common_' prefixes are both removed."""
        assert SemanticCompressionEngine._strip_compression_prefixes("common_common_source_type") == "source_type"

    def test_mixed_prefixes_stripped(self):
        """Alternating common_/varied_ prefixes are fully stripped."""
        assert SemanticCompressionEngine._strip_compression_prefixes("common_varied_source_type") == "source_type"
        assert SemanticCompressionEngine._strip_compression_prefixes("varied_common_source_type") == "source_type"

    def test_triple_prefix_stripped(self):
        """Three levels of prefix nesting are fully stripped."""
        key = "common_common_common_source_type"
        assert SemanticCompressionEngine._strip_compression_prefixes(key) == "source_type"

    def test_variety_count_suffix_stripped(self):
        """_variety_count suffix is removed after prefix stripping."""
        assert (
            SemanticCompressionEngine._strip_compression_prefixes("source_type_variety_count")
            == "source_type"
        )

    def test_prefix_and_suffix_stripped(self):
        """Both prefix and suffix are removed in combination."""
        key = "common_source_type_variety_count"
        assert SemanticCompressionEngine._strip_compression_prefixes(key) == "source_type"

    def test_empty_key(self):
        """Empty string is returned unchanged."""
        assert SemanticCompressionEngine._strip_compression_prefixes("") == ""


# ---------------------------------------------------------------------------
# _aggregate_metadata tests — no prefix nesting after repeated compression
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAggregateMetadataPrefixNesting:
    """Verify _aggregate_metadata does not produce exponentially nested keys."""

    @pytest.fixture
    def engine(self, consolidation_config):
        return SemanticCompressionEngine(consolidation_config)

    def test_fresh_memories_produce_no_double_prefix(self, engine):
        """
        Memories with plain metadata keys produce at most single-prefix output
        (common_/varied_/..._variety_count), never double-prefix.
        """
        memories = [
            _make_memory("h1", "Python list comprehensions content", {"source_type": "python"}),
            _make_memory("h2", "Python generators related content", {"source_type": "python"}),
        ]
        aggregated = engine._aggregate_metadata(memories)

        # common_source_type should exist because both values are "python"
        assert "common_source_type" in aggregated
        # No double-prefix keys should be present
        for key in aggregated:
            assert not key.startswith("common_common_"), f"Double prefix found: {key}"
            assert not key.startswith("varied_varied_"), f"Double prefix found: {key}"
            assert not key.startswith("common_varied_"), f"Double prefix found: {key}"
            assert not key.startswith("varied_common_"), f"Double prefix found: {key}"

    def test_previously_compressed_memories_no_double_prefix(self, engine):
        """
        Memories that already carry common_/varied_ metadata keys (i.e. they
        are themselves outputs of a previous compression) must NOT produce
        double-prefixed keys in the new aggregation.
        """
        memories = [
            _make_memory(
                "h1",
                "Compressed cluster about Python",
                {
                    "common_source_type": "python",
                    "varied_author": ["alice", "bob"],
                    "source_memory_count_variety_count": 3,
                },
            ),
            _make_memory(
                "h2",
                "Another compressed cluster about Python",
                {
                    "common_source_type": "python",
                    "varied_author": ["carol"],
                },
            ),
        ]
        aggregated = engine._aggregate_metadata(memories)

        for key in aggregated:
            assert not key.startswith("common_common_"), f"Double prefix found: {key}"
            assert not key.startswith("varied_varied_"), f"Double prefix found: {key}"
            assert not key.startswith("common_varied_"), f"Double prefix found: {key}"

    def test_three_simulated_compression_runs_no_nesting(self, engine):
        """
        Simulate three successive compression cycles.

        After each cycle, the compressed memory's metadata carries the
        aggregated keys from the previous cycle.  After three cycles, no
        key should contain any prefix nesting (common_common_, etc.).
        """
        # Round 1: raw memories
        raw_memories = [
            _make_memory("r1", "Python list comprehension tutorial", {"source_type": "python"}),
            _make_memory("r2", "Python generator expressions guide", {"source_type": "python"}),
        ]
        round1_meta = engine._aggregate_metadata(raw_memories)
        # e.g. {"common_source_type": "python", ...}

        # Round 2: compressed memories from round 1 are re-compressed
        round2_inputs = [
            _make_memory("c1", "Round 1 compressed A", round1_meta),
            _make_memory("c2", "Round 1 compressed B", {**round1_meta, "extra": "data"}),
        ]
        round2_meta = engine._aggregate_metadata(round2_inputs)

        # Round 3: compressed memories from round 2 are re-compressed
        round3_inputs = [
            _make_memory("d1", "Round 2 compressed A", round2_meta),
            _make_memory("d2", "Round 2 compressed B", {**round2_meta, "more": "info"}),
        ]
        round3_meta = engine._aggregate_metadata(round3_inputs)

        # After three rounds, no key should contain nested prefixes
        for key in round3_meta:
            assert "common_common_" not in key, f"Nested prefix after 3 rounds: {key}"
            assert "varied_varied_" not in key, f"Nested prefix after 3 rounds: {key}"
            assert "common_varied_" not in key, f"Nested prefix after 3 rounds: {key}"
            assert "varied_common_" not in key, f"Nested prefix after 3 rounds: {key}"

    def test_internal_keys_excluded(self, engine):
        """
        Internal consolidation metadata keys must not appear in aggregated output.
        """
        memories = [
            _make_memory(
                "h1",
                "Some memory content with terms",
                {
                    "connection_boost": 1.5,
                    "decay_factor": 0.8,
                    "last_consolidated_at": "2026-01-01T00:00:00",
                    "access_boost": 0.3,
                    "_is_compressed": True,
                    "consolidation_run": "run_123",
                    "source_type": "python",  # normal key — should be included
                },
            ),
            _make_memory(
                "h2",
                "Related memory content about python",
                {
                    "connection_boost": 2.0,
                    "source_type": "python",
                },
            ),
        ]
        aggregated = engine._aggregate_metadata(memories)

        internal_keys = {
            "connection_boost", "decay_factor", "last_consolidated_at",
            "access_boost", "_is_compressed", "consolidation_run",
        }
        for int_key in internal_keys:
            # Should not appear bare or with prefix
            for agg_key in aggregated:
                assert int_key not in agg_key or agg_key == "common_source_type", (
                    f"Internal key '{int_key}' leaked into aggregated output as '{agg_key}'"
                )

        # Normal key should still appear
        assert "common_source_type" in aggregated

    def test_private_underscore_keys_excluded(self, engine):
        """
        Keys starting with '_' (private/internal) must not appear in aggregated output.
        """
        memories = [
            _make_memory("h1", "Memory content alpha", {"_private_key": "secret", "topic": "alpha"}),
            _make_memory("h2", "Memory content beta", {"_private_key": "other", "topic": "alpha"}),
        ]
        aggregated = engine._aggregate_metadata(memories)

        for key in aggregated:
            assert not key.startswith("common__"), "Private underscore key leaked"
            assert not key.startswith("varied__"), "Private underscore key leaked"

        # Normal key should be present
        assert "common_topic" in aggregated


# ---------------------------------------------------------------------------
# Integration-level test via process() — no double-prefix in compressed output
# ---------------------------------------------------------------------------

def _copies_of(compressed_memory: Memory, base_time: float, count: int = 3) -> list:
    """Duplicates of a compressed memory, as a second consolidation run sees it."""
    return [
        Memory(
            content=compressed_memory.content,
            content_hash=f"c2_{i}",
            tags=compressed_memory.tags,
            memory_type=compressed_memory.memory_type,
            embedding=compressed_memory.embedding or [0.1] * 320,
            metadata=dict(compressed_memory.metadata),
            created_at=base_time,
            created_at_iso=datetime.fromtimestamp(base_time).isoformat() + "Z",
        )
        for i in range(count)
    ]


def _assert_no_nested_prefix(metadata: dict) -> None:
    for key in metadata:
        for nested in ("common_common_", "varied_varied_", "common_varied_", "varied_common_"):
            assert nested not in key, f"Nested prefix in {key}"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_process_compression_no_prefix_nesting(consolidation_config):
    """
    Full integration test: run compress twice (simulating repeated consolidation).

    The second pass compresses the output of the first pass.  The resulting
    metadata must not contain nested prefix keys.
    """
    engine = SemanticCompressionEngine(consolidation_config)

    # Build 3 memories with plain metadata
    base_time = datetime.now().timestamp()
    memories = [
        Memory(
            content="Python list comprehensions simplify iteration patterns",
            content_hash=f"orig_{i}",
            tags=["python", "programming"],
            memory_type="learning",
            embedding=[0.1 + i * 0.01] * 320,
            metadata={"domain": "python", "level": "beginner"},
            created_at=base_time - (i * 3600),
            created_at_iso=datetime.fromtimestamp(base_time - i * 3600).isoformat() + "Z",
        )
        for i in range(3)
    ]

    cluster = _make_cluster([m.content_hash for m in memories])

    # First compression pass
    results = await engine.process([cluster], memories)
    assert len(results) == 1
    compressed_memory = results[0].compressed_memory

    _assert_no_nested_prefix(compressed_memory.metadata)

    # Second compression pass: compress the compressed memory with itself
    # duplicated. process() no longer feeds compression output back in (#1492),
    # so the second pass goes through _compress_cluster to keep this check alive.
    second_memories = _copies_of(compressed_memory, base_time)
    second_cluster = _make_cluster([m.content_hash for m in second_memories])
    result2 = await engine._compress_cluster(second_cluster, second_memories)

    _assert_no_nested_prefix(result2.compressed_memory.metadata)
