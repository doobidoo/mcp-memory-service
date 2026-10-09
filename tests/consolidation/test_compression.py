"""Unit tests for the semantic compression engine."""

import pytest
from datetime import datetime, timedelta

from mcp_memory_service.consolidation.compression import (
    SemanticCompressionEngine, 
    CompressionResult
)
from mcp_memory_service.consolidation.base import MemoryCluster
from mcp_memory_service.models.memory import Memory


@pytest.mark.unit
class TestSemanticCompressionEngine:
    """Test the semantic compression system."""
    
    @pytest.fixture
    def compression_engine(self, consolidation_config):
        return SemanticCompressionEngine(consolidation_config)
    
    @pytest.fixture
    def sample_cluster_with_memories(self):
        """Create a sample cluster with corresponding memories."""
        base_time = datetime.now().timestamp()
        
        memories = [
            Memory(
                content="Python list comprehensions provide a concise way to create lists",
                content_hash="hash1",
                tags=["python", "programming", "lists"],
                memory_type="reference",
                embedding=[0.1, 0.2, 0.3] * 107,  # ~320 dim
                created_at=base_time - 86400,
                created_at_iso=datetime.fromtimestamp(base_time - 86400).isoformat() + 'Z'
            ),
            Memory(
                content="List comprehensions in Python are more readable than traditional for loops",
                content_hash="hash2", 
                tags=["python", "readability", "best-practices"],
                memory_type="standard",
                embedding=[0.12, 0.18, 0.32] * 107,
                created_at=base_time - 172800,
                created_at_iso=datetime.fromtimestamp(base_time - 172800).isoformat() + 'Z'
            ),
            Memory(
                content="Example: squares = [x**2 for x in range(10)] creates a list of squares",
                content_hash="hash3",
                tags=["python", "example", "code"],
                memory_type="standard", 
                embedding=[0.11, 0.21, 0.31] * 107,
                created_at=base_time - 259200,
                created_at_iso=datetime.fromtimestamp(base_time - 259200).isoformat() + 'Z'
            ),
            Memory(
                content="Python comprehensions work for lists, sets, and dictionaries",
                content_hash="hash4",
                tags=["python", "comprehensions", "data-structures"],
                memory_type="reference",
                embedding=[0.13, 0.19, 0.29] * 107,
                created_at=base_time - 345600,
                created_at_iso=datetime.fromtimestamp(base_time - 345600).isoformat() + 'Z'
            )
        ]
        
        cluster = MemoryCluster(
            cluster_id="test_cluster",
            memory_hashes=[m.content_hash for m in memories],
            centroid_embedding=[0.12, 0.2, 0.3] * 107,
            coherence_score=0.85,
            created_at=datetime.now(),
            theme_keywords=["python", "comprehensions", "lists", "programming"],
            metadata={"test_cluster": True}
        )
        
        return cluster, memories
    
    @pytest.mark.asyncio
    async def test_basic_compression(self, compression_engine, sample_cluster_with_memories):
        """Test basic compression functionality."""
        cluster, memories = sample_cluster_with_memories
        
        results = await compression_engine.process([cluster], memories)
        
        assert len(results) == 1
        result = results[0]
        
        assert isinstance(result, CompressionResult)
        assert result.cluster_id == "test_cluster"
        assert isinstance(result.compressed_memory, Memory)
        assert result.source_memory_count == 4
        assert 0 < result.compression_ratio < 1  # Should be compressed
        assert len(result.key_concepts) > 0
        assert isinstance(result.temporal_span, dict)
    
    @pytest.mark.asyncio
    async def test_compressed_memory_properties(self, compression_engine, sample_cluster_with_memories):
        """Test properties of the compressed memory object."""
        cluster, memories = sample_cluster_with_memories
        
        results = await compression_engine.process([cluster], memories)
        compressed_memory = results[0].compressed_memory
        
        # Check basic properties
        assert compressed_memory.memory_type == "pattern"  # Updated to match ontology-compliant type
        assert len(compressed_memory.content) <= compression_engine.max_summary_length
        assert len(compressed_memory.content) > 0
        assert compressed_memory.content_hash is not None
        
        # Check tags (should include cluster tags and compression marker)
        assert "compressed_cluster" in compressed_memory.tags or "compressed" in compressed_memory.tags
        
        # Check metadata
        assert "cluster_id" in compressed_memory.metadata
        assert "compression_date" in compressed_memory.metadata
        assert "source_memory_count" in compressed_memory.metadata
        assert "compression_ratio" in compressed_memory.metadata
        assert "key_concepts" in compressed_memory.metadata
        assert "temporal_span" in compressed_memory.metadata
        assert "theme_keywords" in compressed_memory.metadata
        
        # Check embedding (should use cluster centroid)
        assert compressed_memory.embedding == cluster.centroid_embedding
    
    @pytest.mark.asyncio
    async def test_key_concept_extraction(self, compression_engine, sample_cluster_with_memories):
        """Test extraction of key concepts from cluster memories."""
        cluster, memories = sample_cluster_with_memories
        
        key_concepts = await compression_engine._extract_key_concepts(memories, cluster.theme_keywords)
        
        assert isinstance(key_concepts, list)
        assert len(key_concepts) > 0
        
        # Should include theme keywords
        theme_overlap = set(key_concepts).intersection(set(cluster.theme_keywords))
        assert len(theme_overlap) > 0
        
        # Should extract relevant concepts from content
        expected_concepts = {"python", "comprehensions", "lists"}
        found_concepts = set(concept.lower() for concept in key_concepts)
        overlap = expected_concepts.intersection(found_concepts)
        assert len(overlap) > 0
    
    @pytest.mark.asyncio
    async def test_thematic_summary_generation(self, compression_engine, sample_cluster_with_memories):
        """Test generation of thematic summaries."""
        cluster, memories = sample_cluster_with_memories
        
        # Extract key concepts first
        key_concepts = await compression_engine._extract_key_concepts(memories, cluster.theme_keywords)
        
        # Generate summary
        summary = await compression_engine._generate_thematic_summary(memories, key_concepts)
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert len(summary) <= compression_engine.max_summary_length
        
        # Summary should contain information about the cluster
        summary_lower = summary.lower()
        assert "cluster" in summary_lower or str(len(memories)) in summary
        
        # Should mention key concepts
        concept_mentions = sum(1 for concept in key_concepts[:3] if concept.lower() in summary_lower)
        assert concept_mentions > 0
    
    @pytest.mark.asyncio
    async def test_temporal_span_calculation(self, compression_engine, sample_cluster_with_memories):
        """Test calculation of temporal span for memories."""
        cluster, memories = sample_cluster_with_memories
        
        temporal_span = compression_engine._calculate_temporal_span(memories)
        
        assert isinstance(temporal_span, dict)
        assert "start_time" in temporal_span
        assert "end_time" in temporal_span
        assert "span_days" in temporal_span
        assert "span_description" in temporal_span
        assert "start_iso" in temporal_span
        assert "end_iso" in temporal_span
        
        # Check values make sense
        assert temporal_span["start_time"] <= temporal_span["end_time"]
        assert temporal_span["span_days"] >= 0
        assert isinstance(temporal_span["span_description"], str)
    
    @pytest.mark.asyncio
    async def test_tag_aggregation(self, compression_engine, sample_cluster_with_memories):
        """Test aggregation of tags from cluster memories."""
        cluster, memories = sample_cluster_with_memories
        
        aggregated_tags = compression_engine._aggregate_tags(memories)
        
        assert isinstance(aggregated_tags, list)
        assert "cluster" in aggregated_tags
        assert "compressed" in aggregated_tags
        
        # Should include frequent tags from original memories
        original_tags = set()
        for memory in memories:
            original_tags.update(memory.tags)
        
        # Check that some original tags are preserved
        aggregated_set = set(aggregated_tags)
        overlap = original_tags.intersection(aggregated_set)
        assert len(overlap) > 0
    
    @pytest.mark.asyncio
    async def test_metadata_aggregation(self, compression_engine, sample_cluster_with_memories):
        """Test aggregation of metadata from cluster memories."""
        cluster, memories = sample_cluster_with_memories
        
        # Add some metadata to memories
        memories[0].metadata["test_field"] = "value1"
        memories[1].metadata["test_field"] = "value1"  # Same value
        memories[2].metadata["test_field"] = "value2"  # Different value
        memories[3].metadata["unique_field"] = "unique"
        
        aggregated_metadata = compression_engine._aggregate_metadata(memories)
        
        assert isinstance(aggregated_metadata, dict)
        assert "source_memory_hashes" in aggregated_metadata
        
        # Should handle common values
        if "common_test_field" in aggregated_metadata:
            assert aggregated_metadata["common_test_field"] in ["value1", "value2"]
        
        # Should handle varied values
        if "varied_test_field" in aggregated_metadata:
            assert isinstance(aggregated_metadata["varied_test_field"], list)
        
        # Should track variety
        if "unique_field_variety_count" in aggregated_metadata:
            assert aggregated_metadata["unique_field_variety_count"] == 1
    
    @pytest.mark.asyncio
    async def test_compression_ratio_calculation(self, compression_engine, sample_cluster_with_memories):
        """Test compression ratio calculation."""
        cluster, memories = sample_cluster_with_memories
        
        results = await compression_engine.process([cluster], memories)
        result = results[0]
        
        # Calculate expected ratio
        original_size = sum(len(m.content) for m in memories)
        compressed_size = len(result.compressed_memory.content)
        expected_ratio = compressed_size / original_size
        
        assert abs(result.compression_ratio - expected_ratio) < 0.01  # Small tolerance
        assert 0 < result.compression_ratio < 1  # Should be compressed
    
    @pytest.mark.asyncio
    async def test_sentence_splitting(self, compression_engine):
        """Test sentence splitting functionality."""
        text = "This is the first sentence. This is the second sentence! Is this a question? Yes, it is."
        
        sentences = compression_engine._split_into_sentences(text)
        
        assert isinstance(sentences, list)
        assert len(sentences) >= 3  # Should find multiple sentences
        
        # Check that sentences are properly cleaned
        for sentence in sentences:
            assert len(sentence) > 10  # Minimum length filter
            assert sentence.strip() == sentence  # Should be trimmed
    
    @pytest.mark.asyncio
    async def test_empty_cluster_handling(self, compression_engine):
        """Test handling of empty clusters."""
        results = await compression_engine.process([], [])
        assert results == []
    
    @pytest.mark.asyncio
    async def test_single_memory_cluster(self, compression_engine):
        """Test handling of cluster with single memory (should be skipped)."""
        memory = Memory(
            content="Single memory content",
            content_hash="single",
            tags=["test"],
            embedding=[0.1] * 320,
            created_at=datetime.now().timestamp()
        )
        
        cluster = MemoryCluster(
            cluster_id="single_cluster",
            memory_hashes=["single"],
            centroid_embedding=[0.1] * 320,
            coherence_score=1.0,
            created_at=datetime.now(),
            theme_keywords=["test"]
        )
        
        results = await compression_engine.process([cluster], [memory])
        
        # Should skip clusters with insufficient memories
        assert results == []
    
    @pytest.mark.asyncio
    async def test_missing_memories_handling(self, compression_engine):
        """Test handling of cluster referencing missing memories."""
        cluster = MemoryCluster(
            cluster_id="missing_cluster",
            memory_hashes=["missing1", "missing2", "missing3"],
            centroid_embedding=[0.1] * 320,
            coherence_score=0.8,
            created_at=datetime.now(),
            theme_keywords=["missing"]
        )
        
        # Provide empty memories list
        results = await compression_engine.process([cluster], [])
        
        # Should handle missing memories gracefully
        assert results == []
    
    @pytest.mark.asyncio
    async def test_compression_benefit_estimation(self, compression_engine, sample_cluster_with_memories):
        """Test estimation of compression benefits."""
        cluster, memories = sample_cluster_with_memories
        
        benefits = await compression_engine.estimate_compression_benefit([cluster], memories)
        
        assert isinstance(benefits, dict)
        assert "compressible_clusters" in benefits
        assert "total_original_size" in benefits
        assert "estimated_compressed_size" in benefits
        assert "compression_ratio" in benefits
        assert "estimated_savings_bytes" in benefits
        assert "estimated_savings_percent" in benefits
        
        # Check values make sense
        assert benefits["compressible_clusters"] >= 0
        assert benefits["total_original_size"] >= 0
        assert benefits["estimated_compressed_size"] >= 0
        assert 0 <= benefits["compression_ratio"] <= 1
        assert benefits["estimated_savings_bytes"] >= 0
        assert 0 <= benefits["estimated_savings_percent"] <= 100
    
    @pytest.mark.asyncio
    async def test_large_content_truncation(self, compression_engine):
        """Test handling of content that exceeds max summary length."""
        # Create memories with very long content
        long_memories = []
        base_time = datetime.now().timestamp()
        
        for i in range(3):
            # Create content longer than max_summary_length
            long_content = "This is a very long memory content. " * 50  # Much longer than 200 chars
            memory = Memory(
                content=long_content,
                content_hash=f"long_{i}",
                tags=["long", "test"],
                embedding=[0.1 + i*0.1] * 320,
                created_at=base_time - (i * 3600)
            )
            long_memories.append(memory)
        
        cluster = MemoryCluster(
            cluster_id="long_cluster",
            memory_hashes=[m.content_hash for m in long_memories],
            centroid_embedding=[0.2] * 320,
            coherence_score=0.8,
            created_at=datetime.now(),
            theme_keywords=["long", "content"]
        )
        
        results = await compression_engine.process([cluster], long_memories)
        
        if results:
            compressed_content = results[0].compressed_memory.content
            # Should be truncated to max length
            assert len(compressed_content) <= compression_engine.max_summary_length
            
            # Should indicate truncation if content was cut off
            if len(compressed_content) == compression_engine.max_summary_length:
                assert compressed_content.endswith("...")
    
    @pytest.mark.asyncio
    async def test_key_concept_extraction_comprehensive(self, compression_engine):
        """Test comprehensive key concept extraction from memories."""
        # Create memories with various content patterns
        memories = []
        base_time = datetime.now().timestamp()
        
        content_examples = [
            "Check out https://example.com for more info about CamelCaseVariable usage.",
            "Email me at test@example.com if you have questions about the API response.",  
            "The system returns {'status': 'success', 'code': 200} for valid requests.",
            "Today's date is 2024-01-15 and the time is 14:30 for scheduling.",
            "See 'important documentation' for details on snake_case_variable patterns."
        ]
        
        for i, content in enumerate(content_examples):
            memory = Memory(
                content=content,
                content_hash=f"concept_test_{i}",
                tags=["test", "concept", "extraction"],
                embedding=[0.1 + i*0.01] * 320,
                created_at=base_time - (i * 3600)
            )
            memories.append(memory)
        
        theme_keywords = ["test", "API", "documentation", "variable"]
        
        concepts = await compression_engine._extract_key_concepts(memories, theme_keywords)
        
        # Should include theme keywords
        assert any("test" in concepts for concept in [theme_keywords])
        
        # Should extract concepts from content
        assert isinstance(concepts, list)
        assert len(concepts) > 0
        
        # Concepts should be strings
        assert all(isinstance(concept, str) for concept in concepts)
    
    @pytest.mark.asyncio
    async def test_key_concepts_drop_stop_words_and_bare_numbers(self, compression_engine):
        """Noise is filtered whichever source it reaches the concepts from.

        The stop-word check used to run only on the lowercase frequent-word
        source, so a capitalized "With", a theme keyword, or a number matched by
        the ``numbers`` pattern all survived into the summary.
        """
        base_time = datetime.now().timestamp()
        memories = [
            Memory(
                content=(
                    "With Cloudflare in front, That edge cache What the deployment "
                    "needs; a load average of 0.682 showed up in the same window."
                ),
                content_hash="noise_concept_1",
                tags=["test", "concept", "noise"],
                embedding=[0.1] * 320,
                created_at=base_time,
            ),
            Memory(
                content=(
                    "Cloudflare fronts the deployment again, so WITH the same edge "
                    'cache That latency figure stays at 0.682, quoted as " 0.731 ".'
                ),
                content_hash="noise_concept_2",
                tags=["test", "concept", "noise"],
                embedding=[0.12] * 320,
                created_at=base_time - 3600,
            ),
        ]

        # Cloudflare is deliberately absent from the theme keywords: it has to
        # reach the result through the capitalized-terms source off the content.
        concepts = await compression_engine._extract_key_concepts(memories, ["With", "0.695"])

        lowered = [concept.lower() for concept in concepts]
        for noise in ("with", "that", "what", "0.682", "0.695"):
            assert noise not in lowered, f"{noise!r} survived into {concepts}"
        # Quoted text keeps its padding, so a padded number is noise too.
        assert not any(concept.strip() == "0.731" for concept in concepts), concepts
        # A real term from the same memories is untouched by the filter.
        assert "cloudflare" in lowered

    @pytest.mark.asyncio
    async def test_memories_without_timestamps(self, compression_engine):
        """Test handling of memories with timestamps (Memory model auto-sets them)."""
        memories = [
            Memory(
                content="Memory with auto-generated timestamp",
                content_hash="auto_timestamp",
                tags=["test"],
                embedding=[0.1] * 320,
                created_at=None  # Will be auto-set by Memory model
            )
        ]
        
        cluster = MemoryCluster(
            cluster_id="auto_timestamp_cluster",
            memory_hashes=["auto_timestamp"],
            centroid_embedding=[0.1] * 320,
            coherence_score=0.8,
            created_at=datetime.now(),
            theme_keywords=["test"]
        )
        
        # Should handle gracefully without crashing
        temporal_span = compression_engine._calculate_temporal_span(memories)

        # Memory model auto-sets timestamps, so these will be actual values
        assert temporal_span["start_time"] is not None
        assert temporal_span["end_time"] is not None
        assert temporal_span["span_days"] >= 0
        assert isinstance(temporal_span["span_description"], str)


@pytest.mark.unit
class TestAggregateMetadataPrefixNesting:
    """Tests for the prefix-nesting bug fix described in GitHub issue #543.

    Problem: _aggregate_metadata() prefixed metadata keys with ``common_`` or
    ``varied_`` on every compression run.  When a memory produced by compression
    (which already has prefixed keys) was compressed again, the prefixes
    accumulated exponentially:
        Run 1 → common_connection_boost
        Run 2 → common_common_connection_boost
        Run 3 → common_common_common_connection_boost  ... etc.

    A single memory could end up with 261,749 characters of metadata, causing
    MCP tool call failures and database bloat.

    Fix: Strip existing compression prefixes (``common_``, ``varied_``) and
    ``_variety_count`` suffixes from raw keys before re-applying new prefixes.
    Internal consolidation keys are skipped entirely.

    Reference: GitHub issue #543
    """

    @pytest.fixture
    def engine(self, consolidation_config):
        return SemanticCompressionEngine(consolidation_config)

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    @staticmethod
    def _make_memory(content_hash: str, metadata: dict) -> Memory:
        base_time = datetime.now().timestamp()
        return Memory(
            content=f"Compression prefix test memory {content_hash}",
            content_hash=content_hash,
            tags=["test", "compression"],
            memory_type="pattern",
            embedding=[0.1] * 320,
            metadata=metadata,
            created_at=base_time,
            created_at_iso=datetime.fromtimestamp(base_time).isoformat() + 'Z',
        )

    # ------------------------------------------------------------------
    # _strip_compression_prefixes unit tests
    # ------------------------------------------------------------------

    def test_strip_single_common_prefix(self, engine):
        """common_foo → foo (single stripping pass)."""
        assert engine._strip_compression_prefixes("common_foo") == "foo"

    def test_strip_single_varied_prefix(self, engine):
        """varied_foo → foo (single stripping pass)."""
        assert engine._strip_compression_prefixes("varied_foo") == "foo"

    def test_strip_double_common_prefix(self, engine):
        """common_common_foo → foo (two stripping passes)."""
        assert engine._strip_compression_prefixes("common_common_foo") == "foo"

    def test_strip_mixed_prefixes(self, engine):
        """common_varied_common_foo → foo (three stripping passes)."""
        assert engine._strip_compression_prefixes("common_varied_common_foo") == "foo"

    def test_strip_variety_count_suffix(self, engine):
        """foo_variety_count → foo (suffix stripping)."""
        assert engine._strip_compression_prefixes("foo_variety_count") == "foo"

    def test_strip_prefix_then_suffix(self, engine):
        """common_foo_variety_count → foo (prefix then suffix stripped)."""
        assert engine._strip_compression_prefixes("common_foo_variety_count") == "foo"

    def test_plain_key_unchanged(self, engine):
        """Keys without prefix/suffix are returned as-is."""
        assert engine._strip_compression_prefixes("my_key") == "my_key"

    # ------------------------------------------------------------------
    # _aggregate_metadata prefix-nesting tests (issue #543)
    # ------------------------------------------------------------------

    def test_no_double_prefix_on_already_prefixed_keys(self, engine):
        """Keys that already carry common_/varied_ must NOT gain another prefix.

        Simulates the state after one compression run: the output metadata from
        that run (common_source / varied_source / etc.) is fed back into a
        second compression run.  Output keys must still be single-prefixed.

        Validates GitHub issue #543 fix.
        """
        # Memory metadata as it would look after a first compression pass
        post_compression_metadata = {
            "common_source": "database",   # already prefixed common_
            "varied_version": ["1.0", "2.0"],  # already prefixed varied_
            "plain_field": "value",         # not prefixed – normal metadata
        }

        memories = [
            self._make_memory("h1", post_compression_metadata),
            self._make_memory("h2", post_compression_metadata),
        ]

        result = engine._aggregate_metadata(memories)

        # No key in the result may begin with a double prefix
        for key in result:
            assert not key.startswith("common_common_"), (
                f"Double-nested 'common_common_' prefix found in key: {key!r}"
            )
            assert not key.startswith("varied_varied_"), (
                f"Double-nested 'varied_varied_' prefix found in key: {key!r}"
            )
            assert not key.startswith("common_varied_"), (
                f"Mixed double prefix found in key: {key!r}"
            )
            assert not key.startswith("varied_common_"), (
                f"Mixed double prefix found in key: {key!r}"
            )

    def test_triple_nesting_does_not_occur(self, engine):
        """Keys with triple-nested prefixes (common_common_common_…) must not appear.

        Simulates the output from two previous compression runs being fed into a
        third run.

        Validates GitHub issue #543 fix.
        """
        triple_prefixed_metadata = {
            "common_common_connection_boost": "0.5",
            "common_common_decay_factor": "0.9",
        }

        memories = [
            self._make_memory("h1", triple_prefixed_metadata),
            self._make_memory("h2", triple_prefixed_metadata),
        ]

        result = engine._aggregate_metadata(memories)

        for key in result:
            parts = key.split("_")
            leading_prefix_count = 0
            for part in parts:
                if part in ("common", "varied"):
                    leading_prefix_count += 1
                else:
                    break
            assert leading_prefix_count <= 1, (
                f"Key {key!r} has {leading_prefix_count} leading prefix(es); expected at most 1"
            )

    def test_output_keys_have_at_most_one_prefix_level(self, engine):
        """All output keys have zero or exactly one leading prefix level.

        Validates the key invariant stated in GitHub issue #543 fix.
        """
        # Mix of raw, once-prefixed, and twice-prefixed keys
        mixed_metadata = {
            "raw_field": "hello",
            "common_once_prefixed": "world",
            "varied_once_prefixed_too": ["a", "b"],
            "common_common_twice_prefixed": "boom",
            "varied_varied_twice_prefixed": ["x", "y", "z"],
        }

        memories = [
            self._make_memory("hA", mixed_metadata),
            self._make_memory("hB", mixed_metadata),
        ]

        result = engine._aggregate_metadata(memories)

        for key in result:
            if key == "source_memory_hashes":
                continue  # Built-in key, not subject to prefix rules
            # Count contiguous leading prefix tokens
            parts = key.split("_")
            leading_count = 0
            for part in parts:
                if part in ("common", "varied"):
                    leading_count += 1
                else:
                    break
            assert leading_count <= 1, (
                f"Key {key!r} has {leading_count} leading prefix token(s); expected ≤1"
            )

    def test_variety_count_suffix_not_re_suffixed(self, engine):
        """Keys ending with _variety_count must not gain another _variety_count.

        Validates GitHub issue #543 fix for suffix accumulation.
        """
        # Simulate metadata produced by a previous compression run with > 5 unique values
        suffix_metadata = {
            "user_id_variety_count": 42,
        }

        memories = [
            self._make_memory("h1", suffix_metadata),
            self._make_memory("h2", suffix_metadata),
        ]

        result = engine._aggregate_metadata(memories)

        for key in result:
            assert not key.endswith("_variety_count_variety_count"), (
                f"Double _variety_count suffix found in key: {key!r}"
            )

    # ------------------------------------------------------------------
    # Internal-key blocklist tests
    # ------------------------------------------------------------------

    def test_internal_keys_excluded_from_output(self, engine):
        """Internal consolidation metadata keys must not appear in aggregated output.

        These keys hold per-memory state that is meaningless after compression.

        Validates GitHub issue #543 fix (internal key blocklist).
        """
        internal_metadata = {
            "connection_boost": 1.2,
            "decay_factor": 0.8,
            "last_consolidated_at": "2024-01-01T00:00:00Z",
            "access_boost": 0.3,
            "_is_compressed": True,
            "consolidation_run": "daily",
            "normal_field": "keep me",  # Should appear in output
        }

        memories = [
            self._make_memory("i1", internal_metadata),
            self._make_memory("i2", internal_metadata),
        ]

        result = engine._aggregate_metadata(memories)

        # Internal keys (and their prefixed variants) must not appear
        forbidden_bases = {
            "connection_boost", "decay_factor", "last_consolidated_at",
            "access_boost", "_is_compressed", "consolidation_run",
        }
        for key in result:
            clean = engine._strip_compression_prefixes(key)
            assert clean not in forbidden_bases, (
                f"Internal key {clean!r} (from {key!r}) leaked into aggregated metadata"
            )

        # The normal field must still be present
        normal_keys = {k for k in result if "normal_field" in k}
        assert normal_keys, "normal_field was unexpectedly excluded from aggregated metadata"

    def test_private_underscore_keys_excluded(self, engine):
        """Keys starting with '_' (private/internal) must be excluded from output."""
        private_metadata = {
            "_internal_state": "secret",
            "_debug_info": {"level": 3},
            "public_key": "visible",
        }

        memories = [
            self._make_memory("p1", private_metadata),
            self._make_memory("p2", private_metadata),
        ]

        result = engine._aggregate_metadata(memories)

        for key in result:
            clean = engine._strip_compression_prefixes(key)
            assert not clean.startswith("_"), (
                f"Private key {clean!r} (from {key!r}) leaked into aggregated metadata"
            )

    # ------------------------------------------------------------------
    # Regression / idempotency test
    # ------------------------------------------------------------------

    def test_repeated_compression_idempotent_key_depth(self, engine):
        """Simulates 3 sequential compression runs; prefix depth must stay at 1.

        This is the exact regression scenario from GitHub issue #543:
        compound prefixes (common_common_common_…) accumulate across runs,
        causing metadata to balloon to hundreds of thousands of characters.
        """
        # Start with plain metadata
        metadata = {"source": "db", "version": "1.0"}

        for run in range(1, 4):  # Three simulated compression runs
            memories = [
                self._make_memory(f"r{run}_h1", metadata),
                self._make_memory(f"r{run}_h2", metadata),
            ]
            result = engine._aggregate_metadata(memories)

            # The next "run" starts with the output of this run as input metadata
            metadata = {k: v for k, v in result.items() if k != "source_memory_hashes"}

            # Check invariant after each run
            for key in result:
                if key == "source_memory_hashes":
                    continue
                parts = key.split("_")
                leading_count = sum(1 for p in parts[:len(parts)] if p in ("common", "varied"))
                # Count *contiguous* leading prefixes only
                contiguous = 0
                for part in key.split("_"):
                    if part in ("common", "varied"):
                        contiguous += 1
                    else:
                        break
                assert contiguous <= 1, (
                    f"After run {run}, key {key!r} has {contiguous} leading prefix(es)"
                )