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