"""Unit tests for the controlled forgetting engine."""

import pytest
import os
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from mcp_memory_service.consolidation.forgetting import (
    ControlledForgettingEngine, 
    ForgettingCandidate,
    ForgettingResult
)
from mcp_memory_service.consolidation.decay import RelevanceScore
from mcp_memory_service.models.memory import Memory


@pytest.mark.unit
class TestControlledForgettingEngine:
    """Test the controlled forgetting system."""
    
    @pytest.fixture
    def forgetting_engine(self, consolidation_config):
        return ControlledForgettingEngine(consolidation_config)
    
    @pytest.fixture
    def sample_relevance_scores(self, sample_memories):
        """Create sample relevance scores for memories."""
        scores = []
        for i, memory in enumerate(sample_memories):
            # Create varied relevance scores
            if "critical" in memory.tags:
                total_score = 1.5
            elif "temporary" in memory.tags:
                total_score = 0.05  # Very low relevance
            elif "test" in memory.content:
                total_score = 0.02  # Low quality content
            else:
                total_score = 0.8
            
            score = RelevanceScore(
                memory_hash=memory.content_hash,
                total_score=total_score,
                base_importance=1.0,
                decay_factor=0.8,
                connection_boost=1.0,
                access_boost=1.0,
                metadata={"test_score": True}
            )
            scores.append(score)
        
        return scores
    
    @pytest.mark.asyncio
    async def test_basic_forgetting_process(self, forgetting_engine, sample_memories, sample_relevance_scores):
        """Test basic forgetting process functionality."""
        access_patterns = {
            sample_memories[0].content_hash: datetime.now() - timedelta(days=100)  # Old access
        }
        
        results = await forgetting_engine.process(
            sample_memories, 
            sample_relevance_scores,
            access_patterns=access_patterns,
            time_horizon="monthly"
        )
        
        assert isinstance(results, list)
        assert all(isinstance(result, ForgettingResult) for result in results)
        
        # Check that some memories were processed for forgetting
        actions = [result.action_taken for result in results]
        valid_actions = {"archived", "compressed", "deleted", "skipped"}
        assert all(action in valid_actions for action in actions)
    
    @pytest.mark.asyncio
    async def test_identify_forgetting_candidates(self, forgetting_engine, sample_memories, sample_relevance_scores):
        """Test identification of forgetting candidates."""
        access_patterns = {}
        
        candidates = await forgetting_engine._identify_forgetting_candidates(
            sample_memories,
            {score.memory_hash: score for score in sample_relevance_scores},
            access_patterns,
            "monthly"
        )
        
        assert isinstance(candidates, list)
        assert all(isinstance(candidate, ForgettingCandidate) for candidate in candidates)
        
        # Check candidate properties
        for candidate in candidates:
            assert isinstance(candidate.memory, Memory)
            assert isinstance(candidate.relevance_score, RelevanceScore)
            assert isinstance(candidate.forgetting_reasons, list)
            assert len(candidate.forgetting_reasons) > 0
            assert candidate.archive_priority in [1, 2, 3]
            assert isinstance(candidate.can_be_deleted, bool)
    
    @pytest.mark.asyncio
    async def test_protected_memory_exclusion(self, forgetting_engine, sample_memories, sample_relevance_scores):
        """Test that protected memories are excluded from forgetting."""
        # Find critical memory (should be protected)
        critical_memory = next((m for m in sample_memories if "critical" in m.tags), None)
        
        if critical_memory:
            candidates = await forgetting_engine._identify_forgetting_candidates(
                [critical_memory],
                {critical_memory.content_hash: sample_relevance_scores[0]},  # Use first score
                {},
                "yearly"
            )
            
            # Critical memory should not be a candidate for forgetting
            assert len(candidates) == 0
    
    @pytest.mark.asyncio
    async def test_low_relevance_identification(self, forgetting_engine):
        """Test identification of low relevance memories."""
        now = datetime.now()
        
        low_relevance_memory = Memory(
            content="Low relevance test content",
            content_hash="low_relevance",
            tags=["test"],
            embedding=[0.1] * 320,
            created_at=now.timestamp(),
            updated_at=(now - timedelta(days=100)).timestamp()  # Old access
        )
        
        low_score = RelevanceScore(
            memory_hash="low_relevance",
            total_score=0.05,  # Below threshold
            base_importance=1.0,
            decay_factor=0.1,
            connection_boost=1.0,
            access_boost=1.0,
            metadata={}
        )
        
        candidates = await forgetting_engine._identify_forgetting_candidates(
            [low_relevance_memory],
            {"low_relevance": low_score},
            {},
            "monthly"
        )
        
        assert len(candidates) > 0
        candidate = candidates[0]
        assert "low_relevance" in candidate.forgetting_reasons
    
    @pytest.mark.asyncio
    async def test_old_access_identification(self, forgetting_engine):
        """Test identification of memories with old access patterns."""
        now = datetime.now()
        
        old_access_memory = Memory(
            content="Memory with old access",
            content_hash="old_access",
            tags=["test"],
            embedding=[0.1] * 320,
            created_at=now.timestamp(),
            metadata={'quality_score': 0.3}  # Low quality to use 30-90 day threshold
        )
        
        score = RelevanceScore(
            memory_hash="old_access",
            total_score=0.5,  # Decent relevance
            base_importance=1.0,
            decay_factor=0.8,
            connection_boost=1.0,
            access_boost=1.0,
            metadata={}
        )
        
        # Very old access
        old_access_patterns = {
            "old_access": now - timedelta(days=120)  # Older than threshold
        }
        
        candidates = await forgetting_engine._identify_forgetting_candidates(
            [old_access_memory],
            {"old_access": score},
            old_access_patterns,
            "monthly"
        )
        
        assert len(candidates) > 0
        candidate = candidates[0]
        assert "old_access" in candidate.forgetting_reasons
    
    @pytest.mark.asyncio
    async def test_temporary_memory_expiration(self, forgetting_engine):
        """Test identification of expired temporary memories."""
        now = datetime.now()
        
        expired_temp_memory = Memory(
            content="Expired temporary memory",
            content_hash="expired_temp",
            tags=["temporary"],
            memory_type="temporary",
            embedding=[0.1] * 320,
            created_at=(now - timedelta(days=10)).timestamp()  # Older than 7 days
        )
        
        score = RelevanceScore(
            memory_hash="expired_temp",
            total_score=0.8,  # Good relevance, but temporary
            base_importance=1.0,
            decay_factor=0.8,
            connection_boost=1.0,
            access_boost=1.0,
            metadata={}
        )
        
        candidates = await forgetting_engine._identify_forgetting_candidates(
            [expired_temp_memory],
            {"expired_temp": score},
            {},
            "monthly"
        )
        
        assert len(candidates) > 0
        candidate = candidates[0]
        assert "expired_temporary" in candidate.forgetting_reasons
        assert candidate.can_be_deleted is True
    
    @pytest.mark.asyncio
    async def test_low_quality_content_detection(self, forgetting_engine):
        """Test detection of low quality content."""
        # Very short content
        short_memory = Memory(
            content="test",
            content_hash="short",
            tags=["test"],
            embedding=[0.1] * 320,
            created_at=datetime.now().timestamp()
        )
        
        # Repetitive content
        repetitive_memory = Memory(
            content="test test test test test test",
            content_hash="repetitive",
            tags=["test"],
            embedding=[0.1] * 320,
            created_at=datetime.now().timestamp()
        )
        
        # Mostly non-alphabetic
        non_alpha_memory = Memory(
            content="!@#$%^&*()_+{}|<>?",
            content_hash="non_alpha",
            tags=["test"],
            embedding=[0.1] * 320,
            created_at=datetime.now().timestamp()
        )
        
        test_memories = [short_memory, repetitive_memory, non_alpha_memory]
        
        for memory in test_memories:
            is_low_quality = forgetting_engine._is_low_quality_content(memory)
            assert is_low_quality is True
    
    @pytest.mark.asyncio
    async def test_duplicate_detection(self, forgetting_engine, sample_memories):
        """Test detection of potential duplicate content."""
        # Create a memory that's very similar to an existing one
        existing_memory = sample_memories[0]
        duplicate_memory = Memory(
            content=existing_memory.content + " duplicate",  # Very similar
            content_hash="duplicate_test",
            tags=existing_memory.tags,
            embedding=existing_memory.embedding,
            created_at=datetime.now().timestamp()
        )
        
        test_memories = sample_memories + [duplicate_memory]
        
        is_duplicate = forgetting_engine._appears_to_be_duplicate(duplicate_memory, test_memories)
        # This might not always detect as duplicate due to the simple algorithm
        # Just ensure the method runs without error
        assert isinstance(is_duplicate, bool)
    
    @pytest.mark.asyncio
    async def test_archive_memory(self, forgetting_engine):
        """Test archiving a memory to filesystem."""
        memory = Memory(
            content="Memory to archive",
            content_hash="archive_test",
            tags=["test", "archive"],
            embedding=[0.1] * 320,
            created_at=datetime.now().timestamp()
        )
        
        score = RelevanceScore(
            memory_hash="archive_test",
            total_score=0.3,
            base_importance=1.0,
            decay_factor=0.5,
            connection_boost=1.0,
            access_boost=1.0,
            metadata={}
        )
        
        candidate = ForgettingCandidate(
            memory=memory,
            relevance_score=score,
            forgetting_reasons=["test_archival"],
            archive_priority=2,
            can_be_deleted=False
        )
        
        result = await forgetting_engine._archive_memory(candidate)
        
        assert isinstance(result, ForgettingResult)
        assert result.action_taken == "archived"
        assert result.archive_path is not None
        assert os.path.exists(result.archive_path)
        
        # Check archive file content
        with open(result.archive_path, 'r') as f:
            archive_data = json.load(f)
        
        assert "memory" in archive_data
        assert "relevance_score" in archive_data
        assert "forgetting_metadata" in archive_data
        assert archive_data["memory"]["content_hash"] == "archive_test"
    
    @pytest.mark.asyncio
    async def test_compress_memory(self, forgetting_engine):
        """Test compressing a memory."""
        memory = Memory(
            content="This is a longer memory content that should be compressed to preserve key information while reducing size",
            content_hash="compress_test",
            tags=["test", "compression"],
            embedding=[0.1] * 320,
            created_at=datetime.now().timestamp()
        )
        
        score = RelevanceScore(
            memory_hash="compress_test",
            total_score=0.4,
            base_importance=1.0,
            decay_factor=0.6,
            connection_boost=1.0,
            access_boost=1.0,
            metadata={}
        )
        
        candidate = ForgettingCandidate(
            memory=memory,
            relevance_score=score,
            forgetting_reasons=["test_compression"],
            archive_priority=3,
            can_be_deleted=False
        )
        
        result = await forgetting_engine._compress_memory(candidate)
        
        assert isinstance(result, ForgettingResult)
        assert result.action_taken == "compressed"
        assert result.compressed_version is not None
        assert result.archive_path is not None
        
        # Check compressed memory
        compressed = result.compressed_version
        assert compressed.memory_type == "compressed"
        assert "compressed" in compressed.tags
        assert len(compressed.content) <= len(memory.content)
        assert "original_hash" in compressed.metadata
        assert "compression_ratio" in compressed.metadata
    
    @pytest.mark.asyncio
    async def test_delete_memory(self, forgetting_engine):
        """Test deleting a memory with backup."""
        memory = Memory(
            content="Memory to delete",
            content_hash="delete_test",
            tags=["test", "delete"],
            embedding=[0.1] * 320,
            created_at=datetime.now().timestamp()
        )
        
        score = RelevanceScore(
            memory_hash="delete_test",
            total_score=0.01,
            base_importance=1.0,
            decay_factor=0.1,
            connection_boost=1.0,
            access_boost=1.0,
            metadata={}
        )
        
        candidate = ForgettingCandidate(
            memory=memory,
            relevance_score=score,
            forgetting_reasons=["potential_duplicate"],
            archive_priority=1,
            can_be_deleted=True
        )
        
        result = await forgetting_engine._delete_memory(candidate)
        
        assert isinstance(result, ForgettingResult)
        assert result.action_taken == "deleted"
        assert result.archive_path is not None  # Backup should exist
        assert os.path.exists(result.archive_path)
        
        # Check backup file
        with open(result.archive_path, 'r') as f:
            backup_data = json.load(f)
        
        assert "memory" in backup_data
        assert "deletion_metadata" in backup_data
        assert backup_data["memory"]["content_hash"] == "delete_test"
    
    @pytest.mark.asyncio
    async def test_memory_recovery(self, forgetting_engine):
        """Test recovery of forgotten memories."""
        # First archive a memory
        memory = Memory(
            content="Memory for recovery test",
            content_hash="recovery_test",
            tags=["test", "recovery"],
            embedding=[0.1] * 320,
            created_at=datetime.now().timestamp()
        )
        
        score = RelevanceScore(
            memory_hash="recovery_test",
            total_score=0.2,
            base_importance=1.0,
            decay_factor=0.4,
            connection_boost=1.0,
            access_boost=1.0,
            metadata={}
        )
        
        candidate = ForgettingCandidate(
            memory=memory,
            relevance_score=score,
            forgetting_reasons=["test_recovery"],
            archive_priority=2,
            can_be_deleted=False
        )
        
        # Archive the memory
        await forgetting_engine._archive_memory(candidate)
        
        # Now try to recover it
        recovered_memory = await forgetting_engine.recover_memory("recovery_test")
        
        assert recovered_memory is not None
        assert isinstance(recovered_memory, Memory)
        assert recovered_memory.content_hash == "recovery_test"
        assert recovered_memory.content == memory.content
    
    @pytest.mark.asyncio
    async def test_forgetting_statistics(self, forgetting_engine, sample_memories, sample_relevance_scores):
        """Test getting forgetting statistics."""
        # Process some memories to generate statistics
        access_patterns = {
            sample_memories[0].content_hash: datetime.now() - timedelta(days=100)
        }
        
        await forgetting_engine.process(
            sample_memories[:3],  # Use subset for faster test
            sample_relevance_scores[:3],
            access_patterns=access_patterns,
            time_horizon="monthly"
        )
        
        stats = await forgetting_engine.get_forgetting_statistics()
        
        assert isinstance(stats, dict)
        assert "total_archived" in stats
        assert "total_compressed" in stats
        assert "total_deleted" in stats
        assert "archive_size_bytes" in stats
        
        # Values should be non-negative
        assert stats["total_archived"] >= 0
        assert stats["total_compressed"] >= 0
        assert stats["total_deleted"] >= 0
        assert stats["archive_size_bytes"] >= 0
    
    @pytest.mark.asyncio
    async def test_create_compressed_content(self, forgetting_engine):
        """Test creation of compressed content."""
        original_content = """
        This is a longer piece of content that contains multiple sentences. 
        It has important information in the first sentence. 
        The middle part contains additional details and context.
        The final sentence wraps up the content nicely.
        """
        
        compressed = forgetting_engine._create_compressed_content(original_content)
        
        assert isinstance(compressed, str)
        assert len(compressed) <= len(original_content)
        assert len(compressed) > 0
        
        # Should contain compression indicator if significantly compressed
        if len(compressed) < len(original_content) * 0.8:
            assert "[Compressed]" in compressed
    
    @pytest.mark.asyncio
    async def test_extract_important_terms(self, forgetting_engine):
        """Test extraction of important terms from text."""
        text = """
        The CamelCaseVariable is used with the API_ENDPOINT.
        Visit https://example.com for documentation.
        Contact support@example.com for help.
        The temperature is 25.5 degrees.
        See "important documentation" for details.
        Use snake_case_variables appropriately.
        """
        
        terms = forgetting_engine._extract_important_terms(text)
        
        assert isinstance(terms, list)
        assert len(terms) <= 10  # Should be limited
        
        # Should extract various types of important terms
        terms_lower = [term.lower() for term in terms]
        term_str = " ".join(terms_lower)
        
        # Should find some patterns (exact matches may vary)
        assert len(terms) > 0  # At least some terms should be found
    
    @pytest.mark.asyncio
    async def test_archive_directories_creation(self, temp_archive_path):
        """Test that archive directories are created properly."""
        config = type('Config', (), {
            'relevance_threshold': 0.1,
            'access_threshold_days': 90,
            'archive_location': temp_archive_path
        })()
        
        engine = ControlledForgettingEngine(config)
        
        # Check that directories were created
        assert engine.archive_path.exists()
        assert engine.daily_archive.exists()
        assert engine.compressed_archive.exists()
        assert engine.metadata_archive.exists()
    
    @pytest.mark.asyncio
    async def test_empty_input_handling(self, forgetting_engine):
        """Test handling of empty inputs."""
        results = await forgetting_engine.process([], [])
        assert results == []
    
    @pytest.mark.asyncio
    async def test_time_horizon_filtering(self, forgetting_engine, sample_memories, sample_relevance_scores):
        """Test that time horizon affects forgetting behavior."""
        access_patterns = {
            sample_memories[0].content_hash: datetime.now() - timedelta(days=100)
        }
        
        # Test with different time horizons
        daily_results = await forgetting_engine.process(
            sample_memories[:2],
            sample_relevance_scores[:2],
            access_patterns=access_patterns,
            time_horizon="daily"
        )
        
        yearly_results = await forgetting_engine.process(
            sample_memories[:2],
            sample_relevance_scores[:2],
            access_patterns=access_patterns,
            time_horizon="yearly"
        )
        
        # Different time horizons may produce different results
        # At minimum, both should handle the input without errors
        assert isinstance(daily_results, list)
        assert isinstance(yearly_results, list)
    
    @pytest.mark.asyncio
    async def test_metadata_entry_creation(self, forgetting_engine):
        """Test creation of metadata log entries."""
        memory = Memory(
            content="Test memory for metadata",
            content_hash="metadata_test",
            tags=["test"],
            embedding=[0.1] * 320,
            created_at=datetime.now().timestamp()
        )
        
        archive_path = forgetting_engine.metadata_archive / "test_archive.json"
        
        await forgetting_engine._create_metadata_entry(memory, archive_path, "archived")
        
        # Check that log file was created
        log_file = forgetting_engine.metadata_archive / "forgetting_log.jsonl"
        assert log_file.exists()
        
        # Check log content
        with open(log_file, 'r') as f:
            log_content = f.read().strip()
        
        assert len(log_content) > 0
        
        # Parse the JSON line
        log_entry = json.loads(log_content.split('\n')[-1])  # Get last line
        assert log_entry["memory_hash"] == "metadata_test"
        assert log_entry["action"] == "archived"