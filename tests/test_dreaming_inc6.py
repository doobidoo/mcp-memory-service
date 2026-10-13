"""
Test suite for Dreaming INC6: importance-triggered consolidation.

RED tests for the importance-based consolidation trigger complementing cron jobs.
Tests the contract from spec-dreaming-inc6-importance-trigger.md.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from types import SimpleNamespace

from mcp_memory_service.consolidation.scheduler import ConsolidationScheduler
from mcp_memory_service.consolidation.consolidator import DreamInspiredConsolidator


class TestImportanceTriggeredConsolidation:
    """Test importance-based consolidation triggering (INC6)."""
    
    def test_importance_accumulator_tracks_by_theme(self):
        """Test that importance accumulates separately per theme."""
        consolidator = MagicMock(spec=DreamInspiredConsolidator)
        
        with patch.dict('os.environ', {'MCP_IMPORTANCE_TRIGGER_ENABLED': 'true'}):
            scheduler = ConsolidationScheduler(
                consolidator=consolidator,
                schedule_config={"daily": "disabled"},
                enabled=True
            )
        
            # Should have importance tracking per theme
            assert hasattr(scheduler, 'theme_importance'), "Scheduler should track importance per theme"
            
            # Simulate usage events for different themes
            scheduler.record_theme_usage("python", importance=0.3)
            scheduler.record_theme_usage("kubernetes", importance=0.2) 
            scheduler.record_theme_usage("python", importance=0.4)
            
            # Each theme should accumulate independently
            assert scheduler.theme_importance["python"] == pytest.approx(0.7)
            assert scheduler.theme_importance["kubernetes"] == pytest.approx(0.2)

    def test_threshold_trigger_calls_memory_consolidate_with_theme(self):
        """Test that crossing threshold triggers memory_consolidate(theme=X)."""
        consolidator = MagicMock(spec=DreamInspiredConsolidator)
        consolidator.consolidate = AsyncMock()
        
        with patch.dict('os.environ', {'MCP_IMPORTANCE_TRIGGER_ENABLED': 'true'}):
            scheduler = ConsolidationScheduler(
                consolidator=consolidator,
                schedule_config={"daily": "disabled"},
                enabled=True
            )
            
            # Configure threshold  
            threshold = 0.5
            scheduler.importance_threshold = threshold
            
            # Accumulate importance below threshold - no trigger
            scheduler.record_theme_usage("python", importance=0.3)
            consolidator.consolidate.assert_not_called()
            
            # Cross threshold - should trigger consolidation for that theme
            scheduler.record_theme_usage("python", importance=0.3)  # Total: 0.6 > 0.5
            
            # Should call memory_consolidate with theme parameter
            consolidator.consolidate.assert_called_once_with(theme="python")

    def test_cooldown_prevents_repeated_triggers(self):
        """Test that cooldown prevents excessive triggering for same theme."""
        consolidator = MagicMock(spec=DreamInspiredConsolidator) 
        consolidator.consolidate = AsyncMock()
        
        with patch.dict('os.environ', {'MCP_IMPORTANCE_TRIGGER_ENABLED': 'true'}):
            scheduler = ConsolidationScheduler(
                consolidator=consolidator,
                schedule_config={"daily": "disabled"},
                enabled=True
            )
            
            scheduler.importance_threshold = 0.5
            scheduler.cooldown_minutes = 30
            
            # First trigger should work
            scheduler.record_theme_usage("python", importance=0.6)
            assert consolidator.consolidate.call_count == 1
            
            # Immediate second trigger should be blocked by cooldown
            scheduler.record_theme_usage("python", importance=0.6)
            assert consolidator.consolidate.call_count == 1, "Cooldown should prevent repeated triggers"
            
            # After cooldown expires, should trigger again
            with patch('mcp_memory_service.consolidation.scheduler.datetime') as mock_dt:
                mock_dt.now.return_value = datetime.now() + timedelta(minutes=31)
                scheduler.record_theme_usage("python", importance=0.6)
                assert consolidator.consolidate.call_count == 2

    def test_importance_derived_from_usage_events_no_llm(self):
        """Test that importance calculation uses existing usage_events, not LLM."""
        consolidator = MagicMock(spec=DreamInspiredConsolidator)
        
        with patch.dict('os.environ', {'MCP_IMPORTANCE_TRIGGER_ENABLED': 'true'}):
            scheduler = ConsolidationScheduler(
                consolidator=consolidator,
                schedule_config={"daily": "disabled"}, 
                enabled=True
            )
            
            # Mock usage events from database
            mock_usage_events = [
                {"event_type": "search", "tool": "memory_search", "rating": 1, "timestamp": "2024-01-01T10:00:00"},
                {"event_type": "search", "tool": "memory_search", "rating": -1, "timestamp": "2024-01-01T11:00:00"},
                {"event_type": "store", "tool": "memory_store", "rating": 1, "timestamp": "2024-01-01T12:00:00"}
            ]
            
            # Should calculate importance from these signals, no LLM calls
            importance = scheduler.calculate_theme_importance("python", mock_usage_events)
            
            # Should be derived from frequency, rating, recency - not LLM
            assert isinstance(importance, (int, float)), "Importance should be numeric"
            assert importance > 0, "Should derive positive importance from usage signals"
            
            # Verify no LLM/embedding calls were made
            assert not hasattr(scheduler, 'llm_client'), "Should not use LLM for importance calculation"

    def test_opt_in_flag_controls_behavior(self):
        """Test that MCP_IMPORTANCE_TRIGGER_ENABLED controls the feature."""
        consolidator = MagicMock(spec=DreamInspiredConsolidator)
        
        # With flag OFF - should only use cron (current behavior)
        with patch.dict('os.environ', {'MCP_IMPORTANCE_TRIGGER_ENABLED': 'false'}):
            scheduler = ConsolidationScheduler(
                consolidator=consolidator,
                schedule_config={"daily": "disabled"},
                enabled=True
            )
            
            # Should not have importance tracking when disabled
            assert not hasattr(scheduler, 'record_theme_usage'), "Should not have importance tracking when disabled"
        
        # With flag ON - should have importance tracking
        with patch.dict('os.environ', {'MCP_IMPORTANCE_TRIGGER_ENABLED': 'true'}):
            scheduler = ConsolidationScheduler(
                consolidator=consolidator,
                schedule_config={"daily": "disabled"},
                enabled=True
            )
            
            # Should have importance tracking methods when enabled
            assert hasattr(scheduler, 'record_theme_usage'), "Should have importance tracking when enabled"
            assert hasattr(scheduler, 'theme_importance'), "Should track theme importance when enabled"

    def test_complementary_to_cron_not_replacement(self):
        """Test that importance triggering complements cron, doesn't replace it."""
        consolidator = MagicMock(spec=DreamInspiredConsolidator)
        
        with patch.dict('os.environ', {'MCP_IMPORTANCE_TRIGGER_ENABLED': 'true'}):
            scheduler = ConsolidationScheduler(
                consolidator=consolidator,
                schedule_config={"daily": "03:00"},  # Cron enabled
                enabled=True
            )
            
            # Jobs de cron são agendados por _schedule_consolidation_jobs (chamado em start(), que é async).
            # Chamar o agendador diretamente (síncrono) para verificar que o cron é agendado
            # mesmo com importance-trigger ligado (REQ-6.1: complemento, não substituto).
            if scheduler.scheduler:  # If APScheduler available
                scheduler._schedule_consolidation_jobs()
                cron_jobs = [job for job in scheduler.scheduler.get_jobs() if "consolidation_daily" in job.id]
                assert len(cron_jobs) > 0, "Should still have cron jobs when importance triggering is enabled"
            
            # Importance triggering should be ADDITIONAL
            scheduler.record_theme_usage("python", importance=0.6)
            
            # Both mechanisms should coexist
            assert scheduler.importance_threshold is not None, "Should have importance triggering"

    def test_no_embedding_calls_in_importance_calculation(self):
        """Test that importance calculation makes zero embedding calls."""
        consolidator = MagicMock(spec=DreamInspiredConsolidator)
        
        with patch.dict('os.environ', {'MCP_IMPORTANCE_TRIGGER_ENABLED': 'true'}):
            scheduler = ConsolidationScheduler(
                consolidator=consolidator,
                schedule_config={"daily": "disabled"},
                enabled=True
            )
            
            # Track embedding calls
            embedding_call_count = 0
            
            def mock_embedding(*args, **kwargs):
                nonlocal embedding_call_count
                embedding_call_count += 1
                return [0.1, 0.2, 0.3]  # Fake embedding
            
            # calculate_theme_importance deve derivar SÓ de usage_events (sem LLM/embedding).
            # Patch no ponto real de embedding do storage: se for chamado, o contador sobe.
            import mcp_memory_service.storage.sqlite_vec as sv
            with patch.object(sv.SqliteVecMemoryStorage, '_generate_embedding', side_effect=mock_embedding, create=True):
                mock_usage_events = [
                    {"event_type": "search", "rating": 1, "timestamp": "2024-01-01T10:00:00"}
                ]
                result = scheduler.calculate_theme_importance("python", mock_usage_events)

            assert embedding_call_count == 0, "Importance calculation must not call embedding/LLM"
            assert isinstance(result, (int, float)), "importance must be numeric from usage_events"

    def test_theme_convergence_requires_multiple_distinct_seeds(self):
        """Test that consolidation only triggers with ≥2 distinct query seeds per theme."""
        consolidator = MagicMock(spec=DreamInspiredConsolidator)
        consolidator.consolidate = AsyncMock()
        
        with patch.dict('os.environ', {'MCP_IMPORTANCE_TRIGGER_ENABLED': 'true'}):
            scheduler = ConsolidationScheduler(
                consolidator=consolidator,
                schedule_config={"daily": "disabled"},
                enabled=True
            )
            
            scheduler.importance_threshold = 0.5
            
            # Single seed - high importance but no trigger
            scheduler.record_theme_usage("python", importance=0.8, query_seed="async patterns")
            consolidator.consolidate.assert_not_called()
            
            # Second distinct seed - should trigger
            scheduler.record_theme_usage("python", importance=0.1, query_seed="error handling")  
            consolidator.consolidate.assert_called_once_with(theme="python")
            
            # Same seed repeated - should not count as convergence
            consolidator.consolidate.reset_mock()
            scheduler.record_theme_usage("java", importance=0.9, query_seed="spring boot")
            scheduler.record_theme_usage("java", importance=0.9, query_seed="spring boot")  # Same seed
            consolidator.consolidate.assert_not_called()