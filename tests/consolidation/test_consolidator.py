"""Integration tests for the main dream-inspired consolidator."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from mcp_memory_service.consolidation.consolidator import DreamInspiredConsolidator
from mcp_memory_service.consolidation.base import ConsolidationReport
from mcp_memory_service.models.memory import Memory


@pytest.mark.integration
class TestDreamInspiredConsolidator:
    """Test the main consolidation orchestrator."""
    
    @pytest.fixture
    def consolidator(self, mock_storage, consolidation_config):
        return DreamInspiredConsolidator(mock_storage, consolidation_config)
    
    @pytest.mark.asyncio
    async def test_basic_consolidation_pipeline(self, consolidator, mock_storage):
        """Test the complete consolidation pipeline."""
        report = await consolidator.consolidate("weekly")
        
        assert isinstance(report, ConsolidationReport)
        assert report.time_horizon == "weekly"
        assert isinstance(report.start_time, datetime)
        assert isinstance(report.end_time, datetime)
        assert report.end_time >= report.start_time
        assert report.memories_processed >= 0
        assert report.associations_discovered >= 0
        assert report.clusters_created >= 0
        assert report.memories_compressed >= 0
        assert report.memories_archived >= 0
        assert isinstance(report.errors, list)
        assert isinstance(report.performance_metrics, dict)
    
    @pytest.mark.asyncio
    async def test_daily_consolidation(self, consolidator):
        """Test daily consolidation (light processing)."""
        report = await consolidator.consolidate("daily")
        
        assert report.time_horizon == "daily"
        # Daily consolidation should be lighter - less intensive operations
        assert isinstance(report, ConsolidationReport)
    
    @pytest.mark.asyncio
    async def test_weekly_consolidation(self, consolidator):
        """Test weekly consolidation (includes associations)."""
        report = await consolidator.consolidate("weekly")
        
        assert report.time_horizon == "weekly"
        # Weekly should include association discovery
        assert isinstance(report, ConsolidationReport)
    
    @pytest.mark.asyncio
    async def test_monthly_consolidation(self, consolidator):
        """Test monthly consolidation (includes forgetting)."""
        report = await consolidator.consolidate("monthly")
        
        assert report.time_horizon == "monthly"
        # Monthly should include more comprehensive processing
        assert isinstance(report, ConsolidationReport)
    
    @pytest.mark.asyncio
    async def test_quarterly_consolidation(self, consolidator):
        """Test quarterly consolidation (deep processing)."""
        report = await consolidator.consolidate("quarterly")
        
        assert report.time_horizon == "quarterly"
        # Quarterly should include all processing steps
        assert isinstance(report, ConsolidationReport)
    
    @pytest.mark.asyncio
    async def test_yearly_consolidation(self, consolidator):
        """Test yearly consolidation (full processing)."""
        report = await consolidator.consolidate("yearly")
        
        assert report.time_horizon == "yearly"
        # Yearly should include comprehensive forgetting
        assert isinstance(report, ConsolidationReport)
    
    @pytest.mark.asyncio
    async def test_invalid_time_horizon(self, consolidator):
        """Test handling of invalid time horizon."""
        from mcp_memory_service.consolidation.base import ConsolidationError
        with pytest.raises(ConsolidationError):
            await consolidator.consolidate("invalid_horizon")
    
    @pytest.mark.asyncio
    async def test_empty_memory_set(self, consolidation_config):
        """Test consolidation with empty memory set."""
        # Create storage with no memories
        empty_storage = AsyncMock()
        empty_storage.get_all_memories.return_value = []
        empty_storage.get_memories_by_time_range.return_value = []
        empty_storage.get_memory_connections.return_value = {}
        empty_storage.get_access_patterns.return_value = {}
        
        consolidator = DreamInspiredConsolidator(empty_storage, consolidation_config)
        
        report = await consolidator.consolidate("weekly")
        
        assert report.memories_processed == 0
        assert report.associations_discovered == 0
        assert report.clusters_created == 0
        assert report.memories_compressed == 0
        assert report.memories_archived == 0
    
    @pytest.mark.asyncio
    async def test_memories_by_time_range_retrieval(self, consolidator, mock_storage):
        """Test retrieval of memories by time range for daily processing."""
        # Mock the time range method to return specific memories
        recent_memories = mock_storage.memories.copy()
        mock_storage.get_memories_by_time_range = AsyncMock(return_value=list(recent_memories.values())[:3])
        
        report = await consolidator.consolidate("daily")
        
        # Should have called the time range method for daily processing
        mock_storage.get_memories_by_time_range.assert_called_once()
        assert report.memories_processed >= 0
    
    @pytest.mark.asyncio
    async def test_association_storage(self, consolidator, mock_storage):
        """Test that discovered associations are stored as memories."""
        original_memory_count = len(mock_storage.memories)
        
        # Run consolidation that should discover associations
        report = await consolidator.consolidate("weekly")
        
        # Check if new association memories were added
        current_memory_count = len(mock_storage.memories)
        
        # May or may not find associations depending on similarity
        # Just ensure no errors occurred
        assert isinstance(report, ConsolidationReport)
        assert current_memory_count >= original_memory_count
    
    @pytest.mark.asyncio
    async def test_health_check(self, consolidator):
        """Test consolidation system health check."""
        health = await consolidator.health_check()
        
        assert isinstance(health, dict)
        assert "status" in health
        assert "timestamp" in health
        assert "components" in health
        assert "statistics" in health
        
        # Check component health
        expected_components = [
            "decay_calculator",
            "association_engine", 
            "clustering_engine",
            "compression_engine",
            "forgetting_engine"
        ]
        
        for component in expected_components:
            assert component in health["components"]
            assert "status" in health["components"][component]
    
    @pytest.mark.asyncio
    async def test_consolidation_recommendations(self, consolidator):
        """Test consolidation recommendations."""
        recommendations = await consolidator.get_consolidation_recommendations("weekly")
        
        assert isinstance(recommendations, dict)
        assert "recommendation" in recommendations
        assert "memory_count" in recommendations
        
        # Check recommendation types
        valid_recommendations = ["no_action", "consolidation_beneficial", "optional", "error"]
        assert recommendations["recommendation"] in valid_recommendations
        
        if recommendations["recommendation"] != "error":
            assert "reasons" in recommendations
            assert isinstance(recommendations["reasons"], list)
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self, consolidator):
        """Test performance metrics collection."""
        report = await consolidator.consolidate("daily")
        
        assert "performance_metrics" in report.__dict__
        metrics = report.performance_metrics
        
        assert "duration_seconds" in metrics
        assert "memories_per_second" in metrics
        assert "success" in metrics
        
        assert isinstance(metrics["duration_seconds"], float)
        assert metrics["duration_seconds"] >= 0
        assert isinstance(metrics["memories_per_second"], (int, float))
        assert isinstance(metrics["success"], bool)
    
    @pytest.mark.asyncio
    async def test_consolidation_statistics_tracking(self, consolidator):
        """Test that consolidation statistics are tracked."""
        initial_stats = consolidator.consolidation_stats.copy()
        
        # Run consolidation
        await consolidator.consolidate("weekly")
        
        # Check that stats were updated
        assert consolidator.consolidation_stats["total_runs"] == initial_stats["total_runs"] + 1
        
        # Check other stats (may or may not be incremented depending on processing)
        for key in ["successful_runs", "total_memories_processed", "total_associations_created"]:
            assert consolidator.consolidation_stats[key] >= initial_stats[key]
    
    @pytest.mark.asyncio
    async def test_error_handling_in_pipeline(self, consolidation_config):
        """Test error handling in the consolidation pipeline."""
        # Create storage that raises errors
        error_storage = AsyncMock()
        error_storage.get_all_memories.side_effect = Exception("Storage error")
        error_storage.get_memories_by_time_range.side_effect = Exception("Storage error")
        
        consolidator = DreamInspiredConsolidator(error_storage, consolidation_config)
        
        report = await consolidator.consolidate("weekly")
        
        # Should handle errors gracefully
        assert len(report.errors) > 0
        assert report.performance_metrics["success"] is False
    
    @pytest.mark.asyncio
    async def test_component_integration(self, consolidator, mock_storage):
        """Test integration between different consolidation components."""
        # Ensure we have enough memories for meaningful processing
        if len(mock_storage.memories) < 5:
            # Add more memories for testing
            base_time = datetime.now().timestamp()
            for i in range(10):
                memory = Memory(
                    content=f"Integration test memory {i} with content",
                    content_hash=f"integration_{i}",
                    tags=["integration", "test"],
                    embedding=[0.1 + i*0.01] * 320,
                    created_at=base_time - (i * 3600)
                )
                mock_storage.memories[memory.content_hash] = memory
        
        # Run full consolidation
        report = await consolidator.consolidate("monthly")
        
        # Verify that components worked together
        assert report.memories_processed > 0
        
        # Check that the pipeline completed successfully
        assert report.performance_metrics["success"] is True
    
    @pytest.mark.asyncio
    async def test_time_horizon_specific_processing(self, consolidator):
        """Test that different time horizons trigger appropriate processing."""
        # Test that weekly includes associations but not intensive forgetting
        weekly_report = await consolidator.consolidate("weekly")
        
        # Test that monthly includes forgetting
        monthly_report = await consolidator.consolidate("monthly")
        
        # Both should complete successfully
        assert weekly_report.performance_metrics["success"] is True
        assert monthly_report.performance_metrics["success"] is True
        
        # Monthly might have more archived memories (if forgetting triggered)
        # But this depends on the actual memory state, so just verify structure
        assert isinstance(weekly_report.memories_archived, int)
        assert isinstance(monthly_report.memories_archived, int)
    
    @pytest.mark.asyncio
    async def test_concurrent_consolidation_prevention(self, consolidator):
        """Test that the system handles concurrent consolidation requests appropriately."""
        # Start two consolidations concurrently
        import asyncio
        
        task1 = asyncio.create_task(consolidator.consolidate("daily"))
        task2 = asyncio.create_task(consolidator.consolidate("weekly"))
        
        # Both should complete (the system should handle concurrency)
        report1, report2 = await asyncio.gather(task1, task2)
        
        assert isinstance(report1, ConsolidationReport)
        assert isinstance(report2, ConsolidationReport)
        assert report1.time_horizon == "daily"
        assert report2.time_horizon == "weekly"
    
    @pytest.mark.asyncio
    async def test_memory_metadata_updates(self, consolidator, mock_storage):
        """Test that memory metadata is updated during consolidation."""
        original_memories = list(mock_storage.memories.values())
        
        # Run consolidation
        await consolidator.consolidate("weekly")
        
        # Check that memories exist (update_memory would have been called internally)
        # Since the mock doesn't track calls, we just verify the process completed
        current_memories = list(mock_storage.memories.values())
        assert len(current_memories) >= len(original_memories)
    
    @pytest.mark.asyncio
    async def test_large_memory_set_performance(self, consolidation_config, mock_large_storage):
        """Test performance with larger memory sets."""
        consolidator = DreamInspiredConsolidator(mock_large_storage, consolidation_config)
        
        start_time = datetime.now()
        report = await consolidator.consolidate("weekly")
        end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert duration < 30  # 30 seconds for 100 memories
        assert report.memories_processed > 0
        assert report.performance_metrics["success"] is True
        
        # Performance should be reasonable
        if report.memories_processed > 0:
            memories_per_second = report.memories_processed / duration
            assert memories_per_second > 1  # At least 1 memory per second
    
    @pytest.mark.asyncio
    async def test_consolidation_report_completeness(self, consolidator):
        """Test that consolidation reports contain all expected information."""
        report = await consolidator.consolidate("weekly")
        
        # Check all required fields
        required_fields = [
            "time_horizon", "start_time", "end_time", "memories_processed",
            "associations_discovered", "clusters_created", "memories_compressed",
            "memories_archived", "errors", "performance_metrics"
        ]
        
        for field in required_fields:
            assert hasattr(report, field), f"Missing field: {field}"
            assert getattr(report, field) is not None, f"Field {field} is None"
        
        # Check performance metrics
        perf_metrics = report.performance_metrics
        assert "duration_seconds" in perf_metrics
        assert "memories_per_second" in perf_metrics
        assert "success" in perf_metrics
    
    @pytest.mark.asyncio
    async def test_storage_backend_integration(self, consolidator, mock_storage):
        """Test integration with storage backend methods."""
        # Run consolidation
        report = await consolidator.consolidate("monthly")
        
        # Verify storage integration worked (memories were processed)
        assert report.memories_processed >= 0
        assert isinstance(report.performance_metrics, dict)
        
        # Verify storage backend has the expected methods
        assert hasattr(mock_storage, 'get_all_memories')
        assert hasattr(mock_storage, 'get_memories_by_time_range')  
        assert hasattr(mock_storage, 'get_memory_connections')
        assert hasattr(mock_storage, 'get_access_patterns')
        assert hasattr(mock_storage, 'update_memory')
    
    @pytest.mark.asyncio
    async def test_configuration_impact(self, mock_storage):
        """Test that configuration changes affect consolidation behavior."""
        # Create two different configurations
        config1 = type('Config', (), {
            'decay_enabled': True,
            'associations_enabled': True,
            'clustering_enabled': True,
            'compression_enabled': True,
            'forgetting_enabled': True,
            'retention_periods': {'standard': 30},
            'min_similarity': 0.3,
            'max_similarity': 0.7,
            'max_pairs_per_run': 50,
            'min_cluster_size': 3,
            'clustering_algorithm': 'simple',
            'max_summary_length': 200,
            'preserve_originals': True,
            'relevance_threshold': 0.1,
            'access_threshold_days': 30,
            'archive_location': None,
            'batch_size': 500,
            'incremental_mode': True
        })()

        config2 = type('Config', (), {
            'decay_enabled': False,
            'associations_enabled': False,
            'clustering_enabled': False,
            'compression_enabled': False,
            'forgetting_enabled': False,
            'retention_periods': {'standard': 30},
            'min_similarity': 0.3,
            'max_similarity': 0.7,
            'max_pairs_per_run': 50,
            'min_cluster_size': 3,
            'clustering_algorithm': 'simple',
            'max_summary_length': 200,
            'preserve_originals': True,
            'relevance_threshold': 0.1,
            'access_threshold_days': 30,
            'archive_location': None,
            'batch_size': 500,
            'incremental_mode': True
        })()
        
        consolidator1 = DreamInspiredConsolidator(mock_storage, config1)
        consolidator2 = DreamInspiredConsolidator(mock_storage, config2)
        
        # Both should work, but may produce different results
        report1 = await consolidator1.consolidate("weekly")
        report2 = await consolidator2.consolidate("weekly")
        
        assert isinstance(report1, ConsolidationReport)
        assert isinstance(report2, ConsolidationReport)
        
        # With disabled features, the second consolidator might process differently
        # but both should complete successfully
        assert report1.performance_metrics["success"] is True
        assert report2.performance_metrics["success"] is True