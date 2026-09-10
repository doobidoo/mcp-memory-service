"""Tests for ConsolidationHealthMonitor with real storage backends.

These tests verify that the health monitor correctly reports status
when connected to a real sqlite-vec storage, rather than mocked stubs.
"""

import pytest
import pytest_asyncio
import tempfile
import os
from datetime import datetime, timedelta, timezone

from mcp_memory_service.consolidation.health import (
    ConsolidationHealthMonitor,
    HealthStatus,
)
from mcp_memory_service.consolidation.base import ConsolidationConfig
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage


@pytest_asyncio.fixture
async def real_storage():
    """Create a real SqliteVecMemoryStorage in a temp directory."""
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_health.db")

    storage = SqliteVecMemoryStorage(
        db_path=db_path,
        embedding_model="all-MiniLM-L6-v2",
    )
    await storage.initialize()

    yield storage

    try:
        storage.close()
    except Exception:
        pass
    try:
        os.remove(db_path)
        os.rmdir(temp_dir)
    except OSError:
        pass


@pytest.fixture
def health_config():
    """Create a ConsolidationConfig with real field values."""
    config = ConsolidationConfig()
    config.retention_periods = {"decision": 365, "observation": 180}
    config.min_similarity = 0.3
    config.max_similarity = 0.8
    return config


@pytest.mark.integration
class TestHealthMonitorRealStorage:
    """Test health monitor with real sqlite-vec storage."""

    @pytest.mark.asyncio
    async def test_storage_health_check_with_real_backend(self, real_storage, health_config):
        """Verify storage health check works with real sqlite-vec storage."""
        # Create a mock consolidator that exposes the real storage
        class MockConsolidator:
            storage = real_storage
            health_monitor = None

        consolidator = MockConsolidator()
        monitor = ConsolidationHealthMonitor(health_config, consolidator=consolidator)
        consolidator.health_monitor = monitor

        # Run storage health check
        result = await monitor._check_storage_backend_health()

        assert result['status'] == HealthStatus.HEALTHY.value
        assert result['checks']['storage_connection'] == 'connected'
        assert result['checks']['read_operations'] == 'functional'

    @pytest.mark.asyncio
    async def test_overall_health_with_real_storage(self, real_storage, health_config):
        """Verify overall health check passes with real storage."""
        class MockConsolidator:
            storage = real_storage
            health_monitor = None

        consolidator = MockConsolidator()
        monitor = ConsolidationHealthMonitor(health_config, consolidator=consolidator)
        consolidator.health_monitor = monitor

        health = await monitor.check_overall_health()

        # Should be healthy or degraded (not unhealthy)
        assert health['status'] in ('healthy', 'degraded')
        assert 'components' in health
        assert 'storage_backend' in health['components']

    @pytest.mark.asyncio
    async def test_scheduler_health_without_scheduler(self, health_config):
        """Verify scheduler health reports degraded when no scheduler attached."""
        monitor = ConsolidationHealthMonitor(health_config, consolidator=None)

        result = await monitor._check_scheduler_health()

        # Without a scheduler, should report degraded or disabled
        assert result['status'] in (HealthStatus.DEGRADED.value, HealthStatus.UNHEALTHY.value)

    @pytest.mark.asyncio
    async def test_retention_periods_validation(self, health_config):
        """Verify retention_periods validation catches invalid configs."""
        # Valid config
        monitor = ConsolidationHealthMonitor(health_config, consolidator=None)
        result = await monitor._check_decay_calculator_health()
        assert result['status'] == HealthStatus.HEALTHY.value

        # Invalid: negative retention period
        bad_config = ConsolidationConfig()
        bad_config.retention_periods = {"decision": -365}
        monitor2 = ConsolidationHealthMonitor(bad_config, consolidator=None)
        result2 = await monitor2._check_decay_calculator_health()
        assert result2['status'] == HealthStatus.DEGRADED.value

    @pytest.mark.asyncio
    async def test_similarity_range_validation(self, health_config):
        """Verify similarity range validation catches invalid configs."""
        # Valid config (0.3 < 0.8, both in [0,1])
        monitor = ConsolidationHealthMonitor(health_config, consolidator=None)
        result = await monitor._check_association_engine_health()
        assert result['status'] == HealthStatus.HEALTHY.value

        # Invalid: min > max
        bad_config = ConsolidationConfig()
        bad_config.min_similarity = 0.8
        bad_config.max_similarity = 0.3
        monitor2 = ConsolidationHealthMonitor(bad_config, consolidator=None)
        result2 = await monitor2._check_association_engine_health()
        assert result2['status'] == HealthStatus.DEGRADED.value

    @pytest.mark.asyncio
    async def test_closed_storage_reports_unhealthy(self, real_storage, health_config):
        """After close(), get_stats() returns an error dict — health must be UNHEALTHY."""
        from unittest.mock import AsyncMock, MagicMock

        # Wrap real storage in a mock consolidator
        consolidator = MagicMock()
        consolidator.storage = real_storage

        monitor = ConsolidationHealthMonitor(health_config, consolidator=consolidator)

        # Sanity: healthy before close
        result_before = await monitor._check_storage_backend_health()
        assert result_before['status'] == HealthStatus.HEALTHY.value

        # Close the connection (close is async)
        await real_storage.close()

        # get_stats() should now return an error dict
        result_after = await monitor._check_storage_backend_health()
        assert result_after['status'] == HealthStatus.UNHEALTHY.value
        assert result_after['checks']['read_operations'] == 'failing'