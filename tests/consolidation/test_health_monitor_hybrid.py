"""Regression coverage for hybrid primary storage health."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mcp_memory_service.consolidation.base import ConsolidationConfig
from mcp_memory_service.consolidation.health import ConsolidationHealthMonitor, HealthStatus
from mcp_memory_service.storage.hybrid import HybridMemoryStorage


@pytest.mark.asyncio
async def test_hybrid_health_detects_closed_primary(tmp_path):
    """A working hybrid is healthy; a closed SQLite primary is unhealthy."""
    storage = HybridMemoryStorage(str(tmp_path / "hybrid.db"), cloudflare_config={})
    await storage.initialize()
    try:
        monitor = ConsolidationHealthMonitor(
            ConsolidationConfig(), consolidator=SimpleNamespace(storage=storage)
        )
        healthy = await monitor._check_storage_backend_health()
        assert healthy["status"] == HealthStatus.HEALTHY.value
        assert healthy["checks"]["storage_connection"] == "connected"

        await storage.primary.close()
        stats = await storage.get_stats()
        assert "error" in stats["primary_stats"]
        unhealthy = await monitor._check_storage_backend_health()
        assert unhealthy["status"] == HealthStatus.UNHEALTHY.value
        assert unhealthy["checks"]["storage_connection"] == "error"
        assert unhealthy["checks"]["read_operations"] == "failing"
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("stats", [
    {"primary_stats": {"status": "error"}},
    {"error": "unavailable", "primary_stats": {"status": "healthy"}},
])
async def test_hybrid_health_detects_error_status(stats):
    """Both primary status errors and top-level failures stay unhealthy."""
    storage = SimpleNamespace(get_stats=AsyncMock(return_value=stats))
    monitor = ConsolidationHealthMonitor(
        ConsolidationConfig(), consolidator=SimpleNamespace(storage=storage)
    )
    health = await monitor._check_storage_backend_health()
    assert health["status"] == HealthStatus.UNHEALTHY.value
    assert health["checks"]["storage_connection"] == "error"
