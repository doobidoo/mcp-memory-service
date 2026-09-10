"""Backend health probes report observed state, not optimistic defaults."""

from unittest.mock import AsyncMock

import pytest

from mcp_memory_service.storage.cloudflare import CloudflareStorage
from mcp_memory_service.storage.hybrid import HybridMemoryStorage
from mcp_memory_service.storage.milvus import MilvusMemoryStorage


@pytest.mark.asyncio
async def test_milvus_health_probe_uses_client_operation():
    """Milvus probe fails closed on initialization or client failure."""
    storage = MilvusMemoryStorage.__new__(MilvusMemoryStorage)
    storage._ensure_initialized = lambda: True
    storage.collection_name = "memories"
    storage._call_client = AsyncMock(return_value=True)
    assert await storage.health_probe() is True
    storage._call_client.assert_awaited_once()
    storage._call_client = AsyncMock(return_value=False)
    assert await storage.health_probe() is False
    storage._call_client = AsyncMock(side_effect=RuntimeError("offline"))
    assert await storage.health_probe() is False
    storage._ensure_initialized = lambda: False
    assert await storage.health_probe() is False


@pytest.mark.asyncio
async def test_hybrid_health_probe_reflects_primary_failure():
    """Hybrid does not report healthy when its primary probe fails."""
    storage = HybridMemoryStorage.__new__(HybridMemoryStorage)
    storage.primary = type(
        "Primary", (), {"health_probe": AsyncMock(return_value=True)}
    )()
    assert await storage.health_probe() is True
    storage.primary.health_probe.assert_awaited_once()
    storage.primary.health_probe = AsyncMock(return_value=False)
    assert await storage.health_probe() is False
    storage.primary.health_probe = AsyncMock(side_effect=RuntimeError("offline"))
    assert await storage.health_probe() is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"success": False},
        {"success": True, "result": [{}]},
        {"success": True, "result": [{"results": []}]},
    ],
)
async def test_cloudflare_stats_failed_or_empty_d1_is_error(payload):
    """Failed or empty D1 stats cannot become operational zeros."""
    storage = CloudflareStorage.__new__(CloudflareStorage)
    storage.d1_url = "https://d1.example"
    storage.vectorize_index = "index"
    storage.d1_database_id = "db"
    storage.r2_bucket = "bucket"
    storage._retry_request = AsyncMock(
        return_value=type("Response", (), {"json": lambda self: payload})()
    )
    stats = await storage.get_stats()
    assert stats["status"] == "error"


@pytest.mark.asyncio
async def test_cloudflare_stats_preserves_successful_zero_counts():
    """A successful D1 aggregate row with zeros is operational, not an error."""
    storage = CloudflareStorage.__new__(CloudflareStorage)
    storage.d1_url = "https://d1.example"
    storage.vectorize_index = "index"
    storage.d1_database_id = "db"
    storage.r2_bucket = "bucket"
    storage._retry_request = AsyncMock(
        return_value=type(
            "Response",
            (),
            {
                "json": lambda self: {
                    "success": True,
                    "result": [
                        {
                            "results": [
                                {
                                    "total_memories": 0,
                                    "unique_tags": 0,
                                    "memories_this_week": 0,
                                    "total_content_size": 0,
                                    "total_vectors": 0,
                                    "tombstone_count": 0,
                                    "r2_stored_count": 0,
                                }
                            ]
                        }
                    ],
                }
            },
        )()
    )

    stats = await storage.get_stats()

    assert stats["status"] == "operational"
    assert stats["total_memories"] == 0
    assert stats["total_vectors"] == 0
