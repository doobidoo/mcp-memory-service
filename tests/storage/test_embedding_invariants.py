"""Tests for storage embedding invariants and missing embeddings detection (#1225)."""

import os
import shutil
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from mcp_memory_service.consolidation.health import (
    ConsolidationHealthMonitor,
    HealthStatus,
)
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash


@pytest.fixture
def temp_db():
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_invariants.db")
    yield db_path
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_missing_embeddings_in_get_stats(temp_db):
    """Test that get_stats accurately counts active memories missing an embedding."""
    storage = SqliteVecMemoryStorage(temp_db)
    await storage.initialize()

    # Store memory normally
    content = "Invariant test memory"
    mem = Memory(
        content=content,
        content_hash=generate_content_hash(content),
        tags=["test"],
        memory_type="note"
    )
    success, _ = await storage.store(mem)
    assert success is True

    # Check stats reports 0 missing embeddings
    stats = await storage.get_stats()
    assert stats["missing_embeddings"] == 0

    # Simulate dropped embedding row for an active memory
    storage.conn.execute("DELETE FROM memory_embeddings")
    storage.conn.commit()

    stats_after = await storage.get_stats()
    assert stats_after["missing_embeddings"] == 1

    # If memory is soft-deleted, it should NOT count as a missing embedding
    storage.conn.execute("UPDATE memories SET deleted_at = 123456789.0")
    storage.conn.commit()

    stats_deleted = await storage.get_stats()
    assert stats_deleted["missing_embeddings"] == 0

    await storage.close()


@pytest.mark.asyncio
async def test_consolidation_health_flags_missing_embeddings():
    """Test that ConsolidationHealthMonitor flags missing embeddings and degrades health."""
    mock_storage = MagicMock()
    mock_storage.get_stats = AsyncMock(return_value={
        "backend": "sqlite-vec",
        "total_memories": 10,
        "missing_embeddings": 2,
    })

    monitor = ConsolidationHealthMonitor(consolidator=MagicMock(storage=mock_storage))
    health = await monitor._check_storage_backend_health()

    assert health["status"] == HealthStatus.DEGRADED.value
    assert health["checks"]["missing_embeddings"] == 2
