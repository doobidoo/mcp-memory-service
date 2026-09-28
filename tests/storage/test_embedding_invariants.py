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
from mcp_memory_service.web.api.health import detailed_health_check


@pytest.fixture
def temp_db():
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_invariants.db")
    yield db_path
    shutil.rmtree(temp_dir, ignore_errors=True)


class _FailingEmbeddingCheckConnProxy:
    """Proxy for sqlite3.Connection that forces the invariant SELECT check to return None."""

    def __init__(self, real_conn):
        self._conn = real_conn

    def execute(self, sql, *args, **kwargs):
        if isinstance(sql, str) and "SELECT 1 FROM memory_embeddings WHERE rowid" in sql:
            mock_cursor = MagicMock()
            mock_cursor.fetchone.return_value = None
            return mock_cursor
        return self._conn.execute(sql, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._conn, name)


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


@pytest.mark.asyncio
async def test_consolidation_health_flags_hybrid_primary_missing_embeddings():
    """Test that ConsolidationHealthMonitor detects missing embeddings nested inside primary_stats."""
    mock_storage = MagicMock()
    mock_storage.get_stats = AsyncMock(return_value={
        "backend": "hybrid",
        "primary_stats": {
            "missing_embeddings": 5,
        }
    })

    monitor = ConsolidationHealthMonitor(consolidator=MagicMock(storage=mock_storage))
    health = await monitor._check_storage_backend_health()

    assert health["status"] == HealthStatus.DEGRADED.value
    assert health["checks"]["missing_embeddings"] == 5


@pytest.mark.asyncio
async def test_hybrid_storage_get_stats_propagates_missing_embeddings():
    """Test that HybridMemoryStorage propagates missing_embeddings from primary stats."""
    from mcp_memory_service.storage.hybrid import HybridMemoryStorage
    mock_primary = MagicMock()
    mock_primary.get_stats = AsyncMock(return_value={
        "total_memories": 10,
        "missing_embeddings": 4,
        "unique_tags": 2,
    })
    hybrid = object.__new__(HybridMemoryStorage)
    hybrid.primary = mock_primary
    hybrid.secondary = None
    hybrid.sync_service = None

    stats = await hybrid.get_stats()
    assert stats["missing_embeddings"] == 4
    assert stats["primary_stats"]["missing_embeddings"] == 4


@pytest.mark.asyncio
async def test_detailed_health_flags_hybrid_primary_missing_embeddings():
    """Test that /api/health/detailed flags missing embeddings from hybrid primary_stats."""
    mock_storage = MagicMock()
    mock_storage.get_stats = AsyncMock(return_value={
        "storage_backend": "Hybrid (SQLite-vec + Cloudflare)",
        "primary_backend": "SQLite-vec",
        "total_memories": 10,
        "primary_stats": {
            "total_memories": 10,
            "missing_embeddings": 3,
        }
    })
    mock_user = MagicMock()
    res = await detailed_health_check(storage=mock_storage, user=mock_user)
    assert res.status == "degraded"
    assert res.statistics["missing_embeddings"] == 3


@pytest.mark.asyncio
async def test_store_rollback_on_missing_embedding_invariant(temp_db):
    """Test that store() rolls back memory row if embedding row is missing/fails invariant."""
    storage = SqliteVecMemoryStorage(temp_db)
    await storage.initialize()

    content = "Rollback test memory"
    mem = Memory(
        content=content,
        content_hash=generate_content_hash(content),
        tags=["rollback"],
        memory_type="note"
    )

    real_conn = storage.conn
    storage.conn = _FailingEmbeddingCheckConnProxy(real_conn)

    success, message = await storage.store(mem)
    assert success is False
    assert "Invariant violation" in message or "failed" in message.lower()

    # Restore real conn and verify atomic rollback: memory must NOT exist in memories table
    storage.conn = real_conn
    cursor = storage.conn.execute("SELECT COUNT(*) FROM memories WHERE content_hash = ?", (mem.content_hash,))
    assert cursor.fetchone()[0] == 0

    await storage.close()


@pytest.mark.asyncio
async def test_store_batch_rollback_on_missing_embedding_invariant(temp_db):
    """Test that store_batch() rolls back individual items when embedding invariant fails."""
    storage = SqliteVecMemoryStorage(temp_db)
    await storage.initialize()

    content1 = "Batch rollback mem 1"
    mem1 = Memory(
        content=content1,
        content_hash=generate_content_hash(content1),
        tags=["batch1"],
        memory_type="note"
    )
    content2 = "Batch rollback mem 2"
    mem2 = Memory(
        content=content2,
        content_hash=generate_content_hash(content2),
        tags=["batch2"],
        memory_type="note"
    )

    real_conn = storage.conn
    storage.conn = _FailingEmbeddingCheckConnProxy(real_conn)

    results = await storage.store_batch([mem1, mem2])
    assert len(results) == 2
    assert results[0][0] is False
    assert results[1][0] is False
    assert "Invariant violation" in results[0][1] or "Insert failed" in results[0][1]

    # Restore real conn and verify atomic rollback: neither memory was persisted in memories table
    storage.conn = real_conn
    cursor = storage.conn.execute("SELECT COUNT(*) FROM memories")
    assert cursor.fetchone()[0] == 0

    await storage.close()
