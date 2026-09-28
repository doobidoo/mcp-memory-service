"""Tests for embedding integrity detection and health status degradation (Refs #1225)."""

import os
import shutil
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_memory_service.consolidation.health import (
    ConsolidationHealthMonitor,
    HealthStatus,
)
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash
from mcp_memory_service.utils.health_check import _check_embedding_integrity
from mcp_memory_service.web.api.health import detailed_health_check


@pytest.fixture
def temp_db():
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_invariants.db")
    yield db_path
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_embedding_integrity_detection(temp_db):
    """Test that _check_embedding_integrity accurately detects missing embeddings."""
    storage = SqliteVecMemoryStorage(temp_db)
    await storage.initialize()

    # Store memory normally
    content = "Integrity test memory"
    mem = Memory(
        content=content,
        content_hash=generate_content_hash(content),
        tags=["test"],
        memory_type="note"
    )
    success, _ = await storage.store(mem)
    assert success is True

    # 0 missing embeddings initially
    integrity = _check_embedding_integrity(storage.conn)
    assert integrity["missing_embeddings"] == 0

    # Simulate dropped embedding row for active memory
    storage.conn.execute("DELETE FROM memory_embeddings")
    storage.conn.commit()

    integrity_after = _check_embedding_integrity(storage.conn)
    assert integrity_after["missing_embeddings"] == 1

    # Soft-deleted memories should not be counted as missing embeddings
    storage.conn.execute("UPDATE memories SET deleted_at = 123456789.0")
    storage.conn.commit()

    integrity_deleted = _check_embedding_integrity(storage.conn)
    assert integrity_deleted["missing_embeddings"] == 0

    await storage.close()


@pytest.mark.asyncio
async def test_consolidation_health_flags_missing_embeddings(temp_db):
    """Test that ConsolidationHealthMonitor detects missing embeddings via storage.conn."""
    storage = SqliteVecMemoryStorage(temp_db)
    await storage.initialize()

    content = "Consolidation integrity memory"
    mem = Memory(
        content=content,
        content_hash=generate_content_hash(content),
        tags=["test"],
        memory_type="note"
    )
    await storage.store(mem)

    # Delete embedding to create gap
    storage.conn.execute("DELETE FROM memory_embeddings")
    storage.conn.commit()

    monitor = ConsolidationHealthMonitor(consolidator=MagicMock(storage=storage))
    health = await monitor._check_storage_backend_health()

    assert health["status"] == HealthStatus.DEGRADED.value
    assert health["checks"]["missing_embeddings"] == 1

    await storage.close()


@pytest.mark.asyncio
async def test_consolidation_health_flags_hybrid_primary_missing_embeddings(temp_db):
    """Test that ConsolidationHealthMonitor detects missing embeddings via hybrid primary.conn."""
    primary_storage = SqliteVecMemoryStorage(temp_db)
    await primary_storage.initialize()

    content = "Hybrid integrity memory"
    mem = Memory(
        content=content,
        content_hash=generate_content_hash(content),
        tags=["test"],
        memory_type="note"
    )
    await primary_storage.store(mem)

    # Delete embedding to create gap
    primary_storage.conn.execute("DELETE FROM memory_embeddings")
    primary_storage.conn.commit()

    mock_hybrid = MagicMock()
    mock_hybrid.get_stats = AsyncMock(return_value={"backend": "hybrid", "total_memories": 1})
    mock_hybrid.conn = None
    mock_hybrid.primary = primary_storage

    monitor = ConsolidationHealthMonitor(consolidator=MagicMock(storage=mock_hybrid))
    health = await monitor._check_storage_backend_health()

    assert health["status"] == HealthStatus.DEGRADED.value
    assert health["checks"]["missing_embeddings"] == 1

    await primary_storage.close()


@pytest.mark.asyncio
async def test_detailed_health_flags_missing_embeddings(temp_db):
    """Test that /api/health/detailed detects missing embeddings from storage.conn."""
    storage = SqliteVecMemoryStorage(temp_db)
    await storage.initialize()

    content = "Detailed health memory"
    mem = Memory(
        content=content,
        content_hash=generate_content_hash(content),
        tags=["test"],
        memory_type="note"
    )
    await storage.store(mem)

    # Delete embedding to create gap
    storage.conn.execute("DELETE FROM memory_embeddings")
    storage.conn.commit()

    mock_user = MagicMock()
    res = await detailed_health_check(storage=storage, user=mock_user)

    assert res.status == "degraded"
    assert res.statistics["missing_embeddings"] == 1
    assert res.storage["missing_embeddings"] == 1

    await storage.close()


@pytest.mark.asyncio
async def test_detailed_health_flags_hybrid_primary_missing_embeddings(temp_db):
    """Test that /api/health/detailed detects missing embeddings from hybrid primary.conn."""
    primary_storage = SqliteVecMemoryStorage(temp_db)
    await primary_storage.initialize()

    content = "Detailed hybrid health memory"
    mem = Memory(
        content=content,
        content_hash=generate_content_hash(content),
        tags=["test"],
        memory_type="note"
    )
    await primary_storage.store(mem)

    # Delete embedding to create gap
    primary_storage.conn.execute("DELETE FROM memory_embeddings")
    primary_storage.conn.commit()

    mock_hybrid = MagicMock()
    mock_hybrid.get_stats = AsyncMock(return_value={"storage_backend": "Hybrid", "total_memories": 1})
    mock_hybrid.conn = None
    mock_hybrid.primary = primary_storage

    mock_user = MagicMock()
    res = await detailed_health_check(storage=mock_hybrid, user=mock_user)

    assert res.status == "degraded"
    assert res.statistics["missing_embeddings"] == 1
    assert res.storage["missing_embeddings"] == 1

    await primary_storage.close()


@pytest.mark.asyncio
async def test_detailed_health_unverified_integrity_degrades_status(temp_db):
    """Test that /api/health/detailed treats an empty/failed integrity check as degraded and leaves field out."""
    storage = SqliteVecMemoryStorage(temp_db)
    await storage.initialize()

    mock_user = MagicMock()
    with patch("mcp_memory_service.utils.health_check._check_embedding_integrity", return_value={}):
        res = await detailed_health_check(storage=storage, user=mock_user)

    assert res.status == "degraded"
    assert "missing_embeddings" not in res.statistics

    await storage.close()


@pytest.mark.asyncio
async def test_consolidation_health_unverified_integrity_degrades_status(temp_db):
    """Test that ConsolidationHealthMonitor degrades health status when embedding integrity is unverifiable."""
    storage = SqliteVecMemoryStorage(temp_db)
    await storage.initialize()

    monitor = ConsolidationHealthMonitor(consolidator=MagicMock(storage=storage))
    with patch("mcp_memory_service.utils.health_check._check_embedding_integrity", return_value={}):
        health = await monitor._check_storage_backend_health()

    assert health["status"] == HealthStatus.DEGRADED.value
    assert health["checks"]["embedding_integrity"] == "unverifiable"
    assert "missing_embeddings" not in health["checks"]

    await storage.close()


@pytest.mark.asyncio
async def test_health_endpoints_use_run_in_thread(temp_db):
    """Test that health checks offload _check_embedding_integrity via storage._run_in_thread under lock."""
    storage = SqliteVecMemoryStorage(temp_db)
    await storage.initialize()

    with patch.object(storage, "_run_in_thread", wraps=storage._run_in_thread) as spy_run_in_thread:
        mock_user = MagicMock()
        res = await detailed_health_check(storage=storage, user=mock_user)
        assert res.status == "healthy"
        spy_run_in_thread.assert_any_call(_check_embedding_integrity, storage.conn)

    with patch.object(storage, "_run_in_thread", wraps=storage._run_in_thread) as spy_run_in_thread:
        monitor = ConsolidationHealthMonitor(consolidator=MagicMock(storage=storage))
        health = await monitor._check_storage_backend_health()
        assert health["status"] == HealthStatus.HEALTHY.value
        spy_run_in_thread.assert_any_call(_check_embedding_integrity, storage.conn)

    await storage.close()


@pytest.mark.asyncio
async def test_db_health_check_fails_on_store_error():
    """Test that db_health_check.HealthChecker fails when storage.store fails."""
    import sys
    from pathlib import Path
    scripts_path = str(Path(__file__).parent.parent.parent / "scripts" / "database")
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    from db_health_check import HealthChecker

    checker = HealthChecker()
    with patch("mcp_memory_service.storage.sqlite_vec.SqliteVecMemoryStorage.store", new_callable=AsyncMock) as mock_store:
        mock_store.return_value = (False, "Simulated store failure")
        result = await checker.test_embedding_invariants()
        assert result is False
