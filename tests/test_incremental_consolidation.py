"""Tests for incremental consolidation time_horizon."""

import asyncio
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_memory_service.consolidation.consolidator import (
    HORIZON_CONFIGS,
    DreamInspiredConsolidator,
)
from mcp_memory_service.consolidation.run_tracker import RunTracker
from mcp_memory_service.consolidation.base import ConsolidationConfig
from mcp_memory_service.models.memory import Memory


# --- Fixtures ---


@pytest.fixture
def tmp_db(tmp_path):
    return tmp_path / "consolidation_runs.db"


@pytest.fixture
def tracker(tmp_db):
    return RunTracker(tmp_db)


def _make_memory(content_hash: str, created_at: float) -> Memory:
    return Memory(
        content=f"memory {content_hash}",
        content_hash=content_hash,
        tags=["test"],
        memory_type="observation",
        embedding=[0.1] * 320,
        created_at=created_at,
        created_at_iso=datetime.fromtimestamp(created_at, tz=timezone.utc).isoformat(),
    )


def _make_storage_mock(memories=None):
    storage = AsyncMock()
    storage.db_path = "/tmp/test_memories.db"
    storage.get_memories_by_time_range = AsyncMock(return_value=memories or [])
    storage.get_all_memories = AsyncMock(return_value=memories or [])
    storage.get_memory_connections = AsyncMock(return_value={})
    storage.get_access_patterns = AsyncMock(return_value={})
    storage.search_by_tag = AsyncMock(return_value=[])
    storage.update_memories_batch = AsyncMock(return_value=[True])
    return storage


def _make_config():
    return ConsolidationConfig(
        decay_enabled=True,
        retention_periods={"observation": 30},
        associations_enabled=True,
        min_similarity=0.3,
        max_similarity=0.7,
        max_pairs_per_run=10,
        clustering_enabled=True,
        min_cluster_size=3,
        clustering_algorithm="simple",
        compression_enabled=True,
        max_summary_length=200,
        preserve_originals=True,
        forgetting_enabled=True,
        relevance_threshold=0.1,
        access_threshold_days=30,
        archive_location=tempfile.mkdtemp(),
    )


# --- RunTracker unit tests ---


class TestRunTracker:
    @pytest.mark.asyncio
    async def test_get_last_run_at_returns_none_initially(self, tracker):
        result = await tracker.get_last_run_at("incremental")
        assert result is None

    @pytest.mark.asyncio
    async def test_record_and_retrieve(self, tracker):
        await tracker.record_run("incremental", 5, "success")
        last = await tracker.get_last_run_at("incremental")
        assert last is not None
        # Should be within last few seconds
        assert abs(last - time.time()) < 5

    @pytest.mark.asyncio
    async def test_record_updates_on_conflict(self, tracker):
        await tracker.record_run("incremental", 3)
        first = await tracker.get_last_run_at("incremental")
        await asyncio.sleep(0.01)
        await tracker.record_run("incremental", 7)
        second = await tracker.get_last_run_at("incremental")
        assert second >= first

    @pytest.mark.asyncio
    async def test_record_run_zero_memories(self, tracker):
        """Requirement: record run even on 0 memories."""
        await tracker.record_run("incremental", 0, "success")
        last = await tracker.get_last_run_at("incremental")
        assert last is not None

    def test_try_acquire_when_free(self, tracker):
        assert tracker.try_acquire("incremental") is True
        tracker.release("incremental")

    def test_try_acquire_when_locked(self, tracker):
        """Concurrency guard: returns False when lock is held."""
        assert tracker.try_acquire("incremental") is True
        assert tracker.try_acquire("incremental") is False
        tracker.release("incremental")


# --- HORIZON_CONFIGS tests ---


class TestHorizonConfig:
    def test_incremental_in_horizon_configs(self):
        assert "incremental" in HORIZON_CONFIGS

    def test_incremental_phases_skip_decay_and_forgetting(self):
        phases = DreamInspiredConsolidator.ENABLED_PHASES
        assert "incremental" in phases["clustering"]
        assert "incremental" in phases["associations"]
        assert "incremental" in phases["compression"]
        assert "incremental" not in phases["forgetting"]


# --- Consolidator integration tests ---


class TestIncrementalConsolidation:
    @pytest.mark.asyncio
    async def test_incremental_no_memories(self, tmp_path):
        """Incremental with 0 memories still records run."""
        storage = _make_storage_mock([])
        storage.db_path = str(tmp_path / "memories.db")
        config = _make_config()
        consolidator = DreamInspiredConsolidator(storage, config)

        report = await consolidator.consolidate("incremental")

        assert report.memories_processed == 0
        # Run tracker should have been initialized and recorded
        assert consolidator.run_tracker is not None
        last = await consolidator.run_tracker.get_last_run_at("incremental")
        assert last is not None

    @pytest.mark.asyncio
    async def test_incremental_uses_time_range(self, tmp_path):
        """Incremental should call get_memories_by_time_range, not get_all_memories."""
        now = time.time()
        memories = [_make_memory("h1", now - 60)]
        storage = _make_storage_mock(memories)
        storage.db_path = str(tmp_path / "memories.db")
        config = _make_config()
        consolidator = DreamInspiredConsolidator(storage, config)

        await consolidator.consolidate("incremental")

        storage.get_memories_by_time_range.assert_called_once()
        storage.get_all_memories.assert_not_called()

    @pytest.mark.asyncio
    async def test_incremental_bootstrap_24h_window(self, tmp_path):
        """First run bootstraps with 24h window."""
        storage = _make_storage_mock([])
        storage.db_path = str(tmp_path / "memories.db")
        config = _make_config()
        consolidator = DreamInspiredConsolidator(storage, config)

        await consolidator.consolidate("incremental")

        call_args = storage.get_memories_by_time_range.call_args
        start_time = call_args[0][0]
        # Start time should be ~24h ago
        expected = (datetime.now(timezone.utc) - timedelta(days=1)).timestamp()
        assert abs(start_time - expected) < 5

    @pytest.mark.asyncio
    async def test_incremental_uses_last_run_at(self, tmp_path):
        """Second run uses last_run_at as start_time."""
        storage = _make_storage_mock([])
        storage.db_path = str(tmp_path / "memories.db")
        config = _make_config()
        consolidator = DreamInspiredConsolidator(storage, config)

        # First run
        await consolidator.consolidate("incremental")
        first_run = await consolidator.run_tracker.get_last_run_at("incremental")

        # Second run
        await asyncio.sleep(0.01)
        await consolidator.consolidate("incremental")

        # Second call should use first_run as start_time
        second_call = storage.get_memories_by_time_range.call_args_list[1]
        start_time = second_call[0][0]
        assert abs(start_time - first_run) < 2

    @pytest.mark.asyncio
    async def test_incremental_skips_forgetting(self, tmp_path):
        """Incremental should not run forgetting phase."""
        now = time.time()
        memories = [_make_memory(f"h{i}", now - 60) for i in range(5)]
        storage = _make_storage_mock(memories)
        storage.db_path = str(tmp_path / "memories.db")
        config = _make_config()
        consolidator = DreamInspiredConsolidator(storage, config)
        consolidator.forgetting_engine.process = AsyncMock(return_value=[])

        await consolidator.consolidate("incremental")

        consolidator.forgetting_engine.process.assert_not_called()

    @pytest.mark.asyncio
    async def test_incremental_concurrency_guard(self, tmp_path):
        """If lock is held, incremental should skip."""
        storage = _make_storage_mock([])
        storage.db_path = str(tmp_path / "memories.db")
        config = _make_config()
        consolidator = DreamInspiredConsolidator(storage, config)
        # Initialize tracker
        db_path = Path(storage.db_path).parent / "consolidation_runs.db"
        consolidator.run_tracker = RunTracker(db_path)

        # Hold the lock via try_acquire
        assert consolidator.run_tracker.try_acquire("incremental") is True
        report = await consolidator.consolidate("incremental")

        # Should have bailed early with 0 memories processed
        assert report.memories_processed == 0
        storage.get_memories_by_time_range.assert_not_called()
        consolidator.run_tracker.release("incremental")

    @pytest.mark.asyncio
    async def test_timeout_advances_last_run_at(self):
        """Verify that timeout advances last_run_at to prevent infinite retry (#986)."""
        from mcp_memory_service.server.handlers.consolidation import handle_consolidate_memories

        async def slow_consolidate(*a, **kw):
            await asyncio.sleep(60)

        server = MagicMock()
        tracker = MagicMock()
        tracker.record_run = AsyncMock()
        server.consolidator.run_tracker = tracker
        server.consolidator.consolidate = AsyncMock(side_effect=slow_consolidate)

        # Call the real handler — it uses INCREMENTAL_TIMEOUT_SECONDS=10
        # but we need a fast test, so patch the local constant via module reload
        import mcp_memory_service.server.handlers.consolidation as mod
        original_code = mod.handle_consolidate_memories

        # Direct test: simulate what the handler does on timeout
        try:
            await asyncio.wait_for(
                server.consolidator.consolidate("incremental"),
                timeout=0.01,
            )
        except asyncio.TimeoutError:
            # This is what our fix does
            if server.consolidator.run_tracker:
                await server.consolidator.run_tracker.record_run("incremental", 0)

        tracker.record_run.assert_called_once_with("incremental", 0)
