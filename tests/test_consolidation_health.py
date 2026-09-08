"""Tests for ConsolidationHealthMonitor — per-engine checks with negative cases."""

import asyncio
import os
import tempfile
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_memory_service.consolidation.health import (
    ConsolidationHealthMonitor,
    HealthStatus,
    _count_recent,
    _valid_retention_periods,
    _valid_similarity_range,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides):
    """Return a minimal mock config with sane defaults, overridable per test."""
    defaults = dict(
        retention_periods={"short_term": 7, "long_term": 365},
        min_similarity=0.3,
        max_similarity=0.95,
        relevance_threshold=0.5,
        archive_location=None,  # defaults to ~/.mcp_memory_archive in code
        clustering_algorithm="hdbscan",
        schedule_config={"daily": "0 2 * * *", "weekly": "disabled"},
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_consolidator(storage=None, compression_engine=None):
    """Return a minimal mock consolidator."""
    return SimpleNamespace(
        storage=storage,
        compression_engine=compression_engine,
        association_engine=object(),
        clustering_engine=object(),
        forgetting_engine=object(),
        health_monitor=None,
    )


def _make_storage(count_method="count_all_memories"):
    """Return an async-mock storage backend.

    count_method selects which ping method is available.
    """
    storage = MagicMock()
    if count_method == "count_all_memories":
        storage.count_all_memories = AsyncMock(return_value=42)
        # No get_stats attribute
        if hasattr(storage, "get_stats"):
            del storage.get_stats
    elif count_method == "get_stats":
        storage.get_stats = AsyncMock(return_value={"total_memories": 99})
        # No count_all_memories attribute
        if hasattr(storage, "count_all_memories"):
            del storage.count_all_memories
    elif count_method == "none":
        # Remove both ping methods
        if hasattr(storage, "count_all_memories"):
            del storage.count_all_memories
        if hasattr(storage, "get_stats"):
            del storage.get_stats
    return storage


def _run(coro):
    """Run a coroutine in a fresh event loop (Python 3.14 compatible)."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    def test_valid_retention_periods_true(self):
        assert _valid_retention_periods({"daily": 1, "weekly": 7})

    def test_valid_retention_periods_empty_dict(self):
        assert not _valid_retention_periods({})

    def test_valid_retention_periods_non_dict(self):
        assert not _valid_retention_periods(None)

    def test_valid_retention_periods_non_positive_value(self):
        assert not _valid_retention_periods({"daily": 0, "weekly": -1})

    def test_valid_similarity_range_normal(self):
        cfg = SimpleNamespace(min_similarity=0.3, max_similarity=0.9)
        assert _valid_similarity_range(cfg)

    def test_valid_similarity_range_inverted(self):
        cfg = SimpleNamespace(min_similarity=0.9, max_similarity=0.3)
        assert not _valid_similarity_range(cfg)

    def test_valid_similarity_range_equal(self):
        cfg = SimpleNamespace(min_similarity=0.5, max_similarity=0.5)
        assert not _valid_similarity_range(cfg)

    def test_valid_similarity_range_missing(self):
        cfg = SimpleNamespace()
        assert not _valid_similarity_range(cfg)

    def test_count_recent(self):
        now = datetime.now()
        history = [
            {"component": "decay_calculator", "timestamp": now - timedelta(minutes=10)},
            {"component": "decay_calculator", "timestamp": now - timedelta(hours=2)},
            {"component": "association_engine", "timestamp": now},
        ]
        assert _count_recent(history, "decay_calculator", hours=1) == 1

    def test_count_recent_empty(self):
        assert _count_recent([], "anything") == 0


# ---------------------------------------------------------------------------
# Decay calculator health
# ---------------------------------------------------------------------------


class TestDecayCalculatorHealth:
    def test_healthy_when_retention_periods_valid(self):
        cfg = _make_config()
        monitor = ConsolidationHealthMonitor(config=cfg)
        result = _run(monitor._check_decay_calculator_health())
        assert result["status"] == HealthStatus.HEALTHY.value
        assert "configured (2 types)" in result["checks"]["retention_periods"]

    def test_degraded_when_retention_periods_missing(self):
        cfg = _make_config(retention_periods=None)
        monitor = ConsolidationHealthMonitor(config=cfg)
        result = _run(monitor._check_decay_calculator_health())
        assert result["status"] == HealthStatus.DEGRADED.value
        assert "missing" in result["checks"]["retention_periods"]

    def test_degraded_when_retention_periods_empty(self):
        cfg = _make_config(retention_periods={})
        monitor = ConsolidationHealthMonitor(config=cfg)
        result = _run(monitor._check_decay_calculator_health())
        assert result["status"] == HealthStatus.DEGRADED.value

    def test_degraded_when_retention_periods_non_positive(self):
        cfg = _make_config(retention_periods={"daily": 0, "weekly": -1})
        monitor = ConsolidationHealthMonitor(config=cfg)
        result = _run(monitor._check_decay_calculator_health())
        assert result["status"] == HealthStatus.DEGRADED.value


# ---------------------------------------------------------------------------
# Association engine health
# ---------------------------------------------------------------------------


class TestAssociationEngineHealth:
    def test_healthy_with_valid_similarity_range(self):
        cfg = _make_config()
        monitor = ConsolidationHealthMonitor(config=cfg)
        result = _run(monitor._check_association_engine_health())
        assert result["status"] == HealthStatus.HEALTHY.value
        assert "range [0.3, 0.95]" in result["checks"]["similarity_thresholds"]

    def test_degraded_when_similarity_range_inverted(self):
        cfg = _make_config(min_similarity=0.9, max_similarity=0.3)
        monitor = ConsolidationHealthMonitor(config=cfg)
        result = _run(monitor._check_association_engine_health())
        assert result["status"] == HealthStatus.DEGRADED.value

    def test_degraded_when_similarity_range_missing(self):
        cfg = _make_config(min_similarity=None, max_similarity=None)
        monitor = ConsolidationHealthMonitor(config=cfg)
        result = _run(monitor._check_association_engine_health())
        assert result["status"] == HealthStatus.DEGRADED.value


# ---------------------------------------------------------------------------
# Clustering engine health
# ---------------------------------------------------------------------------


class TestClusteringEngineHealth:
    def test_healthy_with_hdbscan(self):
        from mcp_memory_service.consolidation.clustering import SKLEARN_AVAILABLE
        cfg = _make_config(clustering_algorithm="hdbscan")
        monitor = ConsolidationHealthMonitor(config=cfg)
        result = _run(monitor._check_clustering_engine_health())
        assert "checks" in result
        expected = "available" if SKLEARN_AVAILABLE else "unavailable"
        assert result["checks"]["sklearn"] == expected

    def test_healthy_with_fuzzy_cmeans(self):
        from mcp_memory_service.consolidation.clustering import SKLEARN_AVAILABLE
        cfg = _make_config(clustering_algorithm="fuzzy_cmeans")
        monitor = ConsolidationHealthMonitor(config=cfg)
        result = _run(monitor._check_clustering_engine_health())
        expected = "available" if SKLEARN_AVAILABLE else "unavailable"
        assert result["checks"]["sklearn"] == expected


# ---------------------------------------------------------------------------
# Compression engine health
# ---------------------------------------------------------------------------


class TestCompressionEngineHealth:
    def test_healthy_with_no_llm(self):
        cfg = _make_config()
        consolidator = _make_consolidator(compression_engine=SimpleNamespace(llm_client=None))
        monitor = ConsolidationHealthMonitor(config=cfg, consolidator=consolidator)
        result = _run(monitor._check_compression_engine_health())
        assert result["status"] == HealthStatus.HEALTHY.value
        assert "hash fallback" in result["checks"]["llm_backend"]

    def test_healthy_with_llm_configured(self):
        cfg = _make_config()
        engine = SimpleNamespace(llm_client=object())
        consolidator = _make_consolidator(compression_engine=engine)
        monitor = ConsolidationHealthMonitor(config=cfg, consolidator=consolidator)
        result = _run(monitor._check_compression_engine_health())
        assert "configured" in result["checks"]["llm_backend"]

    def test_healthy_without_consolidator(self):
        cfg = _make_config()
        monitor = ConsolidationHealthMonitor(config=cfg, consolidator=None)
        result = _run(monitor._check_compression_engine_health())
        assert result["status"] == HealthStatus.HEALTHY.value


# ---------------------------------------------------------------------------
# Forgetting engine health
# ---------------------------------------------------------------------------


class TestForgettingEngineHealth:
    def test_healthy_with_writable_archive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(archive_location=tmpdir)
            monitor = ConsolidationHealthMonitor(config=cfg)
            result = _run(monitor._check_forgetting_engine_health())
            assert result["status"] == HealthStatus.HEALTHY.value
            assert "accessible" in result["checks"]["archive_storage"]

    def test_degraded_when_archive_not_exists(self):
        cfg = _make_config(archive_location="/nonexistent/path/that/should/not/exist")
        monitor = ConsolidationHealthMonitor(config=cfg)
        result = _run(monitor._check_forgetting_engine_health())
        assert result["status"] == HealthStatus.DEGRADED.value
        assert "does not exist" in result["checks"]["archive_storage"]

    def test_healthy_with_default_archive_location(self):
        cfg = _make_config(archive_location=None)
        monitor = ConsolidationHealthMonitor(config=cfg)
        # Default path (~/.mcp_memory_archive) may or may not exist;
        # this exercises the default branch without crashing
        result = _run(monitor._check_forgetting_engine_health())
        assert result["status"] in (HealthStatus.HEALTHY.value, HealthStatus.DEGRADED.value)


# ---------------------------------------------------------------------------
# Scheduler health
# ---------------------------------------------------------------------------


class TestSchedulerHealth:
    def test_degraded_when_all_horizons_disabled(self):
        cfg = _make_config(schedule_config={"daily": "disabled", "weekly": "disabled"})
        monitor = ConsolidationHealthMonitor(config=cfg)
        result = _run(monitor._check_scheduler_health())
        assert result["status"] == HealthStatus.DEGRADED.value
        assert "disabled" in result["checks"]["scheduler_running"]

    def test_unhealthy_when_scheduler_not_initialized(self):
        cfg = _make_config(schedule_config={"daily": "0 2 * * *"})
        monitor = ConsolidationHealthMonitor(config=cfg)
        # No _scheduler_ref set, schedule_config has enabled entries
        result = _run(monitor._check_scheduler_health())
        assert result["status"] == HealthStatus.UNHEALTHY.value

    def test_healthy_when_scheduler_running(self):
        cfg = _make_config(schedule_config={"daily": "0 2 * * *"})
        monitor = ConsolidationHealthMonitor(config=cfg)

        # Build a fake scheduler with a running APScheduler
        mock_apscheduler = MagicMock()
        mock_apscheduler.running = True
        mock_apscheduler.get_jobs.return_value = [MagicMock(), MagicMock()]

        fake_scheduler = SimpleNamespace(
            scheduler=mock_apscheduler,
            schedule_config={"daily": "0 2 * * *"},
            last_execution_times={"daily": datetime.now() - timedelta(minutes=5)},
            execution_stats={"total_jobs": 10, "successful_jobs": 9, "failed_jobs": 1},
        )
        monitor.attach_scheduler(fake_scheduler)

        result = _run(monitor._check_scheduler_health())
        assert result["status"] == HealthStatus.HEALTHY.value
        assert "active" in result["checks"]["scheduler_running"]
        assert "2 jobs" in result["checks"]["scheduled_jobs"]

    def test_unhealthy_when_scheduler_stopped(self):
        cfg = _make_config(schedule_config={"daily": "0 2 * * *"})
        monitor = ConsolidationHealthMonitor(config=cfg)

        mock_apscheduler = MagicMock()
        mock_apscheduler.running = False

        fake_scheduler = SimpleNamespace(
            scheduler=mock_apscheduler,
            schedule_config={"daily": "0 2 * * *"},
            last_execution_times={},
            execution_stats={},
        )
        monitor.attach_scheduler(fake_scheduler)

        result = _run(monitor._check_scheduler_health())
        assert result["status"] == HealthStatus.UNHEALTHY.value


# ---------------------------------------------------------------------------
# Storage backend health
# ---------------------------------------------------------------------------


class TestStorageBackendHealth:
    def test_healthy_with_count_all_memories(self):
        storage = _make_storage("count_all_memories")
        cfg = _make_config()
        consolidator = _make_consolidator(storage=storage)
        monitor = ConsolidationHealthMonitor(config=cfg, consolidator=consolidator)
        result = _run(monitor._check_storage_backend_health())
        assert result["status"] == HealthStatus.HEALTHY.value
        assert result["checks"]["memory_count"] == 42

    def test_healthy_with_get_stats(self):
        storage = _make_storage("get_stats")
        cfg = _make_config()
        consolidator = _make_consolidator(storage=storage)
        monitor = ConsolidationHealthMonitor(config=cfg, consolidator=consolidator)
        result = _run(monitor._check_storage_backend_health())
        assert result["status"] == HealthStatus.HEALTHY.value
        assert result["checks"]["memory_count"] == 99

    def test_unhealthy_when_no_storage(self):
        cfg = _make_config()
        consolidator = _make_consolidator(storage=None)
        monitor = ConsolidationHealthMonitor(config=cfg, consolidator=consolidator)
        result = _run(monitor._check_storage_backend_health())
        assert result["status"] == HealthStatus.UNHEALTHY.value
        assert "not initialized" in result["checks"]["storage_connection"]

    def test_unhealthy_when_storage_raises(self):
        storage = MagicMock()
        storage.count_all_memories = AsyncMock(side_effect=RuntimeError("connection lost"))
        cfg = _make_config()
        consolidator = _make_consolidator(storage=storage)
        monitor = ConsolidationHealthMonitor(config=cfg, consolidator=consolidator)
        result = _run(monitor._check_storage_backend_health())
        assert result["status"] == HealthStatus.UNHEALTHY.value
        assert "error" in result["checks"]["storage_connection"]

    def test_no_ping_method(self):
        storage = _make_storage("none")
        cfg = _make_config()
        consolidator = _make_consolidator(storage=storage)
        monitor = ConsolidationHealthMonitor(config=cfg, consolidator=consolidator)
        result = _run(monitor._check_storage_backend_health())
        # No ping method, but no exception either — should report unverifiable
        assert result["checks"]["read_operations"] == "unverifiable"


# ---------------------------------------------------------------------------
# Overall health integration
# ---------------------------------------------------------------------------


class TestOverallHealth:
    def test_overall_healthy(self):
        cfg = _make_config()
        storage = _make_storage("count_all_memories")
        consolidator = _make_consolidator(storage=storage)
        monitor = ConsolidationHealthMonitor(config=cfg, consolidator=consolidator)
        # Attach a running scheduler so scheduler check passes
        mock_aps = MagicMock()
        mock_aps.running = True
        mock_aps.get_jobs.return_value = [MagicMock()]
        monitor.attach_scheduler(SimpleNamespace(
            scheduler=mock_aps,
            schedule_config=cfg.schedule_config,
            last_execution_times={"daily": datetime.now()},
            execution_stats={},
        ))
        result = _run(monitor.check_overall_health())
        # At least one component is healthy; overall may be degraded due to
        # disabled horizons or missing archive, but should not be critical
        assert result["status"] in (
            HealthStatus.HEALTHY.value,
            HealthStatus.DEGRADED.value,
        )

    def test_overall_critical_when_consolidator_none(self):
        cfg = _make_config()
        monitor = ConsolidationHealthMonitor(config=cfg, consolidator=None)
        result = _run(monitor.check_overall_health())
        # Storage check returns unhealthy, but overall should still be
        # unhealthy or critical (not crash)
        assert result["status"] in (
            HealthStatus.UNHEALTHY.value,
            HealthStatus.CRITICAL.value,
        )


# ---------------------------------------------------------------------------
# Alert and metric tracking
# ---------------------------------------------------------------------------


class TestAlertsAndMetrics:
    def test_create_alert(self):
        monitor = ConsolidationHealthMonitor()
        monitor._create_alert("test", HealthStatus.DEGRADED, "test message")
        assert len(monitor.alerts) == 1
        assert monitor.alerts[0].component == "test"
        assert not monitor.alerts[0].resolved

    def test_resolve_alert(self):
        monitor = ConsolidationHealthMonitor()
        monitor._create_alert("test", HealthStatus.DEGRADED, "test message")
        alert_id = monitor.alerts[0].alert_id
        monitor.resolve_alert(alert_id)
        assert monitor.alerts[0].resolved
        assert monitor.alerts[0].resolution_timestamp is not None

    def test_record_error_creates_alert(self):
        monitor = ConsolidationHealthMonitor()
        from mcp_memory_service.consolidation.base import ConsolidationError
        monitor.record_error("test", ConsolidationError("test error"))
        assert len(monitor.error_history) == 1
        assert len(monitor.alerts) == 1
