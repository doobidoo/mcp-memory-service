"""Focused truthful consolidation health checks."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from mcp_memory_service.consolidation.health import ConsolidationHealthMonitor
from mcp_memory_service.consolidation.scheduler import ConsolidationScheduler
from mcp_memory_service.storage.base import MemoryStorage


@pytest.mark.asyncio
async def test_storage_health_distinguishes_observed_unverifiable_and_error():
    config = SimpleNamespace()
    storage = SimpleNamespace(health_probe=AsyncMock(return_value=True))
    monitor = ConsolidationHealthMonitor(config, SimpleNamespace(storage=storage))

    assert (await monitor._check_storage_backend_health())["status"] == "healthy"

    monitor.consolidator.storage = SimpleNamespace()
    assert (await monitor._check_storage_backend_health())["status"] == "degraded"

    monitor.consolidator.storage = SimpleNamespace(
        health_probe=AsyncMock(side_effect=RuntimeError("offline"))
    )
    assert (await monitor._check_storage_backend_health())["status"] == "unhealthy"


@pytest.mark.asyncio
async def test_storage_probe_false_exception_and_absence_are_truthful():
    monitor = ConsolidationHealthMonitor(
        SimpleNamespace(),
        SimpleNamespace(
            storage=SimpleNamespace(health_probe=AsyncMock(return_value=False))
        ),
    )
    assert (await monitor._check_storage_backend_health())["status"] == "unhealthy"
    monitor.consolidator.storage = SimpleNamespace(
        health_probe=AsyncMock(side_effect=RuntimeError("offline"))
    )
    assert (await monitor._check_storage_backend_health())["status"] == "unhealthy"
    monitor.consolidator.storage = SimpleNamespace()
    assert (await monitor._check_storage_backend_health())["status"] == "degraded"


@pytest.mark.asyncio
@pytest.mark.parametrize("stats", [{"error": "offline"}, {"status": "error"}])
async def test_base_storage_health_probe_rejects_error_stats(stats):
    """Base health probe does not treat error dictionaries as healthy."""
    storage = SimpleNamespace(get_stats=AsyncMock(return_value=stats))
    assert await MemoryStorage.health_probe(storage) is False


@pytest.mark.asyncio
async def test_forgetting_archive_probe_cleans_up(tmp_path):
    sentinel = tmp_path / "keep"
    sentinel.write_text("keep")
    config = SimpleNamespace(
        forgetting_enabled=True,
        archive_location=str(tmp_path),
        access_threshold_days=30,
        relevance_threshold=0.5,
    )
    monitor = ConsolidationHealthMonitor(
        config, SimpleNamespace(forgetting_engine=object())
    )

    health = await monitor._check_forgetting_engine_health()

    assert health["status"] == "healthy"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["keep"]


@pytest.mark.asyncio
async def test_scheduler_reports_unattached_disabled_and_stopped():
    monitor = ConsolidationHealthMonitor(SimpleNamespace())
    assert (await monitor._check_scheduler_health())["status"] == "unhealthy"
    disabled = ConsolidationHealthMonitor(
        SimpleNamespace(), schedule_config={"daily": "disabled"}
    )
    assert (await disabled._check_scheduler_health())["status"] == "healthy"
    monitor.scheduler = SimpleNamespace(
        get_scheduler_status=AsyncMock(return_value={"enabled": True, "running": False})
    )
    assert (await monitor._check_scheduler_health())["status"] == "unhealthy"
    monitor.scheduler = SimpleNamespace(
        schedule_config={"daily": "0 0 * * *", "weekly": "0 0 * * 0"},
        get_scheduler_status=AsyncMock(
            return_value={
                "enabled": True,
                "running": True,
                "jobs": [
                    {"id": "consolidation_daily"},
                    {"id": "consolidation_weekly"},
                ],
            }
        ),
        last_execution_times={"daily": 1},
        execution_stats={"total_jobs": 2},
    )
    health = await monitor._check_scheduler_health()
    assert health["status"] == "healthy"
    assert health["metrics"]["scheduled_jobs"] == 2
    assert health["metrics"]["last_execution"] == {"daily": 1}


@pytest.mark.asyncio
async def test_scheduler_disabled_status_requires_all_disabled_schedule():
    monitor = ConsolidationHealthMonitor(
        SimpleNamespace(), schedule_config={"daily": "0 0 * * *"}
    )
    monitor.scheduler = SimpleNamespace(
        get_scheduler_status=AsyncMock(
            return_value={"enabled": False, "reason": "APScheduler unavailable"}
        )
    )
    assert (await monitor._check_scheduler_health())["status"] == "unhealthy"
    monitor.schedule_config = {"daily": "disabled"}
    assert (await monitor._check_scheduler_health())["status"] == "healthy"


@pytest.mark.asyncio
async def test_running_scheduler_requires_exact_configured_job_set():
    """Active configs without their jobs are unhealthy; exact sets are healthy."""
    scheduler = SimpleNamespace(
        schedule_config={"daily": "0 0 * * *"},
        last_execution_times={},
        execution_stats={},
        get_scheduler_status=AsyncMock(
            return_value={"enabled": True, "running": True, "jobs": []}
        ),
    )
    monitor = ConsolidationHealthMonitor(SimpleNamespace(), scheduler=scheduler)
    assert (await monitor._check_scheduler_health())["status"] == "unhealthy"
    scheduler.get_scheduler_status = AsyncMock(
        return_value={
            "enabled": True,
            "running": True,
            "jobs": [{"id": "consolidation_daily"}],
        }
    )
    assert (await monitor._check_scheduler_health())["status"] == "healthy"
    scheduler.get_scheduler_status = AsyncMock(
        return_value={
            "enabled": True,
            "running": True,
            "jobs": [{"id": "consolidation_daily"}, {"id": "consolidation_weekly"}],
        }
    )
    assert (await monitor._check_scheduler_health())["status"] == "unhealthy"


@pytest.mark.asyncio
async def test_running_scheduler_without_configuration_is_unverifiable():
    """An attached running scheduler needs an observable configuration."""
    scheduler = SimpleNamespace(
        schedule_config={},
        last_execution_times={},
        execution_stats={},
        get_scheduler_status=AsyncMock(
            return_value={"enabled": True, "running": True, "jobs": []}
        ),
    )
    monitor = ConsolidationHealthMonitor(SimpleNamespace(), scheduler=scheduler)

    health = await monitor._check_scheduler_health()

    assert health["status"] == "unhealthy"
    assert health["checks"]["scheduler"] == (
        "running with no observable schedule configuration"
    )


@pytest.mark.asyncio
async def test_scheduler_rejects_malformed_active_config_without_a_job():
    """An active but invalid schedule cannot be reported as disabled or healthy."""
    scheduler = SimpleNamespace(
        schedule_config={"daily": "not-a-cron"},
        last_execution_times={},
        execution_stats={},
        get_scheduler_status=AsyncMock(
            return_value={"enabled": True, "running": True, "jobs": []}
        ),
    )
    monitor = ConsolidationHealthMonitor(SimpleNamespace(), scheduler=scheduler)

    health = await monitor._check_scheduler_health()

    assert health["status"] == "unhealthy"
    assert health["checks"]["missing_jobs"] == ["consolidation_daily"]


@pytest.mark.asyncio
async def test_update_schedule_synchronizes_live_health_configuration():
    """Runtime updates replace the monitor's fallback schedule configuration."""
    scheduler = ConsolidationScheduler.__new__(ConsolidationScheduler)
    scheduler.scheduler = MagicMock()
    scheduler.schedule_config = {"daily": "disabled"}
    scheduler.consolidator = SimpleNamespace(
        health_monitor=SimpleNamespace(schedule_config={"daily": "disabled"})
    )
    scheduler._schedule_consolidation_jobs = MagicMock()
    scheduler.logger = MagicMock()

    result = await scheduler.update_schedule({"weekly": "SUN 03:00"})

    assert result is True
    assert scheduler.schedule_config == {"weekly": "SUN 03:00"}
    assert scheduler.consolidator.health_monitor.schedule_config == {
        "weekly": "SUN 03:00"
    }
    scheduler._schedule_consolidation_jobs.assert_called_once_with()


@pytest.mark.asyncio
async def test_decay_and_association_validate_observed_configuration():
    valid = SimpleNamespace(
        retention_periods={"short": 1}, min_similarity=0.2, max_similarity=0.8
    )
    monitor = ConsolidationHealthMonitor(
        valid, SimpleNamespace(association_engine=object(), decay_calculator=object())
    )
    assert (await monitor._check_decay_calculator_health())["status"] == "healthy"
    assert (await monitor._check_association_engine_health())["status"] == "healthy"

    monitor.config = SimpleNamespace(
        retention_periods={}, min_similarity=0.9, max_similarity=0.2
    )
    monitor.consolidator.decay_calculator = None
    assert (await monitor._check_decay_calculator_health())["status"] == "unhealthy"
    assert (await monitor._check_association_engine_health())["status"] == "unhealthy"


@pytest.mark.asyncio
async def test_compression_disabled_missing_and_present_engine():
    """Compression has no remote backend: engine plus a usable bound is healthy."""
    monitor = ConsolidationHealthMonitor(
        SimpleNamespace(compression_enabled=False, max_summary_length=500),
        SimpleNamespace(),
    )
    assert (await monitor._check_compression_engine_health())["status"] == "healthy"

    monitor.config = SimpleNamespace(compression_enabled=True, max_summary_length=500)
    missing = await monitor._check_compression_engine_health()
    assert missing["status"] == "unhealthy"
    assert missing["checks"]["compression"] == "engine not attached"

    monitor.consolidator.compression_engine = object()
    present = await monitor._check_compression_engine_health()
    assert present["status"] == "healthy"
    assert present["checks"] == {
        "compression": "engine present",
        "max_summary_length": 500,
    }

    monitor.config = SimpleNamespace(compression_enabled=True, max_summary_length=0)
    invalid = await monitor._check_compression_engine_health()
    assert invalid["status"] == "unhealthy"
    assert invalid["checks"]["max_summary_length"] == "invalid"


@pytest.mark.asyncio
async def test_clustering_reports_resolved_and_unsatisfiable_algorithms(monkeypatch):
    from mcp_memory_service.consolidation import clustering

    monitor = ConsolidationHealthMonitor(
        SimpleNamespace(clustering_algorithm="dbscan"),
        SimpleNamespace(clustering_engine=object()),
    )
    monkeypatch.setattr(clustering, "SKLEARN_AVAILABLE", True)
    monkeypatch.setattr(
        clustering, "resolve_clustering_algorithm", lambda _value: "dbscan"
    )
    healthy = await monitor._check_clustering_engine_health()
    assert healthy["status"] == "healthy"
    assert healthy["checks"]["clustering_algorithm"] == "dbscan"

    from mcp_memory_service.consolidation.base import ConsolidationError

    def unsatisfiable(_value):
        raise ConsolidationError("requires scikit-learn")

    monkeypatch.setattr(clustering, "resolve_clustering_algorithm", unsatisfiable)
    unhealthy = await monitor._check_clustering_engine_health()
    assert unhealthy["status"] == "unhealthy"
    assert unhealthy["checks"]["clustering_algorithm"].startswith("unsatisfiable:")

    monitor.consolidator.clustering_engine = None
    monkeypatch.setattr(
        clustering, "resolve_clustering_algorithm", lambda _value: "dbscan"
    )
    assert (await monitor._check_clustering_engine_health())["status"] == "unhealthy"


@pytest.mark.asyncio
async def test_overall_rollup_uses_component_statuses(monkeypatch):
    monitor = ConsolidationHealthMonitor(SimpleNamespace(), SimpleNamespace())

    async def unhealthy(component):
        return {"status": "unhealthy", "checks": {component: "bad"}, "metrics": {}}

    monkeypatch.setattr(monitor, "_check_component_health", unhealthy)
    assert (await monitor.check_overall_health())["status"] == "unhealthy"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("component_status", "expected"),
    [
        ("healthy", "healthy"),
        ("degraded", "degraded"),
        ("unhealthy", "unhealthy"),
        ("critical", "critical"),
    ],
)
async def test_overall_rollup_respects_component_severity(
    monkeypatch, component_status, expected
):
    monitor = ConsolidationHealthMonitor(SimpleNamespace(), SimpleNamespace())

    async def status(_component):
        return {"status": component_status, "checks": {}, "metrics": {}}

    monkeypatch.setattr(monitor, "_check_component_health", status)
    assert (await monitor.check_overall_health())["status"] == expected


@pytest.mark.asyncio
async def test_forgetting_archive_probe_expands_home_relative_location(
    tmp_path, monkeypatch
):
    """'~/.mcp_memory_archive' is the engine's own fallback and must not read
    as unwritable."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    (tmp_path / ".mcp_memory_archive").mkdir()
    monitor = ConsolidationHealthMonitor(
        SimpleNamespace(
            forgetting_enabled=True,
            archive_location="~/.mcp_memory_archive",
            relevance_threshold=0.1,
            access_threshold_days=90,
        ),
        SimpleNamespace(forgetting_engine=object()),
    )

    health = await monitor._check_forgetting_engine_health()

    assert health["status"] == "healthy"
    assert health["checks"]["archive_storage"] == "writable"
    assert list((tmp_path / ".mcp_memory_archive").iterdir()) == []


@pytest.mark.asyncio
async def test_default_deployment_with_every_component_working_is_healthy(tmp_path):
    """A correct default deployment must roll up green.

    The severity rollup tests below stub ``_check_component_health``, so they
    cannot see a component that is permanently amber on a working install.
    This one runs the real checks against the real shipped defaults, which is
    where such a permanent false alarm would show up.
    """
    from mcp_memory_service.config.consolidation import CONSOLIDATION_CONFIG
    from mcp_memory_service.consolidation.base import ConsolidationConfig

    config = ConsolidationConfig(**CONSOLIDATION_CONFIG)
    config.archive_location = str(tmp_path)
    schedule = {"daily": "02:00", "weekly": "SUN 03:00"}
    consolidator = SimpleNamespace(
        decay_calculator=object(),
        association_engine=object(),
        clustering_engine=object(),
        compression_engine=object(),
        forgetting_engine=object(),
        storage=SimpleNamespace(health_probe=AsyncMock(return_value=True)),
    )
    scheduler = SimpleNamespace(
        schedule_config=schedule,
        last_execution_times={},
        execution_stats={},
        get_scheduler_status=AsyncMock(
            return_value={
                "enabled": True,
                "running": True,
                "jobs": [
                    {"id": "consolidation_daily"},
                    {"id": "consolidation_weekly"},
                ],
            }
        ),
    )
    monitor = ConsolidationHealthMonitor(config, consolidator, scheduler, schedule)

    health = await monitor.check_overall_health()

    amber = {
        name: component["status"]
        for name, component in health["components"].items()
        if component["status"] != "healthy"
    }
    assert amber == {}
    assert health["status"] == "healthy"
