"""Tests for scheduled session harvest job in ConsolidationScheduler.

The scheduler autonomously consolidates and derives beliefs, but never
harvested sessions on its own — the executor (harvest_and_store) existed but
was only reachable through the local-only HTTP handler. These tests pin the
wiring: an opt-in MCP_HARVEST_SCHEDULE registers an in-process harvest job.
Red on main (no such job/method there).
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from mcp_memory_service.consolidation.scheduler import ConsolidationScheduler


def _scheduler(env, monkeypatch):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    consolidator = MagicMock()
    consolidator.storage = MagicMock()
    sched = ConsolidationScheduler(
        consolidator=consolidator,
        schedule_config={},  # no consolidation jobs
        enabled=True,
    )
    return sched


def test_interval_parsing():
    p = ConsolidationScheduler._parse_interval_seconds
    assert p("6h") == 21600
    assert p("30m") == 1800
    assert p("90s") == 90
    assert p("6") == 21600          # bare number = hours
    assert p("2.5h") == 9000
    assert p("garbage") is None
    assert p("") is None


def test_harvest_job_registered_when_env_set(monkeypatch):
    """MCP_HARVEST_SCHEDULE=6h registers a 'session_harvest' job."""
    sched = _scheduler({"MCP_HARVEST_SCHEDULE": "6h"}, monkeypatch)
    assert sched.scheduler is not None
    sched._schedule_harvest_job()
    job = sched.scheduler.get_job("session_harvest")
    assert job is not None, "session_harvest job should be registered when MCP_HARVEST_SCHEDULE is set"


def test_harvest_job_not_registered_by_default(monkeypatch):
    """Unset MCP_HARVEST_SCHEDULE → no harvest job (zero regression, opt-in)."""
    monkeypatch.delenv("MCP_HARVEST_SCHEDULE", raising=False)
    sched = _scheduler({}, monkeypatch)
    sched._schedule_harvest_job()
    assert sched.scheduler.get_job("session_harvest") is None


def test_harvest_job_not_registered_when_disabled(monkeypatch):
    sched = _scheduler({"MCP_HARVEST_SCHEDULE": "disabled"}, monkeypatch)
    sched._schedule_harvest_job()
    assert sched.scheduler.get_job("session_harvest") is None


def test_invalid_schedule_does_not_register(monkeypatch):
    sched = _scheduler({"MCP_HARVEST_SCHEDULE": "notaninterval"}, monkeypatch)
    sched._schedule_harvest_job()
    assert sched.scheduler.get_job("session_harvest") is None


@pytest.mark.asyncio
async def test_run_scheduled_harvest_calls_harvest_and_store(monkeypatch):
    """_run_scheduled_harvest builds a MemoryService + SessionHarvester and
    calls harvest_and_store with dry_run=False, harvesting only pending sessions."""
    sched = _scheduler({"MCP_HARVEST_SESSION_DIR": "/tmp/fake-sessions"}, monkeypatch)

    captured = {}

    class FakeSession:
        def __init__(self, stem): self.stem = stem

    class FakeHarvester:
        def __init__(self, project_dir, memory_service=None):
            captured["project_dir"] = str(project_dir)
            captured["memory_service"] = memory_service
        def _resolve_sessions(self, config):
            return [FakeSession("s1"), FakeSession("s2")]
        async def harvest_and_store(self, config):
            captured["config"] = config
            r = MagicMock(); r.stored = 2; r.found = 4; r.session_id = "s1"
            return [r]

    fake_ms = MagicMock()
    with patch("mcp_memory_service.harvest.harvester.SessionHarvester", FakeHarvester), \
         patch("mcp_memory_service.services.memory_service.MemoryService", lambda s: fake_ms):
        # tracker empty on read; capture update
        sched._read_harvest_tracker = AsyncMock(return_value=set())
        sched._update_harvest_tracker = AsyncMock()
        await sched._run_scheduled_harvest()

    assert captured["project_dir"] == "/tmp/fake-sessions"
    assert captured["config"].dry_run is False
    # only pending session ids passed through
    assert captured["config"].session_ids == ["s1", "s2"]
    sched._update_harvest_tracker.assert_awaited_once()
    assert sched.execution_stats["successful_jobs"] == 1


@pytest.mark.asyncio
async def test_run_scheduled_harvest_skips_already_harvested(monkeypatch):
    """Sessions already in the harvest-tracker are NOT re-harvested (idempotency).
    Regression guard for the G5 finding — without the tracker filter the job
    would re-process everything every cycle and duplicate memories."""
    sched = _scheduler({"MCP_HARVEST_SESSION_DIR": "/tmp/fake"}, monkeypatch)

    called = {"harvest": 0}

    class FakeSession:
        def __init__(self, stem): self.stem = stem

    class FakeHarvester:
        def __init__(self, project_dir, memory_service=None): pass
        def _resolve_sessions(self, config):
            return [FakeSession("s1"), FakeSession("s2")]
        async def harvest_and_store(self, config):
            called["harvest"] += 1
            return []

    with patch("mcp_memory_service.harvest.harvester.SessionHarvester", FakeHarvester), \
         patch("mcp_memory_service.services.memory_service.MemoryService", lambda s: MagicMock()):
        # BOTH sessions already harvested → nothing pending → harvest_and_store not called
        sched._read_harvest_tracker = AsyncMock(return_value={"s1", "s2"})
        sched._update_harvest_tracker = AsyncMock()
        await sched._run_scheduled_harvest()

    assert called["harvest"] == 0, "no pending sessions → harvest_and_store must not run"
    sched._update_harvest_tracker.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_scheduled_harvest_error_does_not_raise(monkeypatch):
    """A failure inside the harvest job must be swallowed (logged), never
    re-raised — otherwise it would tear down the shared scheduler."""
    sched = _scheduler({"MCP_HARVEST_SESSION_DIR": "/tmp/fake"}, monkeypatch)
    with patch("mcp_memory_service.services.memory_service.MemoryService", side_effect=RuntimeError("boom")):
        await sched._run_scheduled_harvest()  # must not raise
    assert sched.execution_stats["failed_jobs"] == 1


@pytest.mark.asyncio
async def test_run_scheduled_harvest_skips_without_storage(monkeypatch):
    """No storage on consolidator → skip gracefully (no crash)."""
    monkeypatch.setenv("MCP_HARVEST_SESSION_DIR", "/tmp/x")
    consolidator = MagicMock()
    consolidator.storage = None
    sched = ConsolidationScheduler(consolidator=consolidator, schedule_config={}, enabled=True)
    await sched._run_scheduled_harvest()  # should not raise
    assert sched.execution_stats["failed_jobs"] == 0
