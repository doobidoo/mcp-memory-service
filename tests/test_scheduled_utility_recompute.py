"""Tests for the scheduled utility_prior recompute job in ConsolidationScheduler.

The learning loop derives a per-belief ``utility_prior`` (frente D / D2) from usage
signals, but ``BeliefService.recompute_utility_priors`` had NO trigger — nothing ever
called it, so on a live DB every belief's ``utility_prior`` stayed NULL and the injection
re-rank's utility term was a no-op (util_mult=1.0 for all). These tests pin the wiring:
an opt-in ``MCP_UTILITY_RECOMPUTE_SCHEDULE`` registers an in-process recompute job that
calls ``recompute_utility_priors``. Red on main (no such job/method there).

Mirrors tests/test_scheduled_harvest.py.
"""

from unittest.mock import MagicMock, patch

import pytest

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


def test_utility_recompute_job_registered_when_env_set(monkeypatch):
    """MCP_UTILITY_RECOMPUTE_SCHEDULE=6h registers a 'utility_recompute' job."""
    sched = _scheduler({"MCP_UTILITY_RECOMPUTE_SCHEDULE": "6h"}, monkeypatch)
    assert sched.scheduler is not None
    sched._schedule_utility_recompute_job()
    job = sched.scheduler.get_job("utility_recompute")
    assert job is not None, "utility_recompute job should be registered when MCP_UTILITY_RECOMPUTE_SCHEDULE is set"


def test_utility_recompute_job_not_registered_by_default(monkeypatch):
    """Unset MCP_UTILITY_RECOMPUTE_SCHEDULE → no job (zero regression, opt-in)."""
    monkeypatch.delenv("MCP_UTILITY_RECOMPUTE_SCHEDULE", raising=False)
    sched = _scheduler({}, monkeypatch)
    sched._schedule_utility_recompute_job()
    assert sched.scheduler.get_job("utility_recompute") is None


def test_utility_recompute_job_not_registered_when_disabled(monkeypatch):
    sched = _scheduler({"MCP_UTILITY_RECOMPUTE_SCHEDULE": "disabled"}, monkeypatch)
    sched._schedule_utility_recompute_job()
    assert sched.scheduler.get_job("utility_recompute") is None


def test_utility_recompute_invalid_schedule_does_not_register(monkeypatch):
    sched = _scheduler({"MCP_UTILITY_RECOMPUTE_SCHEDULE": "notaninterval"}, monkeypatch)
    sched._schedule_utility_recompute_job()
    assert sched.scheduler.get_job("utility_recompute") is None


@pytest.mark.asyncio
async def test_utility_recompute_registered_on_start(monkeypatch):
    """start() wires the utility recompute job alongside the others."""
    sched = _scheduler({"MCP_UTILITY_RECOMPUTE_SCHEDULE": "6h"}, monkeypatch)
    called = {"flag": False}
    orig = sched._schedule_utility_recompute_job

    def spy():
        called["flag"] = True
        return orig()

    sched._schedule_utility_recompute_job = spy
    await sched.start()
    assert called["flag"], "start() must call _schedule_utility_recompute_job"


@pytest.mark.asyncio
async def test_run_utility_recompute_calls_recompute(monkeypatch):
    """_run_utility_recompute builds a BeliefService and calls recompute_utility_priors."""
    sched = _scheduler({}, monkeypatch)
    captured = {}

    class FakeBeliefService:
        def __init__(self, storage):
            captured["storage"] = storage
        async def recompute_utility_priors(self):
            captured["called"] = True
            return {"updated": 7, "errors": []}

    with patch("mcp_memory_service.consolidation.scheduler.BeliefService", FakeBeliefService):
        await sched._run_utility_recompute()

    assert captured.get("called") is True, "recompute_utility_priors must be called"
    assert sched.execution_stats["successful_jobs"] == 1


@pytest.mark.asyncio
async def test_run_utility_recompute_error_does_not_raise(monkeypatch):
    """A failure inside the recompute job must be swallowed, never re-raised —
    otherwise it would tear down the shared scheduler."""
    sched = _scheduler({}, monkeypatch)

    class BoomBeliefService:
        def __init__(self, storage): pass
        async def recompute_utility_priors(self):
            raise RuntimeError("boom")

    with patch("mcp_memory_service.consolidation.scheduler.BeliefService", BoomBeliefService):
        await sched._run_utility_recompute()  # must not raise
    assert sched.execution_stats["failed_jobs"] == 1


@pytest.mark.asyncio
async def test_run_utility_recompute_skips_without_storage(monkeypatch):
    """No storage on consolidator → skip gracefully (no crash)."""
    consolidator = MagicMock()
    consolidator.storage = None
    sched = ConsolidationScheduler(consolidator=consolidator, schedule_config={}, enabled=True)
    await sched._run_utility_recompute()  # should not raise
    assert sched.execution_stats["failed_jobs"] == 0
