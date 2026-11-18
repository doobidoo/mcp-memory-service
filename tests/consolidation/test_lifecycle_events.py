"""Consolidation lifecycle events over SSE (#2011).

consolidate() owns the lifecycle: one ``consolidation_started``, one
``consolidation_phase`` when a phase that runs starts, and exactly one
``consolidation_completed`` with the same ``run_id`` for every announced start,
whoever called it and however the run ended.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from mcp_memory_service.consolidation.base import ConsolidationError
from mcp_memory_service.consolidation.consolidator import (
    CONSOLIDATION_COMPLETED,
    CONSOLIDATION_PHASE,
    CONSOLIDATION_STARTED,
    DreamInspiredConsolidator,
)
from mcp_memory_service.consolidation.run_tracker import RunTracker

PHASE_ORDER = ["relevance", "clustering", "associations", "compression", "forgetting", "finalizing"]


class Recorder:
    """A notifier that keeps every event it receives."""

    def __init__(self):
        self.events = []

    async def __call__(self, event_type, data):
        self.events.append((event_type, dict(data)))

    def of(self, event_type):
        return [data for kind, data in self.events if kind == event_type]

    def phases(self, run_id=None):
        return [d["phase"] for d in self.of(CONSOLIDATION_PHASE) if run_id in (None, d["run_id"])]


@pytest.fixture
def recorder():
    return Recorder()


@pytest.fixture
def consolidator(mock_storage, consolidation_config, recorder):
    return DreamInspiredConsolidator(mock_storage, consolidation_config, notifier=recorder)


def assert_one_lifecycle(recorder, time_horizon, success):
    """Start first, completion last, one of each, one run_id throughout."""
    kinds = [kind for kind, _ in recorder.events]
    assert kinds[0] == CONSOLIDATION_STARTED
    assert kinds[-1] == CONSOLIDATION_COMPLETED
    assert kinds.count(CONSOLIDATION_STARTED) == 1
    assert kinds.count(CONSOLIDATION_COMPLETED) == 1
    run_ids = {data["run_id"] for _, data in recorder.events}
    assert len(run_ids) == 1 and all(run_ids)
    assert {data["time_horizon"] for _, data in recorder.events} == {time_horizon}
    completed = recorder.of(CONSOLIDATION_COMPLETED)[0]
    # The #2006 handler reads these two fields; they keep their names and types.
    assert completed["success"] is success
    assert set(completed) == {"run_id", "time_horizon", "success"}


# ---- What a run announces ----


@pytest.mark.asyncio
async def test_a_run_announces_its_start_each_phase_that_runs_and_one_end(consolidator, recorder):
    report = await consolidator.consolidate("weekly")

    assert report.errors == []
    assert_one_lifecycle(recorder, "weekly", success=True)
    phases = recorder.phases()
    assert phases == [p for p in PHASE_ORDER if p in phases], "phases arrive in schedule order"
    assert phases[0] == "relevance"
    assert phases[-1] == "finalizing", "the run stays busy through finalization"
    assert "forgetting" not in phases, "weekly does not run forgetting"
    assert ("compression" in phases) == (report.clusters_created > 0)


@pytest.mark.asyncio
async def test_disabled_relevance_and_compression_without_clusters_announce_nothing(
    mock_storage, consolidation_config, recorder
):
    consolidation_config.decay_enabled = False
    consolidator = DreamInspiredConsolidator(mock_storage, consolidation_config, notifier=recorder)
    consolidator.clustering_engine.process = AsyncMock(return_value=[])

    await consolidator.consolidate("monthly")

    assert_one_lifecycle(recorder, "monthly", success=True)
    assert recorder.phases() == ["clustering", "associations", "forgetting", "finalizing"]


@pytest.mark.asyncio
async def test_an_empty_selection_still_ends_its_run(consolidator, recorder):
    consolidator._get_memories_for_horizon = AsyncMock(return_value=[])

    report = await consolidator.consolidate("weekly")

    assert report.memories_processed == 0
    assert_one_lifecycle(recorder, "weekly", success=True)
    assert recorder.phases() == []


@pytest.mark.asyncio
async def test_a_report_with_errors_is_not_announced_as_success(consolidator, recorder):
    consolidator._run_phase_schedule = AsyncMock(side_effect=RuntimeError("clustering broke"))

    report = await consolidator.consolidate("weekly")

    assert report.errors, "the error is in the returned report"
    assert_one_lifecycle(recorder, "weekly", success=False)


@pytest.mark.asyncio
async def test_a_raised_configuration_error_still_ends_its_run(consolidator, recorder):
    consolidator._get_memories_for_horizon = AsyncMock(side_effect=ConsolidationError("bad horizon"))

    with pytest.raises(ConsolidationError):
        await consolidator.consolidate("weekly")

    assert_one_lifecycle(recorder, "weekly", success=False)


@pytest.mark.asyncio
async def test_a_cancelled_run_ends_once_as_failed(consolidator, recorder):
    async def never_finishes(*args, **kwargs):
        await asyncio.sleep(60)

    consolidator._run_phase_schedule = never_finishes

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(consolidator.consolidate("weekly"), timeout=0.05)

    assert_one_lifecycle(recorder, "weekly", success=False)


@pytest.mark.asyncio
async def test_a_failing_notifier_does_not_fail_consolidation(mock_storage, consolidation_config):
    calls = []

    async def broken(event_type, data):
        calls.append(event_type)
        raise RuntimeError("SSE unavailable")

    consolidator = DreamInspiredConsolidator(mock_storage, consolidation_config, notifier=broken)

    report = await consolidator.consolidate("weekly")

    assert report.errors == []
    assert report.performance_metrics["success"] is True
    assert calls[0] == CONSOLIDATION_STARTED and calls[-1] == CONSOLIDATION_COMPLETED


@pytest.mark.asyncio
async def test_without_a_notifier_nothing_is_announced_and_the_run_is_unchanged(
    mock_storage, consolidation_config
):
    consolidator = DreamInspiredConsolidator(mock_storage, consolidation_config)
    assert consolidator.notifier is None

    report = await consolidator.consolidate("weekly")

    assert report.errors == []


# ---- Overlapping runs ----


@pytest.mark.asyncio
async def test_a_rejected_concurrent_incremental_run_announces_nothing(
    consolidator, recorder, tmp_path
):
    consolidator.run_tracker = RunTracker(tmp_path / "consolidation_runs.db")
    assert consolidator.run_tracker.try_acquire("incremental")  # the active run holds the slot

    report = await consolidator.consolidate("incremental")

    assert report.errors == ["Skipped: concurrent run in flight"]
    assert recorder.events == [], "no start, and above all no completion for the active run"


@pytest.mark.asyncio
async def test_overlapping_runs_each_end_once_under_their_own_id(consolidator, recorder):
    await asyncio.gather(consolidator.consolidate("weekly"), consolidator.consolidate("monthly"))

    started = recorder.of(CONSOLIDATION_STARTED)
    completed = recorder.of(CONSOLIDATION_COMPLETED)
    assert len(started) == 2 and len(completed) == 2
    assert started[0]["run_id"] != started[1]["run_id"]
    assert sorted(d["run_id"] for d in completed) == sorted(d["run_id"] for d in started)
    for start in started:
        phases = recorder.phases(start["run_id"])
        assert phases and phases[-1] == "finalizing"
        assert all(
            d["time_horizon"] == start["time_horizon"]
            for kind, d in recorder.events if d["run_id"] == start["run_id"]
        )


# ---- The four callers of consolidate() ----


class _FakeSSEManager:
    def __init__(self):
        self.events = []

    async def broadcast_event(self, event):
        self.events.append(event)


@pytest.fixture
def capture_sse(monkeypatch):
    manager = _FakeSSEManager()
    monkeypatch.setattr("mcp_memory_service.web.sse.sse_manager", manager)
    return manager


@pytest.mark.asyncio
async def test_the_web_notifier_broadcasts_system_wide_events(capture_sse):
    from mcp_memory_service.web.sse import broadcast_consolidation_event

    data = {"run_id": "r1", "time_horizon": "daily", "phase": "clustering"}
    await broadcast_consolidation_event(CONSOLIDATION_PHASE, data)

    (event,) = capture_sse.events
    assert event.event_type == CONSOLIDATION_PHASE
    assert event.data == data
    assert event.store is None


@pytest.mark.asyncio
async def test_a_scheduled_run_ends_once_through_consolidate(
    mock_storage, consolidation_config, capture_sse
):
    from mcp_memory_service.consolidation.scheduler import ConsolidationScheduler
    from mcp_memory_service.web.sse import broadcast_consolidation_event

    consolidator = DreamInspiredConsolidator(
        mock_storage, consolidation_config, notifier=broadcast_consolidation_event
    )
    scheduler = ConsolidationScheduler(consolidator, schedule_config={}, enabled=True)

    await scheduler._run_consolidation_job("weekly")

    kinds = [event.event_type for event in capture_sse.events]
    assert kinds[0] == CONSOLIDATION_STARTED
    assert kinds.count(CONSOLIDATION_COMPLETED) == 1, "the scheduler no longer sends its own"
    assert kinds[-1] == CONSOLIDATION_COMPLETED
    assert capture_sse.events[-1].data["success"] is True


@pytest.mark.asyncio
async def test_a_run_started_from_the_api_is_announced(consolidator, recorder, monkeypatch):
    from mcp_memory_service.api import operations

    monkeypatch.setattr(operations, "get_consolidator", lambda: consolidator)

    await operations._consolidate_async("weekly")

    assert_one_lifecycle(recorder, "weekly", success=True)


@pytest.mark.asyncio
async def test_a_run_started_from_the_mcp_tool_is_announced(consolidator, recorder, monkeypatch):
    from mcp_memory_service.server.handlers import consolidation as handlers

    monkeypatch.setattr(handlers, "CONSOLIDATION_ENABLED", True)
    server = MagicMock()
    server.consolidator = consolidator

    await handlers.handle_consolidate_memories(server, {"time_horizon": "weekly"})

    assert_one_lifecycle(recorder, "weekly", success=True)
