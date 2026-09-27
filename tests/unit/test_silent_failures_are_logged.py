"""Three places that used to swallow a failure or a result without a trace.

- scheduler: belief derivation ran and its stats were discarded (#1149)
- server_impl: the session-counter reset failed silently, so the fresh-start
  trigger kept firing with nothing in the log (#1150)
"""
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_job_record_carries_belief_stats(monkeypatch):
    from mcp_memory_service.consolidation import scheduler as sched_mod

    report = SimpleNamespace(
        memories_processed=4, associations_discovered=1, clusters_created=0,
        memories_compressed=0, memories_archived=0, errors=[],
    )
    consolidator = MagicMock()
    consolidator.consolidate = AsyncMock(return_value=report)
    consolidator.storage = MagicMock()

    fake_service = MagicMock()
    fake_service.derive_beliefs = AsyncMock(return_value={"beliefs_derived": 3})
    monkeypatch.setattr(sched_mod, "BeliefService", lambda storage: fake_service)
    monkeypatch.setenv("MCP_BELIEFS_ENABLED", "true")

    scheduler = sched_mod.ConsolidationScheduler(consolidator, schedule_config={}, enabled=False)
    await scheduler._run_consolidation_job("weekly")

    assert scheduler.job_history[-1]["status"] == "success"
    assert scheduler.job_history[-1]["beliefs"] == {"beliefs_derived": 3}


@pytest.mark.asyncio
async def test_session_counter_reset_failure_is_logged(caplog):
    from mcp_memory_service.server import MemoryServer

    fake_self = SimpleNamespace(
        memory_service=SimpleNamespace(
            retrieve_memories=AsyncMock(side_effect=RuntimeError("storage offline")),
        ),
        storage=MagicMock(),
    )
    # server_impl logs through logging_config's own handler and does not propagate,
    # so attach caplog's handler to that logger for the duration of the call.
    target = logging.getLogger("mcp_memory_service.server.logging_config")
    target.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger=target.name):
            await MemoryServer._reset_session_counter(fake_self, "agent-7\nforged")
    finally:
        target.removeHandler(caplog.handler)

    text = caplog.text
    assert "Session counter reset for agent-7" in text
    assert "storage offline" in text
    assert "\nforged" not in text  # the agent id is sanitized before it reaches the log
