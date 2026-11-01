"""Wiring of belief embedding backfill + semantic dedup into the consolidation job.

#1733 added the dedup primitive (library layer) but nothing called it. This PR
wires it into ConsolidationScheduler._run_consolidation_job, right after
derive_beliefs(). These tests pin the contract the maintainer asked for:

- off by default: no MCP_BELIEF_DEDUP -> backfill/dedup never run;
- on + no apply switch: backfill runs, dedup runs DRY-RUN (reports, no writes);
- on + apply switch: dedup runs with dry_run=False (merges);
- the job record carries the backfill and dedup stats;
- a dedup failure is non-fatal (job still succeeds, warning logged).
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_scheduler(monkeypatch, belief_svc):
    from mcp_memory_service.consolidation import scheduler as sched_mod

    report = SimpleNamespace(
        memories_processed=4, associations_discovered=1, clusters_created=0,
        memories_compressed=0, memories_archived=0, errors=[],
    )
    consolidator = MagicMock()
    consolidator.consolidate = AsyncMock(return_value=report)
    consolidator.storage = MagicMock()

    monkeypatch.setattr(sched_mod, "BeliefService", lambda storage, graph=None: belief_svc)
    monkeypatch.setenv("MCP_BELIEFS_ENABLED", "true")
    return sched_mod.ConsolidationScheduler(consolidator, schedule_config={}, enabled=False)


def _belief_svc(env):
    """Fake BeliefService whose _env_flag_on reads the provided env dict, with
    AsyncMock derive_beliefs/backfill and a MagicMock dedup_beliefs."""
    svc = MagicMock()
    svc.derive_beliefs = AsyncMock(return_value={"beliefs_derived": 3})
    svc.backfill_belief_embeddings = AsyncMock(return_value={"filled": 2})
    svc.dedup_beliefs = MagicMock(return_value={"groups": 1, "merged": 2})
    svc._env_flag_on = lambda name: str(env.get(name, "")).lower() in ("true", "1", "yes")
    return svc


@pytest.mark.asyncio
async def test_dedup_off_by_default_not_called(monkeypatch):
    svc = _belief_svc({})  # MCP_BELIEF_DEDUP unset
    scheduler = _make_scheduler(monkeypatch, svc)

    await scheduler._run_consolidation_job("weekly")

    svc.backfill_belief_embeddings.assert_not_called()
    svc.dedup_beliefs.assert_not_called()
    assert "dedup" not in scheduler.job_history[-1]["beliefs"]


@pytest.mark.asyncio
async def test_dedup_on_runs_backfill_then_dry_run(monkeypatch):
    svc = _belief_svc({"MCP_BELIEF_DEDUP": "true"})  # apply switch OFF
    scheduler = _make_scheduler(monkeypatch, svc)

    await scheduler._run_consolidation_job("weekly")

    svc.backfill_belief_embeddings.assert_awaited_once()
    svc.dedup_beliefs.assert_called_once_with(dry_run=True)
    beliefs = scheduler.job_history[-1]["beliefs"]
    assert beliefs["dedup_backfill"] == {"filled": 2}
    assert beliefs["dedup"] == {"groups": 1, "merged": 2}


@pytest.mark.asyncio
async def test_dedup_apply_switch_runs_non_dry(monkeypatch):
    svc = _belief_svc({"MCP_BELIEF_DEDUP": "true", "MCP_BELIEF_DEDUP_APPLY": "true"})
    scheduler = _make_scheduler(monkeypatch, svc)

    await scheduler._run_consolidation_job("weekly")

    svc.dedup_beliefs.assert_called_once_with(dry_run=False)


@pytest.mark.asyncio
async def test_dedup_failure_is_non_fatal(monkeypatch):
    svc = _belief_svc({"MCP_BELIEF_DEDUP": "true"})
    svc.backfill_belief_embeddings = AsyncMock(side_effect=RuntimeError("boom"))
    scheduler = _make_scheduler(monkeypatch, svc)

    await scheduler._run_consolidation_job("weekly")

    # Job still succeeds despite the dedup error; derivation stats survive.
    assert scheduler.job_history[-1]["status"] == "success"
    assert scheduler.job_history[-1]["beliefs"]["beliefs_derived"] == 3
