"""Wiring of belief embedding backfill + semantic dedup into the consolidation job.

#1733 added the dedup primitive (library layer) but nothing called it. This PR
wires it into ConsolidationScheduler._run_consolidation_job, right after
derive_beliefs(). These tests pin the contract the maintainer asked for:

- off by default: no MCP_BELIEF_DEDUP -> backfill/dedup never run;
- on + no apply switch: backfill runs FIRST, dedup runs DRY-RUN (reports, no writes);
- on + apply switch: dedup runs with dry_run=False (merges);
- backfill honours the configured cap (MCP_BELIEF_DEDUP_CAP), not its 1000 default;
- the job record carries the backfill and dedup stats;
- a backfill failure AND a dedup failure are each non-fatal (job still succeeds).
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


def _belief_svc(env, cap=1000, calls=None):
    """Fake BeliefService whose _env_flag_on/_dedup_config read the provided env.

    ``calls`` (optional list) records the ordered sequence of ("backfill"|"dedup")
    so a test can assert backfill ran to completion before dedup.
    """
    svc = MagicMock()
    svc.derive_beliefs = AsyncMock(return_value={"beliefs_derived": 3})

    async def _backfill(cap=1000):
        if calls is not None:
            calls.append("backfill")
        return {"filled": 2, "cap": cap}
    svc.backfill_belief_embeddings = AsyncMock(side_effect=_backfill)

    def _dedup(dry_run=True):
        if calls is not None:
            calls.append("dedup")
        return {"groups": 1, "merged": 2, "dry_run": dry_run}
    svc.dedup_beliefs = MagicMock(side_effect=_dedup)

    svc._env_flag_on = lambda name: str(env.get(name, "")).lower() in ("true", "1", "yes")
    svc._dedup_config = lambda: (0.90, cap)
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
async def test_dedup_on_runs_backfill_before_dedup_dry_run(monkeypatch):
    calls = []
    svc = _belief_svc({"MCP_BELIEF_DEDUP": "true"}, calls=calls)  # apply switch OFF
    scheduler = _make_scheduler(monkeypatch, svc)

    await scheduler._run_consolidation_job("weekly")

    # Order matters: backfill must complete before dedup scans (missing
    # embeddings would otherwise be skipped). Greptile P2.
    assert calls == ["backfill", "dedup"]
    svc.dedup_beliefs.assert_called_once_with(dry_run=True)
    beliefs = scheduler.job_history[-1]["beliefs"]
    assert beliefs["dedup_backfill"]["filled"] == 2
    assert beliefs["dedup"]["groups"] == 1


@pytest.mark.asyncio
async def test_backfill_receives_configured_cap(monkeypatch):
    svc = _belief_svc({"MCP_BELIEF_DEDUP": "true"}, cap=10)
    scheduler = _make_scheduler(monkeypatch, svc)

    await scheduler._run_consolidation_job("weekly")

    # The documented bound must reach backfill, not just dedup (Greptile P2):
    # cap=10 cannot let 1000 embeddings be computed per job.
    svc.backfill_belief_embeddings.assert_awaited_once_with(cap=10)
    assert scheduler.job_history[-1]["beliefs"]["dedup_backfill"]["cap"] == 10


@pytest.mark.asyncio
async def test_dedup_apply_switch_runs_non_dry(monkeypatch):
    svc = _belief_svc({"MCP_BELIEF_DEDUP": "true", "MCP_BELIEF_DEDUP_APPLY": "true"})
    scheduler = _make_scheduler(monkeypatch, svc)

    await scheduler._run_consolidation_job("weekly")

    svc.dedup_beliefs.assert_called_once_with(dry_run=False)


@pytest.mark.asyncio
async def test_backfill_failure_is_non_fatal(monkeypatch):
    svc = _belief_svc({"MCP_BELIEF_DEDUP": "true"})
    svc.backfill_belief_embeddings = AsyncMock(side_effect=RuntimeError("boom"))
    scheduler = _make_scheduler(monkeypatch, svc)

    await scheduler._run_consolidation_job("weekly")

    # Job still succeeds despite the backfill error; derivation stats survive.
    assert scheduler.job_history[-1]["status"] == "success"
    assert scheduler.job_history[-1]["beliefs"]["beliefs_derived"] == 3


@pytest.mark.asyncio
async def test_dedup_failure_is_non_fatal(monkeypatch):
    # Backfill SUCCEEDS so dedup actually runs and is the one that raises —
    # this exercises the advertised dedup-failure path (Greptile P2).
    svc = _belief_svc({"MCP_BELIEF_DEDUP": "true"})
    svc.dedup_beliefs = MagicMock(side_effect=RuntimeError("dedup boom"))
    scheduler = _make_scheduler(monkeypatch, svc)

    await scheduler._run_consolidation_job("weekly")

    assert scheduler.job_history[-1]["status"] == "success"
    assert scheduler.job_history[-1]["beliefs"]["beliefs_derived"] == 3
    # Backfill ran and its stats survived even though dedup blew up.
    svc.backfill_belief_embeddings.assert_awaited_once()
