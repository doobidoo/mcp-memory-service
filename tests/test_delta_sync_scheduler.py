"""
Tests for delta-sync Phase 4d: scheduled sync cycle (pull + push).

Covers CA1-CA5 from spec-delta-sync-fase4d.md (ADR-0028). CA6 (hot E2E) is manual.
Tests exercise the scheduler's sync methods in isolation with fakes — no real APScheduler
timing, no network.
"""

import pytest
import os
from unittest.mock import MagicMock

from mcp_memory_service.consolidation.scheduler import ConsolidationScheduler


class _FakeConsolidator:
    def __init__(self, storage=None):
        self.storage = storage
        self.health_monitor = None


def _make_scheduler():
    # enabled=True builds a real AsyncIOScheduler (no jobs run until .start()); we only
    # inspect add_job registration, never start it.
    sch = ConsolidationScheduler(_FakeConsolidator(storage=MagicMock()), schedule_config={}, enabled=True)
    return sch


# CA1 — opt-in registration
def test_ca1_schedule_registers_when_env_set(monkeypatch):
    monkeypatch.setenv("MCP_SYNC_SCHEDULE", "15m")
    sch = _make_scheduler()
    if sch.scheduler is None:
        pytest.skip("APScheduler not available in this env")
    sch._schedule_sync_job()
    job = sch.scheduler.get_job("delta_sync")
    assert job is not None, "delta_sync job must be registered when MCP_SYNC_SCHEDULE set"


def test_ca1_no_job_when_env_unset(monkeypatch):
    monkeypatch.delenv("MCP_SYNC_SCHEDULE", raising=False)
    sch = _make_scheduler()
    if sch.scheduler is None:
        pytest.skip("APScheduler not available")
    sch._schedule_sync_job()
    assert sch.scheduler.get_job("delta_sync") is None, "no job when env unset (default off)"


def test_ca1_no_job_when_disabled(monkeypatch):
    monkeypatch.setenv("MCP_SYNC_SCHEDULE", "disabled")
    sch = _make_scheduler()
    if sch.scheduler is None:
        pytest.skip("APScheduler not available")
    sch._schedule_sync_job()
    assert sch.scheduler.get_job("delta_sync") is None


# CA2 — invalid interval is safe
def test_ca2_invalid_interval_no_job_no_crash(monkeypatch):
    monkeypatch.setenv("MCP_SYNC_SCHEDULE", "banana")
    sch = _make_scheduler()
    if sch.scheduler is None:
        pytest.skip("APScheduler not available")
    sch._schedule_sync_job()  # must not raise
    assert sch.scheduler.get_job("delta_sync") is None


# CA3 — cycle calls pull before push, per peer
@pytest.mark.asyncio
async def test_ca3_cycle_pull_before_push(monkeypatch):
    calls = []

    async def fake_sync_from_peer(storage, peer, peer_id, limit):
        calls.append(("pull", peer_id)); return MagicMock(events_applied=1)

    async def fake_push_to_peer(storage, peer, peer_id, limit):
        calls.append(("push", peer_id)); return MagicMock(events_pushed=1)

    import mcp_memory_service.storage.sync.orchestrator as orch
    monkeypatch.setattr(orch, "sync_from_peer", fake_sync_from_peer)
    monkeypatch.setattr(orch, "push_to_peer", fake_push_to_peer)

    sch = _make_scheduler()
    # one fake peer
    fake_peer = MagicMock()
    async def _init(): return None
    async def _close(): return None
    fake_peer.initialize = _init
    fake_peer.close = _close
    monkeypatch.setattr(sch, "_resolve_sync_peers", lambda: [("hub-vps", fake_peer)])

    await sch._run_sync_cycle()
    assert calls == [("pull", "hub-vps"), ("push", "hub-vps")], f"got {calls}"


# CA4 — peers from env; unset = zero peers
def test_ca4_peers_from_env(monkeypatch):
    sch = _make_scheduler()
    monkeypatch.delenv("MCP_SYNC_PEERS", raising=False)
    assert sch._resolve_sync_peers() == [], "unset MCP_SYNC_PEERS → no peers"
    monkeypatch.setenv("MCP_SYNC_PEERS", "hub-vps")
    monkeypatch.setenv("MCP_SYNC_PEER_URL", "https://cfnarede.dev/memory")
    monkeypatch.setenv("MCP_API_KEY", "k")
    peers = sch._resolve_sync_peers()
    assert len(peers) == 1 and peers[0][0] == "hub-vps"


# CA5 — a peer error is logged and does not propagate (scheduler survives)
@pytest.mark.asyncio
async def test_ca5_peer_error_does_not_propagate(monkeypatch):
    async def boom(storage, peer, peer_id, limit):
        raise RuntimeError("network down")
    import mcp_memory_service.storage.sync.orchestrator as orch
    monkeypatch.setattr(orch, "sync_from_peer", boom)

    sch = _make_scheduler()
    fake_peer = MagicMock()
    async def _init(): return None
    async def _close(): return None
    fake_peer.initialize = _init
    fake_peer.close = _close
    monkeypatch.setattr(sch, "_resolve_sync_peers", lambda: [("hub-vps", fake_peer)])

    # must NOT raise
    await sch._run_sync_cycle()
    assert sch.execution_stats["failed_jobs"] >= 1
