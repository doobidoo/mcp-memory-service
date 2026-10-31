"""
Issue #1557: Web GUI auto-refresh driven by delta-sync completions.

Covers the backend half of the issue proposal:
- ``create_sync_delta_completed_event`` builds a per-peer completion event
  (event type ``sync_delta_completed`` with pulled/pushed counts).
- The scheduler broadcasts that event over SSE at the end of each scheduled
  cycle — success and failure alike.
- Consolidation completion notifies the dashboard as well.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

import mcp_memory_service.storage.sync.orchestrator as orch
from mcp_memory_service.consolidation.scheduler import ConsolidationScheduler
from mcp_memory_service.web.sse import create_sync_delta_completed_event


class _FakeConsolidator:
    def __init__(self, storage=None):
        self.storage = storage
        self.health_monitor = None


def _make_scheduler():
    sch = ConsolidationScheduler(
        _FakeConsolidator(storage=MagicMock()), schedule_config={}, enabled=True
    )
    return sch


class _FakeSSEManager:
    """Captures every event the scheduler broadcasts."""

    def __init__(self):
        self.events = []

    async def broadcast_event(self, event):
        self.events.append(event)


@pytest.fixture
def capture_sse(monkeypatch):
    manager = _FakeSSEManager()
    monkeypatch.setattr("mcp_memory_service.web.sse.sse_manager", manager)
    return manager


# --- Event helper ---


def test_delta_completed_event_payload():
    event = create_sync_delta_completed_event(
        peer_id="hub-vps", pulled=12, pushed=3, time_taken_seconds=1.234
    )
    assert event.event_type == "sync_delta_completed"
    assert event.data["peer_id"] == "hub-vps"
    assert event.data["pulled"] == 12
    assert event.data["pushed"] == 3
    assert event.data["time_taken_seconds"] == 1.23
    assert event.data["success"] is True
    assert "pulled 12" in event.data["message"]
    assert "pushed 3" in event.data["message"]


def test_delta_completed_event_failure_message():
    event = create_sync_delta_completed_event(
        peer_id="hub-vps", pulled=0, pushed=0, success=False
    )
    assert event.data["success"] is False
    assert event.data["message"] == "Delta-sync with hub-vps failed"


# --- Scheduler broadcasts ---


def _install_fake_peer(sched, monkeypatch, *, compatibility=None):
    """Register one fake peer whose pull/push return the given values."""
    fake_peer = MagicMock()

    async def _init(): return None

    async def _close(): return None

    async def _check(storage): return compatibility if compatibility is not None else {}

    fake_peer.initialize = _init
    fake_peer.close = _close
    fake_peer.check_sync_compatibility = _check
    monkeypatch.setattr(sched, "_resolve_sync_peers", lambda: [("hub-vps", fake_peer)])
    return fake_peer


@pytest.mark.asyncio
async def test_scheduler_broadcasts_delta_completed_on_success(monkeypatch, capture_sse):
    async def fake_sync_from_peer(storage, peer, peer_id, limit):
        return MagicMock(events_received=7, events_applied=7, events_failed=0)

    async def fake_push_to_peer(storage, peer, peer_id, limit):
        return MagicMock(events_pushed=2, events_failed=0)

    monkeypatch.setattr(orch, "sync_from_peer", fake_sync_from_peer)
    monkeypatch.setattr(orch, "push_to_peer", fake_push_to_peer)

    sched = _make_scheduler()
    _install_fake_peer(sched, monkeypatch)
    await sched._run_sync_cycle()

    # First event of the cycle is the SSE notification, not the peer close.
    assert len(capture_sse.events) == 1, f"expected one sync_delta_completed broadcast, got {capture_sse.events}"
    event = capture_sse.events[0]
    assert event.event_type == "sync_delta_completed"
    assert event.data["peer_id"] == "hub-vps"
    assert event.data["pulled"] == 7
    assert event.data["pushed"] == 2
    assert event.data["success"] is True


@pytest.mark.asyncio
async def test_scheduler_broadcasts_delta_completed_on_failure(monkeypatch, capture_sse):
    async def boom(storage, peer, peer_id, limit):
        raise RuntimeError("network down")

    monkeypatch.setattr(orch, "sync_from_peer", boom)
    monkeypatch.setattr(orch, "push_to_peer", boom)

    sched = _make_scheduler()
    _install_fake_peer(sched, monkeypatch)
    await sched._run_sync_cycle()  # must not raise

    assert len(capture_sse.events) == 1, f"expected failure broadcast, got {capture_sse.events}"
    event = capture_sse.events[0]
    assert event.event_type == "sync_delta_completed"
    assert event.data["peer_id"] == "hub-vps"
    assert event.data["success"] is False


@pytest.mark.asyncio
async def test_partial_pull_counts_survive_push_failure(monkeypatch, capture_sse):
    async def pull(*args, **kwargs):
        return MagicMock(events_received=7, events_applied=7, events_failed=0)

    async def push(*args, **kwargs):
        raise RuntimeError('push unavailable')

    monkeypatch.setattr(orch, 'sync_from_peer', pull)
    monkeypatch.setattr(orch, 'push_to_peer', push)
    sched = _make_scheduler()
    _install_fake_peer(sched, monkeypatch)
    await sched._run_sync_cycle()
    event = capture_sse.events[0]
    assert event.data['success'] is False
    assert event.data['pulled'] == 7
    assert event.data['pushed'] is None


@pytest.mark.asyncio
async def test_consolidation_notifies_dashboard(capture_sse):
    from unittest.mock import AsyncMock

    sched = _make_scheduler()
    sched.consolidator.consolidate = AsyncMock(return_value=MagicMock())
    await sched._run_consolidation_job('daily')
    assert capture_sse.events[0].event_type == 'consolidation_completed'
    assert capture_sse.events[0].data == {'time_horizon': 'daily', 'success': True}


@pytest.mark.asyncio
async def test_sse_failure_does_not_change_successful_sync(monkeypatch):
    from unittest.mock import AsyncMock

    sched = _make_scheduler()
    peer = _install_fake_peer(sched, monkeypatch)
    peer.close = AsyncMock()
    monkeypatch.setattr(orch, 'sync_from_peer', AsyncMock(
        return_value=MagicMock(events_received=1, events_applied=1, events_failed=0)))
    monkeypatch.setattr(orch, 'push_to_peer', AsyncMock(
        return_value=MagicMock(events_pushed=0, events_failed=0)))
    monkeypatch.setattr('mcp_memory_service.web.sse.sse_manager.broadcast_event',
                        AsyncMock(side_effect=RuntimeError('SSE unavailable')))
    await sched._run_sync_cycle()
    assert sched.get_sync_peer_status()[0]['cycle_status'] == 'synced'
    assert sched.execution_stats['failed_jobs'] == 0
    peer.close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize('failed_direction', ['pull', 'push'])
async def test_paginated_transport_failure_reports_unknown_counts(
    failed_direction, temp_db_path, monkeypatch, capture_sse
):
    """Run real exchanges: page one commits, then the peer fails page two."""
    from mcp_memory_service.models.memory import Memory
    from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
    from mcp_memory_service.utils.hashing import generate_content_hash

    monkeypatch.setenv('MCP_SYNC_EVENTLOG', 'on')
    storage = SqliteVecMemoryStorage(str(Path(temp_db_path) / "sync.db"))
    await storage.initialize()
    try:
        for index in range(2):
            content = f'Paginated SSE regression {index}'
            await storage.store(Memory(content=content, content_hash=generate_content_hash(content)))
        events = orch._read_local_events_since(storage, 0, 100)
        assert len(events) == 2

        class FailingPeer:
            calls = 0

            async def initialize(self): pass
            async def close(self): pass
            async def check_sync_compatibility(self, local): return {}

            async def get_events_since(self, since_seq, limit):
                if failed_direction != 'pull':
                    return [], since_seq, False
                self.calls += 1
                if self.calls == 1:
                    return events[:1], events[0]['seq'], True
                raise RuntimeError('second pull page unavailable')

            async def push_events(self, batch):
                self.calls += 1
                if self.calls == 1:
                    return {'failed': 0}
                raise RuntimeError('second push page unavailable')

        async def pull(local, peer, peer_id, limit):
            return await orch.sync_from_peer(local, peer, peer_id, limit=1)

        async def push(local, peer, peer_id, limit):
            return await orch.push_to_peer(local, peer, peer_id, limit=1)

        sched = _make_scheduler()
        await sched._run_sync_peer_cycle(storage, 'hub', FailingPeer(), pull, push)
        # The durable cursor proves the first page succeeded before the error.
        if failed_direction == 'pull':
            cursor = storage.conn.execute(
                'SELECT last_seq_seen FROM sync_cursor WHERE peer_id = ?', ('hub',)
            ).fetchone()[0]
        else:
            cursor = orch.get_push_cursor(storage, 'hub')
        assert cursor == events[0]['seq']
        event = capture_sse.events[0]
        assert event.data['success'] is False
        assert event.data['pushed'] is None
        assert event.data['pulled'] == (None if failed_direction == 'pull' else 0)
    finally:
        await storage.close()
