"""
Issue #1773: a supersede that happens after the first sync must reach the other peers.

#1726 made the baseline snapshot and the create path carry ``superseded_by`` /
``superseded_at``, which covers rows that were already superseded when first synced.
These tests cover a supersede made later through each of the three writers, and two
peers superseding the same memory with different winners.
"""

import json
import os
import tempfile

import pytest
import pytest_asyncio

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.storage.sync.apply import apply_remote_event
from mcp_memory_service.storage.sync.orchestrator import _read_local_events_since
from mcp_memory_service.utils.hashing import generate_content_hash


@pytest_asyncio.fixture
async def peers(monkeypatch):
    """Two empty sqlite-vec stores with the event log on."""
    monkeypatch.setenv("MCP_SYNC_EVENTLOG", "true")
    monkeypatch.setenv("MCP_SEMANTIC_DEDUP_ENABLED", "false")
    with tempfile.TemporaryDirectory() as tmp:
        a = SqliteVecMemoryStorage(os.path.join(tmp, "a.db"))
        b = SqliteVecMemoryStorage(os.path.join(tmp, "b.db"))
        await a.initialize()
        await b.initialize()
        yield a, b
        await a.close()
        await b.close()


async def _store(storage, content):
    h = generate_content_hash(content)
    ok, msg = await storage.store(Memory(content=content, content_hash=h, tags=["t"], memory_type="note"))
    assert ok, msg
    return h


def _sync(src, dst):
    """Apply every event src has logged to dst (duplicates are idempotent no-ops)."""
    for event in _read_local_events_since(src, 0, 10_000):
        result = apply_remote_event(dst, event)
        assert result.applied or "Duplicate" in result.reason, result.reason


def _supersede_state(storage, content_hash):
    return storage.conn.execute(
        "SELECT superseded_by, superseded_at FROM memories WHERE content_hash = ?",
        (content_hash,),
    ).fetchone()


async def _supersede(writer, storage, loser, winner_content):
    """Supersede ``loser`` on ``storage`` through ``writer``; return the winner's hash."""
    if writer == "versioned":
        ok, msg, new_hash = await storage.update_memory_versioned(loser, winner_content)
        assert ok, msg
        return new_hash
    winner = await _store(storage, winner_content)
    if writer == "batch":
        assert await storage.mark_superseded_batch([(winner, loser)]) == 1
    else:
        ok, msg = await storage.resolve_conflict(winner, loser)
        assert ok, msg
    return winner


@pytest.mark.asyncio
@pytest.mark.parametrize("writer", ["batch", "resolve_conflict", "versioned"])
async def test_supersede_after_initial_sync_reaches_peer(peers, monkeypatch, writer):
    a, b = peers
    monkeypatch.setenv("MCP_AGENT_ID", "alpha")
    x = await _store(a, "Memory X: the deploy target is staging")
    _sync(a, b)
    assert _supersede_state(b, x) == (None, None)

    monkeypatch.setenv("MCP_AGENT_ID", "alpha")
    winner = await _supersede(writer, a, x, "Memory Y: the deploy target is production")
    superseded_by, superseded_at = _supersede_state(a, x)
    assert superseded_by == winner and superseded_at is not None

    # The event keeps superseded_at out of ``updates``: a peer without #1773 treats
    # superseded_by there as protected and never reads the top-level field.
    row = a.conn.execute(
        "SELECT payload FROM sync_events WHERE content_hash = ? AND op = 'update_metadata'",
        (x,),
    ).fetchone()
    payload = json.loads(row[0])
    assert payload["updates"] == {"superseded_by": winner}
    assert payload["superseded_at"] == superseded_at

    monkeypatch.setenv("MCP_AGENT_ID", "beta")
    _sync(a, b)
    # Same value on both peers: the receiver writes the origin's instant, not its own.
    assert _supersede_state(b, x) == (winner, superseded_at)


@pytest.mark.asyncio
async def test_retried_supersede_logs_no_second_event(peers, monkeypatch):
    a, _ = peers
    monkeypatch.setenv("MCP_AGENT_ID", "alpha")
    x = await _store(a, "Memory X: retry target")
    y = await _store(a, "Memory Y: retry winner")
    assert await a.mark_superseded_batch([(y, x)]) == 1
    assert await a.mark_superseded_batch([(y, x)]) == 1
    count = a.conn.execute(
        "SELECT COUNT(*) FROM sync_events WHERE content_hash = ? AND op = 'update_metadata'",
        (x,),
    ).fetchone()[0]
    assert count == 1


@pytest.mark.asyncio
async def test_conflicting_supersedes_converge_in_both_orders(monkeypatch):
    """Two peers supersede X with different winners; every arrival order ends the same."""
    monkeypatch.setenv("MCP_SYNC_EVENTLOG", "true")
    monkeypatch.setenv("MCP_SEMANTIC_DEDUP_ENABLED", "false")
    with tempfile.TemporaryDirectory() as tmp:
        stores = {}
        for name in ("a", "b", "c", "d"):
            stores[name] = SqliteVecMemoryStorage(os.path.join(tmp, f"{name}.db"))
            await stores[name].initialize()
        a, b, c, d = (stores[n] for n in ("a", "b", "c", "d"))
        try:
            monkeypatch.setenv("MCP_AGENT_ID", "alpha")
            x = await _store(a, "Memory X: contested")
            y = await _store(a, "Memory Y: alpha's replacement")
            z = await _store(a, "Memory Z: beta's replacement")
            for peer in (b, c, d):
                _sync(a, peer)

            monkeypatch.setenv("MCP_AGENT_ID", "alpha")
            assert await a.mark_superseded_batch([(y, x)]) == 1
            monkeypatch.setenv("MCP_AGENT_ID", "beta")
            assert await b.mark_superseded_batch([(z, x)]) == 1

            # c hears alpha first, d hears beta first; then a and b exchange.
            _sync(a, c)
            _sync(b, c)
            _sync(b, d)
            _sync(a, d)
            _sync(b, a)
            _sync(a, b)

            states = {name: _supersede_state(s, x) for name, s in stores.items()}
            assert len(set(states.values())) == 1, states
            superseded_by, superseded_at = states["a"]
            assert superseded_by in (y, z) and superseded_at is not None
        finally:
            for s in stores.values():
                await s.close()


def _metadata(storage, content_hash):
    row = storage.conn.execute(
        "SELECT metadata FROM memories WHERE content_hash = ?", (content_hash,)
    ).fetchone()
    return json.loads(row[0]) if row and row[0] else {}


@pytest.mark.asyncio
async def test_remote_set_and_clear_keep_legacy_metadata_in_step(peers, monkeypatch):
    """The receiver mirrors superseded_by into metadata JSON like the origin, and clears it."""
    a, b = peers
    monkeypatch.setenv("MCP_AGENT_ID", "alpha")
    x = await _store(a, "Memory X: set then cleared")
    y = await _store(a, "Memory Y: temporary winner")
    _sync(a, b)

    ok, msg = await a.update_memory_metadata(x, {"superseded_by": y})
    assert ok, msg
    _sync(a, b)
    assert _supersede_state(b, x)[0] == y
    assert _metadata(b, x).get("superseded_by") == y == _metadata(a, x).get("superseded_by")

    ok, msg = await a.update_memory_metadata(x, {"superseded_by": ""})
    assert ok, msg
    _sync(a, b)
    assert _supersede_state(b, x) == (None, None) == _supersede_state(a, x)
    assert "superseded_by" not in _metadata(b, x)


@pytest.mark.asyncio
async def test_failed_resolve_conflict_keeps_neither_supersede_nor_event(peers, monkeypatch):
    """A failure after the supersede rolls back the supersede and its sync event too."""
    a, _ = peers
    monkeypatch.setenv("MCP_AGENT_ID", "alpha")
    x = await _store(a, "Memory X: loser")
    y = await _store(a, "Memory Y: winner")
    a.conn.execute(
        "CREATE TRIGGER fail_confidence BEFORE UPDATE OF confidence ON memories "
        "BEGIN SELECT RAISE(ABORT, 'confidence update failed'); END"
    )
    a.conn.commit()

    ok, _ = await a.resolve_conflict(y, x)
    assert not ok
    assert _supersede_state(a, x) == (None, None)
    count = a.conn.execute(
        "SELECT COUNT(*) FROM sync_events WHERE content_hash = ? AND op = 'update_metadata'",
        (x,),
    ).fetchone()[0]
    assert count == 0
