"""Ontology overlay rows travel as sync events (#1897, phase 2).

Each overlay write records an ``ontology_upsert`` / ``ontology_delete`` event keyed
``ontology:<kind>:<key>``. A peer resolves the events per key by HLC and writes the
winning row. Peers that do not advertise ``ontology_sync`` never see these events,
so a mixed-version mesh keeps syncing memories.
"""

import asyncio
import json
import os
import shutil
import tempfile

import pytest
import pytest_asyncio

from mcp_memory_service.models import ontology
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.ontology_overlay import ontology_event_hash
from mcp_memory_service.storage.remote_http import RemoteHTTPStorage
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.storage.sync.apply import apply_remote_event
from mcp_memory_service.storage.sync.bootstrap import generate_baseline, install_baseline
from mcp_memory_service.storage.sync.orchestrator import (
    _read_local_events_since,
    push_to_peer,
    sync_from_peer,
    sync_ontology_backlog,
)
from mcp_memory_service.utils.hashing import generate_content_hash
from mcp_memory_service.web.api.sync import get_sync_capabilities
from mcp_memory_service.web.api.sync_events import get_sync_events


@pytest.fixture(autouse=True)
def _eventlog_on(monkeypatch):
    monkeypatch.setenv("MCP_SYNC_EVENTLOG", "on")
    monkeypatch.delenv("MCP_CUSTOM_MEMORY_TYPES", raising=False)
    monkeypatch.setattr(ontology, "_OVERLAY_POLL_SECONDS", 0)


@pytest_asyncio.fixture
async def stores():
    d = tempfile.mkdtemp(prefix="mcp-test-1897-sync-")
    a = SqliteVecMemoryStorage(os.path.join(d, "a.db"))
    b = SqliteVecMemoryStorage(os.path.join(d, "b.db"))
    await a.initialize()
    await b.initialize()
    yield a, b
    await a.close()
    await b.close()
    ontology.clear_ontology_caches()
    shutil.rmtree(d, ignore_errors=True)


class Peer:
    """In-process peer: serves ``store``'s feed and applies what is pushed to it."""

    def __init__(self, store, ontology_sync=True):
        self.store = store
        self.ontology_sync = ontology_sync
        self.feed_filters = []

    async def get_events_since(self, cursor, limit, events_filter=None):
        events_filter = events_filter or ("all" if self.ontology_sync else "memory")
        self.feed_filters.append(events_filter)
        events = _read_local_events_since(self.store, cursor, limit, events_filter)
        next_seq = events[-1]["seq"] if events else cursor
        return events, next_seq, len(events) == limit

    async def push_events(self, events):
        failed = sum(not apply_remote_event(self.store, ev).applied for ev in events)
        return {"applied": len(events) - failed, "failed": failed}


def _rows(store):
    return {
        (k, key): (parent, json.loads(p) if p else None, deleted is not None)
        for k, key, parent, p, deleted in store.conn.execute(
            "SELECT kind, key, parent, payload_json, deleted_at FROM ontology_entries"
        )
    }


def _events(store, prefix="ontology:"):
    return store.conn.execute(
        "SELECT op, content_hash, payload FROM sync_events WHERE content_hash LIKE ? ORDER BY seq",
        (prefix + "%",),
    ).fetchall()


async def _write_overlay(store):
    await store.add_memory_type("legal")
    await store.add_memory_type("postmortem", parent="learning")
    await store.set_retention("insight", 90)
    await store.add_relationship_type("cites", ["research → any"], description="quotes")
    await store.set_type_pattern("decision", "error", "fixes", 0.8)
    await store.remove_memory_type("postmortem")


async def _store_memory(store, content):
    h = generate_content_hash(content)
    await store.store(Memory(content=content, content_hash=h, tags=["sync"], memory_type="note"))
    return h


async def test_every_overlay_write_records_its_row_as_an_event(stores):
    a, _ = stores
    await _write_overlay(a)

    events = _events(a)
    assert [(op, h) for op, h, _ in events] == [
        ("ontology_upsert", "ontology:memory_type:legal"),
        ("ontology_upsert", "ontology:memory_type:postmortem"),
        ("ontology_upsert", "ontology:retention:insight"),
        ("ontology_upsert", "ontology:relationship:cites"),
        ("ontology_upsert", "ontology:type_pattern:decision → error"),
        ("ontology_delete", "ontology:memory_type:postmortem"),
    ]
    assert json.loads(events[1][2]) == {
        "kind": "memory_type", "key": "postmortem", "parent": "learning", "payload": None,
        "deleted_at": None,
    }
    assert json.loads(events[2][2])["payload"] == {"days": 90}
    assert json.loads(events[-1][2])["deleted_at"] is not None


async def test_no_events_without_the_event_log(stores, monkeypatch):
    a, _ = stores
    monkeypatch.delenv("MCP_SYNC_EVENTLOG")
    await a.add_memory_type("legal")
    assert _events(a) == []
    assert ("memory_type", "legal") in _rows(a)


async def test_push_and_pull_carry_the_overlay_to_a_capable_peer(stores):
    a, b = stores
    await _write_overlay(a)

    await push_to_peer(a, Peer(b), peer_id="b")
    assert _rows(b) == _rows(a)

    await b.set_retention("insight", 120)
    await b.remove_relationship_type("cites")
    await sync_from_peer(a, Peer(b), peer_id="b")  # a's own events come back as duplicates
    assert _rows(a) == _rows(b)
    assert _rows(a)[("retention", "insight")][1] == {"days": 120}
    assert _rows(a)[("relationship", "cites")][2] is True


async def test_an_old_peer_gets_memories_and_no_ontology_events(stores):
    a, b = stores
    await a.add_memory_type("legal")
    h = await _store_memory(a, "memory that must still reach an old peer")
    await a.set_retention("insight", 90)

    old = Peer(b, ontology_sync=False)
    result = await push_to_peer(a, old, peer_id="b")
    assert result.events_failed == 0
    assert b.conn.execute("SELECT 1 FROM memories WHERE content_hash = ?", (h,)).fetchone()
    assert _events(b) == []

    pull = await sync_from_peer(b, Peer(a, ontology_sync=False), peer_id="a")
    assert pull.events_failed == 0
    assert _rows(b) == {}


async def test_concurrent_writes_converge_on_the_later_hlc(stores):
    a, b = stores
    await a.set_type_pattern("decision", "error", "fixes", 0.8)
    await asyncio.sleep(0.005)  # HLCs are in ms; within one ms the tie-break decides
    await b.set_type_pattern("decision", "error", "fixes", 0.9)

    await push_to_peer(a, Peer(b), peer_id="b")
    await push_to_peer(b, Peer(a), peer_id="a")

    key = ("type_pattern", "decision → error")
    assert _rows(a)[key] == _rows(b)[key] == (None, {"relationship": "fixes", "confidence": 0.9}, False)


async def test_a_delete_wins_over_an_older_upsert(stores):
    a, b = stores
    await a.add_memory_type("legal")
    await push_to_peer(a, Peer(b), peer_id="b")
    await b.remove_memory_type("legal")
    await push_to_peer(b, Peer(a), peer_id="a")

    assert _rows(a)[("memory_type", "legal")][2] is True


def _event(**overrides):
    event = {
        "schema_version": 1, "agent_id": "peer-x", "event_id": "ev-1",
        "op": "ontology_upsert", "content_hash": ontology_event_hash("memory_type", "legal"),
        "hlc_physical": 1, "hlc_logical": 0, "embedding_model": None, "embedding_dim": None,
        "payload": {"kind": "memory_type", "key": "legal", "parent": None, "payload": None,
                    "deleted_at": None},
    }
    event.update(overrides)
    return event


@pytest.mark.parametrize("overrides", [
    {"content_hash": "ontology:memory_type:other"},  # payload key is not the hashed key
    {"payload": {"kind": "bogus", "key": "legal"}},
    {"payload": {"kind": "memory_type", "key": ""}},
    {"payload": {"kind": "memory_type", "key": "legal", "parent": 5}},
    {"payload": {"kind": "retention", "key": "legal", "payload": [400]},
     "content_hash": "ontology:retention:legal"},
    {"payload": "not an object"},
    {"op": "create"},  # memory op on an ontology hash
])
async def test_malformed_events_are_skippable_and_not_recorded(stores, overrides):
    a, _ = stores
    result = apply_remote_event(a, _event(**overrides))
    assert not result.applied and result.skippable
    assert _events(a) == [] and _rows(a) == {}


async def test_duplicates_are_idempotent_and_altered_replays_rejected(stores):
    a, _ = stores
    assert apply_remote_event(a, _event()).materialized
    again = apply_remote_event(a, _event())
    assert again.applied and not again.materialized

    altered = _event(payload={**_event()["payload"], "parent": "learning"})
    result = apply_remote_event(a, altered)
    assert not result.applied and not result.skippable
    assert _rows(a)[("memory_type", "legal")][0] is None


async def test_an_older_remote_event_does_not_overwrite_a_newer_local_write(stores):
    a, _ = stores
    await a.add_memory_type("legal", parent="learning")
    result = apply_remote_event(a, _event())  # hlc_physical=1, long before the local write
    assert result.applied and not result.materialized
    assert _rows(a)[("memory_type", "legal")][0] == "learning"


async def test_backlog_exchanges_events_hidden_from_the_cursors_once(stores):
    a, b = stores
    await a.add_memory_type("legal")
    await b.set_retention("insight", 90)
    # Both sides synced while the other lacked the capability: cursors moved past the events.
    await push_to_peer(a, Peer(b, ontology_sync=False), peer_id="b")
    await sync_from_peer(a, Peer(b, ontology_sync=False), peer_id="b")
    assert _rows(b).get(("memory_type", "legal")) is None

    peer = Peer(b)
    result = await sync_ontology_backlog(a, peer, peer_id="b")
    assert result["pulled"] == 1  # a then pushes its own event and b's, as a duplicate
    assert _rows(a) == _rows(b) and len(_rows(a)) == 2

    assert await sync_ontology_backlog(a, peer, peer_id="b") == {"pulled": 0, "pushed": 0}
    assert await sync_ontology_backlog(a, Peer(b, ontology_sync=False), peer_id="c") == {
        "pulled": 0, "pushed": 0}


async def test_a_failed_backlog_is_retried_next_cycle(stores):
    a, b = stores
    await a.add_memory_type("legal")

    class Broken(Peer):
        async def push_events(self, events):
            return {"applied": 0, "failed": len(events)}

    assert await sync_ontology_backlog(a, Broken(b), peer_id="b") == {"pulled": 0, "pushed": 0}
    assert await sync_ontology_backlog(a, Peer(b), peer_id="b") == {"pulled": 0, "pushed": 1}


async def test_baseline_carries_the_overlay_only_when_asked(stores):
    a, b = stores
    await _write_overlay(a)

    events, _ = generate_baseline(a)
    assert not [e for e in events if e["content_hash"].startswith("ontology:")]

    events, watermark = generate_baseline(a, include_ontology=True)
    install_baseline(b, events, peer_id="a", watermark=watermark)
    assert _rows(b) == _rows(a)


async def test_feed_and_capabilities_endpoints(stores):
    a, _ = stores
    await a.add_memory_type("legal")
    await _store_memory(a, "memory next to an ontology event in the feed")

    async def ops(events_filter):
        resp = await get_sync_events(since_seq=0, limit=100, events_filter=events_filter,
                                     _auth=True, storage=a)
        return [e.op for e in resp.events]

    assert await ops("memory") == ["create"]
    assert await ops("ontology") == ["ontology_upsert"]
    assert sorted(await ops("all")) == ["create", "ontology_upsert"]

    caps = await get_sync_capabilities(storage=a, user=None)
    assert caps.ontology_sync is True


@pytest.mark.parametrize("advertised, expected", [(True, True), ("true", False), (None, False)])
def test_remote_peer_trusts_only_an_explicit_capability(advertised, expected):
    peer = RemoteHTTPStorage.__new__(RemoteHTTPStorage)
    peer._note_store_scoping({"ontology_sync": advertised})
    assert peer.ontology_sync is expected


# --- Greptile on #1956 ---


@pytest.mark.parametrize("delete_first", [True, False])
async def test_a_delete_wins_an_equal_hlc_tie_whatever_the_agent(stores, delete_first):
    a, _ = stores
    upsert = _event(agent_id="a-agent", event_id="up", hlc_physical=5, payload={
        "kind": "memory_type", "key": "legal", "parent": None, "payload": None, "deleted_at": None})
    delete = _event(agent_id="z-agent", event_id="del", op="ontology_delete", hlc_physical=5,
                    payload={"kind": "memory_type", "key": "legal", "parent": None,
                             "payload": None, "deleted_at": 5.0})
    for event in ([delete, upsert] if delete_first else [upsert, delete]):
        assert apply_remote_event(a, event).applied
    assert _rows(a)[("memory_type", "legal")][2] is True


async def test_an_unknown_envelope_version_is_refused_and_not_skippable(stores):
    a, _ = stores
    result = apply_remote_event(a, _event(schema_version=2))
    assert not result.applied and not result.skippable
    assert _events(a) == [] and _rows(a) == {}

    assert apply_remote_event(a, _event()).applied
    assert a.conn.execute("SELECT schema_version FROM sync_events").fetchone() == (1,)


async def test_backlog_runs_again_after_the_peer_lost_and_regained_the_capability(stores):
    a, b = stores
    await a.add_memory_type("legal")
    assert (await sync_ontology_backlog(a, Peer(b), peer_id="b"))["pushed"] == 1
    await push_to_peer(a, Peer(b), peer_id="b")

    # b is rolled back: a memory-only cycle moves the push cursor past a new ontology event.
    await sync_ontology_backlog(a, Peer(b, ontology_sync=False), peer_id="b")
    await a.set_retention("insight", 90)
    await _store_memory(a, "memory pushed while the peer was on an older version")
    await push_to_peer(a, Peer(b, ontology_sync=False), peer_id="b")
    assert ("retention", "insight") not in _rows(b)

    await sync_ontology_backlog(a, Peer(b), peer_id="b")  # b upgraded again
    assert _rows(b)[("retention", "insight")][1] == {"days": 90}


async def test_an_in_memory_database_sees_a_remote_row_at_once(monkeypatch):
    s = SqliteVecMemoryStorage(":memory:")
    await s.initialize()
    try:
        assert not ontology.validate_memory_type("legal")
        assert apply_remote_event(s, _event()).materialized
        assert ontology.validate_memory_type("legal")
    finally:
        await s.close()
        ontology.clear_ontology_caches()


async def test_a_filtered_feed_ends_when_only_excluded_events_follow(stores):
    a, _ = stores
    await _store_memory(a, "the only memory event in this feed")
    await a.add_memory_type("legal")

    page = await get_sync_events(since_seq=0, limit=1, events_filter="memory", _auth=True, storage=a)
    assert [e.op for e in page.events] == ["create"] and page.has_more is False
    page = await get_sync_events(since_seq=0, limit=1, events_filter="all", _auth=True, storage=a)
    assert page.has_more is True
