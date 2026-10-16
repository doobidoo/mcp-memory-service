# Copyright 2024 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for store-scoped SSE subscriptions.

Exercises SSEManager directly with a fake Request so we don't need to spin up
the full FastAPI app or any storage backend (same approach as
test_sse_replay.py).
"""

from dataclasses import dataclass

import pytest

from mcp_memory_service.web.sse import (
    SSEEvent,
    SSEManager,
    create_memory_deleted_event,
    create_memory_stored_event,
    parse_store_subscription,
)


@dataclass
class _FakeClient:
    host: str = "127.0.0.1"


class _FakeRequest:
    """Minimal fastapi.Request stand-in; only what SSEManager touches."""

    def __init__(self, headers=None, client_host: str = "127.0.0.1"):
        self.headers = headers or {}
        self.client = _FakeClient(host=client_host)


class _FakeQueryParams:
    """Mimics starlette QueryParams.getlist for the parser test."""

    def __init__(self, items):
        self._items = items

    def getlist(self, key):
        return [v for k, v in self._items if k == key]


async def _drain(queue):
    drained = []
    while not queue.empty():
        drained.append(queue.get_nowait())
    return drained


def _stored(events):
    """memory_stored events from a drained queue, in order."""
    return [e for e in events if e.event_type == "memory_stored"]


@pytest.mark.asyncio
async def test_subscription_delivers_only_matching_store():
    """A connection subscribed to one store receives that store's events and
    not another store's."""
    manager = SSEManager(replay_buffer_size=10)
    queue = await manager.add_connection("c1", _FakeRequest(), stores={"default"})

    await manager.broadcast_event(SSEEvent("memory_stored", {"n": "d"}, store="default"))
    await manager.broadcast_event(SSEEvent("memory_stored", {"n": "o"}, store="other"))

    stored = _stored(await _drain(queue))
    assert [e.store for e in stored] == ["default"]


@pytest.mark.asyncio
async def test_firehose_subscription_receives_all_stores():
    """No subscription (stores=None) is the firehose: it still receives every
    store's events — the pre-change default behaviour."""
    manager = SSEManager(replay_buffer_size=10)
    queue = await manager.add_connection("c1", _FakeRequest())  # stores defaults to None

    await manager.broadcast_event(SSEEvent("memory_stored", {"n": "d"}, store="default"))
    await manager.broadcast_event(SSEEvent("memory_stored", {"n": "o"}, store="other"))

    stored = _stored(await _drain(queue))
    assert [e.store for e in stored] == ["default", "other"]


@pytest.mark.asyncio
async def test_system_events_bypass_store_filter():
    """A store-less event (store=None, e.g. a heartbeat) reaches even a
    store-scoped connection."""
    manager = SSEManager(replay_buffer_size=10)
    queue = await manager.add_connection("c1", _FakeRequest(), stores={"default"})

    await manager.broadcast_event(SSEEvent("heartbeat", {"ping": True}))  # store=None
    await manager.broadcast_event(SSEEvent("memory_stored", {"n": "o"}, store="other"))

    events = await _drain(queue)
    assert any(e.event_type == "heartbeat" for e in events)
    assert _stored(events) == []  # the "other" store event was filtered out


@pytest.mark.asyncio
async def test_multiple_store_subscription():
    """A set subscription receives any of its stores, and nothing else."""
    manager = SSEManager(replay_buffer_size=10)
    queue = await manager.add_connection("c1", _FakeRequest(), stores={"a", "b"})

    for name in ("a", "b", "c"):
        await manager.broadcast_event(SSEEvent("memory_stored", {"n": name}, store=name))

    stored = _stored(await _drain(queue))
    assert sorted(e.store for e in stored) == ["a", "b"]


@pytest.mark.asyncio
async def test_replay_respects_store_subscription():
    """Last-Event-ID resume must not replay another store's buffered events to
    a store-scoped subscriber."""
    manager = SSEManager(replay_buffer_size=10)

    anchor = SSEEvent("memory_stored", {"n": 0}, store="a")
    e_a = SSEEvent("memory_stored", {"n": 1}, store="a")
    e_b = SSEEvent("memory_stored", {"n": 2}, store="b")
    for e in (anchor, e_a, e_b):
        await manager.broadcast_event(e)  # no live connections: all buffered

    queue = await manager.add_connection(
        "c-r",
        _FakeRequest(headers={"Last-Event-ID": anchor.event_id}),
        last_event_id=anchor.event_id,
        stores={"a"},
    )
    events = await _drain(queue)

    assert events[0].event_type == "connection_established"
    assert events[0].data["replay"]["status"] == "resumed"
    # Only e_a replays; e_b (store "b") is withheld from the store-"a" subscriber.
    assert events[0].data["replay"]["events_replayed"] == 1
    assert [e.store for e in events[1:]] == ["a"]


@pytest.mark.asyncio
async def test_firehose_replay_receives_all_buffered_stores():
    """A firehose reconnect still replays every buffered store (back-compat)."""
    manager = SSEManager(replay_buffer_size=10)

    anchor = SSEEvent("memory_stored", {"n": 0}, store="a")
    e_a = SSEEvent("memory_stored", {"n": 1}, store="a")
    e_b = SSEEvent("memory_stored", {"n": 2}, store="b")
    for e in (anchor, e_a, e_b):
        await manager.broadcast_event(e)

    queue = await manager.add_connection(
        "c-r",
        _FakeRequest(),
        last_event_id=anchor.event_id,  # stores=None (firehose)
    )
    events = await _drain(queue)
    assert events[0].data["replay"]["events_replayed"] == 2
    assert [e.store for e in events[1:]] == ["a", "b"]


def test_memory_event_helpers_carry_store():
    """The event constructors tag the event and surface the store in payload."""
    stored = create_memory_stored_event({"content_hash": "h", "content": "x"}, store="ml")
    assert stored.store == "ml"
    assert stored.data["store"] == "ml"

    deleted = create_memory_deleted_event("h", True, store="ml")
    assert deleted.store == "ml"
    assert deleted.data["store"] == "ml"


@pytest.mark.asyncio
async def test_exact_name_delivery_roundtrip():
    """A store name with a trailing space is addressable end to end: a
    subscriber to "team " receives that store's events and not "team"'s."""
    manager = SSEManager(replay_buffer_size=10)
    queue = await manager.add_connection("c1", _FakeRequest(), stores={"team "})

    await manager.broadcast_event(SSEEvent("memory_stored", {"n": 1}, store="team "))
    await manager.broadcast_event(SSEEvent("memory_stored", {"n": 2}, store="team"))

    stored = _stored(await _drain(queue))
    assert [e.store for e in stored] == ["team "]


@pytest.mark.asyncio
async def test_legacy_callers_unchanged():
    """Back-compat: pre-change call patterns still work and a legacy (no
    subscription) connection receives store-tagged events — the firehose."""
    # Event helpers called the old way (no store arg): construct and default to
    # store=None, so they broadcast to every connection like before.
    legacy_stored = create_memory_stored_event({"content_hash": "h", "content": "x"})
    legacy_deleted = create_memory_deleted_event("h", True)
    assert legacy_stored.store is None
    assert legacy_deleted.store is None

    # SSEEvent constructed the old way (no store kwarg) still works.
    assert SSEEvent("memory_stored", {"n": 1}).store is None

    # A connection added the old way (no stores kwarg) is a firehose: it must
    # receive a store-tagged event just as it received everything pre-change.
    manager = SSEManager(replay_buffer_size=10)
    queue = await manager.add_connection("legacy", _FakeRequest())
    await manager.broadcast_event(SSEEvent("memory_stored", {"n": 1}, store="some-store"))
    await manager.broadcast_event(legacy_stored)  # store=None, system-style
    stored = _stored(await _drain(queue))
    assert len(stored) == 2  # both delivered; nothing filtered out


def test_parse_store_subscription():
    """Query parsing: repeated params, the 'all' sentinel, empties."""
    assert parse_store_subscription(_FakeQueryParams([])) is None
    assert parse_store_subscription(_FakeQueryParams([("store", "default")])) == {"default"}
    assert parse_store_subscription(_FakeQueryParams([("store", "a"), ("store", "b")])) == {"a", "b"}
    # 'all' anywhere means firehose
    assert parse_store_subscription(_FakeQueryParams([("store", "all")])) is None
    assert parse_store_subscription(_FakeQueryParams([("store", "a"), ("store", "all")])) is None
    # empty values are ignored (not treated as a store named "")
    assert parse_store_subscription(_FakeQueryParams([("store", "")])) is None
    assert parse_store_subscription(_FakeQueryParams([("store", ""), ("store", "a")])) == {"a"}


def test_parse_store_subscription_preserves_exact_names():
    """Names are matched exactly against what the write API stores — no
    trimming or comma-splitting — so spaces and commas stay addressable."""
    assert parse_store_subscription(_FakeQueryParams([("store", "team ")])) == {"team "}
    assert parse_store_subscription(_FakeQueryParams([("store", "a,b")])) == {"a,b"}
    assert parse_store_subscription(_FakeQueryParams([("store", " spaced ")])) == {" spaced "}
