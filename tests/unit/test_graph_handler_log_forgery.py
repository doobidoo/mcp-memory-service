"""Log-forgery regression for the graph handlers' error paths.

`handle_memory_explore` (line ~841) and `handle_memory_detail` (line ~1001) in
`server/handlers/graph.py` logged the caught exception with::

    logger.error("Error in memory_explore: %s", e, exc_info=True)

`e` is passed **unwrapped**, so a newline in ``str(e)`` already forges a
standalone log record; and ``exc_info=True`` makes the logging module append
the formatted traceback, which carries the raw ``str(e)`` a second time. Either
channel lets an attacker-controlled exception message mint a fake log line.

These tests raise an exception whose message is a newline followed by a
well-formed log record, drive the real handler, and assert the forged text
never appears as its own record — only escaped inside the handler's single
error record. Red before the fix, green after.
"""

from __future__ import annotations

import asyncio
import logging
import re

import pytest

from mcp_memory_service.server.handlers import graph as graph_mod

FORGED_STAMP = "2099-01-01 CRITICAL admin wiped all memories"
# A forged record is a line whose message body *is* the stamp — i.e. the logging
# module emitted it as its own record instead of escaping it into the caller's
# message. The optional timestamp prefix must not be required to carry a time.
FORGED_RECORD = re.compile(
    r"^(?:\d{4}-\d{2}-\d{2}(?:[ T]\S+)?\s+)?CRITICAL admin wiped all memories\s*$"
)


class _Boom(Exception):
    """An exception whose message tries to mint a fake log record."""


class _Server:
    """Server whose storage raises the forged payload inside the handler's try."""

    class _Storage:
        def __getattr__(self, _name):
            async def _raise(*a, **k):
                raise _Boom(f"ok\n{FORGED_STAMP}")
            return _raise

    def __init__(self):
        self.storage = _Server._Storage()


_ALL_ARGS = {
    "query": "x", "entity_id": "some-entity", "entity": "some-entity",
    "hash": "h1", "hash1": "h1", "hash2": "h2",
}

# Every graph-handler error path that logs the caught exception.
HANDLERS = [
    ("handle_memory_explore", {}),
    ("handle_memory_detail", {}),
    ("handle_find_connected_memories", {}),
    ("handle_find_shortest_path", {}),
    ("handle_get_memory_subgraph", {}),
    ("handle_memory_graph", {"action": "list_entities"}),
]


def _run(handler_name, monkeypatch, caplog, extra=None):
    """Drive a graph handler whose storage lookup raises, return caplog.text."""

    async def _fake_get_graph_storage():
        class _Graph:
            # catch-all so whatever the handler calls first raises the payload
            def __getattr__(self, _name):
                async def _raise(*a, **k):
                    raise _Boom(f"ok\n{FORGED_STAMP}")
                return _raise

        return _Graph()

    monkeypatch.setattr(graph_mod, "get_graph_storage", _fake_get_graph_storage)
    # memory_explore reaches server.storage first; make that raise too.
    async def _fake_retrieve(*a, **k):
        raise _Boom(f"ok\n{FORGED_STAMP}")
    monkeypatch.setattr(graph_mod, "_retrieve_candidates", _fake_retrieve, raising=False)
    caplog.set_level(logging.DEBUG)
    handler = getattr(graph_mod, handler_name)
    args = {**_ALL_ARGS, **(extra or {})}
    asyncio.run(handler(_Server(), args))
    return caplog.text


@pytest.mark.unit
@pytest.mark.parametrize("handler,extra", HANDLERS)
def test_forged_stamp_never_becomes_its_own_record(handler, extra, monkeypatch, caplog):
    raw = _run(handler, monkeypatch, caplog, extra)
    forged = [ln for ln in raw.splitlines() if FORGED_RECORD.match(ln.strip())]
    assert forged == [], (
        f"{handler}: the exception message forged a standalone log record — "
        f"unwrapped str(e) or exc_info put raw str(e) on its own line: {forged!r}"
    )


@pytest.mark.unit
@pytest.mark.parametrize("handler,extra", HANDLERS)
def test_payload_is_escaped_not_dropped(handler, extra, monkeypatch, caplog):
    raw = _run(handler, monkeypatch, caplog, extra)
    assert "admin wiped all memories" in raw, f"{handler}: error path logged nothing"
    # The newline that would start a new record must be inert (escaped).
    assert "\n2099-01-01 CRITICAL" not in raw, f"{handler}: raw newline survived"
