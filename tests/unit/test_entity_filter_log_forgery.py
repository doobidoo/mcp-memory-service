"""A wrapped log value is not enough while the same call passes exc_info.

Regression for the entity-filter error path in `server/handlers/memory.py`
(#1146 follow-up). Both `%s` arguments were already wrapped in
`_sanitize_log_value`, so the call looked done — but `exc_info=True` makes the
logging module append the formatted traceback, which carries the raw `str(e)`
and reintroduces exactly the newline forgery the wraps exist to stop.

These tests are red without the fix and green with it: they raise an exception
whose message contains a newline followed by a well-formed log record, and
assert that the forged text never appears as its own record.
"""

from __future__ import annotations

import asyncio
import logging
import re

import pytest

from mcp_memory_service.server.handlers import graph as graph_mod
from mcp_memory_service.server.handlers import memory as memory_module

FORGED_STAMP = "2099-01-01 CRITICAL admin wiped all memories"
# A forged record is a line that *is* that stamp: the logging module emitted it
# as a separate record rather than escaping it into the caller's message.
# Matches the payload whether it lands as a bare date or a full timestamp —
# the forged line is the one whose *message body* is the stamp, so the
# timestamp prefix (if any) must not be required to contain a time component.
FORGED_RECORD = re.compile(
    r"^(?:\d{4}-\d{2}-\d{2}(?:[ T]\S+)?\s+)?CRITICAL admin wiped all memories\s*$"
)


class _Boom(Exception):
    """An exception whose message tries to mint a fake log record."""


class _Graph:
    async def find_memories_by_entity(self, _entity_filter):
        raise _Boom(f"ok\n{FORGED_STAMP}")


class _Storage:
    """Enough of a storage backend to reach the entity-filter branch."""

    async def search_memories(self, **_kwargs):
        return {"memories": [{"content_hash": "h1", "content": "hello"}], "total": 1}


class _Server:
    """The handler needs storage initialised so it can reach the filter."""

    async def _ensure_storage_initialized(self):
        return _Storage()


@pytest.fixture
def run_entity_filter_error(monkeypatch, caplog):
    """Drive the failing lookup and return (level_text, raw_text, result)."""

    async def _fake_get_graph_storage():
        return _Graph()

    monkeypatch.setattr(graph_mod, "get_graph_storage", _fake_get_graph_storage)
    caplog.set_level(logging.DEBUG)

    result = asyncio.run(
        memory_module.handle_memory_search(
            _Server(), {"query": "hello", "entity": "some-entity"}
        )
    )

    level_text = "\n".join(f"{r.levelname} {r.getMessage()}" for r in caplog.records)
    return level_text, caplog.text, result


def test_forged_stamp_never_becomes_its_own_record(run_entity_filter_error):
    """The newline in the exception message must not mint a second record."""
    _level, raw, _result = run_entity_filter_error
    forged = [ln for ln in raw.splitlines() if FORGED_RECORD.match(ln.strip())]
    assert forged == [], (
        "the exception message forged a standalone log record — "
        "exc_info or an unwrapped argument put raw str(e) on its own line: "
        f"{forged!r}"
    )


def test_forged_stamp_is_escaped_into_one_record(run_entity_filter_error):
    """The payload still has to appear — escaped, inside the caller's message."""
    _level, raw, _result = run_entity_filter_error
    assert "admin wiped all memories" in raw, "the error path logged nothing"
    # Escaped, so the newline that would start a new record is inert.
    assert "\n2099-01-01 CRITICAL" not in raw


def test_a_single_error_record_describes_the_failure(run_entity_filter_error):
    """One error line names the lookup, so the escape is not achieved by silence."""
    level, _raw, _result = run_entity_filter_error
    errors = [ln for ln in level.splitlines() if ln.startswith("ERROR")]
    assert any("Entity filter lookup failed" in ln for ln in errors), errors


def test_error_record_carries_no_exc_info(run_entity_filter_error, caplog):
    """Restoring exc_info would let logging append the raw traceback again."""
    assert all(r.exc_info is None for r in caplog.records), (
        "a record carries exc_info, so the formatter appends raw str(e)"
    )
