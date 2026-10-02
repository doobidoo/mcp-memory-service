"""What ``MetadataMixin`` logs from outside stays in one record.

``test_log_injection_guard.py`` checks the source of storage/mixins/metadata.py.
This calls two of its methods on a mixin with a stubbed executor and reads
what the logger emitted. ``get_conflicts`` logs the exception text of a
failed query; ``_record_conflicts`` logs the first characters of the hash it
was handed. A newline inside either must come out as a literal ``\\n`` in a
single record, not as a second line that reads like a forged entry. Both
tests fail against ``main``. Part of #1146.
"""

import logging
import types

import pytest

from mcp_memory_service.storage.mixins.metadata import MetadataMixin

LOGGER = "mcp_memory_service.storage.mixins.metadata"
FORGED = "\nINFO forged line"


def _mixin(executor):
    # __init__ belongs to the storage class; the two methods under test touch
    # only self._execute_with_retry and self.conn, which _record_all_conflicts
    # commits on (an empty conflict list issues no other statement).
    mixin = MetadataMixin.__new__(MetadataMixin)
    mixin.conn = types.SimpleNamespace(commit=lambda: None)
    mixin._execute_with_retry = executor
    return mixin


def _messages(caplog):
    return [record.getMessage() for record in caplog.records if record.name == LOGGER]


@pytest.mark.asyncio
async def test_get_conflicts_error_text_stays_in_one_record(caplog):
    async def failing(operation, *args, **kwargs):
        raise RuntimeError("no such table: memory_graph" + FORGED)

    with caplog.at_level(logging.ERROR, logger=LOGGER):
        assert await _mixin(failing).get_conflicts() == []

    messages = _messages(caplog)
    assert len(messages) == 1
    assert "\n" not in messages[0]
    assert messages[0] == "get_conflicts error: no such table: memory_graph\\nINFO forged line"
    assert "\nINFO forged" not in caplog.text


@pytest.mark.asyncio
async def test_record_conflicts_hash_stays_in_one_record(caplog):
    async def run(operation, *args, **kwargs):
        return operation()

    with caplog.at_level(logging.INFO, logger=LOGGER):
        await _mixin(run)._record_conflicts("ab" + FORGED, [])

    messages = _messages(caplog)
    assert len(messages) == 1
    assert "\n" not in messages[0]
    assert messages[0] == "Recorded 0 conflict(s) for ab\\nINFO "
    assert "\nINFO forged" not in caplog.text
