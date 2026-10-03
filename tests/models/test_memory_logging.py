"""
Log-injection tests for models/memory.py (#1146).

Building a Memory logs the memory type and the tags that fail validation. Both
come from the caller or from stored rows, so a newline in one must not reach the
log as a line break.
"""

import logging

from mcp_memory_service.models.memory import Memory

FORGED = "FORGED admin authenticated"
LOGGER = "mcp_memory_service.models.memory"


def _assert_clean(caplog, expected):
    messages = [record.getMessage() for record in caplog.records]
    assert any(expected in m for m in messages)
    assert not any(f"\n{FORGED}" in m for m in messages)


def test_invalid_tag_namespace_log_does_not_carry_newlines(caplog):
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        Memory(content="hello", content_hash="abc", tags=[f"bad\n{FORGED}:value"])

    _assert_clean(caplog, "Tags with invalid namespaces: bad")


def test_invalid_memory_type_log_does_not_carry_newlines(caplog):
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        Memory(content="hello", content_hash="abc", memory_type=f"odd\n{FORGED}")

    _assert_clean(caplog, "Invalid memory_type 'odd")
