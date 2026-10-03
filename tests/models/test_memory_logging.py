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


class _BrokenIsoParser:
    @staticmethod
    def isoparse(value):
        raise ValueError(f"bad timestamp\n{FORGED}")


def test_timestamp_error_logs_do_not_carry_newlines(caplog, monkeypatch):
    from mcp_memory_service.models import memory as memory_module

    monkeypatch.setattr(memory_module, "DATEUTIL_AVAILABLE", True)
    monkeypatch.setattr(memory_module, "dateutil_parser", _BrokenIsoParser, raising=False)

    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        Memory(content="hello", content_hash="abc", created_at=1.0, created_at_iso="x")
        Memory(content="hello", content_hash="abc", created_at_iso="x")
        Memory(content="hello", content_hash="abc", updated_at=1.0, updated_at_iso="x")
        Memory(content="hello", content_hash="abc", updated_at_iso="x")

    _assert_clean(caplog, "Error parsing timestamps: bad timestamp")
    _assert_clean(caplog, "Invalid created_at_iso: bad timestamp")
    _assert_clean(caplog, "Error parsing updated timestamps: bad timestamp")
    _assert_clean(caplog, "Invalid updated_at_iso: bad timestamp")


def test_timestamp_fallback_log_does_not_carry_newlines(caplog, monkeypatch):
    from mcp_memory_service.models import memory as memory_module

    monkeypatch.setattr(memory_module, "DATEUTIL_AVAILABLE", False)

    with caplog.at_level(logging.DEBUG):
        Memory(content="hello", content_hash="abc", created_at_iso=f"not a date\n{FORGED}")

    _assert_clean(caplog, "Failed to parse timestamp 'not a date")
