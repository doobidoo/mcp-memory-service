"""Unit tests for agent id resolution hardening in ``utils/agent_context``.

Covers two defects fixed TDD-style:
- A (write-breaking TypeError): ``metadata={'agent_id': 123}`` made
  ``_mask_agent_id`` call ``len()`` on a non-str; the log call runs even with
  DEBUG off, so the exception broke the memory write (in ``evolve_memory``
  after the new version was already committed).
- B (log injection): the masked value kept 2 raw characters, so
  ``agent_id='a\\nrest'`` could forge a log line; the masked value must pass
  through ``_sanitize_log_value`` (``mcp_memory_service.compat``).

Unit-level only: no transport, no storage.
"""

import logging

import pytest

from mcp_memory_service.utils.agent_context import (
    log_agent_id_resolution,
    reset_current_agent_id,
    resolve_agent_id,
    set_current_agent_id,
)

_LOGGER = "mcp_memory_service.utils.agent_context"


@pytest.fixture(autouse=True)
def _clean_agent_identity(monkeypatch):
    """Isolate the precedence chain: no env, no request header per test."""
    monkeypatch.delenv("MCP_AGENT_ID", raising=False)
    token = set_current_agent_id(None)
    yield
    reset_current_agent_id(token)


def _debug_messages(caplog):
    return [r.getMessage() for r in caplog.records if r.name == _LOGGER]


# ---- Defect A: non-str scalar metadata must not raise, resolution works ----

def test_metadata_int_agent_id_resolves_and_logs_without_raising(caplog):
    """A: metadata agent_id=123 (int) must not raise in resolve or log.

    Declared behavior: the int is normalized via str(value).strip(), so the
    resolved value is '123' with origin 'metadata' (and the write path
    persists the clean str, not the raw int).
    """
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        resolution = resolve_agent_id(metadata={"agent_id": 123})
        log_agent_id_resolution(resolution, context="store_memory")

    assert resolution.agent_id == "123"
    assert resolution.origin == "metadata"
    assert (
        "agent_id resolved (store_memory): origin=metadata value=12***"
        in _debug_messages(caplog)
    )


def test_metadata_non_scalar_agent_id_is_ignored(caplog):
    """A: dict/list agent_id in metadata resolves to None (absent-like)."""
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        for bad in ({"id": 1}, ["zero"], None, "   "):
            resolution = resolve_agent_id(metadata={"agent_id": bad})
            log_agent_id_resolution(resolution, context="evolve_memory")

    assert resolution.agent_id is None
    assert resolution.origin == "none"
    assert (
        "agent_id resolved (evolve_memory): origin=none value=<none>"
        in _debug_messages(caplog)
    )


# ---- Defect B: masked value must not carry raw control characters ----

def test_metadata_agent_id_with_newline_does_not_inject_log_line(caplog):
    """B: agent_id containing '\\n' must not produce a raw newline in the log."""
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        resolution = resolve_agent_id(metadata={"agent_id": "a\nINJECTED"})
        log_agent_id_resolution(resolution, context="evolve_memory")

    assert resolution.agent_id == "a\nINJECTED"
    assert resolution.origin == "metadata"
    messages = _debug_messages(caplog)
    assert messages, "expected a DEBUG resolution line"
    for message in messages:
        assert "\n" not in message
        assert "\r" not in message


# ---- Guard: existing precedence behavior is unchanged ----

def test_explicit_argument_still_wins_over_metadata(caplog):
    """Guard: explicit argument beats metadata, value masked as before."""
    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        resolution = resolve_agent_id(
            explicit="zero", metadata={"agent_id": "kiro"}
        )
        log_agent_id_resolution(resolution, context="store_memory")

    assert (resolution.agent_id, resolution.origin) == ("zero", "argument")
    assert (
        "agent_id resolved (store_memory): origin=argument value=ze***"
        in _debug_messages(caplog)
    )


def test_header_still_resolves_when_no_argument(monkeypatch):
    """Guard: request header (ContextVar) resolves with origin 'header'."""
    token = set_current_agent_id("streamable-agent")
    try:
        resolution = resolve_agent_id()
    finally:
        reset_current_agent_id(token)

    assert (resolution.agent_id, resolution.origin) == (
        "streamable-agent",
        "header",
    )
    # Logging it must also be safe (value longer than 2 chars, masked).
    log_agent_id_resolution(resolution, context="store_memory")
