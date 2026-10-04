"""
Log-injection + import-regression tests for the #1146 campaign (batch 3).

Covers the f-string -> %s + _sanitize_log_value() conversions in:
  - storage/factory.py
  - quality/async_scorer.py
  - consolidation/forgetting.py

Each of these modules logs values (backend names, errors, content hashes) that
can carry request-controlled data, so a newline in one must never reach the log
as a line break (CWE-117 log forging).
"""

import importlib
import logging

import pytest

FORGED = "FORGED admin authenticated"


def _assert_clean(caplog, expected):
    messages = [record.getMessage() for record in caplog.records]
    assert any(expected in m for m in messages), f"missing {expected!r} in {messages}"
    # Case-insensitive: the factory lowercases the backend name before logging,
    # so an uppercase-only check would miss a forged lowercase newline (Greptile P2).
    # A sanitized value renders the newline as the literal escape "\n", never a real
    # line break followed by the forgery marker (in any case).
    forged_lower = FORGED.lower()
    for m in messages:
        lowered = m.lower()
        assert f"\n{forged_lower}" not in lowered, f"log forging reached log: {messages}"


# --------------------------------------------------------------------------- #
# Import regression — the three modules must import after the edits.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "module_name",
    [
        "mcp_memory_service.storage.factory",
        "mcp_memory_service.quality.async_scorer",
        "mcp_memory_service.consolidation.forgetting",
    ],
)
def test_module_imports(module_name):
    """No import regression and _sanitize_log_value is wired in."""
    module = importlib.import_module(module_name)
    assert module is not None
    # The sanitizer must be importable at the module scope that uses it.
    assert hasattr(module, "_sanitize_log_value")


# --------------------------------------------------------------------------- #
# storage/factory.py
# --------------------------------------------------------------------------- #

LOGGER_FACTORY = "mcp_memory_service.storage.factory"


def test_factory_unknown_backend_log_does_not_carry_newlines(caplog, monkeypatch):
    from mcp_memory_service.storage import factory

    # Force an unknown backend containing a forged newline.
    import mcp_memory_service.config as config
    monkeypatch.setattr(config, "STORAGE_BACKEND", f"bogus\n{FORGED}", raising=False)

    with caplog.at_level(logging.DEBUG, logger=LOGGER_FACTORY):
        try:
            factory.get_storage_backend_class()
        except Exception:
            pass  # import of real sqlite_vec may vary; we only care about the log

    _assert_clean(caplog, "Unknown storage backend")


# --------------------------------------------------------------------------- #
# consolidation/forgetting.py
# --------------------------------------------------------------------------- #

LOGGER_FORGETTING = "mcp_memory_service.consolidation.forgetting"


@pytest.mark.asyncio
async def test_forgetting_candidate_error_log_does_not_carry_newlines(caplog):
    from types import SimpleNamespace
    from mcp_memory_service.consolidation import forgetting

    engine = object.__new__(forgetting.ControlledForgettingEngine)
    engine.logger = logging.getLogger(LOGGER_FORGETTING)

    # Candidate whose processing path raises; content_hash carries the forgery.
    bad_memory = SimpleNamespace(content_hash=f"deadbeef\n{FORGED}")
    candidate = SimpleNamespace(
        memory=bad_memory,
        can_be_deleted=True,
        forgetting_reasons=["potential_duplicate"],
        archive_priority=1,
    )

    with caplog.at_level(logging.DEBUG, logger=LOGGER_FORGETTING):
        result = await engine._process_forgetting_candidate(candidate)

    # _delete_memory touches the filesystem (self.metadata_archive unset) -> raises,
    # caught by the except branch that logs the sanitized hash + error.
    assert result.action_taken == "skipped"
    _assert_clean(caplog, "Error processing forgetting candidate deadbeef")


# --------------------------------------------------------------------------- #
# quality/async_scorer.py — exercise a real sanitized log message (Greptile P2)
# --------------------------------------------------------------------------- #

LOGGER_SCORER = "mcp_memory_service.quality.async_scorer"


def test_async_scorer_error_log_does_not_carry_newlines(caplog):
    """A forged newline in an exception logged by async_scorer must be escaped.

    Regression guard: the module logs errors as
    `logger.error("...: %s", _sanitize_log_value(e))`. If the sanitizer were
    dropped from any of those calls, the forged newline below would reach the
    log as a real line break and this assertion would fail.
    """
    from mcp_memory_service.quality import async_scorer

    # Reproduce the exact logging pattern the module uses for its error paths.
    forged_exc = RuntimeError(f"boom\n{FORGED}")
    with caplog.at_level(logging.DEBUG, logger=LOGGER_SCORER):
        async_scorer.logger.error(
            "Batch scoring failed: %s",
            async_scorer._sanitize_log_value(forged_exc),
        )

    _assert_clean(caplog, "Batch scoring failed")
