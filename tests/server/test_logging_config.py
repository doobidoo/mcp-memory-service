"""Strict (claude_desktop) mode decides where records go, LOG_LEVEL decides which.

Strict mode is the fallback whenever no known MCP client is the parent process,
which includes the HTTP server under a service manager. It used to drop every
record below WARNING even when LOG_LEVEL asked for INFO, so the HTTP server's
INFO output never reached its log.
"""

import io
import logging

import pytest

from mcp_memory_service.server import logging_config
from mcp_memory_service.server.logging_config import DualStreamHandler


def _emit(level, client_type="claude_desktop", **kwargs):
    handler = DualStreamHandler(client_type=client_type, **kwargs)
    out, err = io.StringIO(), io.StringIO()
    handler.stdout_handler.setStream(out)
    handler.stderr_handler.setStream(err)
    record = logging.LogRecord("mcp_memory_service.x", level, __file__, 1, "a line", None, None)
    handler.emit(record)
    return out.getvalue(), err.getvalue()


def test_strict_mode_defaults_to_warning():
    assert _emit(logging.INFO) == ("", "")
    out, err = _emit(logging.WARNING)
    assert "a line" in err and out == ""


def test_strict_mode_passes_info_to_stderr_when_asked():
    out, err = _emit(logging.INFO, strict_level=logging.INFO)
    assert "a line" in err
    assert out == ""


def test_strict_mode_never_writes_stdout():
    out, _ = _emit(logging.DEBUG, strict_level=logging.DEBUG)
    assert out == ""


def test_lm_studio_routing_unchanged():
    out, err = _emit(logging.INFO, "lm_studio")
    assert "a line" in out and err == ""


@pytest.fixture
def restore_root():
    root = logging.getLogger()
    level, handlers = root.level, root.handlers[:]
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


@pytest.mark.parametrize("env, expected", [("INFO", logging.INFO), (None, logging.WARNING), ("bogus", logging.WARNING)])
def test_configure_logging_hands_log_level_to_the_handler(monkeypatch, restore_root, env, expected):
    if env is None:
        monkeypatch.delenv("LOG_LEVEL", raising=False)
    else:
        monkeypatch.setenv("LOG_LEVEL", env)
    logging_config.configure_logging()
    root = logging.getLogger()
    assert root.level == expected
    (handler,) = root.handlers
    assert handler.strict_level == expected
