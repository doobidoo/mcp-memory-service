"""What ``mcp_server`` logs from outside stays in one record.

``test_log_injection_guard.py`` checks the source of mcp_server.py. This
drives ``mcp_server_lifespan`` with both caches pre-filled, so no storage is
opened, and reads what the logger emitted. The backend name comes from the
environment and is logged on the storage cache-hit line and on the
graph-tools line; the exception text of a graph storage that fails to
initialise is logged by a ``%``-style warning. A newline inside either must
come out as a literal ``\\n``, not as a second line that reads like a forged
entry. Both tests fail against ``main``.
Part of #1146.
"""

import logging

import pytest

from mcp_memory_service import mcp_server

LOGGER = "mcp_memory_service.mcp_server"
FORGED = "\nINFO forged line"
DB_PATH = "/tmp/memories.db"


class _Storage:
    """Stands in for a cached storage instance; nothing on it is called."""


def _prime(monkeypatch, backend):
    """Fill the module caches so the lifespan takes the hit paths only."""
    storage = _Storage()
    stats = {
        key: ([] if isinstance(value, list) else 0)
        for key, value in mcp_server._CACHE_STATS.items()
    }
    monkeypatch.setattr(mcp_server, "STORAGE_BACKEND", backend)
    monkeypatch.setattr(mcp_server, "SQLITE_VEC_PATH", DB_PATH)
    monkeypatch.setattr(mcp_server, "_STORAGE_CACHE", {f"{backend}:{DB_PATH}": storage})
    monkeypatch.setattr(mcp_server, "_MEMORY_SERVICE_CACHE", {id(storage): object()})
    monkeypatch.setattr(mcp_server, "_GRAPH_STORAGE_CACHE", {})
    monkeypatch.setattr(mcp_server, "_GRAPH_SERVICE_CACHE", {})
    monkeypatch.setattr(mcp_server, "_CACHE_STATS", stats)
    monkeypatch.setattr(mcp_server, "_CACHE_LOCK", None)


def _messages(caplog):
    return [record.getMessage() for record in caplog.records if record.name == LOGGER]


@pytest.mark.asyncio
async def test_backend_name_stays_in_one_record(monkeypatch, caplog):
    # A backend name the graph code does not know takes the "not available" branch.
    _prime(monkeypatch, "sqlite_vec" + FORGED)

    with caplog.at_level(logging.INFO, logger=LOGGER):
        async with mcp_server.mcp_server_lifespan(None) as context:
            assert context.graph_service is not None

    messages = _messages(caplog)
    assert messages, "the lifespan logged nothing"
    assert all("\n" not in message for message in messages)
    assert (
        "✅ Storage Cache HIT - Reusing sqlite_vec\\nINFO forged line instance "
        "(key: sqlite_vec\\nINFO forged line:/tmp/memories.db)"
    ) in messages
    assert "Graph tools not available for sqlite_vec\\nINFO forged line backend (expected)" in messages
    assert "\nINFO forged" not in caplog.text


@pytest.mark.asyncio
async def test_graph_storage_error_text_stays_in_one_record(monkeypatch, caplog):
    _prime(monkeypatch, "sqlite_vec")

    def failing(path):
        raise RuntimeError("unable to open database file" + FORGED)

    monkeypatch.setattr(mcp_server, "GraphStorage", failing)

    with caplog.at_level(logging.INFO, logger=LOGGER):
        async with mcp_server.mcp_server_lifespan(None):
            pass

    messages = _messages(caplog)
    assert all("\n" not in message for message in messages)
    assert (
        "GraphStorage initialization failed (graph tools disabled): "
        "unable to open database file\\nINFO forged line"
    ) in messages
    assert "\nINFO forged" not in caplog.text
