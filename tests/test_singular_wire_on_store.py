"""
Test the on-store WIRE for singular beliefs (dreaming-loop Inc 1).

When MCP_BELIEF_SINGULAR is on and an actionable memory (decision/learning/error)
is stored, handle_store_memory must create a singular belief automatically
(best-effort, non-blocking). This closes the loop so TODAY's fix is injectable the
same day without a manual/batch step.
"""

import os
import tempfile
import pytest

try:
    import sqlite_vec  # noqa: F401
    SQLITE_VEC_AVAILABLE = True
except ImportError:
    SQLITE_VEC_AVAILABLE = False

pytestmark = pytest.mark.skipif(not SQLITE_VEC_AVAILABLE, reason="sqlite-vec not available")

_FIX = ("Fix delta-sync deadlock: content vazio e payload divergente travava o pull em "
        "fail-stop; a correção torna o evento skippable e segue o merge.")


@pytest.fixture
async def server():
    """Minimal server shim with a real storage + memory_service, like the handlers expect."""
    from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
    from mcp_memory_service.services.memory_service import MemoryService

    d = tempfile.mkdtemp()
    storage = SqliteVecMemoryStorage(os.path.join(d, "t.db"))
    await storage.initialize()

    class _Server:
        async def _ensure_storage_initialized(self):
            return storage

    srv = _Server()
    srv.storage = storage
    srv.memory_service = MemoryService(storage)
    yield srv
    await storage.close()


@pytest.mark.asyncio
async def test_store_actionable_wires_singular_belief(server):
    """Storing a type=learning actionable memory with the flag on creates a singular belief."""
    from mcp_memory_service.server.handlers.memory import handle_store_memory

    os.environ["MCP_BELIEF_SINGULAR"] = "true"
    try:
        await handle_store_memory(server, {
            "content": _FIX,
            "tags": ["delta-sync"],
            "metadata": {"type": "learning"},
        })
    finally:
        os.environ.pop("MCP_BELIEF_SINGULAR", None)

    n = server.storage.conn.execute(
        "SELECT COUNT(*) FROM beliefs WHERE status='active' AND metadata LIKE '%singular%' AND content LIKE '%skippable%'"
    ).fetchone()[0]
    assert n == 1, "actionable store must auto-create a singular belief when flag on"


@pytest.mark.asyncio
async def test_store_actionable_no_wire_when_flag_off(server):
    """Flag off => no singular belief (opt-in)."""
    from mcp_memory_service.server.handlers.memory import handle_store_memory

    os.environ.pop("MCP_BELIEF_SINGULAR", None)
    await handle_store_memory(server, {
        "content": _FIX, "tags": ["delta-sync"], "metadata": {"type": "learning"},
    })
    n = server.storage.conn.execute(
        "SELECT COUNT(*) FROM beliefs WHERE content LIKE '%skippable%'"
    ).fetchone()[0]
    assert n == 0, "no singular belief when flag off"


@pytest.mark.asyncio
async def test_store_non_actionable_no_singular(server):
    """A plain note (non-actionable type) does not trigger a singular belief."""
    from mcp_memory_service.server.handlers.memory import handle_store_memory

    os.environ["MCP_BELIEF_SINGULAR"] = "true"
    try:
        await handle_store_memory(server, {
            "content": "uma nota qualquer sem tipo acionável sobre algo corriqueiro aqui",
            "tags": ["x"],
            "metadata": {"type": "note"},
        })
    finally:
        os.environ.pop("MCP_BELIEF_SINGULAR", None)
    n = server.storage.conn.execute("SELECT COUNT(*) FROM beliefs WHERE metadata LIKE '%singular%'").fetchone()[0]
    assert n == 0, "non-actionable type must not create a singular belief"
