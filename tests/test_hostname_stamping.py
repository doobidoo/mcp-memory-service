"""
Tests for server-side hostname stamping on the MCP store path.

Spec: docs/rfc/planned/rfc-mcp-hostname-stamping.md

Problem: MCP_MEMORY_INCLUDE_HOSTNAME is honored ONLY by the Web API
(web/api/memories.py). The MCP store handler reads client_hostname only from the
client argument; it never resolves socket.gethostname() nor consults the flag.
So memories stored via MCP (how Kiro writes) carry no host.

Requirements (RFC §3):
- R1: flag ON + no client_hostname → handler resolves socket.gethostname().
- R2: explicit client_hostname wins over server resolution.
- R3: flag OFF → no hostname (unchanged).
- R4: resolved host persisted in metadata.hostname + source:{host} tag.
- R5: gethostname() failure → store still succeeds, no host (best-effort).

Cover the server-side hostname resolution on the MCP store path.
"""

import os
import shutil
import tempfile

import pytest
import pytest_asyncio

try:
    import sqlite_vec  # noqa: F401
    SQLITE_VEC_AVAILABLE = True
except ImportError:
    SQLITE_VEC_AVAILABLE = False

if SQLITE_VEC_AVAILABLE:
    from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.services.memory_service import MemoryService
from mcp_memory_service.server.handlers.memory import handle_store_memory

pytestmark = pytest.mark.skipif(
    not SQLITE_VEC_AVAILABLE, reason="sqlite-vec not available"
)


class _Server:
    """Minimal stand-in for the MCP server object handle_store_memory expects."""
    def __init__(self, storage):
        self.storage = storage
        self.memory_service = MemoryService(storage)

    async def _ensure_storage_initialized(self):
        return self.storage


@pytest_asyncio.fixture
async def server():
    temp_dir = tempfile.mkdtemp()
    storage = SqliteVecMemoryStorage(os.path.join(temp_dir, "host.db"))
    await storage.initialize()
    try:
        yield _Server(storage)
    finally:
        try:
            await storage.close()
        except Exception:
            pass
        shutil.rmtree(temp_dir, ignore_errors=True)


async def _fetch_latest(storage, content):
    from mcp_memory_service.utils.hashing import generate_content_hash
    return await storage.get_by_hash(generate_content_hash(content))


@pytest.mark.asyncio
async def test_flag_on_resolves_server_hostname(server, monkeypatch):
    """R1: flag ON + no client_hostname → metadata.hostname = socket.gethostname()."""
    import socket
    monkeypatch.setenv("MCP_MEMORY_INCLUDE_HOSTNAME", "true")
    # Config reads the env at import; patch the module-level flag too.
    monkeypatch.setattr("mcp_memory_service.config.INCLUDE_HOSTNAME", True, raising=False)
    monkeypatch.setattr(
        "mcp_memory_service.server.handlers.memory.INCLUDE_HOSTNAME", True, raising=False
    )
    content = "host stamping flag on test"
    await handle_store_memory(server, {"content": content})
    stored = await _fetch_latest(server.storage, content)
    assert stored is not None
    assert (stored.metadata or {}).get("hostname") == socket.gethostname(), (
        f"flag ON must stamp server hostname, got {stored.metadata}"
    )


@pytest.mark.asyncio
async def test_explicit_client_hostname_wins(server, monkeypatch):
    """R2: explicit client_hostname is used unchanged even with flag ON."""
    monkeypatch.setattr(
        "mcp_memory_service.server.handlers.memory.INCLUDE_HOSTNAME", True, raising=False
    )
    content = "host stamping explicit wins test"
    await handle_store_memory(server, {"content": content, "client_hostname": "custom-host"})
    stored = await _fetch_latest(server.storage, content)
    assert (stored.metadata or {}).get("hostname") == "custom-host"


@pytest.mark.asyncio
async def test_flag_off_no_hostname(server, monkeypatch):
    """R3: flag OFF → no hostname (unchanged behavior)."""
    monkeypatch.setattr(
        "mcp_memory_service.server.handlers.memory.INCLUDE_HOSTNAME", False, raising=False
    )
    content = "host stamping flag off test"
    await handle_store_memory(server, {"content": content})
    stored = await _fetch_latest(server.storage, content)
    assert (stored.metadata or {}).get("hostname") is None, (
        f"flag OFF must not stamp host, got {stored.metadata}"
    )


@pytest.mark.asyncio
async def test_resolved_host_adds_source_tag(server, monkeypatch):
    """R4: resolved host also becomes a source:{host} tag."""
    import socket
    monkeypatch.setattr(
        "mcp_memory_service.server.handlers.memory.INCLUDE_HOSTNAME", True, raising=False
    )
    content = "host stamping source tag test"
    await handle_store_memory(server, {"content": content})
    stored = await _fetch_latest(server.storage, content)
    assert f"source:{socket.gethostname()}" in (stored.tags or []), (
        f"resolved host must add source tag, got {stored.tags}"
    )


@pytest.mark.asyncio
async def test_gethostname_failure_is_best_effort(server, monkeypatch):
    """R5: gethostname() raising → store still succeeds, no host."""
    monkeypatch.setattr(
        "mcp_memory_service.server.handlers.memory.INCLUDE_HOSTNAME", True, raising=False
    )

    def _boom():
        raise OSError("no hostname")

    monkeypatch.setattr("socket.gethostname", _boom)
    content = "host stamping best effort test"
    result = await handle_store_memory(server, {"content": content})
    # Store must not fail
    assert result and "Error storing memory" not in result[0].text
    stored = await _fetch_latest(server.storage, content)
    assert stored is not None
    assert (stored.metadata or {}).get("hostname") is None


@pytest.mark.asyncio
async def test_store_session_also_stamps_host(server, monkeypatch):
    """R1 (store_session): session stored via MCP also gets the server host.

    Closes the gap flagged in review (handle_store_session shared the same
    missing resolution as handle_store_memory).
    """
    import socket
    from mcp_memory_service.server.handlers.memory import handle_store_session
    monkeypatch.setattr(
        "mcp_memory_service.server.handlers.memory.INCLUDE_HOSTNAME", True, raising=False
    )
    result = await handle_store_session(
        server,
        {"turns": [{"role": "user", "content": "hello host"},
                   {"role": "assistant", "content": "hi from the session path"}]},
    )
    assert result and "Error" not in result[0].text
    # Find the session memory and assert host stamped.
    # Exclude soft-deleted tombstones (repo requires live-row filter).
    import json
    rows = server.storage.conn.execute(
        "SELECT metadata FROM memories WHERE memory_type='session' AND deleted_at IS NULL"
    ).fetchall()
    assert rows, "a session memory should have been stored"
    assert any(
        (json.loads(r[0]) if r[0] else {}).get("hostname") == socket.gethostname()
        for r in rows
    ), "session memory must carry the server hostname"


@pytest.mark.asyncio
async def test_store_session_chunked_stamps_every_chunk(server, monkeypatch):
    """R1 (chunked session): host is stamped on EVERY chunk, not just single-write.

    The hostname resolution lives inside the chunk loop too; a long session that
    exceeds the chunk threshold must stamp each stored chunk (regression guard
    flagged in review).
    """
    import json
    import socket
    from mcp_memory_service.server.handlers.memory import handle_store_session
    monkeypatch.setattr(
        "mcp_memory_service.server.handlers.memory.INCLUDE_HOSTNAME", True, raising=False
    )
    # chunk_size comes from SESSION_CHUNK_SIZE env (default 1500), not the arg.
    monkeypatch.setenv("SESSION_CHUNK_SIZE", "80")
    # Many turns + tiny chunk_size → forces the chunked branch (multiple chunks).
    turns = [{"role": "user" if i % 2 == 0 else "assistant",
              "content": f"turn number {i} with enough prose to be a real chunk body"}
             for i in range(12)]
    result = await handle_store_session(
        server, {"turns": turns}
    )
    assert result and "Error" not in result[0].text
    rows = server.storage.conn.execute(
        "SELECT metadata, tags FROM memories "
        "WHERE memory_type='session' AND deleted_at IS NULL"
    ).fetchall()
    # Keep only the chunked rows (tag chunk:i/N).
    chunk_rows = [r for r in rows if r[1] and "chunk:" in r[1]]
    assert len(chunk_rows) >= 2, f"expected multiple chunks, got {len(chunk_rows)}"
    assert all(
        (json.loads(r[0]) if r[0] else {}).get("hostname") == socket.gethostname()
        for r in chunk_rows
    ), "every session chunk must carry the server hostname"


@pytest.mark.asyncio
async def test_empty_client_hostname_resolves_server(server, monkeypatch):
    """Edge case: client_hostname='' is treated as absent → server resolves."""
    import socket
    monkeypatch.setattr(
        "mcp_memory_service.server.handlers.memory.INCLUDE_HOSTNAME", True, raising=False
    )
    content = "host stamping empty client hostname test"
    await handle_store_memory(server, {"content": content, "client_hostname": ""})
    stored = await _fetch_latest(server.storage, content)
    assert (stored.metadata or {}).get("hostname") == socket.gethostname()



@pytest.mark.asyncio
async def test_fastmcp_store_memory_stamps_host(server, monkeypatch):
    """R1 (FastMCP tool): the Streamable HTTP store_memory entrypoint stamps host.

    mcp_server.store_memory calls MemoryService.store_memory directly, routed
    through _resolve_hostname. Covers the separately registered tool (Greptile
    r4195956576) so a regression in its forwarding is caught.
    """
    import socket
    from types import SimpleNamespace
    from mcp_memory_service import mcp_server
    monkeypatch.setattr(
        "mcp_memory_service.server.handlers.memory.INCLUDE_HOSTNAME", True, raising=False
    )
    # FastMCP wraps the function; .fn exposes the original coroutine.
    fn = getattr(mcp_server.store_memory, "fn", mcp_server.store_memory)
    # Minimal ctx: ctx.request_context.lifespan_context.memory_service
    ctx = SimpleNamespace(
        request_context=SimpleNamespace(
            lifespan_context=SimpleNamespace(memory_service=server.memory_service)
        )
    )
    content = "fastmcp host stamping test"
    await fn(content=content, ctx=ctx)
    stored = await _fetch_latest(server.storage, content)
    assert stored is not None
    assert (stored.metadata or {}).get("hostname") == socket.gethostname(), (
        f"FastMCP store_memory must stamp the server host, got {stored.metadata}"
    )
