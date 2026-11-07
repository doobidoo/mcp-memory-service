"""Unknown memory types: coerced on a new write, kept on hydration.

Memory() used to coerce an unknown type to 'observation' in __post_init__, which
also runs when storage hydrates a stored row. A node lacking a peer's custom type
then read 'observation' and wrote it back on the next update. Coercion now lives
on the write paths (coerce_memory_type); these tests pin both halves.
"""

from types import SimpleNamespace

import pytest

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.server.handlers.memory import handle_store_memory
from mcp_memory_service.services.memory_service import MemoryService
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash


@pytest.fixture
async def memory_service(temp_db_path):
    storage = SqliteVecMemoryStorage(f"{temp_db_path}/coercion.db")
    await storage.initialize()
    try:
        yield MemoryService(storage)
    finally:
        await storage.close()


async def _stored(memory_service, content):
    memories = await memory_service.storage.get_all_memories()
    match = [m for m in memories if m.content == content]
    assert match, "memory not stored"
    return match[0]


@pytest.mark.asyncio
async def test_service_store_coerces_unknown_type(memory_service):
    result = await memory_service.store_memory(content="service write", memory_type="bogus_type")

    assert result["success"]
    assert (await _stored(memory_service, "service write")).memory_type == "observation"


async def _noop():
    return None


@pytest.fixture
def server(memory_service):
    return SimpleNamespace(memory_service=memory_service, _ensure_storage_initialized=_noop)


@pytest.mark.asyncio
async def test_handler_still_warns_about_coercion(memory_service, server):
    """#843: the MCP response names the coerced type."""
    response = await handle_store_memory(
        server, {"content": "handler write", "metadata": {"type": "bogus_type"}}
    )

    assert "requested memory_type 'bogus_type'" in response[0].text
    assert (await _stored(memory_service, "handler write")).memory_type == "observation"


@pytest.mark.asyncio
async def test_stored_unknown_type_survives_read(memory_service):
    """A row whose type this node does not know reads back unchanged."""
    content = "written by a peer with a custom type"
    peer_memory = Memory(content=content, content_hash=generate_content_hash(content),
                         memory_type="peer_only_type")
    ok, _ = await memory_service.storage.store(peer_memory)
    assert ok

    assert (await _stored(memory_service, content)).memory_type == "peer_only_type"
