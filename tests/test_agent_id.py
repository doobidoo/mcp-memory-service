"""Fase 1 da RFC #1100 — agent_id em memórias (identidade de autor).

Requisitos cobertos (docs/rfc/rfc-agent-id-multi-agent.md):
- R1: memory_store aceita agent_id opcional → persiste em metadata
- R2: sem argumento, preenche de MCP_AGENT_ID
- R3: sem argumento e sem env → agent_id null (comportamento atual)
- R4: Memory.agent_id property retorna metadata.get("agent_id")
"""

import pytest

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.services.memory_service import MemoryService
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash


@pytest.fixture
async def memory_service(temp_db_path):
    storage = SqliteVecMemoryStorage(f"{temp_db_path}/agent_id.db")
    await storage.initialize()
    try:
        yield MemoryService(storage)
    finally:
        await storage.close()


# ---- R4: property no modelo ----

def test_memory_agent_id_property_reads_metadata():
    """R4: Memory.agent_id retorna metadata.get('agent_id')."""
    m = Memory(
        content="x",
        content_hash=generate_content_hash("x"),
        metadata={"agent_id": "zero"},
    )
    assert m.agent_id == "zero"


def test_memory_agent_id_none_when_absent():
    """R4: ausência de agent_id no metadata → None."""
    m = Memory(content="x", content_hash=generate_content_hash("x"))
    assert m.agent_id is None


def test_memory_agent_id_setter_writes_metadata():
    """R4: setter grava em metadata['agent_id']."""
    m = Memory(content="x", content_hash=generate_content_hash("x"))
    m.agent_id = "tpol"
    assert m.metadata["agent_id"] == "tpol"


# ---- R1-R3: store_memory propaga/infere agent_id ----

@pytest.mark.asyncio
async def test_store_persists_explicit_agent_id(memory_service):
    """R1: agent_id explícito é persistido no metadata da memória."""
    result = await memory_service.store_memory(content="explicit author", agent_id="zero")
    stored = await _fetch_stored(memory_service, result)
    assert stored.agent_id == "zero"


@pytest.mark.asyncio
async def test_store_falls_back_to_env(memory_service, monkeypatch):
    """R2: sem argumento, agent_id vem de MCP_AGENT_ID."""
    monkeypatch.setenv("MCP_AGENT_ID", "scotty")
    result = await memory_service.store_memory(content="env author")
    stored = await _fetch_stored(memory_service, result)
    assert stored.agent_id == "scotty"


@pytest.mark.asyncio
async def test_store_null_when_no_signal(memory_service, monkeypatch):
    """R3: sem argumento e sem env → agent_id null (comportamento atual)."""
    monkeypatch.delenv("MCP_AGENT_ID", raising=False)
    result = await memory_service.store_memory(content="anonymous author")
    stored = await _fetch_stored(memory_service, result)
    assert stored.agent_id is None


async def _fetch_stored(memory_service, store_result):
    """Recupera a memória recém-armazenada pelo content_hash do resultado."""
    assert store_result.get("success"), f"store falhou: {store_result!r}"
    content_hash = store_result["memory"]["content_hash"]
    memories = await memory_service.storage.get_all_memories()
    for m in memories:
        if m.content_hash == content_hash:
            return m
    raise AssertionError("memória armazenada não encontrada")


# ---- R1 via caminho real do MCP handler (integração, G5) ----

@pytest.mark.asyncio
async def test_mcp_handler_propagates_agent_id(memory_service, monkeypatch):
    """R1 (integração): agent_id passado como argumento MCP chega ao metadata.

    Exercita o caminho real handle_store_memory -> memory_service.store_memory,
    não apenas o service isolado.
    """
    from types import SimpleNamespace
    from mcp_memory_service.server.handlers.memory import handle_store_memory

    monkeypatch.delenv("MCP_AGENT_ID", raising=False)

    async def _noop():
        return None

    server = SimpleNamespace(
        memory_service=memory_service,
        _ensure_storage_initialized=_noop,
    )

    await handle_store_memory(
        server,
        {"content": "via mcp handler", "agent_id": "zero", "metadata": {}},
    )
    memories = await memory_service.storage.get_all_memories()
    match = [m for m in memories if m.content == "via mcp handler"]
    assert match, "memória não armazenada via handler"
    assert match[0].agent_id == "zero"


def test_setter_none_removes_key_not_pollutes():
    """agent_id = None removes the key (stays unknown), does not store None."""
    m = Memory(content="x", content_hash="h", tags=[])
    m.agent_id = "zero"
    assert m.metadata["agent_id"] == "zero"
    m.agent_id = None
    assert "agent_id" not in m.metadata
    assert m.agent_id is None


@pytest.mark.asyncio
async def test_store_empty_env_does_not_write_agent_id(memory_service, monkeypatch):
    """MCP_AGENT_ID='' (empty) is not a signal: metadata carries no agent_id."""
    monkeypatch.setenv("MCP_AGENT_ID", "")
    result = await memory_service.store_memory(content="empty env agent")
    stored = await _fetch_stored(memory_service, result)
    assert stored.agent_id is None
    assert "agent_id" not in stored.metadata
