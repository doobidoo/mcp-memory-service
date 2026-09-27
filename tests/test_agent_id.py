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


@pytest.mark.asyncio
async def test_raw_metadata_agent_id_is_used_when_no_arg_or_env(memory_service, monkeypatch):
    """metadata['agent_id'] is a valid identity source (the path harvest/bootstrap
    use) — used when no arg/env identity is set."""
    monkeypatch.delenv("MCP_AGENT_ID", raising=False)
    result = await memory_service.store_memory(
        content="authored via metadata",
        metadata={"agent_id": "kiro"},
    )
    stored = await _fetch_stored(memory_service, result)
    assert stored.agent_id == "kiro"


@pytest.mark.asyncio
async def test_explicit_arg_overrides_metadata_agent_id(memory_service, monkeypatch):
    """Precedence: explicit arg wins over a metadata['agent_id']."""
    monkeypatch.delenv("MCP_AGENT_ID", raising=False)
    result = await memory_service.store_memory(
        content="arg beats metadata",
        agent_id="zero",
        metadata={"agent_id": "kiro"},
    )
    stored = await _fetch_stored(memory_service, result)
    assert stored.agent_id == "zero"


# ---- FASE 2: Filtros agent_id em search e list ----

@pytest.mark.asyncio
async def test_search_filters_by_metadata_agent_id(memory_service, monkeypatch):
    """(1) search_memories filtra por metadata.agent_id."""
    monkeypatch.delenv("MCP_AGENT_ID", raising=False)
    
    # Store memórias com agent_ids distintos (conteúdo semanticamente distinto)
    await memory_service.store_memory(content="Database connection pool optimization strategies", agent_id="zero")
    await memory_service.store_memory(content="Machine learning model hyperparameter tuning methods", agent_id="tpol")
    await memory_service.store_memory(content="React component lifecycle optimization techniques", agent_id=None)
    
    # Buscar só memórias do zero - DEVE FALHAR (agent_id não suportado ainda)
    results = await memory_service.storage.search_memories(
        query="optimization", 
        agent_id="zero"
    )
    
    # Deve retornar só a memória do zero
    assert len(results["memories"]) == 1
    assert results["memories"][0]["agent_id"] == "zero"
    assert "Database connection pool" in results["memories"][0]["content"]


@pytest.mark.asyncio  
async def test_search_filters_by_agent_tag(memory_service, monkeypatch):
    """(2) search_memories filtra por tag agent:<id> em memória SEM metadata.agent_id (valida unificação)."""
    monkeypatch.delenv("MCP_AGENT_ID", raising=False)
    
    # Store memória usando o service mas passando tags diretamente
    content = "Kubernetes deployment configuration best practices"
    await memory_service.store_memory(
        content=content,
        tags=["agent:kiro", "devops", "k8s"]
    )
    
    # Store outra memória sem a tag
    await memory_service.store_memory(content="Frontend build optimization webpack strategies", agent_id="zero")
    
    # Buscar por agent_id=kiro deve encontrar a memória com tag agent:kiro - DEVE FALHAR
    results = await memory_service.storage.search_memories(
        query="configuration",
        agent_id="kiro"
    )
    
    assert len(results["memories"]) == 1
    assert results["memories"][0]["content"] == content
    assert "agent:kiro" in results["memories"][0]["tags"]
    assert results["memories"][0].get("agent_id") is None  # metadata não tem agent_id (key might not exist)


@pytest.mark.asyncio
async def test_search_agent_id_none_returns_all(memory_service, monkeypatch):
    """(3) search SEM agent_id retorna TODOS (sem bolha, regressão)."""
    monkeypatch.delenv("MCP_AGENT_ID", raising=False)
    
    # Store memórias de diferentes agentes
    await memory_service.store_memory(content="PostgreSQL query performance optimization techniques", agent_id="zero")
    await memory_service.store_memory(content="Redis caching strategies for web applications", agent_id="tpol") 
    await memory_service.store_memory(content="MongoDB aggregation pipeline design patterns", agent_id=None)
    
    # Buscar sem agent_id deve retornar todas (comportamento atual deve funcionar)
    results = await memory_service.storage.search_memories(query="optimization")
    
    # Deve ter pelo menos as 3 memórias (pode ter mais de outros testes)
    memories = results.get("memories", [])
    contents = [m["content"] for m in memories]
    assert any("PostgreSQL query performance" in c for c in contents)
    assert any("Redis caching strategies" in c for c in contents) 
    assert any("MongoDB aggregation pipeline" in c for c in contents)


@pytest.mark.asyncio
async def test_list_filters_by_agent_id(memory_service, monkeypatch):
    """(4) list_memories filtra por agent_id via SQL."""
    monkeypatch.delenv("MCP_AGENT_ID", raising=False)
    
    # Store memórias com diferentes agent_ids
    await memory_service.store_memory(content="Docker container security scanning methodologies", agent_id="zero")
    await memory_service.store_memory(content="API rate limiting implementation strategies", agent_id="tpol")
    await memory_service.store_memory(content="Load balancer configuration best practices", agent_id=None)
    
    # List só memórias do zero - DEVE FALHAR (agent_id não suportado ainda)
    zero_memories = await memory_service.list_memories(agent_id="zero")
    
    # Deve retornar só a do zero
    assert len(zero_memories["memories"]) == 1
    assert zero_memories["memories"][0].get("agent_id") == "zero"
    assert "Docker container security" in zero_memories["memories"][0]["content"]
    
    # List sem filtro deve retornar todas
    all_memories = await memory_service.list_memories()
    assert len(all_memories["memories"]) >= 3  # Pelo menos as 3 que criamos


# ---- Fixes do review Greptile (PR #1297) ----


@pytest.mark.asyncio
async def test_list_agent_id_wildcard_is_escaped(memory_service, monkeypatch):
    """Greptile P1: agent_id com % não pode agir como wildcard SQL LIKE.

    Um agent_id "%" não deve casar memórias de OUTROS agentes via a tag agent:<id>.
    """
    monkeypatch.delenv("MCP_AGENT_ID", raising=False)
    await memory_service.store_memory(content="Kubernetes ingress controller tuning notes", agent_id="zero")
    await memory_service.store_memory(content="Postgres vacuum autotuning parameters guide", agent_id="tpol")

    # agent_id literal "%" — sem escaping viraria wildcard e casaria tudo
    leaked = await memory_service.list_memories(agent_id="%")
    assert len(leaked["memories"]) == 0, "agent_id '%' vazou memórias de outros agentes (LIKE não escapado)"


@pytest.mark.asyncio
async def test_search_agent_id_over_fetches_in_semantic_mode(memory_service, monkeypatch):
    """Greptile P1: filtro pós-retrieve precisa de over-fetch no modo semântico comum.

    Com várias memórias de outro agente ranqueando alto, uma memória do agente pedido
    não pode ser truncada antes do filtro rodar (limit pequeno).
    """
    monkeypatch.delenv("MCP_AGENT_ID", raising=False)
    # Muitas memórias do tpol, temas variados mas próximos da query (competem no ranking)
    tpol_topics = [
        "Distributed tracing span sampling in microservices",
        "OpenTelemetry collector pipeline configuration",
        "Jaeger backend storage retention tuning",
        "Trace context propagation across async boundaries",
        "Sampling rate tradeoffs for high-throughput services",
        "Tail-based sampling versus head-based sampling",
        "Span attribute cardinality and cost control",
        "Trace exemplars linking metrics to spans",
    ]
    for t in tpol_topics:
        await memory_service.store_memory(content=t, agent_id="tpol")
    # Uma do zero, tema relacionado mas conteúdo distinto (evita dedup 0.92)
    await memory_service.store_memory(
        content="Observability tracing dashboards curated by the zero agent", agent_id="zero"
    )

    # limit=1: sem over-fetch, o retrieve traria só 1 candidato (provavelmente tpol) e o filtro zeraria
    res = await memory_service.storage.search_memories(
        query="distributed tracing sampling observability", limit=1, agent_id="zero"
    )
    assert len(res["memories"]) >= 1, "over-fetch ausente: memória do zero truncada antes do filtro agent_id"
    assert res["memories"][0]["agent_id"] == "zero"


@pytest.mark.asyncio
async def test_search_agent_id_none_still_returns_all(memory_service, monkeypatch):
    """Regressão sem-bolha: sem agent_id, busca retorna de todos os agentes."""
    monkeypatch.delenv("MCP_AGENT_ID", raising=False)
    await memory_service.store_memory(content="GraphQL schema stitching approach", agent_id="zero")
    await memory_service.store_memory(content="GraphQL federation gateway approach", agent_id="tpol")
    res = await memory_service.storage.search_memories(query="graphql approach", limit=10)
    agents = {m.get("agent_id") for m in res["memories"]}
    assert "zero" in agents and "tpol" in agents, "sem agent_id deveria retornar de todos os agentes"
