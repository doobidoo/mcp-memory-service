"""Escopo de `store` no caminho REAL do MCP: `_retrieve_search_results`.

A fatia 1 ajustou `MemoryService`, mas `handle_memory_search` do MCP nao passa por
ele: chama `_retrieve_search_results(storage, arguments, normalize_tags)`. Esse
helper fazia `arguments.get("store", "default")` e os fallbacks (exato e tag) nem
recebiam o escopo — resultado: busca sem `store` continuava presa em "default", e
os fallbacks ficavam cegos aos outros stores.
"""

import hashlib
import os
import shutil
import tempfile

import pytest
import pytest_asyncio

os.environ["MCP_MEMORY_ALLOW_HASH_EMBEDDINGS"] = "1"

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.server.handlers.memory import _retrieve_search_results, _resolve_store_scope
from mcp_memory_service.services.memory_service import normalize_tags
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage

WORK = "kubectl rollout restart deployment payments-api work store runbook"
DEFAULT = "postgres logical replication slot maintenance default store runbook"


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest_asyncio.fixture
async def storage():
    tmp = tempfile.mkdtemp(prefix="mcp-test-handler-scope-")
    st = SqliteVecMemoryStorage(os.path.join(tmp, "test.db"))
    await st.initialize()
    await st.store(Memory(content=WORK, content_hash=_hash(WORK), tags=["deploy"]), store="work")
    await st.store(Memory(content=DEFAULT, content_hash=_hash(DEFAULT), tags=["deploy"]), store="default")
    yield st
    await st.close()
    shutil.rmtree(tmp, ignore_errors=True)


def _contents(result: dict) -> list[str]:
    return [m["content"] for m in result.get("memories", [])]


@pytest.mark.parametrize("mode", ["semantic", "exact", "hybrid", "ranked"])
@pytest.mark.asyncio
async def test_omitted_store_searches_every_store(storage, mode):
    """R1: sem `store`, todo modo enxerga memoria de outro store."""
    result, _ = await _retrieve_search_results(
        storage, {"query": WORK, "mode": mode, "limit": 10}, normalize_tags
    )
    assert WORK in _contents(result), f"modo {mode}: busca sem store nao viu a memoria de 'work'"


@pytest.mark.parametrize("mode", ["semantic", "exact", "hybrid", "ranked"])
@pytest.mark.asyncio
async def test_explicit_store_isolates(storage, mode):
    """R3/R4: `store='default'` nao devolve nada de 'work', em nenhum modo."""
    result, _ = await _retrieve_search_results(
        storage, {"query": WORK, "mode": mode, "limit": 10, "store": "default"}, normalize_tags
    )
    assert WORK not in _contents(result), f"modo {mode}: store='default' vazou memoria de 'work'"


@pytest.mark.asyncio
async def test_tag_fallback_honours_store(storage):
    """R4: o fallback por tag não pode vazar entre partições.

    A tag 'deploy' existe nos dois stores, então só o escopo impede o vazamento.
    O fallback é exercitado direto: ele recebia ``store`` e o descartava.
    """
    from mcp_memory_service.server.handlers.memory import _search_tag_fallback

    scoped = await _search_tag_fallback(storage, "deploy", 10, False, "work")
    assert any(WORK in m["content"] for m in scoped), "fallback não achou a memória do próprio store"
    assert all(DEFAULT not in m["content"] for m in scoped), "fallback vazou memória de 'default'"

    wide = await _search_tag_fallback(storage, "deploy", 10, False, None)
    assert {WORK, DEFAULT} <= {m["content"] for m in wide}, "escopo amplo deveria ver os dois stores"


@pytest.mark.asyncio
async def test_all_alias_keeps_wide_scope(storage):
    """`store='all'` continua significando todos (compatibilidade)."""
    result, _ = await _retrieve_search_results(
        storage, {"query": WORK, "mode": "semantic", "limit": 10, "store": "all"}, normalize_tags
    )
    assert WORK in _contents(result)


def test_resolve_store_scope_rules():
    assert _resolve_store_scope({}) is None
    assert _resolve_store_scope({"store": "all"}) is None
    assert _resolve_store_scope({"store": "default"}) == "default"
    assert _resolve_store_scope({"store": "mir"}) == "mir"
