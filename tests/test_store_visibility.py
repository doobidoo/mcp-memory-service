"""Store visibility (spec-store-visibility, slice 1) — memory_search scope.

RED tests for the implicit ``store="default"`` scope that made memories living in
other stores (``mir``, ``docs-shared``, ``hackathon``, ...) unreachable.

Contract: docs/rfc/planned/spec-store-visibility.md
  R1 - memory_search called WITHOUT ``store`` searches every store.
  R2 - each result carries its origin store as a first-class ``store`` field,
       read from ``memories.store`` (not inferred, not hidden in metadata).
  R3 - an explicit ``store`` keeps restricting the search to that store.
  R5 - the response declares the effective scope (``stores_searched``).

No storage mock: a real SqliteVecMemoryStorage on a temporary database.
"""

import hashlib
import os
import shutil
import tempfile

import pytest
import pytest_asyncio

# Hash embeddings keep the test fast and deterministic (same trick as
# tests/test_store_partition_isolation.py).
os.environ["MCP_MEMORY_ALLOW_HASH_EMBEDDINGS"] = "1"

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.services.memory_service import MemoryService
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage

WORK_TEXT = "kubectl rollout restart deployment payments-api from the work store"
DEFAULT_TEXT = "postgres logical replication slot maintenance runbook in the default store"


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest_asyncio.fixture
async def service():
    """Real storage on a temp DB, seeded with one memory in 'default' and one in 'work'."""
    tmp_dir = tempfile.mkdtemp(prefix="mcp-test-store-visibility-")
    db_path = os.path.join(tmp_dir, "test.db")

    storage = SqliteVecMemoryStorage(db_path)
    await storage.initialize()

    ok_work, msg_work = await storage.store(
        Memory(content=WORK_TEXT, content_hash=_hash(WORK_TEXT), tags=["probe"]),
        store="work",
    )
    assert ok_work, msg_work

    ok_default, msg_default = await storage.store(
        Memory(content=DEFAULT_TEXT, content_hash=_hash(DEFAULT_TEXT), tags=["probe"]),
        store="default",
    )
    assert ok_default, msg_default

    yield MemoryService(storage)

    await storage.close()
    shutil.rmtree(tmp_dir, ignore_errors=True)


def _contents(memories):
    return [m["content"] for m in memories]


@pytest.mark.asyncio
async def test_search_without_store_finds_memory_in_other_store(service):
    """R1+R2: no store argument -> the 'work' memory is reachable and tagged 'work'."""
    result = await service.retrieve_memories(query=WORK_TEXT, n_results=10)

    memories = result["memories"]
    assert WORK_TEXT in _contents(memories), (
        "search without store must span every store (R1): the 'work' memory is invisible"
    )

    hit = next(m for m in memories if m["content"] == WORK_TEXT)
    assert hit["store"] == "work", "each result must expose its origin store (R2)"


@pytest.mark.asyncio
async def test_explicit_default_store_still_excludes_other_stores(service):
    """R3: store='default' stays restricted to the default store."""
    result = await service.retrieve_memories(query=WORK_TEXT, n_results=10, store="default")

    memories = result["memories"]
    assert WORK_TEXT not in _contents(memories), (
        "explicit store='default' must not leak memories from other stores (R3)"
    )
    assert all(m["store"] == "default" for m in memories)


@pytest.mark.asyncio
async def test_explicit_work_store_returns_work_memory(service):
    """R3: store='work' returns the work memory and nothing from 'default'."""
    result = await service.retrieve_memories(query=WORK_TEXT, n_results=10, store="work")

    memories = result["memories"]
    assert WORK_TEXT in _contents(memories), "explicit store='work' must return its memory (R3)"
    assert DEFAULT_TEXT not in _contents(memories), "explicit store='work' must not cross stores"
    assert all(m["store"] == "work" for m in memories)


@pytest.mark.asyncio
async def test_response_declares_searched_stores(service):
    """R5: the response declares the effective scope of the search."""
    all_stores = await service.retrieve_memories(query=WORK_TEXT, n_results=10)
    assert "stores_searched" in all_stores, "response must declare the effective scope (R5)"
    assert set(all_stores["stores_searched"]) >= {"default", "work"}

    scoped = await service.retrieve_memories(query=WORK_TEXT, n_results=10, store="work")
    assert scoped["stores_searched"] == ["work"]


@pytest.mark.asyncio
async def test_storage_retrieve_omitted_store_keeps_default_scope(service):
    """Contract guard: internal callers that OMIT ``store`` keep the 'default' scope.

    Dedup, belief grouping and harvest call ``storage.retrieve(...)`` without a
    store argument. If that omissible default became "every store", writing to
    ``default`` could collide with identical content living in ``mir`` — a silent
    change in the WRITE path (the #1517 territory). Only the search handler asks
    for the wide scope, and it does so by passing ``store=None`` explicitly.
    """
    results = await service.storage.retrieve(WORK_TEXT, n_results=10)
    contents = [r.memory.content for r in results]

    assert WORK_TEXT not in contents, (
        "omitting `store` must keep the historical 'default' scope; "
        "wide scope is opt-in via store=None"
    )

