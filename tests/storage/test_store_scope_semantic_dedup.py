"""Store-scoped semantic and exact duplicate checks (#1517)."""

import pytest

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash


@pytest.fixture
async def storage(temp_db_path, monkeypatch):
    storage = SqliteVecMemoryStorage(f"{temp_db_path}/store_scope.db")
    await storage.initialize()
    storage.semantic_dedup_enabled = True

    # Keep the test deterministic while exercising the real sqlite-vec query.
    embedding = [1.0] + [0.0] * (storage.embedding_dimension - 1)
    monkeypatch.setattr(storage, "_generate_embedding", lambda _content: embedding)

    yield storage
    await storage.close()


def _memory(content: str) -> Memory:
    return Memory(content=content, content_hash=generate_content_hash(content))


@pytest.mark.asyncio
async def test_semantic_dedup_only_rejects_matches_in_target_store(storage):
    default_memory = _memory("same vector in default")
    docs_memory = _memory("same vector in docs")
    duplicate_docs_memory = _memory("another same-vector doc")

    ok, message = await storage.store(default_memory, store="default")
    assert ok, message

    ok, message = await storage.store(docs_memory, store="docs")
    assert ok, message

    ok, message = await storage.store(duplicate_docs_memory, store="docs")
    assert not ok
    assert "semantically similar" in message
    assert docs_memory.content_hash in message


@pytest.mark.asyncio
async def test_exact_duplicate_names_the_store_that_already_has_the_hash(storage):
    content = "identical content must remain globally unique"
    memory = _memory(content)

    ok, message = await storage.store(memory, store="knowledge")
    assert ok, message

    ok, message = await storage.store(_memory(content), store="default")
    assert not ok
    assert "exact match" in message
    assert "knowledge" in message

    batch_results = await storage.store_batch([_memory(content)], store="default")
    assert not batch_results[0][0]
    assert "knowledge" in batch_results[0][1]
