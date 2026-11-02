"""A NULL cosine distance must not break later writes in the same store (#1791).

An empty memory is stored with a zero-length embedding, so
`vec_distance_cosine()` returns NULL for that row. `_check_semantic_duplicate()`
used to compare that NULL to a float and raise `TypeError`, which made every
later write in the same store fail until the empty row was deleted.
"""

import pytest

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash


@pytest.fixture
async def storage(temp_db_path, monkeypatch):
    storage = SqliteVecMemoryStorage(f"{temp_db_path}/null_distance.db")
    await storage.initialize()
    storage.semantic_dedup_enabled = True

    # Empty content embeds to the zero vector, which is what makes sqlite-vec
    # return NULL from vec_distance_cosine() against any later write.
    def _embedding(content: str) -> list[float]:
        if not content.strip():
            return [0.0] * storage.embedding_dimension
        return [1.0] + [0.0] * (storage.embedding_dimension - 1)

    monkeypatch.setattr(storage, "_generate_embedding", _embedding)

    yield storage
    await storage.close()


def _memory(content: str) -> Memory:
    return Memory(content=content, content_hash=generate_content_hash(content))


async def test_null_distance_is_treated_as_no_duplicate(storage):
    stored, message = await storage.store(_memory(""))
    assert stored is True, message

    stored, message = await storage.store(_memory("hello"))

    assert stored is True, message
