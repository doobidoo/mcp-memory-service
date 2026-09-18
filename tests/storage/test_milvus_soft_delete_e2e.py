"""End-to-end test for Milvus soft-delete visibility against a REAL Milvus Lite backend.

Covers #1261: ``get_by_hash()`` must hide soft-deleted rows (matching sqlite_vec
and cloudflare since #1255), while ``is_deleted()`` must still see the tombstone
via its own lookup.

A tombstone reaches Milvus as a ``metadata.deleted_at`` value — Milvus's own
``delete()`` is a hard delete, but hybrid sync propagates a soft-delete from the
primary by writing ``deleted_at`` into metadata via ``update_memory_metadata``.
Both methods consume that value, so both are exercised here against a real
Milvus Lite file (only the embedding is stubbed; the storage layer where the
bug lives is real, which is what the mock-based suites cannot catch).

Run: pytest tests/storage/test_milvus_soft_delete_e2e.py -v
"""

import time
import uuid

import numpy as np
import pytest
import pytest_asyncio

pymilvus = pytest.importorskip("pymilvus")
milvus_lite = pytest.importorskip("milvus_lite")

from mcp_memory_service.models.memory import Memory  # noqa: E402
from mcp_memory_service.storage.milvus import MilvusMemoryStorage  # noqa: E402
from mcp_memory_service.utils.hashing import generate_content_hash  # noqa: E402

_DIM = 8


class _StubEncoder:
    """Deterministic per-text vectors — no model download, no network."""

    def encode(self, texts, convert_to_numpy=True, **kwargs):
        out = []
        for text in texts:
            seed = abs(hash(text)) % (10 ** 6)
            out.append(np.random.default_rng(seed).random(_DIM, dtype=np.float32))
        return np.array(out)


@pytest.fixture(scope="module")
def milvus_db_path(tmp_path_factory):
    return tmp_path_factory.mktemp("milvus_soft_delete_e2e") / "milvus.db"


@pytest_asyncio.fixture
async def storage(milvus_db_path, monkeypatch):
    async def _stub_init_embedding(self):
        self.embedding_model = _StubEncoder()
        self.embedding_dimension = _DIM

    monkeypatch.setattr(
        MilvusMemoryStorage, "_initialize_embedding_model", _stub_init_embedding
    )
    collection_name = f"mcp_softdel_{uuid.uuid4().hex[:12]}"
    instance = MilvusMemoryStorage(
        uri=str(milvus_db_path),
        collection_name=collection_name,
        embedding_model="stub",
    )
    await instance.initialize()
    try:
        yield instance
    finally:
        await instance.close()


async def _store_live(storage, content: str) -> str:
    content_hash = generate_content_hash(content)
    ok, msg = await storage.store(
        Memory(content=content, content_hash=content_hash, tags=["t"], memory_type="note")
    )
    assert ok, msg
    return content_hash


@pytest.mark.asyncio
async def test_get_by_hash_hides_soft_deleted_but_is_deleted_sees_it(storage):
    content_hash = await _store_live(storage, "a memory that will be soft-deleted")

    # Live: visible to get_by_hash, not deleted.
    assert (await storage.get_by_hash(content_hash)) is not None
    assert (await storage.is_deleted(content_hash)) is False

    # Soft-delete as hybrid sync does: write deleted_at into metadata.
    ok, msg = await storage.update_memory_metadata(
        content_hash, {"metadata": {"deleted_at": time.time()}}
    )
    assert ok, msg

    # get_by_hash now hides the tombstone (the #1261 fix) ...
    assert (await storage.get_by_hash(content_hash)) is None
    # ... while is_deleted still reports it, via its own lookup.
    assert (await storage.is_deleted(content_hash)) is True


@pytest.mark.asyncio
async def test_live_memory_is_visible_and_not_deleted(storage):
    content_hash = await _store_live(storage, "a plain live memory")
    got = await storage.get_by_hash(content_hash)
    assert got is not None
    assert got.content == "a plain live memory"
    assert (await storage.is_deleted(content_hash)) is False


@pytest.mark.asyncio
async def test_is_deleted_false_for_absent_memory(storage):
    assert (await storage.is_deleted(generate_content_hash("never stored"))) is False
