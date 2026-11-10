"""End-to-end store confinement (#1879) against a REAL Milvus Lite backend.

test_milvus_store_scope.py pins the calls the storage makes; this file checks
that Milvus itself does what those calls assume: a row inserted without the
``store`` key lands in ``"default"``, an upsert that carries the row's store
keeps it there, and a ``delete`` whose filter names another store removes
nothing. Only the embedding is stubbed.

Run: pytest tests/storage/test_milvus_store_scope_e2e.py -v
"""

import uuid

import numpy as np
import pytest
import pytest_asyncio

pymilvus = pytest.importorskip("pymilvus")
milvus_lite = pytest.importorskip("milvus_lite")

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.milvus import MilvusMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash

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
    return tmp_path_factory.mktemp("milvus_store_scope_e2e") / "milvus.db"


@pytest_asyncio.fixture
async def storage(milvus_db_path, monkeypatch):
    async def _stub_init_embedding(self):
        self.embedding_model = _StubEncoder()
        self.embedding_dimension = _DIM

    monkeypatch.setattr(
        MilvusMemoryStorage, "_initialize_embedding_model", _stub_init_embedding
    )
    monkeypatch.setenv("MCP_SEMANTIC_DEDUP_ENABLED", "false")
    instance = MilvusMemoryStorage(
        uri=str(milvus_db_path),
        collection_name=f"mcp_store_{uuid.uuid4().hex[:12]}",
        embedding_model="stub",
    )
    await instance.initialize()
    try:
        yield instance
    finally:
        await instance.close()


async def _store(storage, content: str, **kwargs) -> str:
    content_hash = generate_content_hash(content)
    ok, msg = await storage.store(
        Memory(content=content, content_hash=content_hash, tags=["t"], memory_type="note"),
        **kwargs,
    )
    assert ok, msg
    return content_hash


async def _stored_in(storage, content_hash: str):
    rows = await storage._call_client(
        "get",
        collection_name=storage.collection_name,
        ids=[content_hash],
        output_fields=["store"],
    )
    return rows[0]["store"] if rows else None


@pytest.mark.asyncio
async def test_store_is_kept_and_confines_the_hash_addressed_calls(storage):
    unnamed = await _store(storage, "written without a store")
    team = await _store(storage, "written to the team store", store="team")
    assert await _stored_in(storage, unnamed) == "default"
    assert await _stored_in(storage, team) == "team"

    # Reads
    assert await storage.get_by_hash(team, store="team") is not None
    assert await storage.get_by_hash(team) is not None
    assert await storage.get_by_hash(team, store="ops") is None
    assert await storage.get_by_hash(team, store="default") is None

    # An update elsewhere is refused; an unscoped one keeps the row where it is.
    ok, _ = await storage.update_memory_metadata(team, {"tags": ["x"]}, store="ops")
    assert ok is False
    ok, msg = await storage.update_memory_metadata(team, {"tags": ["x"]})
    assert ok, msg
    assert await _stored_in(storage, team) == "team"
    assert (await storage.get_by_hash(team, store="team")).tags == ["x"]

    # A delete elsewhere removes nothing; in its own store it goes.
    ok, _ = await storage.delete(team, store="ops")
    assert ok is False
    assert await _stored_in(storage, team) == "team"
    ok, msg = await storage.delete(team, store="team")
    assert ok, msg
    assert await _stored_in(storage, team) is None
