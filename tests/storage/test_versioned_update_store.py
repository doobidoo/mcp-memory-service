"""A versioned update keeps the memory in its store partition.

update_memory_versioned() wrote the new version through store() without a
store argument, so it always landed in "default". The old row was superseded
in its own store, and the memory vanished from that store's searches.
"""

import os
import shutil
import tempfile

import pytest
import pytest_asyncio

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash

os.environ["MCP_MEMORY_ALLOW_HASH_EMBEDDINGS"] = "1"


@pytest_asyncio.fixture
async def storage():
    d = tempfile.mkdtemp(prefix="mcp-test-versioned-store-")
    s = SqliteVecMemoryStorage(os.path.join(d, "test.db"))
    await s.initialize()
    yield s
    await s.close()
    shutil.rmtree(d, ignore_errors=True)


def _store_of(storage, content_hash):
    row = storage.conn.execute(
        "SELECT store FROM memories WHERE content_hash = ?", (content_hash,)
    ).fetchone()
    return row[0]


async def _store(storage, content, store):
    m = Memory(content=content, content_hash=generate_content_hash(content), tags=["t"])
    ok, msg = await storage.store(m, skip_semantic_dedup=True, store=store)
    assert ok, msg
    return m.content_hash


@pytest.mark.asyncio
async def test_new_version_stays_in_the_old_versions_store(storage):
    old = await _store(storage, "The staging cluster runs on three nodes.", "work")

    ok, msg, new_hash = await storage.update_memory_versioned(
        old, "The staging cluster runs on five nodes."
    )

    assert ok, msg
    assert _store_of(storage, new_hash) == "work"


@pytest.mark.asyncio
async def test_versioned_memory_is_still_found_in_its_store(storage):
    old = await _store(storage, "Deploys go out on Tuesdays.", "work")
    await _store(storage, "Groceries: milk and eggs.", "default")

    ok, msg, new_hash = await storage.update_memory_versioned(old, "Deploys go out on Thursdays.")
    assert ok, msg

    work = await storage.retrieve("Deploys go out", n_results=5, store="work")
    assert [r.memory.content_hash for r in work] == [new_hash]

    default = await storage.retrieve("Deploys go out", n_results=5, store="default")
    assert new_hash not in [r.memory.content_hash for r in default]
