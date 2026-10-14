"""
Test for #1521: regenerate_embeddings.py must write each vector into the memory's own
store partition. Before the fix it wrote `(rowid, content_embedding)` with no `store`,
so regenerated vectors landed in the wrong partition and store-scoped KNN never found
them — while the script reported success.

The test exercises the real core functions (_load_memories + _regenerate_all) against an
in-memory sqlite-vec storage, so it goes RED without the fix and GREEN with it.
"""

import importlib.util
import os
import time
from pathlib import Path

import pytest

try:
    import sqlite_vec  # noqa: F401
    SQLITE_VEC_AVAILABLE = True
except ImportError:
    SQLITE_VEC_AVAILABLE = False

if SQLITE_VEC_AVAILABLE:
    from mcp_memory_service.models.memory import Memory
    from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
    from mcp_memory_service.utils.hashing import generate_content_hash

pytestmark = pytest.mark.skipif(not SQLITE_VEC_AVAILABLE, reason="sqlite-vec not available")


def _load_regen_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "maintenance" / "regenerate_embeddings.py"
    spec = importlib.util.spec_from_file_location("regenerate_embeddings", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
async def storage(temp_db_path):
    s = SqliteVecMemoryStorage(os.path.join(temp_db_path, "regen.db"))
    try:
        await s.initialize()
        yield s
    finally:
        await s.close()


@pytest.mark.asyncio
async def test_regenerate_writes_into_named_store_partition(storage):
    """A memory in a named store, after its vector is dropped and regenerated, must have
    the vector back in its OWN partition (not 'default') and be found by a store-scoped
    search. Proves #1521."""
    regen = _load_regen_module()

    content = "Memory that lives in a named store partition"
    h = generate_content_hash(content)
    await storage.store(Memory(content=content, content_hash=h, tags=["x"]), store="projectX")

    # Drop the vector but keep the memory row (the state the script is meant to repair).
    rowid = storage.conn.execute("SELECT id FROM memories WHERE content_hash = ?", (h,)).fetchone()[0]
    storage.conn.execute("DELETE FROM memory_embeddings WHERE rowid = ?", (rowid,))
    storage.conn.commit()

    # Regenerate via the real core functions.
    memories, _ = regen._load_memories(storage)
    success, errors = await regen._regenerate_all(storage, memories)
    storage.conn.commit()
    assert success >= 1 and errors == 0, f"regeneration failed: success={success} errors={errors}"

    # The vector must be in the 'projectX' partition, not 'default'.
    vec = storage.conn.execute(
        "SELECT store FROM memory_embeddings WHERE rowid = ?", (rowid,)
    ).fetchone()
    assert vec is not None and vec[0] == "projectX", \
        f"vector must be in the memory's own store partition, got {vec}"

    # And a store-scoped retrieve finds it.
    results = await storage.retrieve(content, n_results=5, store="projectX")
    hashes = [r.memory.content_hash for r in results] if results else []
    assert h in hashes, "store-scoped retrieve must find the regenerated memory"



@pytest.mark.asyncio
async def test_regenerate_skips_soft_deleted_memories(storage):
    """A soft-deleted (tombstoned) memory must NOT be reloaded by _load_memories nor get its
    vector regenerated — otherwise the maintenance script resurrects it into the vector index.
    Guards the Greptile P2 soft-delete fix (WHERE deleted_at IS NULL)."""
    regen = _load_regen_module()

    # One live memory and one soft-deleted memory.
    live = "A live memory that should be regenerated"
    gone = "A deleted memory that must stay out of the index"
    h_live = generate_content_hash(live)
    h_gone = generate_content_hash(gone)
    await storage.store(Memory(content=live, content_hash=h_live, tags=["x"]))
    await storage.store(Memory(content=gone, content_hash=h_gone, tags=["x"]))

    # Soft-delete the second one and drop every vector (state the script repairs).
    storage.conn.execute(
        "UPDATE memories SET deleted_at = ? WHERE content_hash = ?", (time.time(), h_gone)
    )
    storage.conn.execute("DELETE FROM memory_embeddings")
    storage.conn.commit()

    # _load_memories must return only the live row.
    memories, total = regen._load_memories(storage)
    loaded_hashes = {m["content_hash"] for m in memories}
    assert h_live in loaded_hashes, "live memory must be loaded for regeneration"
    assert h_gone not in loaded_hashes, "soft-deleted memory must not be loaded"
    assert total == 1, f"count must exclude soft-deleted rows, got {total}"

    # And after regeneration the deleted row still has no vector.
    _success, errors = await regen._regenerate_all(storage, memories)
    storage.conn.commit()
    assert errors == 0, f"regeneration errored: {errors}"

    gone_rowid = storage.conn.execute(
        "SELECT id FROM memories WHERE content_hash = ?", (h_gone,)
    ).fetchone()[0]
    vec = storage.conn.execute(
        "SELECT 1 FROM memory_embeddings WHERE rowid = ?", (gone_rowid,)
    ).fetchone()
    assert vec is None, "soft-deleted memory must not get a regenerated vector"
