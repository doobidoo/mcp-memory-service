"""
Regression test for the missing-embedding invariant (#1225).

memories rows must never exist without a matching memory_embeddings row.
The original mechanism: store() guards its savepoint + commit with
`_savepoint_lock`, but the delete/metadata mixins commit the shared
connection without that lock. A commit landing between store's
`INSERT INTO memories` and its embedding insert breaks the savepoint:
the memories row is persisted alone and the RELEASE fails, leaving an
orphan that is invisible to semantic search.
"""

import os
import shutil
import tempfile

import pytest

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils import generate_content_hash


def _make_memory(content: str) -> Memory:
    return Memory(content=content, content_hash=generate_content_hash(content))
def _count_orphans(storage) -> int:
    """Alive memories without embeddings. Tombstones drop their embedding by
    design (delete.py), so deleted_at rows are excluded."""
    cur = storage.conn.execute(
        "SELECT COUNT(*) FROM memories "
        "WHERE deleted_at IS NULL "
        "AND id NOT IN (SELECT rowid FROM memory_embeddings)"
    )
    return cur.fetchone()[0]


@pytest.mark.asyncio
async def test_interleaved_commit_does_not_orphan_a_memory():
    """A foreign commit mid-savepoint must not persist the memories row alone.

    Simulates a delete/metadata coroutine committing the shared connection
    between store()'s memories INSERT and its embedding INSERT.
    """
    tmp = tempfile.mkdtemp()
    storage = SqliteVecMemoryStorage(
        db_path=os.path.join(tmp, "race.db"),
        embedding_model="all-MiniLM-L6-v2",
    )
    await storage.initialize()
    try:
        conn = storage.conn
        fired = {"done": False}

        class _RacingConn:
            """Proxy over the shared connection simulating a sibling
            coroutine committing mid-savepoint (delete/metadata paths)."""

            def __init__(self, real):
                self._real = real

            def execute(self, sql, *args, **kwargs):
                result = self._real.execute(sql, *args, **kwargs)
                if (
                    not fired["done"]
                    and sql.strip().startswith("INSERT INTO memories")
                ):
                    fired["done"] = True
                    self._real.commit()
                return result

            def __getattr__(self, name):
                return getattr(self._real, name)

        storage.conn = _RacingConn(conn)

        # the store may fail loudly because its savepoint was broken; that is
        # fine — the invariant is about the database state, not the return
        try:
            await storage.store(_make_memory("race canary content"))
        except Exception:
            pass

        storage.conn = conn
        conn.commit()

        assert _count_orphans(storage) == 0, (
            "memories row committed without its embedding row "
            "(the #1225 orphan mechanism)"
        )
    finally:
        storage.close()
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.asyncio
async def test_store_then_soft_delete_leaves_no_orphan():
    """Sanity: the normal store/soft-delete cycle never violates the invariant."""
    tmp = tempfile.mkdtemp()
    storage = SqliteVecMemoryStorage(
        db_path=os.path.join(tmp, "cycle.db"),
        embedding_model="all-MiniLM-L6-v2",
    )
    await storage.initialize()
    try:
        mem = _make_memory("cycle canary content")
        ok, _ = await storage.store(mem)
        assert ok
        ok, _ = await storage.delete(mem.content_hash)
        assert ok
        assert _count_orphans(storage) == 0
    finally:
        storage.close()
        shutil.rmtree(tmp, ignore_errors=True)
