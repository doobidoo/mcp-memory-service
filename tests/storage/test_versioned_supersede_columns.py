"""Versioned update must write the migration-011 columns, not just metadata JSON (#1318).

Before this fix `update_memory_versioned` wrote `superseded_by` only into the
metadata JSON blob, while default retrieval, `get_memory_history()`,
`resolve_conflict()` and `mark_superseded_batch()` all read the dedicated
columns from migration 011. Consequences, reproduced here as RED-on-main:

1. the old version stays in default search results (column `superseded_by` NULL);
2. `get_memory_history()` returns a 1-element list for both ends of a chain it
   was written to walk (columns `parent_id`/`version` never set).

The tool descriptions (`memory_update.versioned` = "old memory is marked as
superseded", `memory_search.include_superseded` = "superseded memories are
hidden") promise the column behaviour, so these assert the promise.
"""

import os
import shutil
import tempfile

import pytest
import pytest_asyncio

from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.utils.hashing import generate_content_hash


@pytest.fixture
def temp_storage_dir():
    d = tempfile.mkdtemp(prefix="mcp-test-versioned-")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest_asyncio.fixture
async def storage(temp_storage_dir):
    assert "mcp-test-" in temp_storage_dir
    db_path = os.path.join(temp_storage_dir, "test.db")
    os.environ["MCP_MEMORY_SQLITE_PATH"] = db_path
    os.environ["MCP_MEMORY_STORAGE_BACKEND"] = "sqlite_vec"
    os.environ["MCP_SEMANTIC_DEDUP_ENABLED"] = "false"
    s = SqliteVecMemoryStorage(db_path)
    await s.initialize()
    yield s
    await s.close()


async def _store(storage, content):
    m = Memory(content=content, content_hash=generate_content_hash(content), tags=["__test__"])
    ok, _ = await storage.store(m)
    assert ok
    return m.content_hash


def _columns(storage, content_hash):
    cur = storage.conn.execute(
        "SELECT superseded_by, parent_id, version FROM memories WHERE content_hash = ?",
        (content_hash,),
    )
    row = cur.fetchone()
    return {"superseded_by": row[0], "parent_id": row[1], "version": row[2]}


@pytest.mark.asyncio
async def test_versioned_update_sets_superseded_by_column(storage):
    """The old row's `superseded_by` COLUMN points to the new version."""
    old = await _store(storage, "The nightly backup runs at 02:00.")
    ok, _, new_hash = await storage.update_memory_versioned(
        old, "The nightly backup runs at 03:00.", reason="schedule moved"
    )
    assert ok
    assert _columns(storage, old)["superseded_by"] == new_hash, (
        "old version must have superseded_by set in the COLUMN, not only metadata JSON"
    )


@pytest.mark.asyncio
async def test_versioned_update_sets_parent_and_version_on_new_row(storage):
    """The new row gets parent_id = old hash and version = old + 1."""
    old = await _store(storage, "Python pinned at 3.11.")
    ok, _, new_hash = await storage.update_memory_versioned(old, "Python pinned at 3.14.")
    assert ok
    cols = _columns(storage, new_hash)
    assert cols["parent_id"] == old, "new row must link to its parent via column"
    assert cols["version"] == 2, "new row version must be old version + 1"


@pytest.mark.asyncio
async def test_superseded_old_version_hidden_from_default_search(storage):
    """After a versioned update, the old content is not in default retrieval."""
    old = await _store(storage, "The deploy window is Tuesdays.")
    await storage.update_memory_versioned(old, "The deploy window is Thursdays.")
    results = await storage.retrieve("deploy window", n_results=10)
    hashes = {r.memory.content_hash for r in results}
    assert old not in hashes, "superseded old version must not appear in default search"


@pytest.mark.asyncio
async def test_get_memory_history_links_the_chain(storage):
    """get_memory_history returns the full lineage, oldest-first, once columns are set."""
    old = await _store(storage, "Region is us-east-1.")
    ok, _, h2 = await storage.update_memory_versioned(old, "Region is us-west-2.")
    assert ok
    history = await storage.get_memory_history(old)
    assert len(history) == 2, "history must link both versions via parent_id/version columns"
    assert history[0]["content_hash"] == old
    assert history[-1]["content_hash"] == h2


@pytest.mark.asyncio
async def test_three_version_chain_increments_version_and_links(storage):
    """A 3-version chain: versions increment 1→2→3 and history walks all three (reviewer P2)."""
    h1 = await _store(storage, "Chain step one content.")
    ok2, _, h2 = await storage.update_memory_versioned(h1, "Chain step two content.")
    ok3, _, h3 = await storage.update_memory_versioned(h2, "Chain step three content.")
    assert ok2 and ok3

    assert _columns(storage, h2)["version"] == 2
    assert _columns(storage, h3)["version"] == 3
    assert _columns(storage, h2)["parent_id"] == h1
    assert _columns(storage, h3)["parent_id"] == h2

    # each older version points forward via the column
    assert _columns(storage, h1)["superseded_by"] == h2
    assert _columns(storage, h2)["superseded_by"] == h3

    history = await storage.get_memory_history(h1)
    assert [r["content_hash"] for r in history] == [h1, h2, h3]


@pytest.mark.asyncio
async def test_link_and_supersede_are_consistent(storage):
    """Link (new row) and supersede (old row) always land together — the invariant
    the single transaction guarantees (reviewer P1).

    A successful versioned update must leave BOTH sides set: the old row's
    superseded_by column AND the new row's parent_id/version. There is no valid
    state where one is set without the other (that was the pre-fix inconsistency
    the atomic transaction removes).
    """
    old = await _store(storage, "Atomic original content.")
    ok, _, new_hash = await storage.update_memory_versioned(old, "Atomic new content.")
    assert ok

    old_cols = _columns(storage, old)
    new_cols = _columns(storage, new_hash)
    # Both sides of the link are present together
    assert old_cols["superseded_by"] == new_hash
    assert new_cols["parent_id"] == old
    assert new_cols["version"] == 2
    # Invariant: an old row is superseded iff its successor points back to it
    assert (old_cols["superseded_by"] is not None) == (new_cols["parent_id"] == old)


@pytest.mark.asyncio
async def test_second_update_of_already_superseded_old_is_refused(storage):
    """A second versioned update of an already-superseded row must not fork history
    (Greptile: concurrent updates fork history).

    Supersession is conditional on the old row still being current
    (superseded_by IS NULL). Once superseded, a further update against the SAME old
    hash is refused instead of overwriting the forward link and stranding a successor.
    """
    old = await _store(storage, "Fork guard original content.")
    ok1, _, h2 = await storage.update_memory_versioned(old, "Fork guard v2 content.")
    assert ok1
    assert _columns(storage, old)["superseded_by"] == h2

    # Second update of the SAME (now superseded) old hash must fail...
    ok2, msg, h3 = await storage.update_memory_versioned(old, "Fork guard v3 racing content.")
    assert ok2 is False, "updating an already-superseded row must be refused, not forked"
    # ...and the original forward link is untouched (still points to h2, not overwritten).
    assert _columns(storage, old)["superseded_by"] == h2
