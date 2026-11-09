"""The hybrid sync queue files a memory in the same store on the secondary (#1915, part 2).

HybridMemoryStorage.store() wrote the primary row into the named store but queued
the secondary copy without it, so every memory synced through hybrid landed in
``default`` on Cloudflare. These tests run the real queue: a SQLite-vec primary,
and a CloudflareStorage whose D1 calls run on an in-memory SQLite database.
"""

import os
import shutil
import tempfile
import time
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.hybrid import (
    BackgroundSyncService,
    HybridMemoryStorage,
    SyncOperation,
)
from mcp_memory_service.utils.hashing import generate_content_hash
from tests.unit.test_cloudflare_store_scope import _FakeD1, cloudflare_storage, d1  # noqa: F401


@pytest_asyncio.fixture
async def hybrid(cloudflare_storage, d1):  # noqa: F811
    await cloudflare_storage._initialize_d1_schema()
    tmp = tempfile.mkdtemp(prefix="mcp-test-hybrid-sync-store-")
    storage = HybridMemoryStorage(sqlite_db_path=os.path.join(tmp, "primary.db"))
    await storage.primary.initialize()
    storage.secondary = cloudflare_storage
    # Not started: the tests drain the queue themselves, one operation at a time.
    storage.sync_service = BackgroundSyncService(storage.primary, cloudflare_storage)
    yield storage
    await storage.primary.close()
    shutil.rmtree(tmp, ignore_errors=True)


async def _drain(storage):
    queue = storage.sync_service.operation_queue
    while not queue.empty():
        await storage.sync_service._process_single_operation(queue.get_nowait())


def _memory(content):
    now = time.time()
    return Memory(content=content, content_hash=generate_content_hash(content),
                  tags=["orig"], created_at=now, updated_at=now)


def _primary_store(storage, content_hash):
    return storage.primary.conn.execute(
        "SELECT store FROM memories WHERE content_hash = ?", (content_hash,)
    ).fetchone()[0]


@pytest.mark.asyncio
async def test_queued_store_lands_in_the_named_store_on_cloudflare(hybrid, d1):  # noqa: F811
    memory = _memory("Store B keeps the deploy calendar.")

    ok, msg = await hybrid.store(memory, skip_semantic_dedup=True, store="b")
    assert ok, msg
    await _drain(hybrid)

    assert _primary_store(hybrid, memory.content_hash) == "b"
    assert d1.row(memory.content_hash)["store"] == "b"


@pytest.mark.asyncio
async def test_queued_store_without_a_named_store_stays_default(hybrid, d1):  # noqa: F811
    memory = _memory("Nobody named a store for this one.")

    ok, msg = await hybrid.store(memory, skip_semantic_dedup=True)
    assert ok, msg
    await _drain(hybrid)

    assert d1.row(memory.content_hash)["store"] == "default"


@pytest.mark.asyncio
async def test_operation_queued_without_store_keeps_the_previous_call(hybrid, d1):  # noqa: F811
    # An operation built without ``store`` (any caller other than hybrid.store())
    # behaves as before: the secondary files it in its default store.
    memory = _memory("Queued by a caller that names no store.")
    ok, msg = await hybrid.primary.store(memory, skip_semantic_dedup=True, store="b")
    assert ok, msg

    await hybrid.sync_service._process_single_operation(
        SyncOperation(operation="store", memory=memory)
    )

    assert d1.row(memory.content_hash)["store"] == "default"


@pytest.mark.asyncio
async def test_scoped_delete_through_hybrid_reaches_the_cloudflare_copy(hybrid, d1):  # noqa: F811
    memory = _memory("Store B rotates the on-call roster weekly.")
    await hybrid.store(memory, skip_semantic_dedup=True, store="b")
    await _drain(hybrid)

    ok, msg = await hybrid.delete(memory.content_hash, store="b")
    assert ok, msg
    await _drain(hybrid)

    row = d1.row(memory.content_hash)
    assert row["store"] == "b"
    assert row["deleted_at"] is not None


class _OldSecondary:
    """A secondary (or test double) whose store() predates the ``store`` parameter."""

    def __init__(self):
        self.calls = []

    async def store(self, memory):
        self.calls.append(memory.content_hash)
        return True, "stored"


class _PositionalOnlySecondary(_OldSecondary):
    """``store`` exists but is positional-only, so it cannot be passed as ``store=``."""

    async def store(self, memory, store="default", /):
        self.calls.append(memory.content_hash)
        return True, "stored"


class _Primary:
    async def get_sync_state_for_hash(self, content_hash):
        return None, None


@pytest.mark.asyncio
@pytest.mark.parametrize("store", [None, "default", "b"])
@pytest.mark.parametrize("secondary_type", [_OldSecondary, _PositionalOnlySecondary])
async def test_secondary_without_store_parameter_keeps_the_old_call(store, secondary_type, caplog):
    memory = _memory(f"An old secondary keeps working ({store}, {secondary_type.__name__}).")
    secondary = secondary_type()
    service = BackgroundSyncService(_Primary(), secondary)

    await service._process_single_operation(
        SyncOperation(operation="store", memory=memory, store=store)
    )

    # No TypeError and no retry: the call is the one it got before #1915.
    assert secondary.calls == [memory.content_hash]


async def _legacy_row(storage, d1, content):  # noqa: F811
    """A memory in store "b" on the primary whose D1 row predates #1915 ("default")."""
    memory = _memory(content)
    ok, msg = await storage.primary.store(memory, skip_semantic_dedup=True, store="b")
    assert ok, msg
    ok, msg = await storage.secondary.store(memory)
    assert ok, msg
    assert d1.row(memory.content_hash)["store"] == "default"
    return memory.content_hash


@pytest.mark.asyncio
async def test_scoped_delete_reaches_a_legacy_default_row(hybrid, d1):  # noqa: F811
    # The queued delete is unscoped on purpose: scoping it to "b" would miss this row.
    h = await _legacy_row(hybrid, d1, "Store B archives incident reports monthly.")

    ok, msg = await hybrid.delete(h, store="b")
    assert ok, msg
    await _drain(hybrid)

    assert d1.row(h)["deleted_at"] is not None


@pytest.mark.asyncio
async def test_scoped_update_reaches_a_legacy_default_row(hybrid, d1):  # noqa: F811
    # The queued update is unscoped on purpose, for the same reason as the delete.
    h = await _legacy_row(hybrid, d1, "Store B tracks the vendor contracts.")

    ok, msg = await hybrid.update_memory_metadata(h, {"tags": ["renewed"]}, store="b")
    assert ok, msg
    await _drain(hybrid)

    tag_names = [r[0] for r in d1.db.execute(
        "SELECT t.name FROM tags t JOIN memory_tags mt ON t.id = mt.tag_id "
        "JOIN memories m ON m.id = mt.memory_id WHERE m.content_hash = ?", (h,)
    )]
    assert tag_names == ["renewed"]


@pytest.mark.asyncio
async def test_force_sync_leaves_a_queued_named_store_memory_to_the_queue(
    hybrid, d1, cloudflare_storage, monkeypatch  # noqa: F811
):
    # force_sync() pushes only the primary's default store, so a memory still queued
    # for store "b" is not uploaded into "default" ahead of its queued operation.
    monkeypatch.setattr(cloudflare_storage, "get_stats", AsyncMock(return_value={}))
    named = _memory("Store B lists the release owners.")
    plain = _memory("The default store lists the office hours.")
    await hybrid.store(named, skip_semantic_dedup=True, store="b")
    # Straight into the primary, so the queue holds only the named-store operation.
    await hybrid.primary.store(plain, skip_semantic_dedup=True)

    result = await hybrid.sync_service.force_sync()

    assert result["failed_operations"] == 0, result
    assert d1.row(named.content_hash) is None
    assert d1.row(plain.content_hash)["store"] == "default"

    await _drain(hybrid)
    assert d1.row(named.content_hash)["store"] == "b"
