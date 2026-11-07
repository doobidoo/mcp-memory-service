"""#1890: a restore the sync queue gave up on still reaches Cloudflare via force_sync.

The primary is real sqlite-vec; only the Cloudflare boundary is faked. A restore is told
apart from a local edit by the row's ``created_at``, so the tests run with the event log
on and off: it is off by default, and the outcome must not depend on it.
"""
import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest
import pytest_asyncio

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.services.memory_service import MemoryService
from mcp_memory_service.storage.hybrid import BackgroundSyncService, SyncOperation
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash


class _CloudflareBoundary:
    """Answers the D1 scan and the store calls; everything above it stays real."""

    base_url = "https://cloudflare.invalid"
    d1_database_id = "boundary"

    def __init__(self):
        self.live = {}
        self.tombstones = {}
        self.stored = []
        self.unreachable = False
        self.refuse_restore = False

    async def _retry_request(self, method, url, **kwargs):
        rows = [
            {"content_hash": h, "deleted_at": None} for h in self.live
        ] + [
            {"content_hash": h, "deleted_at": t} for h, t in self.tombstones.items()
        ]
        last_id, limit = kwargs["json"]["params"]
        numbered = [{"id": i, **row} for i, row in enumerate(rows, start=1)]
        response = Mock()
        response.json = Mock(return_value={
            "success": True,
            "result": [{"results": [r for r in numbered if r["id"] > last_id][:limit]}],
        })
        return response

    async def get_stats(self):
        return {"total_memories": len(self.live)}

    async def store(self, memory):
        if self.unreachable:
            raise ConnectionError("Connection timeout")
        self.stored.append(memory.content_hash)
        self.tombstones.pop(memory.content_hash, None)
        self.live[memory.content_hash] = memory
        return True, "stored"

    async def store_if_tombstone_matches(self, memory, expected_deleted_at):
        if self.refuse_restore:
            return False, "D1 write failed"
        if self.tombstones.get(memory.content_hash) != expected_deleted_at:
            return None, "tombstone_changed"
        return await self.store(memory)


def _memory(content, timestamp=None):
    return Memory(
        content=content,
        content_hash=generate_content_hash(content),
        tags=["__test__"],
        created_at=timestamp,
        updated_at=timestamp,
    )


@pytest_asyncio.fixture(params=["off", "on"], ids=["eventlog-off", "eventlog-on"])
async def primary(request, temp_db_path, monkeypatch):
    if request.param == "on":
        monkeypatch.setenv("MCP_SYNC_EVENTLOG", "on")
    else:
        monkeypatch.delenv("MCP_SYNC_EVENTLOG", raising=False)
    storage = SqliteVecMemoryStorage(f"{temp_db_path}/primary.db")
    await storage.initialize()
    yield storage
    await storage.close()


@pytest.fixture
def cloudflare():
    return _CloudflareBoundary()


@pytest.fixture
def sync(primary, cloudflare):
    return BackgroundSyncService(primary, cloudflare, sync_interval=1, batch_size=5)


async def _delete_everywhere(primary, cloudflare, memory):
    """Store, delete, and mirror the delete on Cloudflare; return the remote deleted_at."""
    assert (await primary.store(memory, skip_semantic_dedup=True))[0]
    assert (await primary.delete(memory.content_hash))[0]
    deleted_at = time.time()
    cloudflare.tombstones[memory.content_hash] = deleted_at
    await asyncio.sleep(0.01)  # the restore must land on a later wall clock
    return deleted_at


async def _drop_from_queue(sync, cloudflare, memory):
    """Run a queued restore into a dead secondary until the queue lets go of it."""
    operation = SyncOperation(operation="store", memory=memory, recreate=True)
    cloudflare.unreachable = True
    with patch("mcp_memory_service.storage.hybrid.asyncio.sleep", AsyncMock()):
        while operation.retries < operation.max_retries:
            try:
                await sync._process_single_operation(operation)
            except Exception as error:
                await sync._handle_sync_error(error, operation)
            sync.failed_operations.clear()  # the retry consumer takes it off the deque
    cloudflare.unreachable = False
    assert sync.sync_stats["operations_failed"] == 1


async def _restore(primary, memory):
    result = await MemoryService(storage=primary).store_memory(
        content=memory.content, tags=["__test__"], restore_deleted=True
    )
    assert result["success"], result
    restored = await primary.get_by_hash(memory.content_hash)
    assert restored is not None
    return restored


@pytest.mark.asyncio
async def test_restore_dropped_after_max_retries_reaches_cloudflare(
    primary, cloudflare, sync, unique_content
):
    memory = _memory(unique_content("restored after a delete"))
    await _delete_everywhere(primary, cloudflare, memory)
    restored = await _restore(primary, memory)
    await _drop_from_queue(sync, cloudflare, restored)

    result = await sync.force_sync()

    assert cloudflare.stored == [memory.content_hash]
    assert memory.content_hash not in cloudflare.tombstones
    assert await primary.get_by_hash(memory.content_hash) is not None
    assert result["synced_to_secondary"] == 1
    assert result["skipped_remote_tombstones"] == 0
    assert result["tombstone_conflicts"] == 0
    assert result["tombstone_conflict_hashes"] == []
    assert result["status"] == "completed"

    # The next run has nothing left to do for this hash.
    assert (await sync.force_sync())["synced_to_secondary"] == 0
    assert cloudflare.stored == [memory.content_hash]


@pytest.mark.asyncio
async def test_direct_store_that_revives_a_delete_counts_as_a_restore(
    primary, cloudflare, sync, unique_content
):
    """``recreate=None`` is what a queued plain store would have delivered too."""
    memory = _memory(unique_content("revived by a direct store"))
    await _delete_everywhere(primary, cloudflare, memory)
    assert (
        await primary.store(_memory(memory.content), skip_semantic_dedup=True)
    )[0]

    await sync.force_sync()

    assert cloudflare.stored == [memory.content_hash]


@pytest.mark.asyncio
async def test_a_delete_later_than_the_restore_wins(
    primary, cloudflare, sync, unique_content
):
    memory = _memory(unique_content("deleted again after the restore"))
    await _delete_everywhere(primary, cloudflare, memory)
    restored = await _restore(primary, memory)
    await _drop_from_queue(sync, cloudflare, restored)
    cloudflare.tombstones[memory.content_hash] = time.time() + 1

    result = await sync.force_sync()

    assert cloudflare.stored == []
    assert await primary.is_deleted(memory.content_hash)
    assert result["tombstones_applied"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("created_after_delete", [-100, 0], ids=["older", "equal"])
async def test_a_local_edit_stays_a_conflict(
    primary, cloudflare, sync, unique_content, created_after_delete
):
    """Only a row created strictly after the delete is pushed; an edit just conflicts."""
    deleted_at = time.time() - 50
    memory = _memory(
        unique_content("edited after the remote delete"),
        deleted_at + created_after_delete,
    )
    assert (await primary.store(memory, skip_semantic_dedup=True))[0]
    assert (await primary.update_memory_metadata(
        memory.content_hash, {"tags": ["edited"]}
    ))[0]
    cloudflare.tombstones[memory.content_hash] = deleted_at

    result = await sync.force_sync()

    assert cloudflare.stored == []
    assert memory.content_hash in cloudflare.tombstones
    assert result["tombstone_conflicts"] == 1
    assert result["tombstone_conflict_hashes"] == [memory.content_hash]
    assert result["skipped_remote_tombstones"] == 1
    assert result["status"] == "partial"


@pytest.mark.asyncio
async def test_a_tombstone_that_changed_after_the_scan_is_not_overwritten(
    primary, cloudflare, sync, unique_content
):
    memory = _memory(unique_content("tombstone moved during force sync"))
    await _delete_everywhere(primary, cloudflare, memory)
    restored = await _restore(primary, memory)
    await _drop_from_queue(sync, cloudflare, restored)

    real_compare_and_store = cloudflare.store_if_tombstone_matches

    async def tombstone_moves_first(candidate, expected_deleted_at):
        cloudflare.tombstones[candidate.content_hash] = expected_deleted_at + 1
        return await real_compare_and_store(candidate, expected_deleted_at)

    cloudflare.store_if_tombstone_matches = tombstone_moves_first

    result = await sync.force_sync()

    assert cloudflare.stored == []
    assert result["tombstone_conflicts"] == 1
    assert result["failed_operations"] == 0
    assert result["status"] == "partial"


@pytest.mark.asyncio
async def test_a_failed_push_stays_reported(primary, cloudflare, sync, unique_content):
    memory = _memory(unique_content("restore that cannot be pushed yet"))
    await _delete_everywhere(primary, cloudflare, memory)
    restored = await _restore(primary, memory)
    await _drop_from_queue(sync, cloudflare, restored)
    cloudflare.refuse_restore = True

    result = await sync.force_sync()

    assert result["failed_operations"] == 1
    assert result["tombstone_conflicts"] == 1
    assert result["status"] == "partial"

    cloudflare.refuse_restore = False
    assert (await sync.force_sync())["status"] == "completed"
    assert cloudflare.stored == [memory.content_hash]
