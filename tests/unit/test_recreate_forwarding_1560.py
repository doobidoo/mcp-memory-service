"""#1560 follow-up: the ``recreate`` flag is forwarded only to backends that declare
``supports_recreate_refusal``; everyone else keeps receiving the pre-#1560 ``store()``
call. Also covers: an explicit restore travelling through the hybrid sync queue, refused
chunks surfacing in a partial write, the refusal surfacing after the contradiction
retry, and harvest sessions made only of refusals not being reselected forever.
"""
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_memory_service.consolidation import scheduler as sch
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.services.memory_service import MemoryService
from mcp_memory_service.storage.base import MemoryStorage, RecreateRefused, recreate_kwargs
from mcp_memory_service.storage.hybrid import BackgroundSyncService, SyncOperation
from mcp_memory_service.utils.hashing import generate_content_hash


def _refusal() -> RecreateRefused:
    return RecreateRefused("evt-1", [1_700_000_000_000, 0, "agent-a"])


class _OldSignatureStorage:
    """A backend (or test double) written before #1560: ``store()`` has no ``recreate``."""

    max_content_length = None
    supports_chunking = False

    def __init__(self):
        self.calls = []

    async def store(self, memory, skip_semantic_dedup=False, store="default"):
        self.calls.append({"skip_semantic_dedup": skip_semantic_dedup, "store": store})
        return True, "stored"

    async def get_by_hash(self, h):
        return None


class _RefusingStorage:
    """A backend that keeps a delete record and declares it."""

    max_content_length = None
    supports_chunking = False
    supports_recreate_refusal = True

    def __init__(self, deleted=True):
        self.deleted = deleted
        self.calls = []

    async def store(self, memory, skip_semantic_dedup=False, store="default", recreate=None):
        self.calls.append({"skip_semantic_dedup": skip_semantic_dedup, "recreate": recreate})
        if recreate is False and self.deleted:
            return False, _refusal()
        return True, "stored"

    async def get_by_hash(self, h):
        return None


# --- recreate_kwargs --------------------------------------------------------------

def test_recreate_kwargs_forwards_only_to_backends_that_declare_support():
    assert recreate_kwargs(_OldSignatureStorage(), False) == {}
    assert recreate_kwargs(_OldSignatureStorage(), True) == {}
    assert recreate_kwargs(_RefusingStorage(), None) == {}
    assert recreate_kwargs(_RefusingStorage(), False) == {"recreate": False}
    assert recreate_kwargs(_RefusingStorage(), True) == {"recreate": True}


def test_recreate_kwargs_treats_a_mocked_flag_as_unsupported():
    # ``AsyncMock(spec=MemoryStorage)`` answers a truthy Mock for every property; only an
    # explicit ``True`` opts a backend in.
    assert recreate_kwargs(AsyncMock(spec=MemoryStorage), False) == {}


def test_default_store_batch_does_not_hand_recreate_to_an_old_backend():
    storage = _OldSignatureStorage()
    memories = [Memory(content=f"m{i}", content_hash=generate_content_hash(f"m{i}")) for i in range(3)]
    results = asyncio.run(MemoryStorage.store_batch(storage, memories, recreate=False))
    assert results == [(True, "stored")] * 3
    assert len(storage.calls) == 3


# --- MemoryService.store_memory ---------------------------------------------------

@pytest.mark.asyncio
async def test_store_memory_keeps_the_pre_1560_call_for_an_old_backend():
    """filhocf's reproduction on #1637: a third-party backend / test double with the
    previous ``store()`` signature must not break at call time."""
    storage = _OldSignatureStorage()
    service = MemoryService(storage=storage)
    for restore_deleted in (False, True):
        result = await service.store_memory(content=f"old backend {restore_deleted}",
                                            restore_deleted=restore_deleted)
        assert result["success"] is True, result
    assert storage.calls == [
        {"skip_semantic_dedup": False, "store": "default"},
        {"skip_semantic_dedup": False, "store": "default"},
    ]


@pytest.mark.asyncio
async def test_store_memory_forwards_false_by_default_and_true_on_restore_to_a_refusing_backend():
    storage = _RefusingStorage()
    service = MemoryService(storage=storage)
    refused = await service.store_memory(content="deleted before")
    assert refused["success"] is False
    assert refused["refused_recreate"]["event_id"] == "evt-1"
    restored = await service.store_memory(content="deleted before", restore_deleted=True)
    assert restored["success"] is True
    assert [c["recreate"] for c in storage.calls] == [False, True]


@pytest.mark.asyncio
async def test_refusal_after_the_contradiction_retry_is_still_a_refusal(monkeypatch):
    """Semantic dedup runs before the delete-record check: the first store() answers
    'duplicate', the retry with skip_semantic_dedup=True answers the refusal. The service
    must not fall through to a generic failure (REST 200 instead of 409)."""

    class _DedupThenRefuse(_RefusingStorage):
        async def store(self, memory, skip_semantic_dedup=False, store="default", recreate=None):
            self.calls.append({"skip_semantic_dedup": skip_semantic_dedup, "recreate": recreate})
            if not skip_semantic_dedup:
                return False, "Duplicate content detected (semantically similar to abc123)"
            if recreate is False:
                return False, _refusal()
            return True, "stored"

    storage = _DedupThenRefuse()
    service = MemoryService(storage=storage)
    monkeypatch.setattr(service, "_contradiction_behind_duplicate", AsyncMock(return_value="abc123"))

    result = await service.store_memory(content="X is B")

    assert result["success"] is False
    assert result["refused_recreate"]["event_id"] == "evt-1", result
    assert [c["skip_semantic_dedup"] for c in storage.calls] == [False, True]


@pytest.mark.asyncio
async def test_partial_chunked_write_names_the_refused_chunks():
    storage = AsyncMock(spec=MemoryStorage)
    storage.max_content_length = 100
    storage.supports_chunking = True
    storage.supports_recreate_refusal = True
    storage.store.side_effect = [(True, "stored"), (False, _refusal())] * 5
    service = MemoryService(storage=storage)

    with patch("mcp_memory_service.services.memory_service.ENABLE_AUTO_SPLIT", True):
        result = await service.store_memory(content="C" * 250)

    assert result["success"] is True
    assert result["failed_chunks"] >= 1
    assert result["refused_chunks"], "a skipped deleted chunk must be reported, not folded into success"
    assert result["refused_chunks"][0]["index"] == 1
    assert result["refused_chunks"][0]["event_id"] == "evt-1"
    assert result["refused_chunks"][0]["deleted_at"].endswith("Z")


# --- hybrid: an explicit restore reaches the secondary -----------------------------

class _PrimaryForQueue:
    supports_recreate_refusal = True

    def __init__(self, memory):
        self._memory = memory

    async def get_sync_state_for_hash(self, content_hash):
        return self._memory.updated_at, None


@pytest.mark.asyncio
async def test_queued_store_carries_an_explicit_restore_to_a_refusing_secondary():
    now = time.time()
    memory = Memory(content="restored on primary", content_hash=generate_content_hash("restored on primary"),
                    created_at=now, updated_at=now)
    secondary = _RefusingStorage(deleted=True)
    service = BackgroundSyncService(_PrimaryForQueue(memory), secondary)

    await service._process_single_operation(SyncOperation(operation="store", memory=memory, recreate=True))
    assert secondary.calls[-1]["recreate"] is True

    await service._process_single_operation(SyncOperation(operation="store", memory=memory))
    assert secondary.calls[-1]["recreate"] is None


@pytest.mark.asyncio
async def test_queued_store_never_hands_recreate_to_an_old_secondary():
    now = time.time()
    memory = Memory(content="old secondary", content_hash=generate_content_hash("old secondary"),
                    created_at=now, updated_at=now)
    secondary = _OldSignatureStorage()
    service = BackgroundSyncService(_PrimaryForQueue(memory), secondary)

    await service._process_single_operation(SyncOperation(operation="store", memory=memory, recreate=True))
    assert secondary.calls == [{"skip_semantic_dedup": False, "store": "default"}]


# --- harvest scheduler ------------------------------------------------------------

def _result(session_id, stored, skipped=0, found=None):
    r = MagicMock()
    r.session_id = session_id
    r.stored = stored
    r.skipped = skipped
    r.found = found if found is not None else stored + skipped
    return r


def test_session_made_only_of_refusals_is_tracked_as_handled():
    assert sch.sessions_to_track([_result("sess-A", 0, skipped=3, found=3)]) == {"sess-A"}


def test_session_with_a_candidate_neither_stored_nor_skipped_stays_pending():
    # 3 found, 2 refused, 1 dropped (LLM chain / exception): still retryable.
    assert sch.sessions_to_track([_result("sess-A", 0, skipped=2, found=3)]) == set()


def test_skipped_zero_keeps_the_previous_rule():
    assert sch.sessions_to_track([_result("sess-A", 0, skipped=0, found=0)]) == set()
    assert sch.sessions_to_track([_result("sess-B", 1, skipped=0, found=4)]) == {"sess-B"}
