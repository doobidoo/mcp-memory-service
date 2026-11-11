"""Milvus semantic supersede must not overwrite a row that is no longer current.

The metadata upsert has no compare-and-swap. Two stores can both match the
same memory and both insert. The second link has to notice, leave
superseded_by alone, and drop its new row. No live Milvus server: the
storage object is built with __new__ and the client calls are fakes.
"""

import asyncio

import pytest

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.milvus import MilvusMemoryStorage
from mcp_memory_service.storage.semantic_dedup import SemanticDedupResult


def _storage() -> MilvusMemoryStorage:
    storage = MilvusMemoryStorage.__new__(MilvusMemoryStorage)
    storage._versioned_update_lock = None
    return storage


def _memory(content_hash: str, metadata: dict | None) -> Memory:
    return Memory(content=content_hash, content_hash=content_hash, metadata=metadata or {})


@pytest.mark.asyncio
async def test_link_refuses_when_the_match_is_already_superseded():
    storage = _storage()
    storage.get_by_hash = _async(lambda _hash: _memory("old", {"superseded_by": "winner"}))
    storage.update_memory_metadata = _async(lambda *_args, **_kwargs: (_boom()))

    linked = await storage._link_semantic_supersede("old", "second")

    assert linked is False


@pytest.mark.asyncio
async def test_link_refuses_when_the_match_is_gone():
    storage = _storage()
    storage.get_by_hash = _async(lambda _hash: None)
    storage.update_memory_metadata = _async(lambda *_args, **_kwargs: (_boom()))

    assert await storage._link_semantic_supersede("old", "second") is False


@pytest.mark.asyncio
async def test_second_link_rolls_the_new_row_back():
    storage = _storage()
    removed = []
    storage.get_by_hash = _async(lambda _hash: _memory("old", {"superseded_by": "winner"}))
    storage.update_memory_metadata = _async(lambda *_args, **_kwargs: (_boom()))

    async def _cleanup(content_hash: str) -> bool:
        removed.append(content_hash)
        return True

    storage._cleanup_unlinked_version = _cleanup

    failed = await storage._commit_semantic_supersede("old", "second")

    assert failed is not None
    success, message = failed
    assert success is False
    assert isinstance(message, SemanticDedupResult)
    assert message.outcome == "deduplicated"
    assert message.matched_hash == "old"
    assert removed == ["second"]


@pytest.mark.asyncio
async def test_current_row_is_linked_once():
    storage = _storage()
    written = []
    storage.get_by_hash = _async(lambda _hash: _memory("old", {}))

    async def _update(content_hash, updates, preserve_timestamps=True):
        written.append((content_hash, updates, preserve_timestamps))
        return True, "Updated fields: custom_metadata"

    storage.update_memory_metadata = _update

    assert await storage._link_semantic_supersede("old", "new") is True
    assert written == [(
        "old",
        {"metadata": {"superseded_by": "new", "evolution_reason": "semantic-dedup-supersede"}},
        True,
    )]


@pytest.mark.asyncio
async def test_overlapping_links_keep_the_first_winner():
    """The second linker waits out the first write, then sees the row taken."""
    storage = _storage()
    committed = {}
    entered = asyncio.Event()
    release = asyncio.Event()

    async def _get(content_hash: str):
        winner = committed.get(content_hash)
        metadata = {"superseded_by": winner} if winner else {}
        return _memory(content_hash, metadata)

    async def _update(content_hash, updates, preserve_timestamps=True):
        entered.set()
        await release.wait()
        committed[content_hash] = updates["metadata"]["superseded_by"]
        return True, "ok"

    storage.get_by_hash = _get
    storage.update_memory_metadata = _update

    first = asyncio.create_task(storage._link_semantic_supersede("old", "new-a"))
    await entered.wait()
    second = asyncio.create_task(storage._link_semantic_supersede("old", "new-b"))
    await asyncio.sleep(0)
    assert committed == {}
    release.set()

    assert await first is True
    assert await second is False
    assert committed == {"old": "new-a"}


def _async(fn):
    async def _call(*args, **kwargs):
        return fn(*args, **kwargs)
    return _call


def _boom():
    raise AssertionError("update_memory_metadata must not run")
