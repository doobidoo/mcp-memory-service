"""Milvus semantic supersede must not overwrite a row that is no longer current.

The metadata upsert has no compare-and-swap. Two stores can both match the
same memory and both insert. The second link has to notice, leave
superseded_by alone, and drop its new row. A soft-deleted row is not a match
in either dedup mode. No live Milvus server: the storage object is built with
__new__ and the client calls are fakes.
"""

import asyncio
import json
import math
import time

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
    storage.get_by_hash = _async(lambda _hash, store=None: _memory("old", {"superseded_by": "winner"}))
    storage.update_memory_metadata = _async(lambda *_args, **_kwargs: (_boom()))

    linked = await storage._link_semantic_supersede("old", "second")

    assert linked is False


@pytest.mark.asyncio
async def test_link_refuses_when_the_match_is_gone():
    storage = _storage()
    storage.get_by_hash = _async(lambda _hash, store=None: None)
    storage.update_memory_metadata = _async(lambda *_args, **_kwargs: (_boom()))

    assert await storage._link_semantic_supersede("old", "second") is False


@pytest.mark.asyncio
async def test_second_link_rolls_the_new_row_back():
    storage = _storage()
    removed = []
    storage.get_by_hash = _async(lambda _hash, store=None: _memory("old", {"superseded_by": "winner"}))
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
    storage.get_by_hash = _async(lambda _hash, store=None: _memory("old", {}))

    async def _update(content_hash, updates, preserve_timestamps=True, store=None):
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

    async def _get(content_hash: str, store: str | None = None):
        winner = committed.get(content_hash)
        metadata = {"superseded_by": winner} if winner else {}
        return _memory(content_hash, metadata)

    async def _update(content_hash, updates, preserve_timestamps=True, store=None):
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


def test_sibling_chunk_is_not_a_supersede_target():
    storage = _storage()
    sibling = {
        "id": "chunk-1",
        "vector": [1.0, 0.0, 0.0],
        "content": "first chunk",
        "metadata": json.dumps({"original_hash": "whole-document"}),
    }
    other = {
        "id": "other-fact",
        "vector": [0.9, math.sqrt(1.0 - 0.81), 0.0],
        "content": "a different fact",
        "metadata": "{}",
    }

    only_sibling = storage._choose_semantic_hit(
        [sibling], [1.0, 0.0, 0.0], 1.0, 0.85, True,
        exclude_original_hash="whole-document",
    )
    assert only_sibling[0] is False

    hit = storage._choose_semantic_hit(
        [sibling, other], [1.0, 0.0, 0.0], 1.0, 0.85, True,
        exclude_original_hash="whole-document",
    )
    assert hit[0] is True
    assert hit[1] == "other-fact"


def _vector_at(cosine: float) -> list[float]:
    sine = math.sqrt(max(0.0, 1.0 - cosine * cosine))
    return [cosine, sine, 0.0]


def _deleted_row(cosine: float) -> dict:
    return {
        "id": "deleted-old",
        "vector": _vector_at(cosine),
        "content": "deleted fact",
        "metadata": json.dumps({"deleted_at": time.time()}),
        "created_at": time.time(),
    }


def _dedup_storage(mode: str) -> MilvusMemoryStorage:
    storage = _storage()
    storage._initialized = True
    storage.client = object()
    storage.collection_name = "unit"
    storage.semantic_dedup_enabled = True
    storage.semantic_dedup_mode = mode
    storage.semantic_dedup_threshold = 0.85
    storage.semantic_dedup_near_identical_threshold = 0.97
    storage.semantic_dedup_time_window = 24
    storage._generate_embedding = lambda _content: [1.0, 0.0, 0.0]
    storage._has_content_lower = False
    storage._memory_to_entity = lambda memory, embedding, *_args, **_kwargs: {"id": memory.content_hash}

    async def _no_conflicts(*_args, **_kwargs):
        return []

    storage._detect_conflicts = _no_conflicts

    async def _scan(filter_expr, fields):
        # Supersede mode reads the whole window; route it through the test's query mock.
        return await storage._call_client("query", filter=filter_expr, output_fields=fields)

    storage._scan_dedup_rows = _scan
    return storage


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,cosine", [
    ("reject", 0.99),
    ("reject", 0.90),
    ("supersede", 0.99),
    ("supersede", 0.90),
])
async def test_deleted_memory_does_not_block_a_new_store(mode, cosine):
    storage = _dedup_storage(mode)
    calls = {"insert": 0, "cleanup": 0}

    async def _call(method_name, *_args, **_kwargs):
        if method_name == "query":
            return [_deleted_row(cosine)]
        if method_name == "insert":
            calls["insert"] += 1
        return []

    async def _cleanup(content_hash: str) -> bool:
        calls["cleanup"] += 1
        return True

    storage._call_client = _call
    storage._cleanup_unlinked_version = _cleanup
    memory = Memory(content=f"new fact {mode} {cosine}", content_hash="new-hash")

    ok, message = await storage.store(memory)

    assert ok is True, message
    assert calls["insert"] == 1
    assert calls["cleanup"] == 0
    assert not isinstance(message, SemanticDedupResult)


def test_other_store_row_is_not_a_supersede_target():
    """A close memory in another store must not be chosen as the replacement."""
    storage = _storage()
    storage._has_store = True
    other = {
        "id": "docs-fact",
        "vector": [1.0, 0.0, 0.0],
        "content": "a fact in docs",
        "store": "docs",
        "metadata": "{}",
    }
    same = {
        "id": "default-fact",
        "vector": _vector_at(0.90),
        "content": "a fact in default",
        "store": "default",
        "metadata": "{}",
    }

    only_other = storage._choose_semantic_hit(
        [other], [1.0, 0.0, 0.0], 1.0, 0.85, True, store="default",
    )
    assert only_other[0] is False

    hit = storage._choose_semantic_hit(
        [other, same], [1.0, 0.0, 0.0], 1.0, 0.85, True, store="default",
    )
    assert hit[0] is True
    assert hit[1] == "default-fact"


def test_dedup_query_filter_confines_to_the_destination_store():
    storage = _storage()
    storage._has_store = True
    assert storage._semantic_dedup_filter(100.0, "default") == (
        'created_at >= 100.0 and store == "default"'
    )
    assert storage._semantic_dedup_filter(100.0, "docs") == (
        'created_at >= 100.0 and store == "docs"'
    )


@pytest.mark.asyncio
async def test_link_refuses_when_the_match_is_in_another_store():
    storage = _storage()

    async def _get(content_hash: str, store: str | None = None):
        if store == "default":
            return None
        return _memory(content_hash, {})

    storage.get_by_hash = _get
    storage.update_memory_metadata = _async(lambda *_args, **_kwargs: (_boom()))

    assert await storage._link_semantic_supersede("old", "new", store="default") is False


@pytest.mark.asyncio
async def test_store_does_not_supersede_a_memory_in_another_store():
    """Storing into default must not retire a close fact that lives in docs."""
    storage = _dedup_storage("supersede")
    storage._has_store = True
    queried = {}

    async def _call(method_name, *_args, **kwargs):
        if method_name == "query":
            queried["filter"] = kwargs.get("filter")
            return [{
                "id": "docs-old",
                "vector": _vector_at(0.90),
                "content": "a fact in docs",
                "store": "docs",
                "metadata": "{}",
                "created_at": time.time(),
            }]
        if method_name in {"insert", "get"}:
            return []
        return []

    storage._call_client = _call
    storage.update_memory_metadata = _async(lambda *_args, **_kwargs: (_boom()))
    storage.get_by_hash = _async(lambda _hash, store=None: None)
    memory = Memory(content="updated fact in default", content_hash="new-hash")

    ok, message = await storage.store(memory, store="default")

    assert ok is True, message
    assert 'store == "default"' in (queried.get("filter") or "")
    assert not isinstance(message, SemanticDedupResult)


def _live_row(row_id: str, cosine: float, **metadata) -> dict:
    return {
        "id": row_id,
        "vector": _vector_at(cosine),
        "content": f"fact {row_id}",
        "metadata": json.dumps(metadata),
        "created_at": time.time(),
    }


@pytest.mark.asyncio
async def test_supersede_reads_the_whole_window_not_a_50_row_sample():
    """The closest live match past the first 50 rows still wins."""
    storage = _dedup_storage("supersede")
    filler = [_live_row(f"dead-{i}", 0.99, deleted_at=1.0) for i in range(40)]
    filler += [_live_row(f"old-{i}", 0.99, superseded_by="x") for i in range(20)]
    weak = _live_row("weak", 0.86)
    best = _live_row("best", 0.93)
    rows = filler + [weak] + [_live_row(f"far-{i}", 0.10) for i in range(30)] + [best]
    seen = {}

    async def _scan(filter_expr, fields):
        seen["scanned"] = True
        return rows

    async def _no_sample(*_args, **_kwargs):
        raise AssertionError("supersede mode must not use the 50-row sample")

    storage._scan_dedup_rows = _scan
    storage._call_client = _no_sample

    match = await storage._check_semantic_duplicate(
        "new fact", time_window_hours=24, similarity_threshold=0.85, current_only=True,
    )

    assert seen.get("scanned") is True
    assert match[0] is True
    assert match[1] == "best"


@pytest.mark.asyncio
async def test_supersede_sees_a_near_identical_match_beyond_the_sample():
    storage = _dedup_storage("supersede")
    rows = [_live_row(f"far-{i}", 0.10) for i in range(80)] + [_live_row("twin", 0.99)]

    async def _scan(filter_expr, fields):
        return rows

    storage._scan_dedup_rows = _scan

    match = await storage._check_semantic_duplicate(
        "new fact", time_window_hours=24, similarity_threshold=0.85, current_only=True,
    )

    assert match[1] == "twin"
    assert match.similarity >= 0.97


@pytest.mark.asyncio
async def test_reject_mode_keeps_the_bounded_sample():
    storage = _dedup_storage("reject")
    calls = {}

    async def _call(method_name, *_args, **kwargs):
        calls[method_name] = kwargs
        return []

    async def _no_scan(*_args, **_kwargs):
        raise AssertionError("reject mode should not scan the whole window")

    storage._call_client = _call
    storage._scan_dedup_rows = _no_scan

    await storage._check_semantic_duplicate(
        "new fact", time_window_hours=24, similarity_threshold=0.85, current_only=False,
    )

    assert calls["query"]["limit"] == 50


def test_dedup_drain_reads_every_batch_once():
    storage = _storage()
    storage.collection_name = "unit"

    class _Iter:
        def __init__(self):
            self.batches = [[{"id": "a"}, {"id": "b"}], [{"id": "c"}], [{"id": "c"}], []]
            self.closed = False

        def next(self):
            return self.batches.pop(0)

        def close(self):
            self.closed = True

    it = _Iter()

    class _Client:
        def query_iterator(self, **kwargs):
            assert kwargs["filter"] == "created_at >= 1"
            assert "vector" in kwargs["output_fields"]
            return it

    storage.client = _Client()
    rows = storage._drain_dedup_rows("created_at >= 1", ["id", "vector"])

    assert [r["id"] for r in rows] == ["a", "b", "c"]
    assert it.closed is True


@pytest.mark.asyncio
async def test_link_to_the_same_replacement_counts_as_linked():
    """A second save of the same text finds its own hash already linked."""
    storage = _storage()
    storage.get_by_hash = _async(lambda _hash, store=None: _memory("old", {"superseded_by": "new"}))
    storage.update_memory_metadata = _async(lambda *_args, **_kwargs: (_boom()))

    assert await storage._link_semantic_supersede("old", "new") is True


@pytest.mark.asyncio
async def test_commit_keeps_the_replacement_the_old_row_points_to():
    storage = _storage()
    removed = []
    storage.get_by_hash = _async(lambda _hash, store=None: _memory("old", {"superseded_by": "new"}))
    storage.update_memory_metadata = _async(lambda *_args, **_kwargs: (_boom()))

    async def _cleanup(content_hash: str) -> bool:
        removed.append(content_hash)
        return True

    storage._cleanup_unlinked_version = _cleanup

    assert await storage._commit_semantic_supersede("old", "new") is None
    assert removed == []


@pytest.mark.asyncio
async def test_commit_still_rolls_back_when_another_hash_won():
    storage = _storage()
    removed = []
    storage.get_by_hash = _async(lambda _hash, store=None: _memory("old", {"superseded_by": "other"}))
    storage.update_memory_metadata = _async(lambda *_args, **_kwargs: (_boom()))

    async def _cleanup(content_hash: str) -> bool:
        removed.append(content_hash)
        return True

    storage._cleanup_unlinked_version = _cleanup

    success, message = await storage._commit_semantic_supersede("old", "new")

    assert success is False
    assert isinstance(message, SemanticDedupResult)
    assert message.outcome == "deduplicated"
    assert removed == ["new"]


@pytest.mark.asyncio
async def test_commit_rechecks_the_link_before_deleting():
    """The write failed here, but another process linked the same hash."""
    storage = _storage()
    removed = []
    reads = []

    async def _get(content_hash: str, store: str | None = None):
        reads.append(content_hash)
        metadata = {} if len(reads) == 1 else {"superseded_by": "new"}
        return _memory(content_hash, metadata)

    async def _update(*_args, **_kwargs):
        return False, "write lost"

    async def _cleanup(content_hash: str) -> bool:
        removed.append(content_hash)
        return True

    storage.get_by_hash = _get
    storage.update_memory_metadata = _update
    storage._cleanup_unlinked_version = _cleanup

    assert await storage._commit_semantic_supersede("old", "new") is None
    assert removed == []
    assert reads == ["old", "old"]


@pytest.mark.asyncio
async def test_commit_keeps_the_row_when_the_recheck_fails():
    storage = _storage()
    removed = []
    reads = []

    async def _get(content_hash: str, store: str | None = None):
        reads.append(content_hash)
        if len(reads) > 1:
            raise RuntimeError("milvus unavailable")
        return _memory(content_hash, {})

    async def _update(*_args, **_kwargs):
        return False, "write lost"

    async def _cleanup(content_hash: str) -> bool:
        removed.append(content_hash)
        return True

    storage.get_by_hash = _get
    storage.update_memory_metadata = _update
    storage._cleanup_unlinked_version = _cleanup

    success, message = await storage._commit_semantic_supersede("old", "new")

    assert success is False
    assert "could not be rolled back" in message
    assert removed == []


@pytest.mark.asyncio
async def test_concurrent_saves_of_the_same_update_keep_the_replacement():
    """Both saves match the same old row; one row is stored and stays linked."""
    storage = _dedup_storage("supersede")
    rows = {"old": {}}
    inserted = []
    removed = []
    both_matched = asyncio.Event()
    scans = []

    async def _scan(filter_expr, fields):
        scans.append(filter_expr)
        if len(scans) == 2:
            both_matched.set()
        await both_matched.wait()
        return [_live_row("old", 0.90)]

    async def _call(method_name, *_args, **kwargs):
        if method_name == "insert":
            for entity in kwargs["data"]:
                inserted.append(entity["id"])
                rows[entity["id"]] = {}
        return []

    async def _get(content_hash: str, store: str | None = None):
        if content_hash not in rows:
            return None
        return _memory(content_hash, rows[content_hash])

    async def _update(content_hash, updates, preserve_timestamps=True, store=None):
        rows[content_hash] = dict(rows[content_hash], **updates["metadata"])
        return True, "ok"

    async def _cleanup(content_hash: str) -> bool:
        removed.append(content_hash)
        rows.pop(content_hash, None)
        return True

    storage._scan_dedup_rows = _scan
    storage._call_client = _call
    storage.get_by_hash = _get
    storage.update_memory_metadata = _update
    storage._cleanup_unlinked_version = _cleanup

    first, second = await asyncio.gather(
        storage.store(Memory(content="updated fact", content_hash="new")),
        storage.store(Memory(content="updated fact", content_hash="new")),
    )

    outcomes = sorted([first[0], second[0]])
    assert outcomes == [False, True]
    loser = first if first[0] is False else second
    assert loser[1] == "Duplicate content detected (exact match)"
    assert inserted == ["new"]
    assert removed == []
    assert rows["old"]["superseded_by"] == "new"
    assert "new" in rows
