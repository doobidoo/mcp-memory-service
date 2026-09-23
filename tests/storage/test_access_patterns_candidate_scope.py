"""
get_access_patterns() must be bounded by the consolidation candidate window (#1289).

#1288 removed the old `LIMIT 100` so frequently-accessed memories are no longer dropped
from the access-pattern base. That fix is correct, but it left the query unbounded: every
consolidation run loaded all live, ever-accessed memories even though the decay and
forgetting consumers only look up hashes inside the current bounded batch.

These tests cover the candidate-scoped contract (option 1 in #1289):

1. With `content_hashes`, only those hashes come back — and soft-deleted rows stay hidden.
2. An empty sequence means "no candidates" and must not fall back to the full population.
3. `None` keeps the previous unbounded behaviour, so existing callers are unaffected.
4. A window wider than SQLite's host-parameter cap is chunked, not truncated or raised.
5. The consolidator passes its candidate window down, and still works against a backend
   whose `get_access_patterns` predates the parameter.
6. The window is passed in the form the backend's signature accepts: by name to a
   `**kwargs` forwarder or a keyword-only parameter, by position to a `*args` forwarder
   or a positional-only parameter.
"""

import hashlib
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage


@pytest_asyncio.fixture
async def storage():
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_access_scope.db"
        store = SqliteVecMemoryStorage(str(db_path))
        await store.initialize()
        try:
            yield store
        finally:
            await store.close()


def _make_memory(content: str) -> Memory:
    return Memory(
        content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        tags=[],
    )


async def _store_accessed(storage, contents):
    """Store each memory and stamp a distinct last_accessed on it."""
    memories = []
    base = time.time() - 10_000
    for i, c in enumerate(contents):
        m = _make_memory(c)
        await storage.store(m)
        memories.append(m)

    def _stamp():
        for i, m in enumerate(memories):
            storage.conn.execute(
                "UPDATE memories SET last_accessed = ? WHERE content_hash = ?",
                (base + i, m.content_hash),
            )
        storage.conn.commit()

    await storage._execute_with_retry(_stamp)
    return memories


@pytest.mark.asyncio
async def test_scoped_query_returns_only_the_candidate_window(storage):
    memories = await _store_accessed(storage, [f"scoped memory {i}" for i in range(10)])
    window = [m.content_hash for m in memories[:3]]

    patterns = await storage.get_access_patterns(window)

    assert set(patterns) == set(window), (
        "get_access_patterns(window) must return exactly the requested hashes, "
        f"got {len(patterns)} keys instead of {len(window)}"
    )
    for value in patterns.values():
        assert isinstance(value, datetime)
        assert value.tzinfo is not None


@pytest.mark.asyncio
async def test_scoped_query_still_hides_soft_deleted_rows(storage):
    memories = await _store_accessed(storage, [f"deleted scope {i}" for i in range(3)])
    window = [m.content_hash for m in memories]
    await storage.delete(memories[0].content_hash)

    patterns = await storage.get_access_patterns(window)

    assert memories[0].content_hash not in patterns, (
        "a soft-deleted memory must stay out of the access patterns even when it is "
        "named explicitly in the candidate window"
    )
    assert set(patterns) == {m.content_hash for m in memories[1:]}


@pytest.mark.asyncio
async def test_empty_window_returns_nothing_rather_than_everything(storage):
    await _store_accessed(storage, [f"empty scope {i}" for i in range(5)])

    assert await storage.get_access_patterns([]) == {}, (
        "an empty candidate window means 'no candidates'; falling back to the full "
        "population would reintroduce exactly the unbounded load this change removes"
    )


@pytest.mark.asyncio
async def test_none_keeps_the_unbounded_contract(storage):
    memories = await _store_accessed(storage, [f"unbounded {i}" for i in range(7)])

    patterns = await storage.get_access_patterns()

    assert set(patterns) == {m.content_hash for m in memories}


@pytest.mark.asyncio
async def test_window_larger_than_the_sqlite_parameter_cap_is_chunked(storage):
    """A batch wider than SQLITE_MAX_VARIABLE_NUMBER must not raise or truncate."""
    from mcp_memory_service.storage.mixins.retrieve import _IN_CLAUSE_CHUNK

    n = _IN_CLAUSE_CHUNK + 25
    memories = await _store_accessed(storage, [f"chunked {i}" for i in range(n)])
    window = [m.content_hash for m in memories]

    patterns = await storage.get_access_patterns(window)

    assert len(patterns) == n, (
        f"a {n}-hash window spans more than one chunk of {_IN_CLAUSE_CHUNK}; "
        f"got {len(patterns)} rows back"
    )


@pytest.mark.asyncio
async def test_consolidator_passes_the_candidate_window_down():
    from mcp_memory_service.consolidation.consolidator import DreamInspiredConsolidator

    backend = AsyncMock()

    async def _patterns(content_hashes=None):
        return {h: datetime.now(tz=timezone.utc) for h in (content_hashes or [])}

    backend.get_access_patterns = AsyncMock(side_effect=_patterns)
    consolidator = DreamInspiredConsolidator.__new__(DreamInspiredConsolidator)
    consolidator.storage = backend

    window = ["aaa", "bbb"]
    got = await consolidator._get_access_patterns(window)

    backend.get_access_patterns.assert_awaited_once_with(content_hashes=window)
    assert set(got) == set(window)


@pytest.mark.asyncio
async def test_consolidator_falls_back_for_backends_without_the_parameter():
    """Third-party backends that predate #1289 must keep working."""
    import logging

    from mcp_memory_service.consolidation.consolidator import DreamInspiredConsolidator

    class LegacyBackend:
        def __init__(self):
            self.calls = []

        async def get_access_patterns(self):
            self.calls.append("no-args")
            return {"legacy": datetime.now(tz=timezone.utc)}

    backend = LegacyBackend()
    consolidator = DreamInspiredConsolidator.__new__(DreamInspiredConsolidator)
    consolidator.storage = backend
    consolidator.logger = logging.getLogger(__name__)

    got = await consolidator._get_access_patterns(["aaa", "bbb"])

    assert backend.calls == ["no-args"]
    assert set(got) == {"legacy"}


async def _scoped_patterns(content_hashes=None):
    return {h: datetime.now(tz=timezone.utc) for h in (content_hashes or [])}


class _KwargsForwarder:
    """A proxy that forwards by name only, e.g. an instrumentation wrapper."""

    async def get_access_patterns(self, **kwargs):
        return await _scoped_patterns(**kwargs)


class _KeywordOnly:
    async def get_access_patterns(self, *, content_hashes=None):
        return await _scoped_patterns(content_hashes)


class _ArgsForwarder:
    async def get_access_patterns(self, *args):
        return await _scoped_patterns(*args)


class _PositionalOnly:
    async def get_access_patterns(self, content_hashes=None, /):
        return await _scoped_patterns(content_hashes)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "backend_cls",
    [_KwargsForwarder, _KeywordOnly, _ArgsForwarder, _PositionalOnly],
    ids=["kwargs-forwarder", "keyword-only", "args-forwarder", "positional-only"],
)
async def test_consolidator_passes_the_window_in_the_form_the_signature_accepts(backend_cls):
    """Accepting the window and passing it must agree.

    A `**kwargs` forwarder or a keyword-only `content_hashes` takes the window only by
    name; a `*args` forwarder or a positional-only parameter only by position. Passing it
    in the other form raises `TypeError` and fails the whole consolidation run.
    """
    import logging

    from mcp_memory_service.consolidation.consolidator import DreamInspiredConsolidator

    consolidator = DreamInspiredConsolidator.__new__(DreamInspiredConsolidator)
    consolidator.storage = backend_cls()
    consolidator.logger = logging.getLogger(__name__)

    window = ["aaa", "bbb"]
    got = await consolidator._get_access_patterns(window)

    assert set(got) == set(window)
