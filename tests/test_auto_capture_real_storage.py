"""Regression tests for #1758, against real sqlite-vec storage.

``AutoCaptureService.capture()`` unpacks ``(evolved, child_hash)`` from
``SessionHarvester._try_evolve()``. That method used to return a bare bool,
so with ``dry_run=False`` every candidate raised ``TypeError`` inside the
per-candidate ``try`` and nothing was stored. ``tests/test_auto_capture.py``
patches ``_try_evolve`` with a two-tuple, which is why the suite stayed green.

These tests leave ``_try_evolve`` unpatched and run it on a real
``SqliteVecMemoryStorage``, so the return shape that ``capture()`` sees is
the one the method really produces.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from mcp_memory_service.harvest.auto_capture import AutoCaptureService
from mcp_memory_service.harvest.models import HarvestConfig
from mcp_memory_service.services.memory_service import MemoryService
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage

TEXT = "I decided to use WAL mode for concurrent SQLite because readers were blocking writers."
# Same decision, reworded: close enough to the first to be evolved, not stored again.
TEXT_REWORDED = "I decided to use WAL mode for concurrent SQLite because the readers kept blocking writers."

CONFIG = HarvestConfig(similarity_threshold=0.85, min_confidence_to_evolve=0.3)


@pytest.fixture
async def memory_service(temp_db_path):
    storage = SqliteVecMemoryStorage(f"{temp_db_path}/auto_capture.db")
    await storage.initialize()
    try:
        yield MemoryService(storage)
    finally:
        await storage.close()


async def _require_similarity(memory_service):
    """Skip when the embedding backend cannot see TEXT and TEXT_REWORDED as close.

    With ONNX embeddings (the ``sqlite`` extra, as in CI) they score ~0.998.
    The hash fallback used when no embedding backend is installed does not see
    reworded text as similar, so the evolve path cannot be reached there.
    """
    storage = memory_service.storage
    probe = await memory_service.store_memory(content=TEXT + " [probe]", tags=["probe"])
    hits = await storage.retrieve(TEXT_REWORDED, n_results=1, min_confidence=CONFIG.min_confidence_to_evolve)
    await storage.delete(probe["memory"]["content_hash"])
    if not hits or hits[0].relevance_score <= CONFIG.similarity_threshold:
        pytest.skip("embedding backend does not rate the reworded text as similar (hash fallback?)")


async def _capture(service, text, link):
    with patch(
        "mcp_memory_service.harvest.auto_capture._get_graph_instance_async",
        AsyncMock(return_value=None),
    ), patch(
        "mcp_memory_service.harvest.auto_capture._link_derived_from",
        link,
    ):
        return await service.capture(
            text,
            parent_hash="parent0001758",
            dry_run=False,
            harvest_config=CONFIG,
        )


@pytest.mark.asyncio
async def test_capture_stores_candidate_in_empty_store(memory_service):
    """No similar memory: _try_evolve reports not-evolved and capture() stores.

    On main this stored 0 of N candidates.
    """
    service = AutoCaptureService(memory_service=memory_service, min_confidence=0.6)
    link = AsyncMock()

    result = await _capture(service, TEXT, link)

    assert result.candidates, "the text should yield at least one candidate"
    assert result.stored == len(result.candidates)
    assert result.evolved == 0
    assert len(result.stored_hashes) == len(result.candidates)
    for content_hash in result.stored_hashes:
        assert await memory_service.storage.get_by_hash(content_hash) is not None
    link.assert_awaited()


@pytest.mark.asyncio
async def test_capture_evolves_similar_memory_and_links_child(memory_service):
    """A similar memory exists: capture() evolves it and links the child hash.

    Covers the path #1758 broke the other way round: the child hash that
    _try_evolve now returns is what reaches _link_derived_from, and it is a
    memory that exists in storage.
    """
    await _require_similarity(memory_service)
    service = AutoCaptureService(memory_service=memory_service, min_confidence=0.6)
    first = await _capture(service, TEXT, AsyncMock())
    assert first.stored == len(first.candidates) >= 1

    link = AsyncMock()
    second = await _capture(service, TEXT_REWORDED, link)

    assert second.candidates
    assert second.evolved >= 1
    assert second.stored == len(second.candidates)
    child_hashes = [call.args[1] for call in link.await_args_list]
    assert child_hashes, "an evolved candidate should be linked to its parent"
    for child_hash in child_hashes:
        assert child_hash not in first.stored_hashes
        assert await memory_service.storage.get_by_hash(child_hash) is not None


@pytest.mark.asyncio
async def test_hashless_evolution_is_not_stored_twice(memory_service):
    """evolve_memory() may report success without a hash.

    _try_evolve must still read as evolved, or capture() falls through to
    store_memory() and stores the candidate a second time.
    """
    await _require_similarity(memory_service)
    service = AutoCaptureService(memory_service=memory_service, min_confidence=0.6)
    first = await _capture(service, TEXT, AsyncMock())
    assert first.stored >= 1

    memory_service.evolve_memory = AsyncMock(return_value=(True, "ok", None))
    store_spy = AsyncMock(wraps=memory_service.store_memory)
    memory_service.store_memory = store_spy
    link = AsyncMock()

    second = await _capture(service, TEXT_REWORDED, link)

    assert second.evolved == len(second.candidates) >= 1
    store_spy.assert_not_awaited()
    link.assert_not_awaited()  # no child hash to link


@pytest.mark.asyncio
@pytest.mark.parametrize("existing_store", ["default", "docs"])
async def test_capture_and_evolution_stay_in_requested_store(
    memory_service, monkeypatch, existing_store
):
    """Auto-capture must not evolve a similar memory in another store."""
    storage = memory_service.storage
    vector = storage._generate_embedding("stable embedding for store isolation")
    monkeypatch.setattr(storage, "_generate_embedding", lambda _text: vector)

    existing_content = (
        "The existing store contains a prior SQLite concurrency decision."
    )
    existing = await memory_service.store_memory(
        content=existing_content,
        tags=["existing"],
        store=existing_store,
    )
    assert existing["success"]
    existing_hash = existing["memory"]["content_hash"]

    service = AutoCaptureService(memory_service=memory_service, min_confidence=0.6)
    result = await service.capture(
        TEXT,
        parent_hash=None,
        dry_run=False,
        link_parent=False,
        store="docs",
        harvest_config=CONFIG,
    )

    assert result.candidates
    assert result.stored == len(result.candidates)
    assert result.evolved == (len(result.candidates) if existing_store == "docs" else 0)

    old_row = storage.conn.execute(
        "SELECT content, store, superseded_by FROM memories WHERE content_hash = ?",
        (existing_hash,),
    ).fetchone()
    assert old_row[0] == existing_content
    assert old_row[1] == existing_store
    assert bool(old_row[2]) is (existing_store == "docs")

    for candidate in result.candidates:
        rows = storage.conn.execute(
            "SELECT store FROM memories WHERE content = ? AND deleted_at IS NULL",
            (candidate.content,),
        ).fetchall()
        assert rows
        assert {row[0] for row in rows} == {"docs"}


@pytest.mark.asyncio
async def test_store_handler_keeps_auto_capture_in_requested_store(
    memory_service, monkeypatch
):
    """The MCP store handler must carry its store into real auto-capture."""
    from mcp_memory_service.server.handlers.memory import handle_store_memory

    handler_text = TEXT + " The migration starts next week."
    candidates = AutoCaptureService(
        memory_service=memory_service, min_confidence=0.6
    ).extract_from_text(handler_text)
    assert candidates

    storage = memory_service.storage
    vector = storage._generate_embedding("stable embedding for store isolation")
    monkeypatch.setattr(storage, "_generate_embedding", lambda _text: vector)

    existing_content = "A prior default-store SQLite concurrency decision."
    existing = await memory_service.store_memory(
        content=existing_content, tags=["existing"], store="default"
    )
    assert existing["success"]
    existing_hash = existing["memory"]["content_hash"]

    monkeypatch.setattr(
        "mcp_memory_service.harvest.auto_capture._get_graph_instance_async",
        AsyncMock(return_value=None),
    )
    server = SimpleNamespace(
        _ensure_storage_initialized=AsyncMock(),
        memory_service=memory_service,
        storage=storage,
    )
    response = await handle_store_memory(
        server,
        {
            "content": handler_text,
            "store": "docs",
            "auto_extract": True,
            "min_extract_confidence": 0.6,
        },
    )

    assert "Auto-capture:" in response[0].text
    row = storage.conn.execute(
        "SELECT store, superseded_by FROM memories WHERE content_hash = ?",
        (existing_hash,),
    ).fetchone()
    assert row == ("default", None)
    for candidate in candidates:
        rows = storage.conn.execute(
            "SELECT store FROM memories WHERE content = ? AND deleted_at IS NULL",
            (candidate.content,),
        ).fetchall()
        assert rows
        assert {store for (store,) in rows} == {"docs"}
