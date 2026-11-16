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


async def _capture(service, text, link, store=None):
    store_kw = {"store": store} if store is not None else {}
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
            **store_kw,
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


# --- store confinement (GHSA-989p-hfjf-q97q) ---------------------------------
#
# capture() took no store: a candidate extracted from a memory in another store
# was compared with "default", replaced a similar memory there or was inserted
# there. Tests by RISCON Labs, split up.

async def _seed_default(memory_service, monkeypatch):
    """One memory in "default" that every candidate is a near-duplicate of."""
    storage = memory_service.storage
    unit_vec = [1.0] + [0.0] * (storage.embedding_dimension - 1)
    monkeypatch.setattr(storage, "_generate_embedding", lambda _text: list(unit_vec))
    baseline = await memory_service.store_memory(
        content=TEXT, tags=["baseline"], memory_type="decision", store="default"
    )
    assert baseline["success"]
    return baseline["memory"]["content_hash"]


def _default_rows(storage):
    return storage.conn.execute(
        "SELECT content_hash, superseded_by FROM memories "
        "WHERE store = 'default' AND deleted_at IS NULL"
    ).fetchall()


@pytest.mark.asyncio
async def test_capture_into_a_store_leaves_default_untouched(memory_service, monkeypatch):
    default_hash = await _seed_default(memory_service, monkeypatch)
    storage = memory_service.storage
    service = AutoCaptureService(memory_service=memory_service, min_confidence=0.6)

    captured = await _capture(service, TEXT_REWORDED, AsyncMock(), store="project_a")

    assert captured.stored >= 1 and captured.evolved == 0
    assert _default_rows(storage) == [(default_hash, None)]
    for child_hash in captured.stored_hashes:
        assert await storage.get_by_hash(child_hash, store="project_a") is not None
        assert await storage.get_by_hash(child_hash, store="default") is None


@pytest.mark.asyncio
async def test_capture_evolves_inside_its_own_store(memory_service, monkeypatch):
    default_hash = await _seed_default(memory_service, monkeypatch)
    service = AutoCaptureService(memory_service=memory_service, min_confidence=0.6)
    await _capture(service, TEXT_REWORDED, AsyncMock(), store="project_a")

    again = await _capture(
        service,
        "I decided to use WAL mode for concurrent SQLite because multiple readers "
        "kept blocking writers.",
        AsyncMock(),
        store="project_a",
    )

    assert again.evolved >= 1
    assert _default_rows(memory_service.storage) == [(default_hash, None)]


@pytest.mark.asyncio
async def test_evolve_memory_refuses_a_store_where_get_by_hash_is_unscoped(
    memory_service, monkeypatch
):
    """Where get_by_hash ignores ``store``, confinement cannot be shown: refuse."""
    default_hash = await _seed_default(memory_service, monkeypatch)
    monkeypatch.setattr(memory_service.storage, "get_by_hash_is_store_scoped", False)

    ok, _msg, child = await memory_service.evolve_memory(
        default_hash, "Attempted cross-store evolution", store="project_a"
    )

    assert not ok and child is None
    assert _default_rows(memory_service.storage) == [(default_hash, None)]


@pytest.mark.asyncio
async def test_evolve_memory_refuses_a_hash_from_another_store(memory_service):
    """The service refuses on its own, whatever its caller checked before."""
    stored = await memory_service.store_memory(
        content=TEXT, tags=["baseline"], memory_type="decision", store="default"
    )
    default_hash = stored["memory"]["content_hash"]

    ok, _msg, child = await memory_service.evolve_memory(
        default_hash, "Rewritten from another store", store="project_a"
    )

    assert not ok and child is None
    assert _default_rows(memory_service.storage) == [(default_hash, None)]


class _Server:
    def __init__(self, memory_service):
        self.memory_service = memory_service
        self.storage = memory_service.storage

    async def _ensure_storage_initialized(self):
        return self.storage


@pytest.mark.asyncio
async def test_unnamed_store_still_evolves_where_get_by_hash_is_unscoped(
    memory_service, monkeypatch
):
    """handle_store_memory falls back to store "default". Forwarding that fallback
    would switch evolution off for every caller on a backend whose get_by_hash
    ignores ``store`` (Cloudflare, Milvus), named store or not."""
    from mcp_memory_service.server.handlers.memory import handle_store_memory

    storage = memory_service.storage
    dim = storage.embedding_dimension
    note = "Session notes.\n" + TEXT_REWORDED
    monkeypatch.setattr(
        storage, "_generate_embedding",
        lambda text: ([0.0, 1.0] if text == note else [1.0, 0.0]) + [0.0] * (dim - 2),
    )
    monkeypatch.setattr(storage, "get_by_hash_is_store_scoped", False)
    baseline = await memory_service.store_memory(
        content=TEXT, tags=["baseline"], memory_type="decision"
    )
    assert baseline["success"]

    out = await handle_store_memory(_Server(memory_service), {
        "content": note, "auto_extract": True, "min_extract_confidence": 0.5,
    })

    assert "1 evolved" in out[0].text
