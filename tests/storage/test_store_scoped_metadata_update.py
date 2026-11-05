"""Hash-addressed updates are confined to the claimed store (#1875, part of #1683).

update_memory_metadata() and update_memory_versioned() used to find the row by
content_hash alone. The handler guarded them with a get_by_hash pre-check, but
the check and the write were separate steps. With ``store`` passed down, the
lookup and the UPDATE themselves carry ``AND store = ?``, so a claim on one
store cannot change a memory in another, whoever the caller is.
"""

import os
import shutil
import tempfile

import pytest
import pytest_asyncio

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.hybrid import HybridMemoryStorage
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash

os.environ["MCP_MEMORY_ALLOW_HASH_EMBEDDINGS"] = "1"


@pytest_asyncio.fixture
async def storage():
    d = tempfile.mkdtemp(prefix="mcp-test-store-scoped-update-")
    s = SqliteVecMemoryStorage(os.path.join(d, "test.db"))
    await s.initialize()
    yield s
    await s.close()
    shutil.rmtree(d, ignore_errors=True)


async def _store(storage, content, store):
    m = Memory(content=content, content_hash=generate_content_hash(content), tags=["orig"])
    ok, msg = await storage.store(m, skip_semantic_dedup=True, store=store)
    assert ok, msg
    return m.content_hash


def _tags_of(storage, content_hash):
    return storage.conn.execute(
        "SELECT tags FROM memories WHERE content_hash = ?", (content_hash,)
    ).fetchone()[0]


def _row_count(storage):
    return storage.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]


@pytest.mark.asyncio
async def test_metadata_update_from_another_store_is_not_found(storage):
    h = await _store(storage, "Store B keeps the deploy calendar.", "b")

    ok, msg = await storage.update_memory_metadata(h, {"tags": ["hijacked"]}, store="a")

    assert not ok
    assert "not found" in msg
    assert _tags_of(storage, h) == "orig"


@pytest.mark.asyncio
@pytest.mark.parametrize("store", ["b", None, "all"])
async def test_metadata_update_in_own_store_or_unscoped_succeeds(storage, store):
    h = await _store(storage, f"Store B keeps the release notes ({store}).", "b")

    ok, msg = await storage.update_memory_metadata(h, {"tags": ["updated"]}, store=store)

    assert ok, msg
    assert _tags_of(storage, h) == "updated"


@pytest.mark.asyncio
async def test_versioned_update_from_another_store_is_not_found(storage):
    h = await _store(storage, "Store B runs backups at 02:00.", "b")
    before = _row_count(storage)

    ok, msg, new_hash = await storage.update_memory_versioned(
        h, "Store B runs backups at 03:00.", store="a"
    )

    assert not ok
    assert "not found" in msg
    assert new_hash is None
    assert _row_count(storage) == before


@pytest.mark.asyncio
async def test_versioned_update_in_own_store_succeeds(storage):
    h = await _store(storage, "Store B rotates keys monthly.", "b")

    ok, msg, new_hash = await storage.update_memory_versioned(
        h, "Store B rotates keys weekly.", store="b"
    )

    assert ok, msg
    assert new_hash and new_hash != h


def _hybrid_over(primary):
    # Only the primary path matters here; no Cloudflare secondary is set up.
    hybrid = object.__new__(HybridMemoryStorage)
    hybrid.primary = primary
    hybrid.sync_service = None
    hybrid._sync_paused = False
    return hybrid


@pytest.mark.asyncio
async def test_hybrid_passes_store_to_primary(storage):
    h = await _store(storage, "Store B owns the on-call rota.", "b")
    hybrid = _hybrid_over(storage)

    ok, msg = await hybrid.update_memory_metadata(h, {"tags": ["hijacked"]}, store="a")
    assert not ok
    assert _tags_of(storage, h) == "orig"

    ok, msg = await hybrid.update_memory_metadata(h, {"tags": ["updated"]}, store="b")
    assert ok, msg
    assert _tags_of(storage, h) == "updated"


class _Server:
    def __init__(self, storage):
        self._storage = storage

    async def _ensure_storage_initialized(self):
        return self._storage


@pytest.mark.asyncio
async def test_handler_passes_claimed_store_to_storage(storage, monkeypatch):
    """The handler hands the claimed store to the storage call itself.

    The get_by_hash pre-check still runs first; this pins that the write does
    not depend on it alone.
    """
    from mcp_memory_service.server.handlers.memory import handle_update_memory_metadata

    h = await _store(storage, "Store B tracks the incident log.", "b")
    seen = {}
    original = storage.update_memory_metadata

    async def spy(*args, **kwargs):
        seen.update(kwargs)
        return await original(*args, **kwargs)

    monkeypatch.setattr(storage, "update_memory_metadata", spy)

    await handle_update_memory_metadata(
        _Server(storage), {"content_hash": h, "updates": {"tags": ["x"]}, "store": "b"}
    )

    assert seen.get("store") == "b"


class _RecordingStorage:
    """Stand-in for a backend; records the update call's keyword arguments."""

    def __init__(self, scoped):
        self.get_by_hash_is_store_scoped = scoped
        self.update_kwargs = None

    async def get_by_hash(self, content_hash, store=None):
        return Memory(content="x", content_hash=content_hash, tags=["orig"])

    async def update_memory_metadata(self, **kwargs):
        self.update_kwargs = kwargs
        return False, "stop here"


@pytest.mark.asyncio
@pytest.mark.parametrize("scoped, expected", [(True, "b"), (False, None)])
async def test_rest_update_passes_store_only_to_scoped_backends(scoped, expected):
    """PUT /memories/{hash} confines the write itself on backends that can.

    A backend that ignores ``store`` (cloudflare, milvus, remote_http) does not
    accept the keyword yet, so it must not receive it.
    """
    from mcp_memory_service.web.api.memories import MemoryUpdateRequest, update_memory

    storage = _RecordingStorage(scoped)
    await update_memory(
        "abc", MemoryUpdateRequest(tags=["x"]), store="b", storage=storage, user=None
    )

    assert storage.update_kwargs.get("store") == expected
