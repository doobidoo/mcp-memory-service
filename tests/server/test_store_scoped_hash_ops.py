"""Per-store confinement of hash-addressed updates (memory_update)."""
import pytest

from mcp_memory_service.server.handlers.memory import (
    handle_delete_memory,
    handle_update_memory_metadata,
)


class _Storage:
    def __init__(self, *, scoped: bool, found: bool):
        self.get_by_hash_is_store_scoped = scoped
        self._found = found
        self.updated = False
        self.update_kwargs = None

    async def get_by_hash(self, content_hash, store=None):
        return object() if self._found else None

    async def update_memory_metadata(self, **_kwargs):
        self.updated = True
        self.update_kwargs = _kwargs
        return (True, "updated")


class _Server:
    def __init__(self, storage):
        self.storage = storage

    async def _ensure_storage_initialized(self):
        return self.storage


class _DeleteService:
    def __init__(self):
        self.kwargs = None

    async def delete_memory(self, content_hash, **kwargs):
        self.kwargs = kwargs
        return {"success": True, "content_hash": content_hash}


class _DeleteServer:
    def __init__(self):
        self.memory_service = _DeleteService()

    async def _ensure_storage_initialized(self):
        return None


@pytest.mark.asyncio
async def test_update_fails_closed_on_store_ignoring_backend():
    """A store-scoped update is refused on a backend whose get_by_hash ignores
    store — it cannot prove the hash is confined to the claimed partition. The
    refusal raises (-> isError), so it cannot look like a completed write."""
    storage = _Storage(scoped=False, found=True)
    with pytest.raises(PermissionError, match="not supported on this storage backend"):
        await handle_update_memory_metadata(
            _Server(storage),
            {"content_hash": "abc", "updates": {"tags": ["t"]}, "store": "projectA"},
        )
    assert storage.updated is False


@pytest.mark.asyncio
async def test_update_forwards_claimed_store_to_storage_write():
    storage = _Storage(scoped=True, found=True)
    await handle_update_memory_metadata(
        _Server(storage),
        {"content_hash": "abc", "updates": {"tags": ["t"]}, "store": "projectA"},
    )

    assert storage.updated is True
    assert storage.update_kwargs["store"] == "projectA"


@pytest.mark.asyncio
async def test_delete_all_store_is_rejected_as_non_read_scope():
    server = _DeleteServer()

    with pytest.raises(PermissionError, match="only valid for read scopes"):
        await handle_delete_memory(server, {"content_hash": "abc", "store": "all"})

    assert server.memory_service.kwargs is None


@pytest.mark.asyncio
async def test_update_all_store_is_rejected_as_non_read_scope():
    storage = _Storage(scoped=True, found=True)

    with pytest.raises(PermissionError, match="only valid for read scopes"):
        await handle_update_memory_metadata(
            _Server(storage),
            {"content_hash": "abc", "updates": {"tags": ["t"]}, "store": "all"},
        )

    assert storage.updated is False


@pytest.mark.asyncio
async def test_update_not_found_when_hash_absent_from_claimed_store():
    """On a store-honoring backend, a hash absent from the claimed store raises
    not-found before any write happens (surfaces as a tool error, not success)."""
    storage = _Storage(scoped=True, found=False)
    with pytest.raises(PermissionError, match="not found in the requested store"):
        await handle_update_memory_metadata(
            _Server(storage),
            {"content_hash": "abc", "updates": {"tags": ["t"]}, "store": "projectA"},
        )
    assert storage.updated is False
