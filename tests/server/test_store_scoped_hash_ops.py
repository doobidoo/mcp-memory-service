"""Per-store confinement of hash-addressed updates (memory_update).

The transport gate authorizes the *claimed* store, but the update resolves by
content_hash. The handler pre-resolves the hash within the claimed store:

every backend's get_by_hash confines to ``store`` (remote_http refuses a named
store against a peer without store_scoped_access, which also reads as
missing), so a hash from another store is reported as not found before any
write. The handler no longer reads the capability flag (#2120).
"""
import pytest

from mcp_memory_service.server.handlers.memory import handle_update_memory_metadata


class _Storage:
    def __init__(self, *, scoped: bool, found: bool):
        self.get_by_hash_is_store_scoped = scoped
        self._found = found
        self.updated = False
        self.lookup_stores = []
        self.update_kwargs = None

    async def get_by_hash(self, content_hash, store=None):
        self.lookup_stores.append(store)
        return object() if self._found else None

    async def update_memory_metadata(self, **kwargs):
        self.updated = True
        self.update_kwargs = kwargs
        return (True, "updated")


class _Server:
    def __init__(self, storage):
        self.storage = storage

    async def _ensure_storage_initialized(self):
        return self.storage


@pytest.mark.asyncio
async def test_update_confines_lookup_and_write_whatever_the_flag():
    """The flag no longer gates the handler: the claimed store goes to the
    lookup and to the write on every backend."""
    storage = _Storage(scoped=False, found=True)
    await handle_update_memory_metadata(
        _Server(storage),
        {"content_hash": "abc", "updates": {"tags": ["t"]}, "store": "projectA"},
    )
    assert storage.lookup_stores == ["projectA"]
    assert storage.update_kwargs.get("store") == "projectA"


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
