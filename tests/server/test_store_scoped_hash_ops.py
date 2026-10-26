"""Per-store confinement of hash-addressed updates (memory_update).

The transport gate authorizes the *claimed* store, but the update resolves by
content_hash. The handler pre-resolves the hash within the claimed store:

- backends whose get_by_hash honors ``store`` (get_by_hash_is_store_scoped=True)
  report a hash from another store as not-found, before any write;
- backends that accept but ignore ``store`` can't prove confinement, so a
  store-scoped update fails closed (refused).
"""
import pytest

from mcp_memory_service.server.handlers.memory import handle_update_memory_metadata


class _Storage:
    def __init__(self, *, scoped: bool, found: bool):
        self.get_by_hash_is_store_scoped = scoped
        self._found = found
        self.updated = False

    async def get_by_hash(self, content_hash, store=None):
        return object() if self._found else None

    async def update_memory_metadata(self, **_kwargs):
        self.updated = True
        return (True, "updated")


class _Server:
    def __init__(self, storage):
        self.storage = storage

    async def _ensure_storage_initialized(self):
        return self.storage


def _text(result):
    return result[0].text.lower()


@pytest.mark.asyncio
async def test_update_fails_closed_on_store_ignoring_backend():
    """A store-scoped update is refused on a backend whose get_by_hash ignores
    store — it cannot prove the hash is confined to the claimed partition."""
    storage = _Storage(scoped=False, found=True)
    out = await handle_update_memory_metadata(
        _Server(storage),
        {"content_hash": "abc", "updates": {"tags": ["t"]}, "store": "projectA"},
    )
    assert "not supported on this storage backend" in _text(out)
    assert storage.updated is False


@pytest.mark.asyncio
async def test_update_not_found_when_hash_absent_from_claimed_store():
    """On a store-honoring backend, a hash absent from the claimed store is
    reported not-found before any write happens."""
    storage = _Storage(scoped=True, found=False)
    out = await handle_update_memory_metadata(
        _Server(storage),
        {"content_hash": "abc", "updates": {"tags": ["t"]}, "store": "projectA"},
    )
    assert "not found in the requested store" in _text(out)
    assert storage.updated is False
