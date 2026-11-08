"""RemoteHTTPStorage confines hash-addressed calls to ``store`` (#1879, part of #1683).

The peer is simulated with an httpx MockTransport that answers like the REST API:
a missing ``store`` query parameter means ``"default"``, ``"all"`` means every
store, and a memory in another store is a 404. The older peer ignores ``store``
and does not advertise ``store_scoped_access``.
"""

import logging

import httpx
import pytest

from mcp_memory_service.storage.remote_http import RemoteHTTPStorage

HASH_WORK = "a" * 64
HASH_DEFAULT = "b" * 64

BASE_CAPABILITIES = {
    "sync_protocol": 2,
    "sync_protocol_min": 1,
    "event_schema_version": 1,
    "embedding_model": "test-model",
    "embedding_dimension": 384,
    "version": "11.14.0",
}


def _memory_json(content_hash, store):
    return {
        "content": f"memory in {store}",
        "content_hash": content_hash,
        "tags": [],
        "memory_type": "observation",
        "metadata": {},
    }


async def _peer(*, scoped):
    """A peer holding HASH_WORK in store "work" and HASH_DEFAULT in "default"."""
    stores = {HASH_WORK: "work", HASH_DEFAULT: "default"}
    requested = []

    def handler(request):
        store_param = request.url.params.get("store")
        requested.append((request.method, request.url.path, store_param))
        path = request.url.path
        if path == "/api/sync/capabilities":
            capabilities = dict(BASE_CAPABILITIES)
            if scoped:
                capabilities["store_scoped_access"] = True
            return httpx.Response(200, json=capabilities, request=request)
        if path.startswith("/api/memories/"):
            content_hash = path.rsplit("/", 1)[1]
            owner = stores.get(content_hash)
            wanted = store_param or "default"
            if scoped and request.method in ("PUT", "DELETE") and wanted == "all":
                return httpx.Response(400, json={"detail": "read scopes only"}, request=request)
            visible = owner is not None and (
                not scoped or wanted == "all" or owner == wanted
            )
            if not visible:
                return httpx.Response(404, json={"detail": "Memory not found"}, request=request)
            if request.method == "GET":
                return httpx.Response(200, json=_memory_json(content_hash, owner), request=request)
            if request.method == "PUT":
                return httpx.Response(200, json={"success": True}, request=request)
            if request.method == "DELETE":
                del stores[content_hash]
                return httpx.Response(200, json={"success": True}, request=request)
        return httpx.Response(404, json={"detail": "not found"}, request=request)

    storage = RemoteHTTPStorage("https://peer.test")
    await storage.client.aclose()
    storage.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return storage, requested, stores


def _memory_calls(requested):
    return [r for r in requested if r[1].startswith("/api/memories/")]


@pytest.mark.asyncio
async def test_flag_follows_the_peer_advertisement():
    scoped, _, _ = await _peer(scoped=True)
    older, _, _ = await _peer(scoped=False)
    assert scoped.get_by_hash_is_store_scoped is False
    await scoped.initialize()
    await older.initialize()
    assert scoped.get_by_hash_is_store_scoped is True
    assert older.get_by_hash_is_store_scoped is False


@pytest.mark.asyncio
async def test_scoped_peer_receives_store_and_refuses_cross_store_access():
    storage, requested, stores = await _peer(scoped=True)
    await storage.initialize()

    assert (await storage.get_by_hash(HASH_WORK, store="work")).content_hash == HASH_WORK
    assert await storage.get_by_hash(HASH_WORK, store="other") is None

    ok, _ = await storage.update_memory_metadata(HASH_WORK, {"tags": ["x"]}, store="work")
    assert ok is True
    ok, message = await storage.update_memory_metadata(HASH_WORK, {"tags": ["x"]}, store="other")
    assert ok is False and "404" in message

    ok, message = await storage.delete(HASH_WORK, store="other")
    assert ok is False and "404" in message
    assert HASH_WORK in stores

    assert _memory_calls(requested) == [
        ("GET", f"/api/memories/{HASH_WORK}", "work"),
        ("GET", f"/api/memories/{HASH_WORK}", "other"),
        ("PUT", f"/api/memories/{HASH_WORK}", "work"),
        ("PUT", f"/api/memories/{HASH_WORK}", "other"),
        ("DELETE", f"/api/memories/{HASH_WORK}", "other"),
    ]


@pytest.mark.asyncio
async def test_older_peer_refuses_named_store_without_a_request(caplog):
    storage, requested, stores = await _peer(scoped=False)
    await storage.initialize()

    with caplog.at_level(logging.WARNING, logger="mcp_memory_service.storage.remote_http"):
        assert await storage.get_by_hash(HASH_WORK, store="work") is None
        ok, update_message = await storage.update_memory_metadata(
            HASH_WORK, {"tags": ["x"]}, store="work"
        )
        ok_delete, _ = await storage.delete(HASH_WORK, store="work")

    assert ok is False and ok_delete is False
    assert "store_scoped_access" in update_message
    assert _memory_calls(requested) == []
    assert HASH_WORK in stores
    warnings = [r for r in caplog.records if "store_scoped_access" in r.getMessage()]
    assert len(warnings) == 1
    assert "peer.test" in warnings[0].getMessage()


@pytest.mark.asyncio
async def test_peer_without_capabilities_endpoint_fails_closed():
    def handler(request):
        return httpx.Response(404, json={"detail": "not found"}, request=request)

    storage = RemoteHTTPStorage("https://peer.test")
    await storage.client.aclose()
    storage.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await storage.initialize()

    assert storage.get_by_hash_is_store_scoped is False
    assert await storage.get_by_hash(HASH_WORK, store="work") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("scoped", [True, False])
async def test_unscoped_and_default_calls_send_no_store(scoped):
    storage, requested, _ = await _peer(scoped=scoped)
    await storage.initialize()

    assert await storage.get_by_hash(HASH_DEFAULT) is not None
    assert await storage.get_by_hash(HASH_DEFAULT, store="default") is not None
    ok, _ = await storage.update_memory_metadata(HASH_DEFAULT, {"tags": ["x"]}, store="default")
    assert ok is True
    ok, _ = await storage.delete(HASH_DEFAULT, store="default")
    assert ok is True

    assert [r[2] for r in _memory_calls(requested)] == [None, None, None, None]


@pytest.mark.asyncio
@pytest.mark.parametrize("scoped", [True, False])
async def test_all_stores_read_is_passed_through(scoped):
    storage, requested, _ = await _peer(scoped=scoped)
    await storage.initialize()

    memory = await storage.get_by_hash(HASH_WORK, store="all")

    assert memory is not None and memory.content_hash == HASH_WORK
    assert _memory_calls(requested) == [("GET", f"/api/memories/{HASH_WORK}", "all")]


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ["older", "no-capabilities", "unreachable"])
async def test_flag_is_cleared_when_reinitialize_cannot_confirm(answer):
    storage, _, _ = await _peer(scoped=True)
    await storage.initialize()
    assert storage.get_by_hash_is_store_scoped is True

    requested = []
    if answer == "older":
        older, requested, _ = await _peer(scoped=False)
        transport_client = older.client
    else:
        def handler(request):
            requested.append((request.method, request.url.path, request.url.params.get("store")))
            if answer == "unreachable":
                raise httpx.ConnectError("connection refused", request=request)
            return httpx.Response(404, json={"detail": "not found"}, request=request)

        transport_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await storage.client.aclose()
    storage.client = transport_client
    await storage.initialize()

    assert storage.get_by_hash_is_store_scoped is False
    assert await storage.get_by_hash(HASH_WORK, store="work") is None
    assert _memory_calls(requested) == []
