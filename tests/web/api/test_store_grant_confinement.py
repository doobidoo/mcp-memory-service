"""
A per-store grant (read:<store> / write:<store>) authorizes only calls that
really stay inside that store.

#1682 introduced per-store scopes and gated every request on the store the
caller *claims*. Three holes were left behind it:

- tools that ignore ``store`` (memory_quality, memory_cleanup, memory_resolve,
  ...) accepted a claimed store and then acted on every partition;
- memory_search's ``include_beliefs`` and ``fallback`` read across partitions
  even when the main search is store-scoped;
- on backends whose ``get_by_hash`` ignores ``store`` (cloudflare, milvus,
  remote_http), hash-addressed REST routes returned another store's memory.

These drive the real /mcp shim and the real REST routes against sqlite-vec,
with the authenticated user replaced by one holding exactly the given scope.
"""

import os

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash
from mcp_memory_service.web.dependencies import set_storage


@pytest_asyncio.fixture
async def storage(temp_db_path, monkeypatch):
    monkeypatch.setenv("MCP_SEMANTIC_DEDUP_ENABLED", "false")
    s = SqliteVecMemoryStorage(os.path.join(temp_db_path, "test.db"))
    await s.initialize()
    yield s
    await s.close()


@pytest_asyncio.fixture
async def other_store_hash(storage):
    """A memory that lives in projectB, which a projectA credential must not reach."""
    content = "belongs to projectB only"
    memory = Memory(
        content=content,
        content_hash=generate_content_hash(content),
        tags=["b", "privateprojectb"],
    )
    ok, _ = await storage.store(memory, store="projectB")
    assert ok
    return memory.content_hash


@pytest.fixture
def make_client(storage, monkeypatch):
    from mcp_memory_service.web.app import app
    from mcp_memory_service.web.oauth import middleware
    from mcp_memory_service.web.api import mcp

    monkeypatch.setattr(middleware, "API_KEY", None)
    monkeypatch.setattr(middleware, "OAUTH_ENABLED", False)
    monkeypatch.setattr(middleware, "ALLOW_ANONYMOUS_ACCESS", True)
    from mcp_memory_service.web import dependencies
    # Restored on teardown, so this storage cannot answer later tests' lookups.
    monkeypatch.setattr(dependencies, "_storage", None)
    set_storage(storage)
    monkeypatch.setattr(mcp, "_memory_server", None)

    def _make(scope):
        async def mock_user():
            return middleware.AuthenticationResult(
                authenticated=True, client_id="c", scope=scope, auth_method="test",
            )

        app.dependency_overrides[middleware.get_current_user] = mock_user
        app.dependency_overrides[middleware.require_read_access] = mock_user
        return TestClient(app)

    yield _make
    app.dependency_overrides.clear()


def _ignore_store_like_cloudflare(storage, monkeypatch):
    """Make the sqlite backend behave like cloudflare/milvus/remote_http:
    get_by_hash accepts ``store`` and ignores it, and says so."""
    original = storage.get_by_hash

    async def get_by_hash(content_hash, store=None):
        return await original(content_hash)

    monkeypatch.setattr(storage, "get_by_hash", get_by_hash)
    monkeypatch.setattr(storage, "get_by_hash_is_store_scoped", False)
    monkeypatch.setattr(storage, "supports_store_scoped_access", False)


def _call(client, name, arguments):
    return client.post("/mcp", json={
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    })


def _refused(response):
    return response.status_code == 403 and response.json()["error"]["code"] == -32003


# --------------------------------------------------------------------------
# /mcp shim: tool allowlist and cross-store arguments
# --------------------------------------------------------------------------

@pytest.mark.integration
def test_store_grant_cannot_reach_a_tool_that_ignores_store(make_client, other_store_hash):
    """memory_quality has no store parameter: claiming one must not unlock it."""
    client = make_client("read:projectA write:projectA")
    response = _call(client, "memory_quality", {
        "action": "rate", "content_hash": other_store_hash, "rating": "-1", "store": "projectA",
    })
    assert _refused(response)


@pytest.mark.integration
@pytest.mark.parametrize("flag", ["include_beliefs", "fallback"])
@pytest.mark.asyncio
async def test_store_grant_search_options_do_not_cross_partitions(
    make_client, storage, other_store_hash, flag
):
    client = make_client("read:projectA")
    if flag == "include_beliefs":
        storage.conn.execute(
            "INSERT INTO beliefs "
            "(belief_hash, content, confidence, status, created_at, updated_at, "
            "derived_from, contradicted_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "b" * 64,
                "belief based on projectB only",
                0.9,
                "active",
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
                f'["{other_store_hash}"]',
                "[]",
            ),
        )
        storage.conn.commit()
        arguments = {
            "query": "no matching memories",
            "store": "projectA",
            "include_beliefs": True,
        }
    else:
        arguments = {
            "query": "belongs to projectB only",
            "store": "projectA",
            "fallback": True,
        }

    response = _call(client, "memory_search", arguments)
    assert response.status_code == 200
    assert "error" not in response.json()
    rendered = " ".join(
        item.get("text", "")
        for item in response.json().get("result", {}).get("content", [])
    )
    assert other_store_hash not in rendered
    if flag == "include_beliefs":
        assert "belief based on projectB only" not in rendered
    else:
        assert "No memories found for query" in rendered


@pytest.mark.integration
@pytest.mark.asyncio
async def test_store_grant_still_searches_its_own_store(make_client, storage):
    memory = Memory(
        content="belongs to projectA only",
        content_hash=generate_content_hash("belongs to projectA only"),
        tags=["privateprojecta"],
    )
    ok, _ = await storage.store(memory, store="projectA")
    assert ok
    client = make_client("read:projectA")
    response = _call(
        client,
        "memory_search",
        {
            "query": "belongs to projectA only",
            "store": "projectA",
            "fallback": True,
        },
    )
    assert response.status_code == 200
    assert "error" not in response.json()
    rendered = " ".join(
        item.get("text", "")
        for item in response.json().get("result", {}).get("content", [])
    )
    assert memory.content_hash in rendered


@pytest.mark.integration
@pytest.mark.asyncio
async def test_store_grant_tag_fallback_does_not_cross_partitions(
    make_client, other_store_hash
):
    response = _call(
        make_client("read:projectA"),
        "memory_search",
        {
            "query": "privateprojectb",
            "store": "projectA",
            "fallback": True,
        },
    )
    assert response.status_code == 200
    rendered = " ".join(
        item.get("text", "")
        for item in response.json().get("result", {}).get("content", [])
    )
    assert other_store_hash not in rendered
    assert "No memories found for query" in rendered


@pytest.mark.integration
@pytest.mark.asyncio
async def test_default_search_does_not_append_other_store_beliefs(
    make_client, storage, other_store_hash
):
    default_memory = Memory(
        content="belongs to the default store",
        content_hash=generate_content_hash("belongs to the default store"),
        tags=["default"],
    )
    ok, _ = await storage.store(default_memory, store="default")
    assert ok
    for belief_hash, content, source_hash in (
        (
            "d" * 64,
            "belief based on the default store",
            default_memory.content_hash,
        ),
        (
            "e" * 64,
            "belief based on projectB only",
            other_store_hash,
        ),
    ):
        storage.conn.execute(
            "INSERT INTO beliefs "
            "(belief_hash, content, confidence, status, created_at, updated_at, "
            "derived_from, contradicted_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                belief_hash,
                content,
                0.9,
                "active",
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
                f'["{source_hash}"]',
                "[]",
            ),
        )
    storage.conn.commit()

    response = _call(
        make_client("read"),
        "memory_search",
        {"query": "no matching memories", "include_beliefs": True},
    )
    assert response.status_code == 200
    rendered = " ".join(
        item.get("text", "")
        for item in response.json().get("result", {}).get("content", [])
    )
    assert "belief based on the default store" in rendered
    assert "belief based on projectB only" not in rendered


@pytest.mark.integration
def test_global_grant_is_unchanged_for_store_ignoring_tools(make_client, other_store_hash):
    client = make_client("read write")
    response = _call(client, "memory_quality", {
        "action": "get", "content_hash": other_store_hash, "store": "projectA",
    })
    assert not _refused(response)


@pytest.mark.integration
def test_store_grant_refused_on_a_backend_that_cannot_confine(make_client, storage, monkeypatch):
    monkeypatch.setattr(storage, "get_by_hash_is_store_scoped", False)
    monkeypatch.setattr(storage, "supports_store_scoped_access", False)
    client = make_client("read:projectA write:projectA")
    response = _call(client, "memory_delete", {"content_hash": "deadbeef" * 8, "store": "projectA"})
    assert _refused(response)


# --------------------------------------------------------------------------
# REST: hash-addressed routes on a backend that cannot confine
# --------------------------------------------------------------------------

@pytest.mark.integration
def test_rest_store_grant_refused_on_a_backend_that_cannot_confine(
    make_client, storage, other_store_hash, monkeypatch,
):
    """On cloudflare/milvus/remote_http, get_by_hash ignores store, so a
    read:projectA token would read projectB's memory by hash."""
    _ignore_store_like_cloudflare(storage, monkeypatch)
    client = make_client("read:projectA")
    response = client.get(f"/api/memories/{other_store_hash}", params={"store": "projectA"})
    assert response.status_code == 403


@pytest.mark.integration
def test_rest_store_grant_confined_on_a_backend_that_can(make_client, other_store_hash):
    client = make_client("read:projectA")
    response = client.get(f"/api/memories/{other_store_hash}", params={"store": "projectA"})
    assert response.status_code == 404


@pytest.mark.integration
def test_rest_global_grant_unchanged_on_a_backend_that_cannot_confine(
    make_client, storage, other_store_hash, monkeypatch,
):
    _ignore_store_like_cloudflare(storage, monkeypatch)
    client = make_client("read")
    response = client.get(f"/api/memories/{other_store_hash}", params={"store": "projectB"})
    assert response.status_code == 200
