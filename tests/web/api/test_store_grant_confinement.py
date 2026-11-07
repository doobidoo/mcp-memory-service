"""
A per-store grant (read:<store> / write:<store>) authorizes only calls that
really stay inside that store.

#1682 introduced per-store scopes and gated every request on the store the
caller *claims*. Three holes were left behind it:

- tools that ignore ``store`` (memory_quality, memory_cleanup, memory_resolve,
  ...) accepted a claimed store and then acted on every partition;
- memory_search's ``include_beliefs`` and ``fallback`` read across partitions
  even when the main search is store-scoped (both are filtered by store since
  #1877 and #1876);
- on backends whose ``get_by_hash`` ignores ``store`` (cloudflare, milvus,
  remote_http), hash-addressed REST routes returned another store's memory.

These drive the real /mcp shim and the real REST routes against sqlite-vec,
with the authenticated user replaced by one holding exactly the given scope.
"""

import os
import tempfile

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash
from mcp_memory_service.web.dependencies import set_storage


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, "test.db")


@pytest_asyncio.fixture
async def storage(temp_db, monkeypatch):
    monkeypatch.setenv("MCP_SEMANTIC_DEDUP_ENABLED", "false")
    s = SqliteVecMemoryStorage(temp_db)
    await s.initialize()
    yield s
    await s.close()


@pytest_asyncio.fixture
async def other_store_hash(storage):
    """A memory that lives in projectB, which a projectA credential must not reach."""
    content = "belongs to projectB only"
    memory = Memory(content=content, content_hash=generate_content_hash(content), tags=["b"])
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


def _call(client, name, arguments, headers=None):
    return client.post("/mcp", json={
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }, headers=headers or {})


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


@pytest_asyncio.fixture
async def hash_memories_in_two_stores(storage):
    hashes = {}
    for store in ("projectA", "projectB"):
        text = f"hash lookup memory in {store}"
        content_hash = generate_content_hash(text)
        memory = Memory(content=text, content_hash=content_hash, tags=["hash-test"])
        ok, _ = await storage.store(memory, store=store)
        assert ok
        hashes[store] = content_hash
    return hashes


@pytest.mark.integration
def test_store_grant_hash_lookup_stays_in_its_store(make_client, hash_memories_in_two_stores):
    client = make_client("read:projectA")

    own = _call(client, "memory_search", {
        "content_hash": hash_memories_in_two_stores["projectA"], "store": "projectA",
    })
    assert own.status_code == 200
    own_body = own.json()
    assert "error" not in own_body
    own_text = "".join(c.get("text", "") for c in own_body["result"]["content"])
    assert "hash lookup memory in projectA" in own_text

    other = _call(client, "memory_search", {
        "content_hash": hash_memories_in_two_stores["projectB"], "store": "projectA",
    })
    assert other.status_code == 200
    other_body = other.json()
    assert "error" not in other_body
    other_text = "".join(c.get("text", "") for c in other_body["result"]["content"])
    assert "No memories found" in other_text
    assert "hash lookup memory in projectB" not in other_text


@pytest.mark.integration
def test_hash_lookup_separates_agent_header_from_explicit_filter(
    make_client, hash_memories_in_two_stores
):
    client = make_client("read:projectA")
    content_hash = hash_memories_in_two_stores["projectA"]
    headers = {"X-Agent-ID": "transport-agent"}

    direct = _call(client, "memory_search", {
        "content_hash": content_hash, "store": "projectA",
    }, headers=headers)
    direct_text = "".join(
        c.get("text", "") for c in direct.json()["result"]["content"]
    )
    assert "hash lookup memory in projectA" in direct_text

    explicit = _call(client, "memory_search", {
        "content_hash": content_hash, "store": "projectA", "agent_id": "author-a",
    }, headers=headers)
    explicit_text = "".join(
        c.get("text", "") for c in explicit.json()["result"]["content"]
    )
    assert explicit_text == (
        "Error: content_hash cannot be combined with search arguments: agent_id"
    )


@pytest_asyncio.fixture
async def memories_by_agent(storage):
    for agent_id in ("transport-agent", "other-agent"):
        text = f"author filter probe {agent_id}"
        memory = Memory(
            content=text,
            content_hash=generate_content_hash(text),
            metadata={"agent_id": agent_id},
        )
        ok, _ = await storage.store(memory)
        assert ok


@pytest.mark.integration
def test_agent_header_still_filters_text_search_and_list(make_client, memories_by_agent):
    client = make_client("read")
    headers = {"X-Agent-ID": "transport-agent"}

    search = _call(client, "memory_search", {
        "query": "author filter probe", "mode": "exact",
    }, headers=headers)
    search_text = "".join(
        c.get("text", "") for c in search.json()["result"]["content"]
    )
    assert "author filter probe transport-agent" in search_text
    assert "author filter probe other-agent" not in search_text

    listed = _call(client, "memory_list", {"page_size": 50}, headers=headers)
    list_text = listed.json()["result"]["content"][0]["text"]
    assert "author filter probe transport-agent" in list_text
    assert "author filter probe other-agent" not in list_text

    unfiltered = _call(client, "memory_search", {
        "query": "author filter probe", "mode": "exact",
    })
    unfiltered_text = "".join(
        c.get("text", "") for c in unfiltered.json()["result"]["content"]
    )
    assert "author filter probe transport-agent" in unfiltered_text
    assert "author filter probe other-agent" in unfiltered_text


@pytest_asyncio.fixture
async def zebra_in_three_stores(storage):
    """The same word in projectA, projectB and default, reachable by the exact
    and the tag fallback alike."""
    for store in ("projectA", "projectB", "default"):
        text = f"zebra crossing note for {store}"
        memory = Memory(content=text, content_hash=generate_content_hash(text), tags=["zebra"])
        ok, _ = await storage.store(memory, store=store)
        assert ok


@pytest.mark.integration
def test_store_grant_fallback_stays_in_its_store(make_client, zebra_in_three_stores):
    # The exact/tag fallback searches the claimed store since #1876.
    client = make_client("read:projectA")
    response = _call(client, "memory_search", {"query": "zebra", "store": "projectA", "fallback": True})
    assert response.status_code == 200
    body = response.json()
    assert "error" not in body
    text = "".join(c.get("text", "") for c in body["result"]["content"])
    assert "note for projectA" in text
    assert "note for projectB" not in text
    assert "note for default" not in text


@pytest_asyncio.fixture
async def beliefs_in_two_stores(storage):
    """One belief backed by a projectA memory, one by a projectB memory."""
    from datetime import datetime, timezone

    from mcp_memory_service.consolidation.belief_service import BeliefService

    svc = BeliefService(storage)
    for store, text in (("projectA", "alpha deploys nightly"), ("projectB", "bravo deploys weekly")):
        memory = Memory(content=text, content_hash=generate_content_hash(text), tags=["t"])
        ok, _ = await storage.store(memory, store=store)
        assert ok
        await svc._create_belief(
            f"belief-{store}", f"Belief from {store}", 0.9, "active",
            [memory.content_hash], [], datetime.now(timezone.utc),
        )


@pytest.mark.integration
def test_store_grant_searches_with_beliefs_in_its_own_store(make_client, beliefs_in_two_stores):
    # Beliefs are filtered by the claimed store since #1877.
    client = make_client("read:projectA")
    response = _call(client, "memory_search", {"query": "deploys", "store": "projectA", "include_beliefs": True})
    assert response.status_code == 200
    body = response.json()
    assert "error" not in body
    text = "".join(c.get("text", "") for c in body["result"]["content"])
    assert "Belief from projectA" in text
    assert "Belief from projectB" not in text


@pytest.mark.integration
def test_store_grant_still_searches_its_own_store(make_client):
    client = make_client("read:projectA")
    response = _call(client, "memory_search", {"query": "x", "store": "projectA"})
    assert response.status_code == 200
    assert "error" not in response.json()


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
    client = make_client("read:projectA write:projectA")
    response = _call(client, "memory_delete", {"content_hash": "deadbeef" * 8, "store": "projectA"})
    assert _refused(response)


@pytest.mark.integration
def test_global_hash_lookup_fails_closed_for_scopes_backend_cannot_enforce(
    make_client, storage, hash_memories_in_two_stores, monkeypatch,
):
    _ignore_store_like_cloudflare(storage, monkeypatch)
    client = make_client("read")
    error = (
        "Error: store-scoped content_hash lookup is not supported on this "
        "storage backend"
    )
    for arguments in (
        {"content_hash": hash_memories_in_two_stores["projectA"]},
        {
            "content_hash": hash_memories_in_two_stores["projectA"],
            "store": "projectA",
        },
    ):
        response = _call(client, "memory_search", arguments)
        text = response.json()["result"]["content"][0]["text"]
        assert text == error

    cross_store = _call(client, "memory_search", {
        "content_hash": hash_memories_in_two_stores["projectB"], "store": "all",
    })
    cross_store_text = cross_store.json()["result"]["content"][0]["text"]
    assert "hash lookup memory in projectB" in cross_store_text


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
