"""A per-store credential on Cloudflare stays inside its store (#1915).

With ``get_by_hash_is_store_scoped`` set on Cloudflare, a ``read:a`` credential
is granted ``memory_search`` and ``memory_list`` for ``store="a"``. These tests
drive the real /mcp shim against ``CloudflareStorage`` with D1 replaced by an
in-memory SQLite database and Vectorize answering every query with every stored
vector, so only the store predicate keeps another store's memories out.
"""

import sqlite3
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.cloudflare import CloudflareStorage
from mcp_memory_service.utils.hashing import generate_content_hash
from mcp_memory_service.web.dependencies import set_storage
from tests.unit.test_cloudflare_store_scope import _FakeD1

TEXT_A = "store a keeps the zebra rollout notes"
TEXT_B = "store b keeps the zebra incident notes"


class _FakeCloudflare(_FakeD1):
    """D1 on SQLite; Vectorize returns every stored vector for any query."""

    def __init__(self, storage):
        super().__init__(storage)
        # TestClient runs the app on another thread than the fixture.
        self.db.close()
        self.db = sqlite3.connect(":memory:", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.vector_upserts = []

    async def __call__(self, method, url, **kwargs):
        if url == f"{self.storage.vectorize_url}/query":
            ids = [r[0] for r in self.db.execute("SELECT vector_id FROM memories").fetchall()]
            matches = [{"id": vector_id, "score": 0.9} for vector_id in ids]
            return Mock(json=Mock(return_value={"success": True, "result": {"matches": matches}}))
        return await super().__call__(method, url, **kwargs)


@pytest.fixture
def cloudflare_storage():
    return CloudflareStorage(
        api_token="test-token",
        account_id="test-account",
        vectorize_index="test-index",
        d1_database_id="test-db",
        r2_bucket="test-bucket",
        embedding_model="@cf/baai/bge-base-en-v1.5",
    )


@pytest_asyncio.fixture
async def cloudflare(cloudflare_storage, monkeypatch):
    monkeypatch.setenv("MCP_SEMANTIC_DEDUP_ENABLED", "false")
    fake = _FakeCloudflare(cloudflare_storage)
    monkeypatch.setattr(cloudflare_storage, "_retry_request", fake)
    upsert = AsyncMock()
    monkeypatch.setattr(cloudflare_storage, "_store_vectorize_vector", upsert)
    for name in ("_delete_vectorize_vector", "_store_r2_content", "_delete_r2_content"):
        monkeypatch.setattr(cloudflare_storage, name, AsyncMock())
    monkeypatch.setattr(cloudflare_storage, "_generate_embedding", AsyncMock(return_value=[0.1]))
    await cloudflare_storage._initialize_d1_schema()
    for text, store in ((TEXT_A, "a"), (TEXT_B, "b")):
        memory = Memory(content=text, content_hash=generate_content_hash(text), tags=["zebra"])
        ok, msg = await cloudflare_storage.store(memory, store=store)
        assert ok, msg
    fake.vector_upserts = upsert
    yield fake
    fake.db.close()


@pytest.fixture
def make_client(cloudflare_storage, cloudflare, monkeypatch):
    from mcp_memory_service.web.app import app
    from mcp_memory_service.web.oauth import middleware
    from mcp_memory_service.web.api import mcp
    from mcp_memory_service.web import dependencies

    monkeypatch.setattr(middleware, "API_KEY", None)
    monkeypatch.setattr(middleware, "OAUTH_ENABLED", False)
    monkeypatch.setattr(middleware, "ALLOW_ANONYMOUS_ACCESS", True)
    monkeypatch.setattr(dependencies, "_storage", None)
    set_storage(cloudflare_storage)
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


def _call(client, name, arguments):
    return client.post("/mcp", json={
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    })


def _text(response):
    assert response.status_code == 200, response.text
    body = response.json()
    assert "error" not in body, body
    return " ".join(part.get("text", "") for part in body["result"]["content"])


def test_cloudflare_confines_get_by_hash_to_store():
    assert CloudflareStorage.get_by_hash_is_store_scoped is True


@pytest.mark.integration
def test_read_grant_searches_only_its_store(make_client):
    client = make_client("read:a")
    text = _text(_call(client, "memory_search", {"query": "zebra notes", "store": "a"}))
    assert "rollout" in text
    assert "incident" not in text


@pytest.mark.integration
def test_read_grant_lists_only_its_store(make_client):
    client = make_client("read:a")
    text = _text(_call(client, "memory_list", {"store": "a"}))
    assert "rollout" in text
    assert "incident" not in text


@pytest.mark.integration
def test_read_grant_cannot_name_another_store(make_client):
    client = make_client("read:a")
    response = _call(client, "memory_search", {"query": "zebra notes", "store": "b"})
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_duplicate_in_another_store_is_refused_before_vectorize(cloudflare_storage, cloudflare):
    """Storing store b's content under store a must not touch b's vector."""
    cloudflare.vector_upserts.reset_mock()
    memory = Memory(content=TEXT_B, content_hash=generate_content_hash(TEXT_B), tags=["x"])

    ok, msg = await cloudflare_storage.store(memory, store="a")

    assert not ok
    assert msg.startswith("Duplicate content detected (exact match)")
    cloudflare.vector_upserts.assert_not_called()
    row = cloudflare.row(memory.content_hash)
    assert row["store"] == "b" and row["deleted_at"] is None


@pytest.mark.asyncio
async def test_tombstone_of_another_store_is_not_taken_over(cloudflare_storage, cloudflare):
    """Store a cannot purge store b's tombstone by storing the same content."""
    content_hash = generate_content_hash(TEXT_B)
    ok, msg = await cloudflare_storage.delete(content_hash, store="b")
    assert ok, msg
    cloudflare.vector_upserts.reset_mock()

    memory = Memory(content=TEXT_B, content_hash=content_hash, tags=["x"])
    ok, msg = await cloudflare_storage.store(memory, store="a")

    assert not ok
    assert msg == "Duplicate content detected (exact match); already exists in another store"
    cloudflare.vector_upserts.assert_not_called()
    row = cloudflare.row(content_hash)
    assert row["store"] == "b" and row["deleted_at"] is not None


@pytest.mark.asyncio
async def test_own_tombstone_can_be_stored_again(cloudflare_storage, cloudflare):
    content_hash = generate_content_hash(TEXT_A)
    ok, msg = await cloudflare_storage.delete(content_hash, store="a")
    assert ok, msg

    memory = Memory(content=TEXT_A, content_hash=content_hash, tags=["again"])
    ok, msg = await cloudflare_storage.store(memory, store="a")

    assert ok, msg
    row = cloudflare.row(content_hash)
    assert row["store"] == "a" and row["deleted_at"] is None


@pytest.mark.integration
def test_rest_delete_is_confined_to_the_granted_store(make_client, cloudflare_storage, monkeypatch):
    """The delete itself carries the store, not only the lookup before it."""
    original = cloudflare_storage.delete
    calls = []

    async def delete(content_hash, store=None):
        calls.append(store)
        return await original(content_hash, store=store)

    monkeypatch.setattr(cloudflare_storage, "delete", delete)
    client = make_client("read:a write:a")
    content_hash = generate_content_hash(TEXT_A)

    response = client.delete(f"/api/memories/{content_hash}", params={"store": "a"})

    assert response.status_code == 200, response.text
    assert calls == ["a"]
