"""Cloudflare confines hash-addressed operations to the claimed store (#1878, part of #1683).

get_by_hash, update_memory_metadata and delete looked a memory up by
content_hash alone. D1 now keeps a ``store`` column, store() writes it, and the
three hash-addressed paths add ``AND store = ?`` when a store is named. These
tests run the real SQL against an in-memory SQLite database standing in for D1.
"""

import sqlite3
from unittest.mock import AsyncMock, Mock

import pytest

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.cloudflare import CloudflareStorage
from mcp_memory_service.utils.hashing import generate_content_hash


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


class _FakeD1:
    """Answer D1 /query calls by running the SQL on an in-memory SQLite database."""

    def __init__(self, storage):
        self.storage = storage
        self.db = sqlite3.connect(":memory:")
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.statements = []

    async def __call__(self, method, url, **kwargs):
        assert method == "POST" and url == f"{self.storage.d1_url}/query"
        payload = kwargs["json"]
        if "batch" in payload:
            return Mock(json=Mock(return_value=self._batch(payload["batch"])))
        sql = payload["sql"]
        self.statements.append(sql)
        try:
            if sql.count(";") > 1:
                self.db.executescript(sql)
                rows, row_id = [], 0
            else:
                cursor = self.db.execute(sql, payload.get("params", []))
                rows = [dict(row) for row in cursor.fetchall()]
                row_id = cursor.lastrowid
            self.db.commit()
            result = {"success": True, "result": [{"results": rows, "meta": {"last_row_id": row_id}}]}
        except sqlite3.Error as error:
            result = {"success": False, "errors": [str(error)]}
        return Mock(json=Mock(return_value=result))

    def _batch(self, statements):
        # D1 runs a batch as one transaction and reports one result set per statement.
        result_sets = []
        try:
            for statement in statements:
                self.statements.append(statement["sql"])
                cursor = self.db.execute(statement["sql"], statement.get("params", []))
                rows = [dict(row) for row in cursor.fetchall()]
                result_sets.append({"success": True, "results": rows})
            self.db.commit()
        except sqlite3.Error as error:
            self.db.rollback()
            return {"success": False, "errors": [str(error)]}
        return {"success": True, "result": result_sets}

    def row(self, content_hash):
        return self.db.execute(
            "SELECT * FROM memories WHERE content_hash = ?", (content_hash,)
        ).fetchone()


@pytest.fixture
def d1(cloudflare_storage, monkeypatch):
    fake = _FakeD1(cloudflare_storage)
    monkeypatch.setattr(cloudflare_storage, "_retry_request", fake)
    for name in ("_store_vectorize_vector", "_delete_vectorize_vector",
                 "_store_r2_content", "_delete_r2_content"):
        monkeypatch.setattr(cloudflare_storage, name, AsyncMock())
    monkeypatch.setattr(cloudflare_storage, "_generate_embedding", AsyncMock(return_value=[0.1]))
    yield fake
    fake.db.close()


async def _store(storage, content, store):
    memory = Memory(content=content, content_hash=generate_content_hash(content), tags=["orig"])
    ok, msg = await storage.store(memory, store=store)
    assert ok, msg
    return memory.content_hash


@pytest.mark.asyncio
async def test_store_writes_the_named_store(cloudflare_storage, d1):
    await cloudflare_storage._initialize_d1_schema()
    named = await _store(cloudflare_storage, "Store B keeps the deploy calendar.", "b")
    default = await _store(cloudflare_storage, "Nobody named a store for this one.", "default")

    assert d1.row(named)["store"] == "b"
    assert d1.row(default)["store"] == "default"


@pytest.mark.asyncio
async def test_get_by_hash_from_another_store_is_not_found(cloudflare_storage, d1):
    await cloudflare_storage._initialize_d1_schema()
    h = await _store(cloudflare_storage, "Store B keeps the deploy calendar.", "b")

    assert await cloudflare_storage.get_by_hash(h, store="a") is None

    same = await cloudflare_storage.get_by_hash(h, store="b")
    assert same is not None and same.content_hash == h
    # None and "all" stay unscoped, as on the other backends.
    assert await cloudflare_storage.get_by_hash(h) is not None
    assert await cloudflare_storage.get_by_hash(h, store="all") is not None


@pytest.mark.asyncio
async def test_metadata_update_from_another_store_is_not_found(cloudflare_storage, d1):
    await cloudflare_storage._initialize_d1_schema()
    h = await _store(cloudflare_storage, "Store B keeps the deploy calendar.", "b")
    before = tuple(d1.row(h))

    ok, msg = await cloudflare_storage.update_memory_metadata(
        h, {"tags": ["hijacked"], "metadata": {"owner": "a"}}, store="a"
    )

    assert not ok
    assert "not found" in msg.lower()
    assert tuple(d1.row(h)) == before
    tag_names = [r[0] for r in d1.db.execute(
        "SELECT t.name FROM tags t JOIN memory_tags mt ON t.id = mt.tag_id"
    ).fetchall()]
    assert tag_names == ["orig"]


@pytest.mark.asyncio
async def test_metadata_update_in_the_same_store_applies(cloudflare_storage, d1):
    await cloudflare_storage._initialize_d1_schema()
    h = await _store(cloudflare_storage, "Store B keeps the deploy calendar.", "b")

    ok, msg = await cloudflare_storage.update_memory_metadata(
        h, {"metadata": {"owner": "b"}}, store="b"
    )

    assert ok, msg
    assert '"owner": "b"' in d1.row(h)["metadata_json"]


@pytest.mark.asyncio
async def test_tag_update_in_the_same_store_binds_store_before_tag_params(cloudflare_storage, d1):
    # With tags, the UPDATE carries tag parameters in RETURNING after the store
    # predicate, so the bind order is worth pinning.
    await cloudflare_storage._initialize_d1_schema()
    h = await _store(cloudflare_storage, "Store B keeps the deploy calendar.", "b")

    ok, msg = await cloudflare_storage.update_memory_metadata(h, {"tags": ["renamed"]}, store="b")

    assert ok, msg
    tag_names = [r[0] for r in d1.db.execute(
        "SELECT t.name FROM tags t JOIN memory_tags mt ON t.id = mt.tag_id"
    ).fetchall()]
    assert tag_names == ["renamed"]


@pytest.mark.asyncio
async def test_tombstone_restore_keeps_the_row_store(cloudflare_storage, d1):
    await cloudflare_storage._initialize_d1_schema()
    content = "Store B keeps the deploy calendar."
    h = await _store(cloudflare_storage, content, "b")
    assert (await cloudflare_storage.delete(h, store="b"))[0]
    deleted_at = d1.row(h)["deleted_at"]

    memory = Memory(content=content, content_hash=h, tags=["orig"])
    ok, msg = await cloudflare_storage.store_if_tombstone_matches(memory, deleted_at)

    assert ok, msg
    assert d1.row(h)["deleted_at"] is None
    assert d1.row(h)["store"] == "b"


@pytest.mark.asyncio
async def test_delete_from_another_store_is_refused(cloudflare_storage, d1):
    await cloudflare_storage._initialize_d1_schema()
    h = await _store(cloudflare_storage, "Store B keeps the deploy calendar.", "b")

    ok, msg = await cloudflare_storage.delete(h, store="a")

    assert not ok
    assert "not found" in msg.lower()
    assert d1.row(h)["deleted_at"] is None
    # The vector and R2 object must stay too: they are removed before D1.
    cloudflare_storage._delete_vectorize_vector.assert_not_awaited()
    cloudflare_storage._delete_r2_content.assert_not_awaited()

    ok, msg = await cloudflare_storage.delete(h, store="b")
    assert ok, msg
    assert d1.row(h)["deleted_at"] is not None


_OLD_SCHEMA = """
CREATE TABLE memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content_hash TEXT UNIQUE NOT NULL,
    content TEXT NOT NULL,
    memory_type TEXT,
    created_at REAL NOT NULL,
    created_at_iso TEXT NOT NULL,
    updated_at REAL,
    updated_at_iso TEXT,
    metadata_json TEXT,
    vector_id TEXT UNIQUE,
    content_size INTEGER DEFAULT 0,
    r2_key TEXT,
    tags TEXT,
    deleted_at REAL DEFAULT NULL
);
INSERT INTO memories (content_hash, content, created_at, created_at_iso, vector_id)
VALUES ('pre-existing', 'written before the store column', 1, '1970-01-01T00:00:01Z', 'pre-existing');
"""


@pytest.mark.asyncio
async def test_migration_backfills_default_and_is_idempotent(cloudflare_storage, d1):
    d1.db.executescript(_OLD_SCHEMA)

    await cloudflare_storage._initialize_d1_schema()

    assert d1.row("pre-existing")["store"] == "default"
    assert await cloudflare_storage.get_by_hash("pre-existing", store="default") is not None
    assert await cloudflare_storage.get_by_hash("pre-existing", store="b") is None

    # A database that already has the column is left alone.
    d1.statements.clear()
    await cloudflare_storage._initialize_d1_schema()
    assert not [s for s in d1.statements if "ALTER TABLE" in s.upper()]
    assert d1.row("pre-existing")["store"] == "default"


# --- Read paths confined to the claimed store (#1915 point 1) -----------------
#
# retrieve and recall go through Vectorize, which holds every store's vectors,
# and then load each match from D1 by vector_id; the store predicate sits on
# that load. The other reads are plain D1 SQL.

_CONTENT_A = "Store A keeps the incident runbook."
_CONTENT_B = "Store B keeps the deploy calendar."


@pytest.fixture
async def two_stores(cloudflare_storage, d1, monkeypatch):
    """One memory in store "a", one in store "b", both tagged "shared"."""
    await cloudflare_storage._initialize_d1_schema()
    hashes = {}
    for store, content in (("a", _CONTENT_A), ("b", _CONTENT_B)):
        memory = Memory(content=content, content_hash=generate_content_hash(content),
                        tags=["shared", f"only-{store}"], memory_type="note")
        ok, msg = await cloudflare_storage.store(memory, store=store)
        assert ok, msg
        hashes[store] = memory.content_hash

    # Vectorize knows nothing about stores: every query matches both vectors.
    d1_request = cloudflare_storage._retry_request

    async def request(method, url, **kwargs):
        if url == f"{cloudflare_storage.vectorize_url}/query":
            matches = [{"id": h, "score": 0.9} for h in hashes.values()]
            return Mock(json=Mock(return_value={"success": True, "result": {"matches": matches}}))
        return await d1_request(method, url, **kwargs)

    monkeypatch.setattr(cloudflare_storage, "_retry_request", request)
    monkeypatch.setattr(cloudflare_storage, "_persist_access_metadata", AsyncMock())
    return hashes


def _hashes(items):
    return {getattr(item, "memory", item).content_hash for item in items}


@pytest.mark.asyncio
async def test_retrieve_loads_only_matches_from_the_claimed_store(cloudflare_storage, two_stores):
    a, b = two_stores["a"], two_stores["b"]

    assert _hashes(await cloudflare_storage.retrieve("runbook", n_results=5, store="a")) == {a}
    assert _hashes(await cloudflare_storage.retrieve("runbook", n_results=5, store="b")) == {b}
    assert _hashes(await cloudflare_storage.retrieve("runbook", n_results=5, store=None)) == {a, b}
    assert _hashes(await cloudflare_storage.retrieve("runbook", n_results=5, store="all")) == {a, b}


@pytest.mark.asyncio
async def test_recall_is_confined_on_both_paths(cloudflare_storage, two_stores):
    a, b = two_stores["a"], two_stores["b"]

    # Semantic path: Vectorize, then the D1 load by vector_id.
    assert _hashes(await cloudflare_storage.recall("runbook", store="a")) == {a}
    # Time-only path, with and without a window: plain D1.
    assert _hashes(await cloudflare_storage.recall(None, store="b")) == {b}
    assert _hashes(await cloudflare_storage.recall(None, start_timestamp=0, store="b")) == {b}
    assert _hashes(await cloudflare_storage.recall(None)) == {a, b}
    assert _hashes(await cloudflare_storage.recall(None, store="all")) == {a, b}


@pytest.mark.asyncio
async def test_listing_and_counting_are_confined(cloudflare_storage, two_stores):
    a, b = two_stores["a"], two_stores["b"]

    assert _hashes(await cloudflare_storage.get_all_memories(store="a")) == {a}
    assert _hashes(await cloudflare_storage.get_all_memories(tags=["shared"], store="b")) == {b}
    assert _hashes(await cloudflare_storage.get_all_memories(store=None)) == {a, b}
    assert _hashes(await cloudflare_storage.get_all_memories(store="all")) == {a, b}

    assert await cloudflare_storage.count_all_memories(store="a") == 1
    assert await cloudflare_storage.count_all_memories(tags=["shared"], store="b") == 1
    assert await cloudflare_storage.count_all_memories(store=None) == 2
    assert await cloudflare_storage.count_all_memories(store="all") == 2


@pytest.mark.asyncio
async def test_time_range_and_exact_content_are_confined(cloudflare_storage, two_stores):
    a, b = two_stores["a"], two_stores["b"]

    window = (0, 4_000_000_000)
    assert _hashes(await cloudflare_storage.get_memories_by_time_range(*window, store="a")) == {a}
    assert _hashes(await cloudflare_storage.get_memories_by_time_range(*window)) == {a, b}

    # "keeps the" is in both memories.
    assert _hashes(await cloudflare_storage.get_by_exact_content("keeps the", store="b")) == {b}
    assert _hashes(await cloudflare_storage.get_by_exact_content("keeps the", store="all")) == {a, b}


@pytest.mark.asyncio
async def test_tag_counts_and_timestamps_are_confined(cloudflare_storage, two_stores):
    counts = {row["tag"]: row["count"] for row in
              await cloudflare_storage.get_all_tags_with_counts(store="a")}
    assert counts == {"shared": 1, "only-a": 1}

    unscoped = {row["tag"]: row["count"] for row in
                await cloudflare_storage.get_all_tags_with_counts()}
    assert unscoped == {"shared": 2, "only-a": 1, "only-b": 1}

    assert len(await cloudflare_storage.get_memory_timestamps(store="a")) == 1
    assert len(await cloudflare_storage.get_memory_timestamps(days=1, store="a")) == 1
    assert len(await cloudflare_storage.get_memory_timestamps()) == 2


@pytest.mark.asyncio
async def test_delete_by_tags_leaves_other_stores_alone(cloudflare_storage, d1, two_stores):
    a, b = two_stores["a"], two_stores["b"]

    count, _, deleted = await cloudflare_storage.delete_by_tags(["shared"], store="a")

    assert (count, deleted) == (1, [a])
    assert d1.row(a)["deleted_at"] is not None
    assert d1.row(b)["deleted_at"] is None


@pytest.mark.asyncio
async def test_omitted_store_reads_default_only(cloudflare_storage, two_stores):
    # retrieve, get_all_memories and count_all_memories default to "default",
    # as on sqlite-vec; only None and "all" read every store.
    default = await _store(cloudflare_storage, "Nobody named a store for this one.", "default")
    two_stores["default"] = default  # the fake Vectorize matches it too

    assert _hashes(await cloudflare_storage.retrieve("anything", n_results=5)) == {default}
    assert _hashes(await cloudflare_storage.get_all_memories()) == {default}
    assert await cloudflare_storage.count_all_memories() == 1


@pytest.mark.asyncio
async def test_store_binds_in_order_with_every_other_filter(cloudflare_storage, two_stores):
    a = two_stores["a"]
    filters = dict(memory_type="note", tags=["shared", "only-a"], tag_match="all")

    assert _hashes(await cloudflare_storage.get_all_memories(
        limit=5, offset=0, store="a", **filters)) == {a}
    assert await cloudflare_storage.count_all_memories(store="a", **filters) == 1
    # Store "b" holds a "note" tagged "shared" but not "only-a".
    assert await cloudflare_storage.count_all_memories(store="b", **filters) == 0

    # The store is a bound parameter, never part of the SQL text.
    hostile = "a' OR '1'='1"
    assert await cloudflare_storage.get_all_memories(store=hostile, **filters) == []
    assert await cloudflare_storage.count_all_memories(store=hostile, **filters) == 0
