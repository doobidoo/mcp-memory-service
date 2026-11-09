"""Access write-back must not overwrite concurrent metadata updates (#1464).

retrieve() reads a row, calls record_access() on that snapshot and then writes
the access metadata back. The write-back used to replace the whole metadata
JSON with the snapshot, so a key that update_memory_metadata() committed in
between was lost. It now writes only access_count, last_accessed_at and
access_queries. The interleaving is forced by call order.
"""

import json
import sqlite3
from unittest.mock import Mock

import pytest

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.cloudflare import CloudflareStorage
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash


async def _stored(storage, content):
    content_hash = generate_content_hash(content)
    ok, message = await storage.store(
        Memory(content=content, content_hash=content_hash, tags=["t"])
    )
    assert ok, message
    return content_hash


def _stored_metadata(storage, content_hash):
    row = storage.conn.execute(
        "SELECT metadata, last_accessed FROM memories WHERE content_hash = ?",
        (content_hash,),
    ).fetchone()
    return json.loads(row[0]), row[1]


@pytest.mark.asyncio
async def test_access_writeback_keeps_concurrent_metadata_update(temp_db_path, unique_content):
    storage = SqliteVecMemoryStorage(f"{temp_db_path}/test.db")
    await storage.initialize()
    try:
        content_hash = await _stored(storage, unique_content("Release checklist for staging"))

        # retrieve(): snapshot + record_access
        snapshot = await storage.get_by_hash(content_hash)
        snapshot.record_access("release checklist")

        # A metadata update commits before the write-back.
        ok, message = await storage.update_memory_metadata(
            content_hash, {"metadata": {"reviewed": True}}
        )
        assert ok, message

        await storage._persist_access_metadata_batch([snapshot])

        metadata, last_accessed = _stored_metadata(storage, content_hash)
        assert metadata.get("reviewed") is True, (
            f"#1464: concurrent metadata key lost by access write-back: {sorted(metadata)}"
        )
        assert metadata["access_count"] == snapshot.metadata["access_count"] == 1
        assert metadata["last_accessed_at"] == snapshot.metadata["last_accessed_at"]
        assert metadata["access_queries"] == snapshot.metadata["access_queries"]
        assert last_accessed == int(snapshot.metadata["last_accessed_at"])
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_retrieve_writeback_keeps_metadata_written_after_snapshot(
    temp_db_path, unique_content
):
    """A snapshot without access_queries leaves stored access_queries alone."""
    storage = SqliteVecMemoryStorage(f"{temp_db_path}/test.db")
    await storage.initialize()
    try:
        content_hash = await _stored(storage, unique_content("Rotation policy for API keys"))

        snapshot = await storage.get_by_hash(content_hash)
        snapshot.record_access()  # no query: no access_queries key
        assert "access_queries" not in snapshot.metadata

        ok, message = await storage.update_memory_metadata(
            content_hash,
            {"metadata": {"owner": "ops", "access_queries": [{"query": "q", "timestamp": 1.0}]}},
        )
        assert ok, message

        await storage._persist_access_metadata_batch([snapshot])

        metadata, _ = _stored_metadata(storage, content_hash)
        assert metadata["owner"] == "ops"
        assert metadata["access_queries"] == [{"query": "q", "timestamp": 1.0}]
        assert metadata["access_count"] == 1
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("stored", ["[1, 2]", "not json", None])
async def test_access_writeback_replaces_non_object_metadata(
    temp_db_path, unique_content, stored
):
    """Non-object or invalid stored metadata falls back to {} like _safe_json_loads."""
    storage = SqliteVecMemoryStorage(f"{temp_db_path}/test.db")
    await storage.initialize()
    try:
        content_hash = await _stored(storage, unique_content("Backup window"))
        snapshot = await storage.get_by_hash(content_hash)
        snapshot.record_access("backup")
        storage.conn.execute(
            "UPDATE memories SET metadata = ? WHERE content_hash = ?", (stored, content_hash)
        )
        storage.conn.commit()

        await storage._persist_access_metadata_batch([snapshot])

        metadata, _ = _stored_metadata(storage, content_hash)
        assert metadata == {
            "access_count": 1,
            "last_accessed_at": snapshot.metadata["last_accessed_at"],
            "access_queries": snapshot.metadata["access_queries"],
        }
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_access_writeback_skips_soft_deleted(temp_db_path, unique_content):
    storage = SqliteVecMemoryStorage(f"{temp_db_path}/test.db")
    await storage.initialize()
    try:
        content_hash = await _stored(storage, unique_content("Deprecated runbook"))
        snapshot = await storage.get_by_hash(content_hash)
        snapshot.record_access("runbook")
        storage.conn.execute(
            "UPDATE memories SET deleted_at = 1.0 WHERE content_hash = ?", (content_hash,)
        )
        storage.conn.commit()

        await storage._persist_access_metadata_batch([snapshot])

        metadata, last_accessed = _stored_metadata(storage, content_hash)
        assert "access_count" not in metadata
        assert last_accessed is None
    finally:
        await storage.close()


class _FakeD1:
    """Answer D1 /query calls by running the SQL on an in-memory SQLite database."""

    def __init__(self, d1_url):
        self.d1_url = d1_url
        self.db = sqlite3.connect(":memory:")
        self.db.execute(
            "CREATE TABLE memories (content_hash TEXT PRIMARY KEY, metadata_json TEXT)"
        )

    async def __call__(self, method, url, **kwargs):
        assert method == "POST" and url == f"{self.d1_url}/query"
        payload = kwargs["json"]
        self.db.execute(payload["sql"], payload.get("params", []))
        self.db.commit()
        return Mock(json=Mock(return_value={"success": True, "result": [{"results": []}]}))

    def metadata(self, content_hash):
        row = self.db.execute(
            "SELECT metadata_json FROM memories WHERE content_hash = ?", (content_hash,)
        ).fetchone()
        return json.loads(row[0])


@pytest.mark.asyncio
async def test_cloudflare_access_writeback_keeps_concurrent_metadata_update():
    storage = CloudflareStorage(
        api_token="test-token",
        account_id="test-account",
        vectorize_index="test-index",
        d1_database_id="test-db",
        r2_bucket="test-bucket",
        embedding_model="@cf/baai/bge-base-en-v1.5",
    )
    d1 = _FakeD1(storage.d1_url)
    storage._retry_request = d1

    content = "Incident review notes"
    content_hash = generate_content_hash(content)
    snapshot = Memory(content=content, content_hash=content_hash, tags=["t"])
    snapshot.record_access("incident")
    # Stored row already carries a key written after the snapshot was read.
    d1.db.execute(
        "INSERT INTO memories VALUES (?, ?)", (content_hash, json.dumps({"reviewed": True}))
    )

    await storage._persist_access_metadata(snapshot)

    metadata = d1.metadata(content_hash)
    assert metadata.get("reviewed") is True, (
        f"#1464: Cloudflare access write-back lost a metadata key: {sorted(metadata)}"
    )
    assert metadata["access_count"] == 1
    assert metadata["access_queries"] == snapshot.metadata["access_queries"]
