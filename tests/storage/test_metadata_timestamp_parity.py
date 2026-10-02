"""Timestamp semantics must agree between sqlite-vec and Cloudflare D1."""

import json
import sqlite3
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.cloudflare import CloudflareStorage
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash


def timestamp_fields(db, content_hash):
    """Read persisted timestamps rather than inspect how an update was built."""
    return tuple(
        db.execute(
            "SELECT created_at, created_at_iso, updated_at, updated_at_iso "
            "FROM memories WHERE content_hash = ?",
            (content_hash,),
        ).fetchone()
    )


@pytest_asyncio.fixture(params=["sqlite_vec", "cloudflare"])
async def metadata_storage(request, temp_db_path, monkeypatch):
    """Use real storage methods, replacing only D1's HTTP boundary with SQLite."""
    if request.param == "sqlite_vec":
        storage = SqliteVecMemoryStorage(f"{temp_db_path}/metadata.db")
        await storage.initialize()
        db = storage.conn
        metadata_column = "metadata"
    else:
        storage = CloudflareStorage(
            api_token="test-token",
            account_id="test-account",
            vectorize_index="test-index",
            d1_database_id="test-db",
        )
        db = sqlite3.connect(":memory:")
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        metadata_column = "metadata_json"

        async def query_d1(method, url, **kwargs):
            """Execute the backend's SQL and return the D1 response envelope."""
            assert method == "POST" and url == f"{storage.d1_url}/query"
            payload = kwargs["json"]
            try:
                if payload["sql"].count(";") > 1:
                    db.executescript(payload["sql"])
                    rows, row_id = [], 0
                else:
                    cursor = db.execute(payload["sql"], payload.get("params", []))
                    rows = [dict(row) for row in cursor.fetchall()]
                    row_id = cursor.lastrowid
                db.commit()
                result = {
                    "success": True,
                    "result": [{"results": rows, "meta": {"last_row_id": row_id}}],
                }
            except sqlite3.Error as error:
                result = {"success": False, "errors": [str(error)]}
            return httpx.Response(200, json=result)

        monkeypatch.setattr(storage, "_retry_request", query_d1)
        await storage._initialize_d1_schema()

    created_at, updated_at = 1700000000.0, 1710000000.0
    memory = Memory(
        content="Memory whose metadata will be annotated",
        content_hash=generate_content_hash("Memory whose metadata will be annotated"),
        tags=["original"],
        memory_type="observation",
        metadata={"source": "original"},
        created_at=created_at,
        created_at_iso=datetime.fromtimestamp(created_at, timezone.utc).isoformat(),
        updated_at=updated_at,
        updated_at_iso=datetime.fromtimestamp(updated_at, timezone.utc).isoformat(),
    )
    try:
        if request.param == "sqlite_vec":
            success, message = await storage.store(memory)
            assert success, message
        else:
            await storage._store_d1_memory(
                memory, "test-vector", len(memory.content), None, memory.content
            )
        yield storage, db, memory.content_hash, metadata_column
    finally:
        await storage.close()
        if request.param == "cloudflare":
            db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("use_default", [True, False])
async def test_metadata_only_preserves_timestamps(metadata_storage, use_default):
    """Annotations must persist without making an old memory appear recent."""
    storage, db, content_hash, metadata_column = metadata_storage
    before = timestamp_fields(db, content_hash)
    options = {} if use_default else {"preserve_timestamps": True}

    success, message = await storage.update_memory_metadata(
        content_hash, {"metadata": {"annotated": True}}, **options
    )

    assert success, message
    assert timestamp_fields(db, content_hash) == before
    metadata = db.execute(
        f"SELECT {metadata_column} FROM memories WHERE content_hash = ?",
        (content_hash,),
    ).fetchone()[0]
    assert json.loads(metadata)["annotated"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "updates",
    [{"tags": []}, {"memory_type": "decision"}, {"content": "replacement"}],
)
async def test_structural_updates_advance_updated_at(metadata_storage, updates):
    """The preservation flag still allows structural fields to advance time."""
    storage, db, content_hash, _ = metadata_storage
    before = timestamp_fields(db, content_hash)
    started_at = time.time()

    success, message = await storage.update_memory_metadata(
        content_hash, updates, preserve_timestamps=True
    )

    assert success, message
    after = timestamp_fields(db, content_hash)
    assert after[:2] == before[:2]
    assert started_at <= after[2] <= time.time()
    assert after[3] != before[3]


@pytest.mark.asyncio
async def test_metadata_without_preservation_advances_time(metadata_storage):
    """Callers can still opt into a fresh update time for metadata changes."""
    storage, db, content_hash, _ = metadata_storage
    before = timestamp_fields(db, content_hash)
    started_at = time.time()

    success, message = await storage.update_memory_metadata(
        content_hash, {"metadata": {"annotated": True}}, preserve_timestamps=False
    )

    assert success, message
    after = timestamp_fields(db, content_hash)
    assert after[:2] == before[:2]
    assert started_at <= after[2] <= time.time()
    assert after[3] != before[3]


@pytest.mark.asyncio
@pytest.mark.parametrize("preserve_timestamps", [True, False])
async def test_explicit_sync_timestamps(metadata_storage, preserve_timestamps):
    """Source timestamps are applied only when preservation is disabled."""
    storage, db, content_hash, _ = metadata_storage
    before = timestamp_fields(db, content_hash)
    source = (
        1600000000.0,
        "2020-09-13T12:26:40Z",
        1650000000.0,
        "2022-04-15T05:20:00Z",
    )
    updates = dict(
        zip(("created_at", "created_at_iso", "updated_at", "updated_at_iso"), source)
    )
    updates["metadata"] = {"synced": True}

    success, message = await storage.update_memory_metadata(
        content_hash, updates, preserve_timestamps=preserve_timestamps
    )

    assert success, message
    assert timestamp_fields(db, content_hash) == (
        before if preserve_timestamps else source
    )


@pytest.mark.asyncio
async def test_cloudflare_reports_failed_metadata_update(monkeypatch):
    """A D1 network failure must remain a failed update, not a successful no-op."""
    storage = CloudflareStorage("test-token", "test-account", "test-index", "test-db")
    monkeypatch.setattr(
        storage,
        "_retry_request",
        AsyncMock(side_effect=httpx.ConnectError("D1 unavailable")),
    )

    success, message = await storage.update_memory_metadata(
        "test-hash", {"metadata": {"annotated": True}}, preserve_timestamps=True
    )

    assert not success
    assert "D1 unavailable" in message
