"""Regression tests for issue #1538 tombstone conflict resolution."""

import asyncio
import sqlite3
import time
from types import SimpleNamespace

import pytest

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.cloudflare import CloudflareStorage
from mcp_memory_service.storage.hybrid import (
    HybridMemoryStorage,
    _reconcile_remote_tombstones,
)
from mcp_memory_service.utils.hashing import generate_content_hash


class CloudflareStub:
    """Small Cloudflare boundary fake; hybrid and SQLite logic stays real."""

    def __init__(self):
        self.tombstones = {}
        self.stored = []

    async def get_tombstone_for_hash(self, content_hash):
        return self.tombstones.get(content_hash)

    async def store(self, memory):
        self.stored.append(memory)
        self.tombstones.pop(memory.content_hash, None)
        return True, "stored"

    async def store_if_tombstone_matches(self, memory, expected_deleted_at):
        if self.tombstones.get(memory.content_hash) != expected_deleted_at:
            return None, "tombstone_changed"
        return await self.store(memory)


async def _make_hybrid(temp_db_path):
    hybrid = HybridMemoryStorage(f"{temp_db_path}/primary.db")
    await hybrid.primary.initialize()
    hybrid.secondary = CloudflareStub()
    hybrid.secondary_backend_type = "cloudflare"
    return hybrid


async def _store_local_memory(primary, unique_content, updated_at):
    content = unique_content("local memory newer than a remote tombstone")
    memory = Memory(
        content=content,
        content_hash=generate_content_hash(content),
        tags=["__test__"],
        created_at=updated_at,
        updated_at=updated_at,
    )
    success, message = await primary.store(memory, skip_semantic_dedup=True)
    assert success, message
    return memory


@pytest.mark.asyncio
async def test_tombstone_reconciliation_returns_conflicting_hashes(
    temp_db_path, unique_content
):
    hybrid = await _make_hybrid(temp_db_path)
    try:
        local = await _store_local_memory(
            hybrid.primary, unique_content, time.time() - 10
        )

        result = await _reconcile_remote_tombstones(
            hybrid.primary, {local.content_hash: local.updated_at - 1}
        )

        assert result["tombstone_conflicts"] == 1
        assert result["tombstone_conflict_hashes"] == [local.content_hash]
    finally:
        await hybrid.primary.close()


@pytest.mark.asyncio
async def test_keep_local_pushes_current_local_memory_and_clears_remote_tombstone(
    temp_db_path, unique_content
):
    hybrid = await _make_hybrid(temp_db_path)
    try:
        local = await _store_local_memory(
            hybrid.primary, unique_content, time.time() - 10
        )
        deleted_at = local.updated_at - 1
        hybrid.secondary.tombstones[local.content_hash] = deleted_at

        result = await hybrid.resolve_tombstone_conflict(
            local.content_hash, "keep_local"
        )

        assert result["success"] is True
        assert result["status"] == "resolved"
        assert result["action"] == "keep_local"
        assert hybrid.secondary.tombstones == {}
        assert len(hybrid.secondary.stored) == 1
        pushed = hybrid.secondary.stored[0]
        assert pushed.content_hash == local.content_hash
        assert pushed.content == local.content
        assert pushed.created_at == local.created_at
        assert pushed.created_at_iso == local.created_at_iso
        assert pushed.updated_at == local.updated_at
        assert pushed.updated_at_iso == local.updated_at_iso
        assert (await hybrid.primary.get_sync_state_for_hash(local.content_hash))[
            1
        ] is None
    finally:
        await hybrid.primary.close()


@pytest.mark.asyncio
async def test_accept_delete_soft_deletes_local_using_conditional_timestamp(
    temp_db_path, unique_content
):
    hybrid = await _make_hybrid(temp_db_path)
    try:
        local = await _store_local_memory(
            hybrid.primary, unique_content, time.time() - 10
        )
        hybrid.secondary.tombstones[local.content_hash] = local.updated_at - 1

        result = await hybrid.resolve_tombstone_conflict(
            local.content_hash, "accept_delete"
        )

        assert result["success"] is True
        assert result["status"] == "resolved"
        assert result["action"] == "accept_delete"
        assert hybrid.secondary.stored == []
        _, deleted_at = await hybrid.primary.get_sync_state_for_hash(local.content_hash)
        assert deleted_at is not None
        assert await hybrid.primary.get_by_hash(local.content_hash) is None
    finally:
        await hybrid.primary.close()


@pytest.mark.asyncio
async def test_accept_delete_preserves_a_local_edit_that_races_resolution(
    temp_db_path, unique_content, monkeypatch
):
    hybrid = await _make_hybrid(temp_db_path)
    try:
        local = await _store_local_memory(
            hybrid.primary, unique_content, time.time() - 10
        )
        hybrid.secondary.tombstones[local.content_hash] = local.updated_at - 1
        original_delete = hybrid.primary.delete_if_not_updated_after

        async def edit_before_conditional_delete(content_hash, expected_updated_at):
            hybrid.primary.conn.execute(
                "UPDATE memories SET updated_at = ? WHERE content_hash = ?",
                (expected_updated_at + 1, content_hash),
            )
            hybrid.primary.conn.commit()
            return await original_delete(content_hash, expected_updated_at)

        monkeypatch.setattr(
            hybrid.primary,
            "delete_if_not_updated_after",
            edit_before_conditional_delete,
        )

        result = await hybrid.resolve_tombstone_conflict(
            local.content_hash, "accept_delete"
        )

        assert result["success"] is False
        assert result["status"] == "conflict_changed"
        updated_at, deleted_at = await hybrid.primary.get_sync_state_for_hash(
            local.content_hash
        )
        assert updated_at == pytest.approx(local.updated_at + 1)
        assert deleted_at is None
    finally:
        await hybrid.primary.close()


@pytest.mark.asyncio
async def test_resolution_is_noop_if_remote_tombstone_is_already_gone(
    temp_db_path, unique_content
):
    hybrid = await _make_hybrid(temp_db_path)
    try:
        local = await _store_local_memory(
            hybrid.primary, unique_content, time.time() - 10
        )

        result = await hybrid.resolve_tombstone_conflict(
            local.content_hash, "keep_local"
        )

        assert result["success"] is True
        assert result["status"] == "no_conflict"
        assert hybrid.secondary.stored == []
        assert (await hybrid.primary.get_sync_state_for_hash(local.content_hash))[
            1
        ] is None
    finally:
        await hybrid.primary.close()


@pytest.mark.asyncio
async def test_resolution_is_noop_when_remote_tombstone_is_not_older_than_local(
    temp_db_path, unique_content
):
    hybrid = await _make_hybrid(temp_db_path)
    try:
        local = await _store_local_memory(
            hybrid.primary, unique_content, time.time() - 10
        )
        hybrid.secondary.tombstones[local.content_hash] = local.updated_at + 1

        result = await hybrid.resolve_tombstone_conflict(
            local.content_hash, "accept_delete"
        )

        assert result["success"] is True
        assert result["status"] == "no_conflict"
        assert hybrid.secondary.stored == []
        assert (await hybrid.primary.get_sync_state_for_hash(local.content_hash))[
            1
        ] is None
    finally:
        await hybrid.primary.close()


@pytest.mark.asyncio
async def test_keep_local_rechecks_remote_tombstone_before_upload(
    temp_db_path, unique_content, monkeypatch
):
    hybrid = await _make_hybrid(temp_db_path)
    try:
        local = await _store_local_memory(
            hybrid.primary, unique_content, time.time() - 10
        )
        hybrid.secondary.tombstones[local.content_hash] = local.updated_at - 1
        original_get_by_hash = hybrid.primary.get_by_hash

        async def clear_tombstone_during_snapshot(content_hash, store=None):
            memory = await original_get_by_hash(content_hash, store=store)
            hybrid.secondary.tombstones.pop(content_hash, None)
            return memory

        monkeypatch.setattr(
            hybrid.primary, "get_by_hash", clear_tombstone_during_snapshot
        )

        result = await hybrid.resolve_tombstone_conflict(
            local.content_hash, "keep_local"
        )

        assert result["success"] is True
        assert result["status"] == "no_conflict"
        assert hybrid.secondary.stored == []
    finally:
        await hybrid.primary.close()


@pytest.mark.asyncio
async def test_keep_local_and_accept_delete_are_serialized_for_same_conflict(
    temp_db_path, unique_content, monkeypatch
):
    hybrid = await _make_hybrid(temp_db_path)
    try:
        local = await _store_local_memory(
            hybrid.primary, unique_content, time.time() - 10
        )
        hybrid.secondary.tombstones[local.content_hash] = local.updated_at - 1
        original_store = hybrid.secondary.store
        store_started = asyncio.Event()
        finish_store = asyncio.Event()

        async def pause_remote_store(memory):
            store_started.set()
            await finish_store.wait()
            return await original_store(memory)

        monkeypatch.setattr(hybrid.secondary, "store", pause_remote_store)
        keep_task = asyncio.create_task(
            hybrid.resolve_tombstone_conflict(local.content_hash, "keep_local")
        )
        delete_task = None
        try:
            await store_started.wait()
            delete_task = asyncio.create_task(
                hybrid.resolve_tombstone_conflict(local.content_hash, "accept_delete")
            )
            await asyncio.sleep(0)

            assert not delete_task.done()
        finally:
            finish_store.set()

        keep_result = await keep_task
        delete_result = await delete_task

        assert keep_result["status"] == "resolved"
        assert delete_result["status"] == "no_conflict"
        assert await hybrid.primary.get_by_hash(local.content_hash) is not None
        assert hybrid.secondary.tombstones == {}
        assert len(hybrid.secondary.stored) == 1
    finally:
        await hybrid.primary.close()


@pytest.mark.asyncio
async def test_keep_local_and_accept_delete_are_serialized_across_hybrid_instances(
    temp_db_path, unique_content, monkeypatch
):
    keep_hybrid = await _make_hybrid(temp_db_path)
    delete_hybrid = await _make_hybrid(temp_db_path)
    await delete_hybrid.primary.initialize()
    delete_hybrid.secondary = keep_hybrid.secondary
    try:
        local = await _store_local_memory(
            keep_hybrid.primary, unique_content, time.time() - 10
        )
        deleted_at = local.updated_at - 1
        keep_hybrid.secondary.tombstones[local.content_hash] = deleted_at
        store_started = asyncio.Event()
        finish_store = asyncio.Event()
        original_store = keep_hybrid.secondary.store_if_tombstone_matches

        async def pause_remote_store(memory, expected_deleted_at):
            store_started.set()
            await finish_store.wait()
            return await original_store(memory, expected_deleted_at)

        monkeypatch.setattr(
            keep_hybrid.secondary,
            "store_if_tombstone_matches",
            pause_remote_store,
        )
        keep_task = asyncio.create_task(
            keep_hybrid.resolve_tombstone_conflict(local.content_hash, "keep_local")
        )
        delete_task = None
        try:
            await store_started.wait()
            delete_task = asyncio.create_task(
                delete_hybrid.resolve_tombstone_conflict(
                    local.content_hash, "accept_delete"
                )
            )
            await asyncio.sleep(0)

            assert not delete_task.done()
        finally:
            finish_store.set()

        keep_result, delete_result = await asyncio.gather(keep_task, delete_task)

        assert keep_result["status"] == "resolved"
        assert delete_result["status"] == "no_conflict"
        assert await keep_hybrid.primary.get_by_hash(local.content_hash) is not None
        assert keep_hybrid.secondary.tombstones == {}
        assert len(keep_hybrid.secondary.stored) == 1
    finally:
        await keep_hybrid.primary.close()
        await delete_hybrid.primary.close()


@pytest.mark.asyncio
async def test_keep_local_does_not_overwrite_a_changed_remote_tombstone(
    temp_db_path, unique_content, monkeypatch
):
    hybrid = await _make_hybrid(temp_db_path)
    try:
        local = await _store_local_memory(
            hybrid.primary, unique_content, time.time() - 10
        )
        hybrid.secondary.tombstones[local.content_hash] = local.updated_at - 1
        original_conditional_store = hybrid.secondary.store_if_tombstone_matches

        async def change_tombstone_then_store(memory, expected_deleted_at):
            hybrid.secondary.tombstones[memory.content_hash] = expected_deleted_at + 1
            return await original_conditional_store(memory, expected_deleted_at)

        monkeypatch.setattr(
            hybrid.secondary,
            "store_if_tombstone_matches",
            change_tombstone_then_store,
        )

        result = await hybrid.resolve_tombstone_conflict(
            local.content_hash, "keep_local"
        )

        assert result["success"] is False
        assert result["status"] == "conflict_changed"
        assert hybrid.secondary.stored == []
        assert await hybrid.primary.get_by_hash(local.content_hash) is not None
    finally:
        await hybrid.primary.close()


@pytest.mark.asyncio
async def test_remote_tombstone_read_error_fails_closed_without_local_mutation(
    temp_db_path, unique_content, monkeypatch
):
    hybrid = await _make_hybrid(temp_db_path)
    try:
        local = await _store_local_memory(
            hybrid.primary, unique_content, time.time() - 10
        )

        async def fail_tombstone_read(content_hash):
            raise RuntimeError("D1 unavailable")

        monkeypatch.setattr(
            hybrid.secondary, "get_tombstone_for_hash", fail_tombstone_read
        )

        with pytest.raises(RuntimeError, match="D1 unavailable"):
            await hybrid.resolve_tombstone_conflict(local.content_hash, "accept_delete")

        assert (await hybrid.primary.get_sync_state_for_hash(local.content_hash))[
            1
        ] is None
    finally:
        await hybrid.primary.close()


@pytest.mark.asyncio
async def test_keep_local_reloads_if_local_memory_changes_during_read(
    temp_db_path, unique_content, monkeypatch
):
    hybrid = await _make_hybrid(temp_db_path)
    try:
        local = await _store_local_memory(
            hybrid.primary, unique_content, time.time() - 10
        )
        hybrid.secondary.tombstones[local.content_hash] = local.updated_at - 1
        original_get_by_hash = hybrid.primary.get_by_hash
        calls = 0

        async def edit_after_first_read(content_hash, store=None):
            nonlocal calls
            memory = await original_get_by_hash(content_hash, store=store)
            calls += 1
            if calls == 1:
                hybrid.primary.conn.execute(
                    "UPDATE memories SET updated_at = ? WHERE content_hash = ?",
                    (local.updated_at + 1, content_hash),
                )
                hybrid.primary.conn.commit()
            return memory

        monkeypatch.setattr(hybrid.primary, "get_by_hash", edit_after_first_read)

        result = await hybrid.resolve_tombstone_conflict(
            local.content_hash, "keep_local"
        )

        assert result["success"] is True
        assert result["status"] == "resolved"
        assert len(hybrid.secondary.stored) == 1
        assert hybrid.secondary.stored[0].updated_at == pytest.approx(
            local.updated_at + 1
        )
    finally:
        await hybrid.primary.close()


@pytest.mark.asyncio
async def test_cloudflare_tombstone_lookup_uses_hash_parameter_and_reads_deleted_at(
    monkeypatch,
):
    storage = CloudflareStorage.__new__(CloudflareStorage)
    storage.d1_url = "https://example.invalid/d1/database/test"
    seen = {}

    async def fake_request(method, url, **kwargs):
        seen.update(method=method, url=url, **kwargs)
        return SimpleNamespace(
            json=lambda: {
                "success": True,
                "result": [{"results": [{"deleted_at": 123.5}]}],
            }
        )

    monkeypatch.setattr(storage, "_retry_request", fake_request)

    deleted_at = await storage.get_tombstone_for_hash("hash-with-?&-characters")

    assert deleted_at == 123.5
    assert seen["json"]["sql"] == (
        "SELECT deleted_at FROM memories WHERE content_hash = ? LIMIT 1"
    )
    assert seen["json"]["params"] == ["hash-with-?&-characters"]


@pytest.mark.asyncio
async def test_cloudflare_conditional_store_does_not_insert_after_tombstone_changes(
    monkeypatch,
):
    storage = CloudflareStorage.__new__(CloudflareStorage)
    storage.d1_url = "https://example.invalid/d1/database/test"
    calls = []

    async def changed_tombstone_request(method, url, **kwargs):
        calls.append(kwargs["json"])
        return SimpleNamespace(
            json=lambda: {
                "success": True,
                "result": [
                    {"success": True, "meta": {"changes": 0}, "results": []},
                    {"success": True, "meta": {"changes": 0}, "results": []},
                    {"success": True, "results": []},
                    {"success": True, "meta": {"changes": 0}, "results": []},
                    {"success": True, "results": []},
                ],
            }
        )

    monkeypatch.setattr(storage, "_retry_request", changed_tombstone_request)
    memory = Memory(
        content="conditional tombstone test",
        content_hash="hash-123",
        created_at=100.0,
        updated_at=101.0,
        tags=[],
    )

    stored = await storage._store_d1_memory(
        memory,
        vector_id="hash-123",
        content_size=len(memory.content),
        r2_key=None,
        stored_content=memory.content,
        expected_deleted_at=99.0,
    )

    assert stored is False
    assert len(calls) == 1
    batch = calls[0]["batch"]
    assert "deleted_at = ?" in batch[3]["sql"]
    assert batch[3]["params"][-4:-2] == ["hash-123", 99.0]
    assert len(batch) == 5
    assert "json_each" in batch[1]["sql"]
    assert "json_each" in batch[2]["sql"]
    assert "json_each" in batch[3]["sql"]


@pytest.mark.asyncio
async def test_cloudflare_rejected_tombstone_restore_cleans_staged_remote_artifacts(
    monkeypatch,
):
    storage = CloudflareStorage.__new__(CloudflareStorage)
    storage.r2_bucket = "test-bucket"
    storage.large_content_threshold = 1
    storage.vectorize_url = "https://example.invalid/vectorize"
    stored = {}
    deleted = {"vectors": [], "r2": []}

    async def generate_embedding(content):
        return [0.1, 0.2]

    async def store_r2(key, content):
        stored["r2"] = (key, content)

    async def store_vector(vector_id, embedding, metadata):
        stored["vector"] = (vector_id, embedding, metadata)

    async def reject_d1_restore(memory, vector_id, content_size, r2_key, content, **kwargs):
        stored["d1"] = (vector_id, r2_key)
        return False

    async def delete_vector(vector_id):
        deleted["vectors"].append(vector_id)

    async def delete_r2(key):
        deleted["r2"].append(key)

    monkeypatch.setattr(storage, "_generate_embedding", generate_embedding)
    monkeypatch.setattr(storage, "_store_r2_content", store_r2)
    monkeypatch.setattr(storage, "_store_vectorize_vector", store_vector)
    monkeypatch.setattr(storage, "_store_d1_memory", reject_d1_restore)
    monkeypatch.setattr(storage, "_delete_vectorize_vector", delete_vector)
    monkeypatch.setattr(storage, "_delete_r2_content", delete_r2)

    memory = Memory(
        content="large staged tombstone restore",
        content_hash="stable-content-hash",
        created_at=100.0,
        updated_at=101.0,
        tags=["__test__"],
    )
    success, message = await storage._store_memory(
        memory, expected_deleted_at=99.0
    )

    assert success is None
    assert message == "tombstone_changed"
    staged_vector_id, staged_r2_key = stored["d1"]
    assert staged_vector_id != memory.content_hash
    assert len(staged_vector_id.encode("utf-8")) <= 64
    assert staged_r2_key != f"content/{memory.content_hash}.txt"
    assert stored["vector"][0] == staged_vector_id
    assert stored["r2"][0] == staged_r2_key
    assert deleted == {"vectors": [staged_vector_id], "r2": [staged_r2_key]}


@pytest.mark.asyncio
async def test_cloudflare_restore_keeps_tombstone_when_tag_link_step_fails(
    monkeypatch,
):
    storage = CloudflareStorage.__new__(CloudflareStorage)
    storage.d1_url = "https://example.invalid/d1/database/test"
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.executescript(
        """
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
        CREATE TABLE tags (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL);
        CREATE TABLE memory_tags (
            memory_id INTEGER,
            tag_id INTEGER,
            PRIMARY KEY (memory_id, tag_id)
        );
        """
    )
    memory = Memory(
        content="restore only after tags are linked",
        content_hash="hash-tag-failure",
        created_at=100.0,
        updated_at=101.0,
        tags=["required-tag"],
    )
    db.execute(
        """
        INSERT INTO memories (
            content_hash, content, created_at, created_at_iso,
            updated_at, updated_at_iso, vector_id, deleted_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            memory.content_hash,
            memory.content,
            memory.created_at,
            memory.created_at_iso,
            memory.updated_at,
            memory.updated_at_iso,
            memory.content_hash,
            99.0,
        ),
    )
    db.commit()

    async def partially_failed_batch(method, url, **kwargs):
        results = []
        for index, statement in enumerate(kwargs["json"]["batch"]):
            if index == 2:
                results.append(
                    {"success": False, "meta": {"changes": 0}, "results": []}
                )
                continue
            cursor = db.execute(statement["sql"], statement.get("params", []))
            rows = [dict(row) for row in cursor.fetchall()] if cursor.description else []
            results.append(
                {
                    "success": True,
                    "results": rows,
                    "meta": {"changes": cursor.rowcount},
                }
            )
        db.commit()
        return SimpleNamespace(json=lambda: {"success": True, "result": results})

    monkeypatch.setattr(storage, "_retry_request", partially_failed_batch)
    try:
        with pytest.raises(RuntimeError, match="did not complete"):
            await storage._store_d1_memory(
                memory,
                vector_id="staged-vector",
                content_size=len(memory.content),
                r2_key=None,
                stored_content=memory.content,
                expected_deleted_at=99.0,
            )

        row = db.execute(
            "SELECT deleted_at FROM memories WHERE content_hash = ?",
            (memory.content_hash,),
        ).fetchone()
        assert row["deleted_at"] == 99.0
        assert db.execute("SELECT COUNT(*) FROM memory_tags").fetchone()[0] == 0
    finally:
        db.close()


@pytest.mark.asyncio
async def test_cloudflare_tombstone_restore_survives_retry_after_d1_commit(
    monkeypatch,
):
    storage = CloudflareStorage.__new__(CloudflareStorage)
    storage.d1_url = "https://example.invalid/d1/database/test"
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.executescript(
        """
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
        CREATE TABLE tags (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL);
        CREATE TABLE memory_tags (
            memory_id INTEGER,
            tag_id INTEGER,
            PRIMARY KEY (memory_id, tag_id)
        );
        """
    )
    memory = Memory(
        content="restore after a lost D1 response",
        content_hash="hash-restore-retry",
        created_at=100.0,
        created_at_iso="1970-01-01T00:01:40+00:00",
        updated_at=101.0,
        updated_at_iso="1970-01-01T00:01:41+00:00",
        tags=["retry-safe"],
    )
    db.execute(
        """
        INSERT INTO memories (
            content_hash, content, created_at, created_at_iso,
            updated_at, updated_at_iso, vector_id, deleted_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            memory.content_hash,
            memory.content,
            memory.created_at,
            memory.created_at_iso,
            memory.updated_at,
            memory.updated_at_iso,
            memory.content_hash,
            99.0,
        ),
    )
    db.commit()
    request_calls = 0
    batch_attempts = 0

    async def apply_request(kwargs):
        body = kwargs["json"]
        statements = body.get("batch") or [body]
        results = []
        db.execute("BEGIN")
        try:
            for statement in statements:
                cursor = db.execute(statement["sql"], statement.get("params", []))
                rows = [dict(row) for row in cursor.fetchall()] if cursor.description else []
                results.append(
                    {
                        "success": True,
                        "results": rows,
                        "meta": {"changes": cursor.rowcount},
                    }
                )
            db.commit()
        except sqlite3.Error:
            db.rollback()
            raise
        return results

    async def retry_request(method, url, **kwargs):
        nonlocal batch_attempts, request_calls
        request_calls += 1
        results = await apply_request(kwargs)
        if "batch" in kwargs["json"] and batch_attempts == 0:
            # Simulate the D1 commit succeeding but the response being lost.
            batch_attempts += 1
            results = await apply_request(kwargs)
        if "batch" in kwargs["json"]:
            batch_attempts += 1
        return SimpleNamespace(json=lambda: {"success": True, "result": results})

    monkeypatch.setattr(storage, "_retry_request", retry_request)
    try:
        stored = await storage._store_d1_memory(
            memory,
            vector_id=memory.content_hash,
            content_size=len(memory.content),
            r2_key=None,
            stored_content=memory.content,
            expected_deleted_at=99.0,
        )

        assert stored is True
        assert batch_attempts == 2
        assert request_calls == 1  # restore and tag links share one D1 batch
        row = db.execute(
            "SELECT content_hash, content, deleted_at FROM memories"
        ).fetchone()
        assert tuple(row) == (memory.content_hash, memory.content, None)
        linked_tags = db.execute(
            "SELECT COUNT(*) FROM memory_tags WHERE memory_id = 1"
        ).fetchone()[0]
        assert linked_tags == 1
    finally:
        db.close()


@pytest.mark.asyncio
async def test_cloudflare_tombstone_lookup_fails_closed_on_d1_error(monkeypatch):
    storage = CloudflareStorage.__new__(CloudflareStorage)
    storage.d1_url = "https://example.invalid/d1/database/test"

    async def failed_request(*args, **kwargs):
        return SimpleNamespace(json=lambda: {"success": False})

    monkeypatch.setattr(storage, "_retry_request", failed_request)

    with pytest.raises(RuntimeError, match="tombstone state"):
        await storage.get_tombstone_for_hash("memory-hash")


@pytest.mark.asyncio
async def test_cloudflare_tombstone_lookup_fails_closed_on_malformed_response(
    monkeypatch,
):
    storage = CloudflareStorage.__new__(CloudflareStorage)
    storage.d1_url = "https://example.invalid/d1/database/test"

    async def malformed_request(*args, **kwargs):
        return SimpleNamespace(json=lambda: {"success": True})

    monkeypatch.setattr(storage, "_retry_request", malformed_request)

    with pytest.raises(RuntimeError, match="invalid tombstone response"):
        await storage.get_tombstone_for_hash("memory-hash")
