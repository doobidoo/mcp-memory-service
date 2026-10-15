"""Regression tests for issue #1538 tombstone conflict resolution."""

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
