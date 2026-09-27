"""
Tests for MemoryService.evolve_memory().

storage.update_memory_versioned() writes the new version straight into
storage, so a caller that uses it directly skips everything store_memory()
does after a write. evolve_memory() is the service-level entry point that
gives an evolved memory the same treatment as a stored one.
"""

import os
import shutil
import tempfile
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from mcp_memory_service.services.memory_service import MemoryService
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage

SERVICE = "mcp_memory_service.services.memory_service"


@pytest_asyncio.fixture
async def memory_service():
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_evolve.db")
    storage = SqliteVecMemoryStorage(db_path)
    try:
        await storage.initialize()
        yield MemoryService(storage)
    finally:
        if getattr(storage, "conn", None):
            storage.conn.close()
        shutil.rmtree(temp_dir, ignore_errors=True)


async def _store_original(svc):
    resp = await svc.store_memory(
        content="The backup job runs nightly at 02:00 against the NAS share.",
        tags=["backup"],
        memory_type="observation",
    )
    assert resp["success"]
    return resp["memory"]["content_hash"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_evolve_memory_writes_caller_metadata(memory_service, monkeypatch):
    monkeypatch.delenv("MCP_AGENT_ID", raising=False)
    old_hash = await _store_original(memory_service)

    ok, _msg, new_hash = await memory_service.evolve_memory(
        old_hash,
        "The backup job runs nightly at 03:00 against the NAS share.",
        tags=["backup", "session-harvest"],
        memory_type="observation",
        metadata={"source": "harvest", "harvest_method": "llm", "agent_id": "omp"},
        reason="test",
    )

    assert ok and new_hash and new_hash != old_hash
    new = await memory_service.storage.get_by_hash(new_hash)
    assert new.metadata["source"] == "harvest"
    assert new.metadata["harvest_method"] == "llm"
    assert new.metadata["agent_id"] == "omp"
    assert "session-harvest" in new.tags


@pytest.mark.unit
@pytest.mark.asyncio
async def test_evolve_memory_resolves_agent_id_from_env(memory_service, monkeypatch):
    monkeypatch.setenv("MCP_AGENT_ID", "claude-code")
    old_hash = await _store_original(memory_service)

    ok, _msg, new_hash = await memory_service.evolve_memory(
        old_hash, "The backup job runs nightly at 04:00 against the NAS share."
    )

    assert ok
    new = await memory_service.storage.get_by_hash(new_hash)
    assert new.metadata["agent_id"] == "claude-code"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_evolve_memory_runs_post_store_steps(memory_service):
    old_hash = await _store_original(memory_service)
    scorer = AsyncMock()
    memory_service._maybe_link_entities = AsyncMock()
    memory_service._plugin_registry.fire = AsyncMock()

    with patch(f"{SERVICE}.MCP_QUALITY_BOOST_ENABLED", True), \
         patch(f"{SERVICE}.async_scorer.score_memory", scorer):
        ok, _msg, new_hash = await memory_service.evolve_memory(
            old_hash, "The backup job runs nightly at 05:00 against the NAS share."
        )

    assert ok
    scored = scorer.await_args.args[0]
    assert scored.content_hash == new_hash
    assert memory_service._maybe_link_entities.await_args.args[0].content_hash == new_hash
    event, payload = memory_service._plugin_registry.fire.await_args.args
    assert event == "on_store" and payload["content_hash"] == new_hash


@pytest.mark.unit
@pytest.mark.asyncio
async def test_evolve_memory_reports_metadata_write_failure(memory_service):
    """The new version is committed before its metadata is written. If that
    write fails (e.g. database locked), say so instead of reporting a clean
    evolution, and still run the post-store steps on the committed version."""
    old_hash = await _store_original(memory_service)
    memory_service.storage.update_memory_metadata = AsyncMock(return_value=(False, "database is locked"))
    memory_service._run_post_store_steps = AsyncMock()

    ok, msg, new_hash = await memory_service.evolve_memory(
        old_hash,
        "The backup job runs nightly at 06:00 against the NAS share.",
        metadata={"source": "harvest"},
    )

    assert ok and new_hash
    assert "metadata" in msg and "database is locked" in msg
    memory_service._run_post_store_steps.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_evolve_memory_scores_even_if_reread_fails(memory_service):
    """A failed re-read after commit must not skip scoring: the post-store
    steps run on the version as written."""
    old_hash = await _store_original(memory_service)
    memory_service.storage.get_by_hash = AsyncMock(return_value=None)
    memory_service._run_post_store_steps = AsyncMock()
    content = "The backup job runs nightly at 07:00 against the NAS share."

    ok, _msg, new_hash = await memory_service.evolve_memory(
        old_hash, content, tags=["backup"], memory_type="observation",
        metadata={"source": "harvest"},
    )

    assert ok
    written = memory_service._run_post_store_steps.await_args.args[0]
    assert written.content_hash == new_hash
    assert written.content == content
    assert written.metadata["source"] == "harvest"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_evolve_memory_reread_fallback_keeps_inherited_fields(memory_service):
    """When the caller omits tags/type, storage inherits them from the old
    memory; the re-read fallback must report those, not empty values."""
    old_hash = await _store_original(memory_service)
    real_get = memory_service.storage.get_by_hash
    new_version_reads = []

    async def get_by_hash(h):
        if h == old_hash:
            return await real_get(h)
        new_version_reads.append(h)
        return None

    memory_service.storage.get_by_hash = get_by_hash
    memory_service._run_post_store_steps = AsyncMock()

    ok, _msg, new_hash = await memory_service.evolve_memory(
        old_hash, "The backup job runs nightly at 08:00 against the NAS share."
    )

    assert ok and new_version_reads == [new_hash]
    written = memory_service._run_post_store_steps.await_args.args[0]
    assert "backup" in written.tags
    assert written.memory_type == "observation"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_evolve_memory_failure_skips_post_store_steps(memory_service):
    memory_service._maybe_link_entities = AsyncMock()
    memory_service._plugin_registry.fire = AsyncMock()

    ok, msg, new_hash = await memory_service.evolve_memory("missing-hash", "anything")

    assert not ok and new_hash is None and "not found" in msg
    memory_service._maybe_link_entities.assert_not_awaited()
    memory_service._plugin_registry.fire.assert_not_awaited()
