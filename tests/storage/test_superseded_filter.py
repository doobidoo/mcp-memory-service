"""Tests for include_superseded retrieval filter + auto-mark on contradiction (#732)."""
import os
import hashlib
import pytest
import pytest_asyncio
import tempfile
import shutil
from unittest.mock import AsyncMock, MagicMock

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage


def _make_memory(content, tags=None, created_at=None):
    test_tags = list(tags or [])
    if "__test__" not in test_tags:
        test_tags.append("__test__")
    content_hash = hashlib.sha256(content.strip().lower().encode("utf-8")).hexdigest()
    return Memory(
        content=content,
        content_hash=content_hash,
        tags=test_tags,
        created_at=created_at,
    )


@pytest.fixture
def temp_storage_dir():
    d = tempfile.mkdtemp(prefix="mcp-test-superseded-")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest_asyncio.fixture
async def storage(temp_storage_dir):
    assert "mcp-test-" in temp_storage_dir
    db_path = os.path.join(temp_storage_dir, "test.db")
    os.environ["MCP_MEMORY_SQLITE_PATH"] = db_path
    os.environ["MCP_MEMORY_STORAGE_BACKEND"] = "sqlite_vec"
    os.environ["MCP_SEMANTIC_DEDUP_ENABLED"] = "false"
    s = SqliteVecMemoryStorage(db_path)
    await s.initialize()
    yield s
    try:
        if s.conn:
            s.conn.execute("DELETE FROM memories WHERE tags LIKE '%__test__%'")
            s.conn.commit()
    except Exception:
        pass
    await s.close()


async def _store(storage, content, tags=None, created_at=None):
    mem = _make_memory(content, tags, created_at=created_at)
    ok, msg = await storage.store(mem, skip_semantic_dedup=True)
    assert ok, f"Failed to store: {msg}"
    return mem.content_hash


def _mark_superseded(storage, loser_hash, winner_hash):
    """Directly mark a memory as superseded in the DB."""
    storage.conn.execute(
        "UPDATE memories SET superseded_by = ? WHERE content_hash = ? AND deleted_at IS NULL",
        (winner_hash, loser_hash),
    )
    storage.conn.commit()


class TestRetrieveSupersededFilter:

    @pytest.mark.asyncio
    async def test_retrieve_excludes_superseded_by_default(self, storage):
        """Default retrieve() should not return superseded memories."""
        h1 = await _store(storage, "The database uses PostgreSQL 15")
        h2 = await _store(storage, "The database uses MySQL 8.0")
        _mark_superseded(storage, h1, h2)

        results = await storage.retrieve("database", n_results=10)
        result_hashes = {r.memory.content_hash for r in results}
        assert h1 not in result_hashes
        assert h2 in result_hashes

    @pytest.mark.asyncio
    async def test_retrieve_includes_superseded_when_flag_set(self, storage):
        """retrieve(include_superseded=True) should return superseded memories."""
        h1 = await _store(storage, "The database uses PostgreSQL 15")
        h2 = await _store(storage, "The database uses MySQL 8.0")
        _mark_superseded(storage, h1, h2)

        results = await storage.retrieve("database", n_results=10, include_superseded=True)
        result_hashes = {r.memory.content_hash for r in results}
        assert h1 in result_hashes
        assert h2 in result_hashes

    @pytest.mark.asyncio
    async def test_non_superseded_always_returned(self, storage):
        """Non-superseded memories should appear regardless of the flag."""
        h1 = await _store(storage, "The server runs on port 8080")

        results_default = await storage.retrieve("server port", n_results=10)
        results_with = await storage.retrieve("server port", n_results=10, include_superseded=True)

        assert any(r.memory.content_hash == h1 for r in results_default)
        assert any(r.memory.content_hash == h1 for r in results_with)


class TestAutoSupersedOnContradiction:

    @pytest.mark.asyncio
    async def test_auto_supersede_marks_older_memory(self, storage):
        """When a contradicts edge is detected, the older memory should be auto-superseded."""
        older_ts = 1000000.0
        newer_ts = 2000000.0
        h_old = await _store(storage, "The API uses REST architecture", created_at=older_ts)
        h_new = await _store(storage, "The API uses GraphQL architecture", created_at=newer_ts)

        marked = await storage.mark_superseded_batch([(h_new, h_old)])
        assert marked == 1

        # Verify older memory is superseded
        row = storage.conn.execute(
            "SELECT superseded_by FROM memories WHERE content_hash = ?", (h_old,)
        ).fetchone()
        assert row is not None
        assert row[0] == h_new

        # Verify newer memory is NOT superseded
        row2 = storage.conn.execute(
            "SELECT superseded_by FROM memories WHERE content_hash = ?", (h_new,)
        ).fetchone()
        assert row2 is not None
        assert row2[0] is None or row2[0] == ""

    @pytest.mark.asyncio
    async def test_auto_supersede_reverse_order(self, storage):
        """mark_superseded_batch correctly handles winner/loser regardless of order."""
        older_ts = 1000000.0
        newer_ts = 2000000.0
        h_old = await _store(storage, "Deploy target is AWS", created_at=older_ts)
        h_new = await _store(storage, "Deploy target is GCP", created_at=newer_ts)

        # Winner is h_new, loser is h_old
        marked = await storage.mark_superseded_batch([(h_new, h_old)])
        assert marked == 1

        row = storage.conn.execute(
            "SELECT superseded_by FROM memories WHERE content_hash = ?", (h_old,)
        ).fetchone()
        assert row is not None
        assert row[0] == h_new
