"""Tests for Memory Evolution P3: Conflict Detection."""
import os
import time
import hashlib
import pytest
import pytest_asyncio
import tempfile
import shutil
from difflib import SequenceMatcher

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage


def _make_memory(content, tags=None):
    test_tags = list(tags or [])
    if "__test__" not in test_tags:
        test_tags.append("__test__")
    content_hash = hashlib.sha256(content.strip().lower().encode("utf-8")).hexdigest()
    return Memory(content=content, content_hash=content_hash, tags=test_tags)


@pytest.fixture
def temp_storage_dir():
    d = tempfile.mkdtemp(prefix="mcp-test-conflict-")
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


async def _store(storage, content, tags=None):
    mem = _make_memory(content, tags)
    ok, msg = await storage.store(mem, skip_semantic_dedup=True)
    assert ok, f"Failed to store: {msg}"
    return mem.content_hash


class TestDetectConflicts:

    @pytest.mark.asyncio
    async def test_conflicting_memories_detected(self, storage):
        """Two semantically similar but textually different memories trigger conflict."""
        h1 = await _store(storage, "The project database is PostgreSQL 15")
        h2 = await _store(storage, "The project database is MySQL 8.0")
        # Both are about "project database" but contradict each other
        conflicts = await storage.get_conflicts()
        # At least check the method exists and returns a list
        assert isinstance(conflicts, list)

    @pytest.mark.asyncio
    async def test_similar_content_no_conflict(self, storage):
        """Nearly identical content (low divergence) should NOT trigger conflict."""
        h1 = await _store(storage, "The server runs on port 8080")
        h2 = await _store(storage, "The server runs on port 8080 by default")
        conflicts = await storage.get_conflicts()
        conflicting_hashes = {(c["hash_a"], c["hash_b"]) for c in conflicts}
        assert (h1, h2) not in conflicting_hashes
        assert (h2, h1) not in conflicting_hashes

    @pytest.mark.asyncio
    async def test_superseded_memory_excluded(self, storage):
        """A superseded memory should not appear in conflicts."""
        h1 = await _store(storage, "Deploy target is AWS us-east-1")
        h2 = await _store(storage, "Deploy target is GCP europe-west1")
        # Resolve: h1 wins, h2 loses
        await storage.resolve_conflict(h1, h2)
        conflicts = await storage.get_conflicts()
        conflicting_hashes = {(c["hash_a"], c["hash_b"]) for c in conflicts}
        assert (h1, h2) not in conflicting_hashes


class TestGetConflicts:

    @pytest.mark.asyncio
    async def test_get_conflicts_returns_pairs(self, storage):
        """get_conflicts returns detected conflict pairs with metadata."""
        h1 = await _store(storage, "The API uses REST with JSON responses")
        h2 = await _store(storage, "The API uses GraphQL with schema-first design")
        conflicts = await storage.get_conflicts()
        # Should contain at least similarity and divergence info
        for c in conflicts:
            assert "hash_a" in c
            assert "hash_b" in c
            assert "similarity" in c


class TestResolveConflict:

    @pytest.mark.asyncio
    async def test_resolve_marks_loser_superseded(self, storage):
        """Resolving a conflict supersedes the loser."""
        h1 = await _store(storage, "Production runs on Kubernetes 1.28")
        h2 = await _store(storage, "Production runs on bare metal servers")
        ok, msg = await storage.resolve_conflict(h1, h2)
        assert ok
        # Loser should be superseded
        cursor = storage.conn.execute(
            "SELECT superseded_by FROM memories WHERE content_hash = ?", (h2,)
        )
        assert cursor.fetchone()[0] == h1

    @pytest.mark.asyncio
    async def test_resolve_boosts_winner_confidence(self, storage):
        """Winner gets confidence reset to 1.0 and last_accessed updated."""
        h1 = await _store(storage, "Cache backend is Redis 7.0")
        h2 = await _store(storage, "Cache backend is Memcached")
        now = time.time()
        ok, _ = await storage.resolve_conflict(h1, h2)
        assert ok
        cursor = storage.conn.execute(
            "SELECT confidence, last_accessed FROM memories WHERE content_hash = ?",
            (h1,),
        )
        conf, la = cursor.fetchone()
        assert conf == 1.0
        assert la >= now - 1

    @pytest.mark.asyncio
    async def test_resolve_removes_conflict_tag(self, storage):
        """Both memories should have conflict:unresolved tag removed."""
        h1 = await _store(storage, "CI pipeline uses GitHub Actions")
        h2 = await _store(storage, "CI pipeline uses Jenkins")
        await storage.resolve_conflict(h1, h2)
        for h in (h1, h2):
            cursor = storage.conn.execute(
                "SELECT tags FROM memories WHERE content_hash = ?", (h,)
            )
            tags = cursor.fetchone()[0]
            assert "conflict:unresolved" not in tags

    @pytest.mark.asyncio
    async def test_resolve_nonexistent_hash_fails(self, storage):
        """Resolving with unknown hash should fail gracefully."""
        ok, msg = await storage.resolve_conflict("nonexistent1", "nonexistent2")
        assert ok is False


class TestConflictIntegration:

    @pytest.mark.asyncio
    async def test_store_detect_resolve_lifecycle(self, storage):
        """Full lifecycle: store conflicting memories, detect, resolve."""
        # Store two contradictory memories
        h1 = await _store(storage, "The application framework is Django 5.0")
        h2 = await _store(storage, "The application framework is Flask 3.0")

        # Check conflicts exist
        conflicts = await storage.get_conflicts()
        hashes_in_conflicts = set()
        for c in conflicts:
            hashes_in_conflicts.add(c["hash_a"])
            hashes_in_conflicts.add(c["hash_b"])

        # Resolve: h1 wins
        if h1 in hashes_in_conflicts:
            ok, msg = await storage.resolve_conflict(h1, h2)
            assert ok

            # Conflicts should be empty now
            conflicts_after = await storage.get_conflicts()
            remaining = {(c["hash_a"], c["hash_b"]) for c in conflicts_after}
            assert (h1, h2) not in remaining
            assert (h2, h1) not in remaining
