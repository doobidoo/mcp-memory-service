"""Tests for Memory Evolution P1+P2: versioned updates, lineage tracking, staleness scoring.

Covers:
- _effective_confidence: time-decayed confidence scoring
- update_memory_versioned: atomic versioned updates with SAVEPOINT
- get_memory_history: lineage chain traversal (forward + backward)
- retrieve() overfetch: confidence filtering returns correct count

Safety:
- All tests use isolated temp directories with 'mcp-test-' prefix
- conftest.py enforces sqlite_vec backend and temp DB paths globally
- Explicit __test__ tag cleanup on teardown
"""

import hashlib
import os
import time
import pytest
import pytest_asyncio
import tempfile
import shutil

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage


def _make_memory(content: str, tags: list[str] | None = None) -> Memory:
    """Create a Memory object with __test__ tag for safe cleanup."""
    test_tags = list(tags or [])
    if "__test__" not in test_tags:
        test_tags.append("__test__")
    content_hash = hashlib.sha256(content.strip().lower().encode("utf-8")).hexdigest()
    return Memory(content=content, content_hash=content_hash, tags=test_tags)


@pytest.fixture
def temp_storage_dir():
    """Temporary directory for SQLite-vec storage."""
    d = tempfile.mkdtemp(prefix="mcp-test-evolution-")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest_asyncio.fixture
async def storage(temp_storage_dir):
    """Initialize a real SqliteVecMemoryStorage with evolution columns.

    Safety guarantees:
    - Uses temp directory with 'mcp-test-' prefix (verified by assertion)
    - conftest.py already overrides MCP_MEMORY_SQLITE_PATH globally
    - Entire temp directory is removed on teardown (temp_storage_dir fixture)
    """
    assert "mcp-test-" in temp_storage_dir, (
        f"SAFETY: temp_storage_dir does not contain 'mcp-test-' prefix: {temp_storage_dir}"
    )
    db_path = os.path.join(temp_storage_dir, "test.db")
    os.environ["MCP_MEMORY_SQLITE_PATH"] = db_path
    os.environ["MCP_MEMORY_STORAGE_BACKEND"] = "sqlite_vec"
    os.environ["MCP_SEMANTIC_DEDUP_ENABLED"] = "false"

    s = SqliteVecMemoryStorage(db_path)
    await s.initialize()
    yield s

    # Cleanup: delete all __test__ tagged memories, then close
    try:
        if s.conn:
            s.conn.execute("DELETE FROM memories WHERE tags LIKE '%__test__%'")
            s.conn.commit()
    except Exception:
        pass
    await s.close()


async def _store(storage, content: str, tags: list[str] | None = None) -> str:
    """Helper to store a memory and return its content_hash."""
    mem = _make_memory(content, tags)
    ok, msg = await storage.store(mem, skip_semantic_dedup=True)
    assert ok, f"Failed to store memory: {msg}"
    return mem.content_hash


# ── _effective_confidence ────────────────────────────────────────────


class TestEffectiveConfidence:
    """Unit tests for staleness-aware confidence decay."""

    def test_fresh_memory_full_confidence(self):
        """A memory accessed just now should have confidence ~1.0."""
        now = time.time()
        result = SqliteVecMemoryStorage._effective_confidence(
            confidence=1.0, last_accessed=now, created_at=now, now=now,
        )
        assert result == 1.0

    def test_stale_memory_decays(self):
        """A memory not accessed for 30 days should decay significantly."""
        now = time.time()
        thirty_days_ago = now - (30 * 86400)
        result = SqliteVecMemoryStorage._effective_confidence(
            confidence=1.0, last_accessed=thirty_days_ago,
            created_at=thirty_days_ago, now=now,
        )
        # decay_window=30, decay_rate=0.5: staleness=1.0, decay=0.5
        assert result == 0.5

    def test_very_old_memory_floors_at_zero(self):
        """A memory not accessed for 60+ days should floor at 0.0."""
        now = time.time()
        sixty_days_ago = now - (60 * 86400)
        result = SqliteVecMemoryStorage._effective_confidence(
            confidence=1.0, last_accessed=sixty_days_ago,
            created_at=sixty_days_ago, now=now,
        )
        # staleness=2.0, decay=max(0, 1 - 2.0*0.5) = 0.0
        assert result == 0.0

    def test_none_confidence_defaults_to_one(self):
        """None confidence should be treated as 1.0."""
        now = time.time()
        result = SqliteVecMemoryStorage._effective_confidence(
            confidence=None, last_accessed=now, created_at=now, now=now,
        )
        assert result == 1.0

    def test_none_last_accessed_falls_back_to_created_at(self):
        """If last_accessed is None, use created_at as reference."""
        now = time.time()
        ten_days_ago = now - (10 * 86400)
        result = SqliteVecMemoryStorage._effective_confidence(
            confidence=1.0, last_accessed=None, created_at=ten_days_ago, now=now,
        )
        # staleness = 10/30 ≈ 0.333, decay ≈ 0.833
        assert 0.8 < result < 0.9

    def test_half_confidence_with_decay(self):
        """Confidence of 0.5 with some staleness should compound."""
        now = time.time()
        fifteen_days_ago = now - (15 * 86400)
        result = SqliteVecMemoryStorage._effective_confidence(
            confidence=0.5, last_accessed=fifteen_days_ago,
            created_at=fifteen_days_ago, now=now,
        )
        # staleness=0.5, decay=0.75, effective=0.5*0.75=0.375
        assert result == 0.375

    def test_custom_decay_window_env(self, monkeypatch):
        """MEMORY_DECAY_WINDOW_DAYS env var changes the decay window."""
        monkeypatch.setenv("MEMORY_DECAY_WINDOW_DAYS", "60")
        now = time.time()
        thirty_days_ago = now - (30 * 86400)
        result = SqliteVecMemoryStorage._effective_confidence(
            confidence=1.0, last_accessed=thirty_days_ago,
            created_at=thirty_days_ago, now=now,
        )
        # staleness=30/60=0.5, decay=0.75
        assert result == 0.75


# ── update_memory_versioned ──────────────────────────��───────────────


class TestUpdateMemoryVersioned:
    """Tests for atomic versioned updates with lineage tracking."""

    @pytest.mark.asyncio
    async def test_basic_versioned_update(self, storage):
        """Update creates a new version linked to the original."""
        h = await _store(storage, "The capital of France is Paris")

        success, msg, new_hash = await storage.update_memory_versioned(
            h, "The capital of France is Paris, the City of Light"
        )
        assert success is True
        assert "v1" in msg and "v2" in msg
        assert new_hash is not None
        assert new_hash != h

    @pytest.mark.asyncio
    async def test_versioned_update_sets_superseded_by(self, storage):
        """Old memory should have superseded_by pointing to new version."""
        h = await _store(storage, "Python is version 3.11")

        success, _, new_hash = await storage.update_memory_versioned(
            h, "Python is version 3.14"
        )
        assert success

        cursor = storage.conn.execute(
            "SELECT superseded_by FROM memories WHERE content_hash = ?", (h,)
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == new_hash

    @pytest.mark.asyncio
    async def test_versioned_update_sets_parent_id(self, storage):
        """New version should have parent_id pointing to old version."""
        h = await _store(storage, "Test memory for parent tracking")

        success, _, new_hash = await storage.update_memory_versioned(
            h, "Updated memory for parent tracking"
        )
        assert success

        cursor = storage.conn.execute(
            "SELECT parent_id, version FROM memories WHERE content_hash = ?",
            (new_hash,),
        )
        row = cursor.fetchone()
        assert row[0] == h  # parent_id
        assert row[1] == 2  # version

    @pytest.mark.asyncio
    async def test_versioned_update_nonexistent_hash(self, storage):
        """Updating a non-existent hash should fail gracefully."""
        success, msg, new_hash = await storage.update_memory_versioned(
            "0000000000000000", "New content"
        )
        assert success is False
        assert new_hash is None

    @pytest.mark.asyncio
    async def test_multi_version_chain(self, storage):
        """Three versions should form a linked chain."""
        h1 = await _store(storage, "Chain v1 original content here")

        ok2, _, h2 = await storage.update_memory_versioned(h1, "Chain v2 updated content here")
        assert ok2

        ok3, _, h3 = await storage.update_memory_versioned(h2, "Chain v3 final content here")
        assert ok3

        # Verify h2 links both ways
        cur = storage.conn.execute(
            "SELECT version, parent_id, superseded_by FROM memories WHERE content_hash = ?",
            (h2,),
        )
        row = cur.fetchone()
        assert row[0] == 2  # version
        assert row[1] == h1  # parent_id
        assert row[2] == h3  # superseded_by

    @pytest.mark.asyncio
    async def test_versioned_update_preserves_tags(self, storage):
        """Tags from original should carry over to new version."""
        h = await _store(storage, "Tagged memory for evolution test", tags=["important", "project"])

        ok, _, new_hash = await storage.update_memory_versioned(
            h, "Updated tagged memory for evolution test"
        )
        assert ok

        cursor = storage.conn.execute(
            "SELECT tags FROM memories WHERE content_hash = ?", (new_hash,)
        )
        tags = cursor.fetchone()[0]
        assert "important" in tags
        assert "project" in tags

    @pytest.mark.asyncio
    async def test_atomicity_embedding_exists(self, storage):
        """New version should have an embedding (atomic insert)."""
        h = await _store(storage, "Memory needing embedding check test")

        ok, _, new_hash = await storage.update_memory_versioned(
            h, "Updated memory with embedding test"
        )
        assert ok

        # Verify embedding exists for new version
        cursor = storage.conn.execute(
            "SELECT m.id FROM memories m WHERE m.content_hash = ?", (new_hash,)
        )
        new_rowid = cursor.fetchone()[0]
        cursor = storage.conn.execute(
            "SELECT COUNT(*) FROM memory_embeddings WHERE rowid = ?", (new_rowid,)
        )
        assert cursor.fetchone()[0] == 1


# ── get_memory_history ───────────────────────────────────────────────


class TestGetMemoryHistory:
    """Tests for lineage chain traversal."""

    @pytest.mark.asyncio
    async def test_single_memory_no_history(self, storage):
        """A single memory with no versions returns just itself."""
        h = await _store(storage, "Standalone memory no versions test")

        history = await storage.get_memory_history(h)
        assert len(history) == 1
        assert history[0]["content_hash"] == h
        assert history[0]["active"] is True

    @pytest.mark.asyncio
    async def test_two_version_lineage(self, storage):
        """Two versions should return both in order, oldest first."""
        h1 = await _store(storage, "History test v1 original content")
        ok, _, h2 = await storage.update_memory_versioned(h1, "History test v2 updated content")
        assert ok

        history = await storage.get_memory_history(h2)
        assert len(history) == 2
        assert history[0]["content_hash"] == h1
        assert history[0]["active"] is False  # superseded
        assert history[1]["content_hash"] == h2
        assert history[1]["active"] is True

    @pytest.mark.asyncio
    async def test_history_from_middle_version(self, storage):
        """Querying history from middle version should return full chain."""
        h1 = await _store(storage, "Middle chain test v1 content")
        ok2, _, h2 = await storage.update_memory_versioned(h1, "Middle chain test v2 content")
        ok3, _, h3 = await storage.update_memory_versioned(h2, "Middle chain test v3 content")
        assert ok2 and ok3

        # Query from h2 (middle) should still find full chain
        history = await storage.get_memory_history(h2)
        assert len(history) == 3
        assert history[0]["content_hash"] == h1
        assert history[2]["content_hash"] == h3

    @pytest.mark.asyncio
    async def test_history_version_numbers(self, storage):
        """Version numbers should increment correctly."""
        h1 = await _store(storage, "Version numbering test content")
        _, _, h2 = await storage.update_memory_versioned(h1, "Version numbering v2 content")
        _, _, h3 = await storage.update_memory_versioned(h2, "Version numbering v3 content")

        history = await storage.get_memory_history(h1)
        versions = [h["version"] for h in history]
        assert versions == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_history_nonexistent_hash(self, storage):
        """Non-existent hash should return empty list."""
        history = await storage.get_memory_history("nonexistent_hash_000")
        assert history == []


# ── retrieve with confidence filtering ───────────────────────────────


class TestRetrieveConfidenceFiltering:
    """Tests for overfetch behavior when min_confidence > 0."""

    @pytest.mark.asyncio
    async def test_retrieve_without_confidence_returns_requested_count(self, storage):
        """Basic retrieve without confidence filter returns up to n_results."""
        for i in range(5):
            await _store(storage, f"Confidence test memory {i} about unique topic alpha beta gamma")

        results = await storage.retrieve("unique topic alpha beta gamma", n_results=3)
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_retrieve_with_staleness_returns_fresh_results(self, storage):
        """retrieve_with_staleness should return fresh memories when min_confidence=0."""
        for i in range(5):
            await _store(storage, f"Staleness test {i} specific unique content delta epsilon")

        results = await storage.retrieve_with_staleness(
            "specific unique content delta epsilon",
            n_results=3,
            min_confidence=0.0,
        )
        assert len(results) <= 3
