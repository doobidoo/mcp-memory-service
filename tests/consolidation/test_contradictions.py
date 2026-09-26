"""Tests for temporal contradiction detection.

The mocked tests use ``AsyncMock(spec=...)`` against the real storage classes,
so a call to a method that does not exist raises ``AttributeError`` instead of
being silently invented by the mock. An unspecced ``AsyncMock`` is how
``storage.add_graph_edge`` (which never existed) passed this suite while the
feature did nothing in production.

``TestAgainstRealStorage`` runs the detector end-to-end on a real
``SqliteVecMemoryStorage`` + ``GraphStorage`` pair and checks the two effects
the feature exists for: the older memory drops out of default retrieval, and a
``contradicts`` edge lands in the graph.
"""

import hashlib
import os

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from mcp_memory_service.consolidation.contradictions import (
    detect_contradictions,
    check_contradiction_on_store,
)
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.graph import GraphStorage
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage

_MOD = "mcp_memory_service.consolidation.contradictions"


def _make_memory(content_hash, content, memory_type="observation", created_at=None, metadata=None, tags=None):
    """Create a mock Memory dataclass instance."""
    m = MagicMock()
    m.content_hash = content_hash
    m.content = content
    m.memory_type = memory_type
    m.created_at = created_at or 0.0
    m.metadata = metadata or {}
    m.tags = tags or []
    return m


def _hit(content_hash, similarity, created_at, memory_type="observation"):
    """A search_memories() result entry (Memory.to_dict() + similarity_score)."""
    return {
        "content_hash": content_hash,
        "similarity_score": similarity,
        "type": memory_type,
        "created_at": created_at,
    }


def _spec_storage():
    """Storage mock restricted to the real SqliteVecMemoryStorage surface."""
    storage = AsyncMock(spec=SqliteVecMemoryStorage)
    storage.mark_superseded_batch = AsyncMock(side_effect=lambda pairs: len(pairs))
    return storage


def _spec_graph():
    graph = AsyncMock(spec=GraphStorage)
    graph.store_association = AsyncMock(return_value=True)
    return graph


OLD_T = 1735689600.0
NEW_T = 1746057600.0


@pytest.fixture
def mock_storage():
    storage = _spec_storage()
    storage.get_all_memories = AsyncMock(return_value=[
        _make_memory("hash_old", "The sky is blue", "observation", created_at=OLD_T),
        _make_memory("hash_new", "The sky is red", "observation", created_at=NEW_T),
    ])

    # Each memory finds itself (similarity 1.0) plus the other one in the band.
    async def _search(query, limit):
        if query == "The sky is blue":
            return {"memories": [_hit("hash_old", 1.0, OLD_T), _hit("hash_new", 0.6, NEW_T)]}
        return {"memories": [_hit("hash_new", 1.0, NEW_T), _hit("hash_old", 0.6, OLD_T)]}

    storage.search_memories = AsyncMock(side_effect=_search)
    return storage


@pytest.fixture
def mock_storage_no_contradiction():
    storage = _spec_storage()
    storage.get_all_memories = AsyncMock(return_value=[
        _make_memory("hash1", "Hello", "note", created_at=OLD_T),
    ])
    # Returns only self — should be filtered out
    storage.search_memories = AsyncMock(return_value={
        "memories": [_hit("hash1", 1.0, OLD_T, "note")]
    })
    return storage


class TestDetectContradictions:
    @pytest.mark.asyncio
    @patch(f"{_MOD}.CONTRADICTION_ENABLED", True)
    async def test_detects_contradiction_dry_run(self, mock_storage):
        graph = _spec_graph()
        result = await detect_contradictions(mock_storage, dry_run=True, graph=graph)
        assert result["pairs_detected"] == 1
        assert result["dry_run"] is True
        assert result["edges_created"] == 0
        assert result["superseded_marked"] == 0
        mock_storage.mark_superseded_batch.assert_not_called()
        graph.store_association.assert_not_called()

    @pytest.mark.asyncio
    @patch(f"{_MOD}.CONTRADICTION_ENABLED", True)
    async def test_live_marks_older_superseded_via_storage_api(self, mock_storage):
        graph = _spec_graph()
        result = await detect_contradictions(mock_storage, dry_run=False, graph=graph)

        assert result["pairs_detected"] == 1
        assert result["superseded_marked"] == 1
        # (winner, loser) — the newer memory supersedes the older one.
        mock_storage.mark_superseded_batch.assert_awaited_once_with([("hash_new", "hash_old")])
        mock_storage.update_memory_metadata.assert_not_called()

    @pytest.mark.asyncio
    @patch(f"{_MOD}.CONTRADICTION_ENABLED", True)
    async def test_live_writes_contradicts_edge(self, mock_storage):
        graph = _spec_graph()
        result = await detect_contradictions(mock_storage, dry_run=False, graph=graph)

        assert result["edges_created"] == 1
        graph.store_association.assert_awaited_once()
        kwargs = graph.store_association.call_args.kwargs
        assert kwargs["source_hash"] == "hash_new"
        assert kwargs["target_hash"] == "hash_old"
        assert kwargs["relationship_type"] == "contradicts"
        assert kwargs["similarity"] == pytest.approx(0.6)

    @pytest.mark.asyncio
    @patch(f"{_MOD}.CONTRADICTION_ENABLED", True)
    async def test_pair_seen_from_both_sides_counted_once(self, mock_storage):
        result = await detect_contradictions(mock_storage, dry_run=True)
        assert result["pairs_detected"] == 1
        assert len(result["pairs"]) == 1

    @pytest.mark.asyncio
    @patch(f"{_MOD}.CONTRADICTION_ENABLED", True)
    async def test_live_without_graph_still_supersedes(self, mock_storage):
        result = await detect_contradictions(mock_storage, dry_run=False, graph=None)
        assert result["superseded_marked"] == 1
        assert result["edges_created"] == 0
        assert result["graph_available"] is False

    @pytest.mark.asyncio
    @patch(f"{_MOD}.CONTRADICTION_ENABLED", True)
    async def test_memory_hidden_from_retrieval_cannot_supersede(self):
        """A memory that search does not return for its own content (already
        superseded, or deleted) must not be used to hide another memory."""
        storage = _spec_storage()
        storage.get_all_memories = AsyncMock(return_value=[
            _make_memory("hash_hidden_new", "Newer but already superseded", created_at=NEW_T),
        ])
        # Search excludes superseded rows, so the memory does not find itself.
        storage.search_memories = AsyncMock(return_value={
            "memories": [_hit("hash_live_old", 0.6, OLD_T)]
        })
        result = await detect_contradictions(storage, dry_run=False, graph=_spec_graph())
        assert result["pairs_detected"] == 0
        storage.mark_superseded_batch.assert_not_called()

    @pytest.mark.asyncio
    @patch(f"{_MOD}.CONTRADICTION_ENABLED", True)
    async def test_memory_superseded_this_run_cannot_win(self):
        """B supersedes A; D (older than A) must not then be superseded by A,
        which is already hidden."""
        storage = _spec_storage()
        storage.get_all_memories = AsyncMock(return_value=[
            _make_memory("hash_b", "B", created_at=NEW_T),
            _make_memory("hash_d", "D", created_at=OLD_T - 1000),
            _make_memory("hash_a", "A", created_at=OLD_T),
        ])

        async def _search(query, limit):
            if query == "B":
                return {"memories": [_hit("hash_b", 1.0, NEW_T), _hit("hash_a", 0.6, OLD_T)]}
            return {"memories": [_hit("hash_d", 1.0, OLD_T - 1000), _hit("hash_a", 0.6, OLD_T)]}

        storage.search_memories = AsyncMock(side_effect=_search)
        result = await detect_contradictions(storage, dry_run=False)
        assert result["pairs_detected"] == 1
        storage.mark_superseded_batch.assert_awaited_once_with([("hash_b", "hash_a")])

    @pytest.mark.asyncio
    @patch(f"{_MOD}.CONTRADICTION_ENABLED", True)
    @pytest.mark.parametrize("tag", ["critical", "important", "reference", "permanent"])
    async def test_protected_older_memory_is_never_superseded(self, mock_storage, tag):
        """Same protection forgetting and decay apply (consolidation/base.py)."""
        mock_storage.get_all_memories = AsyncMock(return_value=[
            _make_memory("hash_old", "The sky is blue", "observation", created_at=OLD_T, tags=[tag]),
            _make_memory("hash_new", "The sky is red", "observation", created_at=NEW_T),
        ])
        result = await detect_contradictions(mock_storage, dry_run=False, graph=_spec_graph())
        assert result["pairs_detected"] == 0
        assert result["protected_skipped"] == 1
        mock_storage.mark_superseded_batch.assert_not_called()

    @pytest.mark.asyncio
    @patch(f"{_MOD}.CONTRADICTION_ENABLED", True)
    async def test_protected_newer_memory_can_still_supersede(self, mock_storage):
        mock_storage.get_all_memories = AsyncMock(return_value=[
            _make_memory("hash_old", "The sky is blue", "observation", created_at=OLD_T),
            _make_memory("hash_new", "The sky is red", "observation", created_at=NEW_T, tags=["important"]),
        ])
        result = await detect_contradictions(mock_storage, dry_run=False, graph=_spec_graph())
        assert result["superseded_marked"] == 1
        mock_storage.mark_superseded_batch.assert_awaited_once_with([("hash_new", "hash_old")])

    @pytest.mark.asyncio
    @patch(f"{_MOD}.CONTRADICTION_ENABLED", True)
    async def test_older_memory_outside_scan_is_looked_up(self):
        """get_all_memories() can be capped (Milvus: 16,384 newest). An older
        memory returned by search but absent from the scan is fetched by hash
        instead of being skipped."""
        storage = _spec_storage()
        storage.get_all_memories = AsyncMock(return_value=[
            _make_memory("hash_new", "The sky is red", created_at=NEW_T),
        ])
        storage.search_memories = AsyncMock(return_value={
            "memories": [_hit("hash_new", 1.0, NEW_T), _hit("hash_old", 0.6, OLD_T)]
        })
        storage.get_by_hash = AsyncMock(return_value=_make_memory("hash_old", "The sky is blue", created_at=OLD_T))

        result = await detect_contradictions(storage, dry_run=False)

        storage.get_by_hash.assert_awaited_once_with("hash_old")
        storage.mark_superseded_batch.assert_awaited_once_with([("hash_new", "hash_old")])
        assert result["protected_skipped"] == 0

    @pytest.mark.asyncio
    @patch(f"{_MOD}.CONTRADICTION_ENABLED", True)
    async def test_older_memory_outside_scan_still_protected(self):
        storage = _spec_storage()
        storage.get_all_memories = AsyncMock(return_value=[
            _make_memory("hash_new", "The sky is red", created_at=NEW_T),
        ])
        storage.search_memories = AsyncMock(return_value={
            "memories": [_hit("hash_new", 1.0, NEW_T), _hit("hash_old", 0.6, OLD_T)]
        })
        storage.get_by_hash = AsyncMock(
            return_value=_make_memory("hash_old", "The sky is blue", created_at=OLD_T, tags=["reference"])
        )

        result = await detect_contradictions(storage, dry_run=False)

        assert result["protected_skipped"] == 1
        storage.mark_superseded_batch.assert_not_called()

    @pytest.mark.asyncio
    @patch(f"{_MOD}.CONTRADICTION_ENABLED", True)
    async def test_outside_scan_lookup_is_cached(self):
        """A protected memory outside the scan stays eligible for later pairs;
        it must be fetched once per run, not once per pair."""
        storage = _spec_storage()
        storage.get_all_memories = AsyncMock(return_value=[
            _make_memory("hash_b", "B", created_at=NEW_T),
            _make_memory("hash_c", "C", created_at=NEW_T + 1),
        ])

        async def _search(query, limit):
            own = "hash_b" if query == "B" else "hash_c"
            return {"memories": [_hit(own, 1.0, NEW_T), _hit("hash_old", 0.6, OLD_T)]}

        storage.search_memories = AsyncMock(side_effect=_search)
        storage.get_by_hash = AsyncMock(
            return_value=_make_memory("hash_old", "old", created_at=OLD_T, tags=["permanent"])
        )

        result = await detect_contradictions(storage, dry_run=False)

        assert result["protected_skipped"] == 2
        storage.get_by_hash.assert_awaited_once_with("hash_old")

    @pytest.mark.asyncio
    @patch(f"{_MOD}.CONTRADICTION_ENABLED", True)
    async def test_lookup_miss_is_retried_not_cached(self):
        """Milvus get_by_hash() returns None on a transient backend error, so a
        miss must not be cached: a later pair retries and can still supersede."""
        storage = _spec_storage()
        storage.get_all_memories = AsyncMock(return_value=[
            _make_memory("hash_b", "B", created_at=NEW_T),
            _make_memory("hash_c", "C", created_at=NEW_T + 1),
        ])

        async def _search(query, limit):
            own = "hash_b" if query == "B" else "hash_c"
            return {"memories": [_hit(own, 1.0, NEW_T), _hit("hash_old", 0.6, OLD_T)]}

        storage.search_memories = AsyncMock(side_effect=_search)
        storage.get_by_hash = AsyncMock(
            side_effect=[None, _make_memory("hash_old", "old", created_at=OLD_T)]
        )

        result = await detect_contradictions(storage, dry_run=False)

        assert storage.get_by_hash.await_count == 2
        assert result["protected_skipped"] == 1
        storage.mark_superseded_batch.assert_awaited_once_with([("hash_c", "hash_old")])

    @pytest.mark.asyncio
    @patch(f"{_MOD}.CONTRADICTION_ENABLED", True)
    async def test_failed_edge_write_is_reported(self, mock_storage):
        graph = _spec_graph()
        graph.store_association = AsyncMock(return_value=False)
        result = await detect_contradictions(mock_storage, dry_run=False, graph=graph)
        assert result["edges_created"] == 0
        assert result["edge_failures"] == 1

    @pytest.mark.asyncio
    @patch(f"{_MOD}.CONTRADICTION_ENABLED", True)
    async def test_transient_search_error_is_counted_not_hidden(self, mock_storage):
        mock_storage.search_memories = AsyncMock(side_effect=RuntimeError("database is locked"))
        result = await detect_contradictions(mock_storage, dry_run=True)
        assert result["search_errors"] == 2
        assert "database is locked" in result["last_search_error"]

    @pytest.mark.asyncio
    @patch(f"{_MOD}.CONTRADICTION_ENABLED", True)
    async def test_programming_error_is_not_swallowed_per_memory(self, mock_storage):
        """AttributeError/TypeError mean the code is calling the storage API
        wrong — surface them in the result instead of skipping the memory."""
        mock_storage.search_memories = AsyncMock(side_effect=AttributeError("no such method"))
        result = await detect_contradictions(mock_storage, dry_run=True)
        assert "no such method" in result.get("error", "")

    @pytest.mark.asyncio
    @patch(f"{_MOD}.CONTRADICTION_ENABLED", False)
    async def test_skipped_when_disabled(self, mock_storage):
        result = await detect_contradictions(mock_storage, dry_run=True)
        assert result["skipped"] is True

    @pytest.mark.asyncio
    @patch(f"{_MOD}.CONTRADICTION_ENABLED", True)
    async def test_no_contradiction_found(self, mock_storage_no_contradiction):
        result = await detect_contradictions(mock_storage_no_contradiction, dry_run=True)
        assert result["pairs_detected"] == 0

    @pytest.mark.asyncio
    @patch(f"{_MOD}.CONTRADICTION_ENABLED", True)
    async def test_empty_memories(self):
        storage = _spec_storage()
        storage.get_all_memories = AsyncMock(return_value=[])
        result = await detect_contradictions(storage, dry_run=True)
        assert "No memories" in result.get("message", "")


class TestCheckContradictionOnStore:
    @pytest.mark.asyncio
    @patch(f"{_MOD}.CONTRADICTION_ON_STORE", True)
    async def test_finds_contradiction(self):
        storage = _spec_storage()
        storage.search_memories = AsyncMock(return_value={
            "memories": [_hit("existing_hash", 0.55, OLD_T)]
        })
        storage.get_by_hash = AsyncMock(return_value=_make_memory("existing_hash", "old"))
        graph = _spec_graph()

        result = await check_contradiction_on_store(
            storage, "New contradicting content", "new_hash", graph=graph
        )
        assert result is not None
        assert "contradicts" in result
        storage.mark_superseded_batch.assert_awaited_once_with([("new_hash", "existing_hash")])
        graph.store_association.assert_awaited_once()
        assert graph.store_association.call_args.kwargs["relationship_type"] == "contradicts"

    @pytest.mark.asyncio
    @patch(f"{_MOD}.CONTRADICTION_ON_STORE", True)
    async def test_protected_existing_memory_is_not_superseded(self):
        storage = _spec_storage()
        storage.search_memories = AsyncMock(return_value={
            "memories": [_hit("existing_hash", 0.55, OLD_T)]
        })
        storage.get_by_hash = AsyncMock(
            return_value=_make_memory("existing_hash", "old", tags=["critical"])
        )
        result = await check_contradiction_on_store(storage, "new", "new_hash", graph=_spec_graph())
        assert result is None
        storage.mark_superseded_batch.assert_not_called()

    @pytest.mark.asyncio
    @patch(f"{_MOD}.CONTRADICTION_ON_STORE", False)
    async def test_skipped_when_disabled(self):
        storage = _spec_storage()
        result = await check_contradiction_on_store(storage, "content", "hash")
        assert result is None

    @pytest.mark.asyncio
    @patch(f"{_MOD}.CONTRADICTION_ON_STORE", True)
    async def test_no_contradiction(self):
        storage = _spec_storage()
        storage.search_memories = AsyncMock(return_value={
            # too similar (dedup zone, above SIMILARITY_MAX=0.75)
            "memories": [_hit("other", 0.9, OLD_T, "note")]
        })
        result = await check_contradiction_on_store(storage, "content", "hash")
        assert result is None
        storage.mark_superseded_batch.assert_not_called()


# ── End-to-end against real storage ──────────────────────────────────


@pytest_asyncio.fixture
async def real_storage(temp_db_path, monkeypatch):
    db_path = os.path.join(temp_db_path, "test.db")
    monkeypatch.setenv("MCP_SEMANTIC_DEDUP_ENABLED", "false")
    s = SqliteVecMemoryStorage(db_path)
    await s.initialize()
    g = GraphStorage(db_path)
    yield s, g
    await g.close()
    await s.close()


def _real_memory(content: str, created_at: float) -> Memory:
    content_hash = hashlib.sha256(content.strip().lower().encode("utf-8")).hexdigest()
    return Memory(
        content=content,
        content_hash=content_hash,
        tags=["__test__"],
        memory_type="note",
        created_at=created_at,
    )


class TestAgainstRealStorage:
    @pytest.mark.asyncio
    @patch(f"{_MOD}.CONTRADICTION_ENABLED", True)
    # Widen the band so two related sentences land in it regardless of the
    # exact embedding model — the band itself is not what this test checks.
    @patch(f"{_MOD}.SIMILARITY_MIN", 0.0)
    @patch(f"{_MOD}.SIMILARITY_MAX", 0.99)
    async def test_older_memory_hidden_and_edge_written(self, real_storage):
        storage, graph = real_storage
        older = _real_memory("The production memory backend is sqlite_vec.", OLD_T)
        newer = _real_memory("The production memory backend is now Milvus.", NEW_T)
        for m in (older, newer):
            ok, msg = await storage.store(m)
            assert ok, msg

        result = await detect_contradictions(storage, dry_run=False, graph=graph)

        assert "error" not in result, result
        assert result["pairs_detected"] == 1
        assert result["superseded_marked"] == 1
        assert result["edges_created"] == 1

        default = await storage.search_memories(query="production memory backend", limit=10)
        default_hashes = {m["content_hash"] for m in default["memories"]}
        assert newer.content_hash in default_hashes
        assert older.content_hash not in default_hashes

        with_superseded = await storage.search_memories(
            query="production memory backend", limit=10, include_superseded=True
        )
        assert older.content_hash in {m["content_hash"] for m in with_superseded["memories"]}

        rel_types = await graph.get_relationship_types(older.content_hash)
        assert rel_types.get("contradicts", 0) >= 1

    @pytest.mark.asyncio
    @patch(f"{_MOD}.CONTRADICTION_ENABLED", True)
    @patch(f"{_MOD}.SIMILARITY_MIN", 0.0)
    @patch(f"{_MOD}.SIMILARITY_MAX", 0.99)
    async def test_protected_older_memory_stays_visible(self, real_storage):
        storage, graph = real_storage
        older = _real_memory("The production memory backend is sqlite_vec.", OLD_T)
        older.tags = ["__test__", "important"]
        newer = _real_memory("The production memory backend is now Milvus.", NEW_T)
        for m in (older, newer):
            ok, msg = await storage.store(m)
            assert ok, msg

        result = await detect_contradictions(storage, dry_run=False, graph=graph)

        assert result["pairs_detected"] == 0
        assert result["protected_skipped"] == 1
        default = await storage.search_memories(query="production memory backend", limit=10)
        assert older.content_hash in {m["content_hash"] for m in default["memories"]}

    @pytest.mark.asyncio
    @patch(f"{_MOD}.CONTRADICTION_ENABLED", True)
    @patch(f"{_MOD}.SIMILARITY_MIN", 0.0)
    @patch(f"{_MOD}.SIMILARITY_MAX", 0.99)
    async def test_second_run_is_idempotent(self, real_storage):
        storage, graph = real_storage
        for m in (
            _real_memory("The production memory backend is sqlite_vec.", OLD_T),
            _real_memory("The production memory backend is now Milvus.", NEW_T),
        ):
            ok, msg = await storage.store(m)
            assert ok, msg

        first = await detect_contradictions(storage, dry_run=False, graph=graph)
        second = await detect_contradictions(storage, dry_run=False, graph=graph)
        assert first["pairs_detected"] == 1
        assert second["pairs_detected"] == 0
