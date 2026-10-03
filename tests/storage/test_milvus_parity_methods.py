# Copyright 2024 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Milvus implementations of the storage contract the API assumed everywhere else.

Each of these previously fell through to a MemoryStorage base stub or was
missing outright, which showed up at the API boundary as an empty
/api/analytics/relationship-types, a 500 on /api/analytics/graph-visualization,
a 501 on /api/manage/untagged/count, sampled (wrong) type counts, or
"versioned updates are not supported by the current storage backend".

Follows the mocked-client approach of test_milvus_web_api_methods.py rather
than pytest.importorskip("pymilvus"), so these run on CI runners that install
only .[dev,sqlite]. End-to-end coverage against real Milvus Lite lives in
tests/test_milvus_graph_entities.py.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.milvus import MilvusDeleteError, MilvusMemoryStorage


def _make_storage(uri: str = "./milvus.db") -> MilvusMemoryStorage:
    """Return a MilvusMemoryStorage with __init__ skipped and the client mocked."""
    storage = MilvusMemoryStorage.__new__(MilvusMemoryStorage)
    storage.uri = uri
    storage.collection_name = "unit_test_collection"
    storage.embedding_dimension = 4
    storage.embedding_model_name = "test-model"
    storage.embedding_model = MagicMock()
    storage._initialized = True
    storage.client = MagicMock()
    storage._has_content_lower = True
    storage._has_bm25 = False
    storage._has_access_collection = False
    storage._lock = None
    storage._call_client = AsyncMock()
    storage._generate_embedding = MagicMock(return_value=[0.1, 0.2, 0.3, 0.4])
    storage._fetch_live_hashes = AsyncMock(side_effect=lambda hashes: set(hashes))
    return storage


def _edge(
    source: str,
    target: str,
    rel_type: str = "related",
    similarity: float = 0.8,
) -> Dict[str, Any]:
    return {
        "id": f"{source}:{target}",
        "source_hash": source,
        "target_hash": target,
        "relationship_type": rel_type,
        "similarity": similarity,
        "connection_types": '["semantic"]',
    }


def _memory(content_hash: str, content: str = "body", **kw) -> Memory:
    now = time.time()
    return Memory(
        content=content,
        content_hash=content_hash,
        tags=kw.get("tags", ["t"]),
        memory_type=kw.get("memory_type", "note"),
        metadata=kw.get("metadata", {}),
        created_at=now,
        updated_at=now,
    )


# -- get_relationship_type_distribution ---------------------------------------


class TestRelationshipTypeDistribution:

    @pytest.mark.asyncio
    async def test_counts_by_type_descending(self):
        storage = _make_storage()
        storage._drain_all_graph_edges = AsyncMock(return_value=[
            {"id": "1", "relationship_type": "related"},
            {"id": "2", "relationship_type": "causes"},
            {"id": "3", "relationship_type": "related"},
            {"id": "4", "relationship_type": "related"},
        ])

        result = await storage.get_relationship_type_distribution()

        assert result == {"related": 3, "causes": 1}
        assert list(result) == ["related", "causes"]

    @pytest.mark.asyncio
    async def test_missing_type_counted_as_untyped(self):
        """Matches the SQLite CASE that maps NULL/'' to 'untyped'."""
        storage = _make_storage()
        storage._drain_all_graph_edges = AsyncMock(return_value=[
            {"id": "1", "relationship_type": None},
            {"id": "2", "relationship_type": ""},
        ])

        assert await storage.get_relationship_type_distribution() == {"untyped": 2}

    @pytest.mark.asyncio
    async def test_no_graph_collection_returns_empty(self):
        storage = _make_storage()
        storage._drain_all_graph_edges = AsyncMock(return_value=[])

        assert await storage.get_relationship_type_distribution() == {}


# -- get_graph_visualization_data ---------------------------------------------


class TestGraphVisualizationData:

    @pytest.mark.asyncio
    async def test_returns_meta_key_when_empty(self):
        """The API response model requires meta; omitting it surfaced as a 500."""
        storage = _make_storage()
        storage._drain_all_graph_edges = AsyncMock(return_value=[])

        result = await storage.get_graph_visualization_data(limit=10, min_connections=2)

        assert result["nodes"] == []
        assert result["edges"] == []
        assert result["meta"] == {
            "total_nodes": 0,
            "total_edges": 0,
            "min_connections": 2,
            "limit": 10,
        }

    @pytest.mark.asyncio
    async def test_builds_nodes_and_edges(self):
        storage = _make_storage()
        storage._drain_all_graph_edges = AsyncMock(return_value=[
            _edge("a", "b", "causes", 0.9),
            _edge("a", "c", "related", 0.7),
        ])
        storage._fetch_memories_by_hashes = AsyncMock(return_value=[
            _memory("a", "alpha"), _memory("b", "bravo"), _memory("c", "charlie"),
        ])

        result = await storage.get_graph_visualization_data()

        # Nodes are memory rows that are edge sources, matching SQLite's
        # INNER JOIN on source_hash. b/c are targets only, so their candidate
        # rows keep 'a' connection-eligible but are not rendered as nodes.
        assert {n["id"] for n in result["nodes"]} == {"a"}
        node_a = next(n for n in result["nodes"] if n["id"] == "a")
        assert node_a["connections"] == 2
        assert node_a["type"] == "note"
        assert result["meta"]["total_nodes"] == 1
        assert result["meta"]["total_edges"] == 0

    @pytest.mark.asyncio
    async def test_min_connections_filters_nodes(self):
        storage = _make_storage()
        storage._drain_all_graph_edges = AsyncMock(return_value=[
            _edge("a", "b"), _edge("a", "c"), _edge("d", "e"),
        ])
        storage._fetch_memories_by_hashes = AsyncMock(
            side_effect=lambda hashes: [_memory(h) for h in hashes]
        )

        result = await storage.get_graph_visualization_data(min_connections=2)

        # 'a' has 2 distinct targets; 'd' has 1 and is excluded.
        assert [n["id"] for n in result["nodes"]] == ["a"]

    @pytest.mark.asyncio
    async def test_connection_count_uses_distinct_targets(self):
        """Duplicate edges to the same target count once, as COUNT(DISTINCT) does."""
        storage = _make_storage()
        storage._drain_all_graph_edges = AsyncMock(return_value=[
            _edge("a", "b", "causes"), _edge("a", "b", "related"),
        ])
        storage._fetch_memories_by_hashes = AsyncMock(
            side_effect=lambda hashes: [_memory(h) for h in hashes]
        )

        result = await storage.get_graph_visualization_data()
        node_a = next(n for n in result["nodes"] if n["id"] == "a")
        assert node_a["connections"] == 1

    @pytest.mark.asyncio
    async def test_edges_to_unrendered_nodes_dropped(self):
        """No dangling edges: both endpoints must be in the node set."""
        storage = _make_storage()
        storage._drain_all_graph_edges = AsyncMock(return_value=[_edge("a", "b")])
        storage._fetch_memories_by_hashes = AsyncMock(
            return_value=[_memory("a"), _memory("b")]
        )

        result = await storage.get_graph_visualization_data()

        assert [n["id"] for n in result["nodes"]] == ["a"]
        assert result["edges"] == []

    @pytest.mark.asyncio
    async def test_deleted_targets_do_not_consume_node_limit(self):
        """Ranking must ignore edges to rows that no longer exist."""
        storage = _make_storage()
        storage._drain_all_graph_edges = AsyncMock(return_value=[
            _edge("dead-source", "gone-1"),
            _edge("dead-source", "gone-2"),
            _edge("live-source", "live-target"),
        ])
        live_hashes = {"live-source", "live-target"}
        storage._fetch_live_hashes = AsyncMock(
            side_effect=lambda hashes: set(hashes) & live_hashes
        )
        storage._fetch_memories_by_hashes = AsyncMock(
            side_effect=lambda hashes: [_memory(h) for h in hashes]
        )

        result = await storage.get_graph_visualization_data(limit=1)

        assert [n["id"] for n in result["nodes"]] == ["live-source"]
        assert result["nodes"][0]["connections"] == 1
        assert result["edges"] == []

    @pytest.mark.asyncio
    async def test_bounded_graph_loads_only_selected_source_records(self):
        """A small graph request must not fetch full rows for every endpoint."""
        storage = _make_storage()
        storage._drain_all_graph_edges = AsyncMock(return_value=[
            _edge("a", "b"), _edge("c", "d"), _edge("e", "f"),
        ])
        storage._fetch_memories_by_hashes = AsyncMock(
            side_effect=lambda hashes: [_memory(h) for h in hashes]
        )

        result = await storage.get_graph_visualization_data(limit=1)

        requested = storage._fetch_memories_by_hashes.await_args.args[0]
        assert len(requested) == 1
        assert len(result["nodes"]) == 1

    @pytest.mark.asyncio
    async def test_quality_score_read_from_metadata(self):
        storage = _make_storage()
        storage._drain_all_graph_edges = AsyncMock(return_value=[_edge("a", "b")])
        storage._fetch_memories_by_hashes = AsyncMock(return_value=[
            _memory("a", metadata={"quality_score": 0.91}), _memory("b"),
        ])

        result = await storage.get_graph_visualization_data()
        node_a = next(n for n in result["nodes"] if n["id"] == "a")
        assert node_a["quality_score"] == 0.91

    @pytest.mark.asyncio
    async def test_entity_edges_do_not_count_as_connections(self):
        """has_entity targets are not memory nodes and must not affect the cut."""
        storage = _make_storage()
        storage._drain_all_graph_edges = AsyncMock(return_value=[
            _edge("a", "ent:entity-key", "has_entity"),
        ])
        storage._fetch_memories_by_hashes = AsyncMock(return_value=[_memory("a")])

        result = await storage.get_graph_visualization_data()

        assert result["nodes"] == []
        assert result["edges"] == []


# -- get_type_counts ----------------------------------------------------------


class TestGetTypeCounts:

    @pytest.mark.asyncio
    async def test_groups_by_memory_type(self):
        storage = _make_storage()
        storage._drain_rows = AsyncMock(return_value=[
                {"id": "1", "memory_type": "note"},
                {"id": "2", "memory_type": "note"},
                {"id": "3", "memory_type": "decision"},
        ])

        result = await storage.get_type_counts()

        assert result == {"note": 2, "decision": 1}

    @pytest.mark.asyncio
    async def test_missing_type_is_untyped(self):
        storage = _make_storage()
        storage._drain_rows = AsyncMock(return_value=[
            {"id": "1", "memory_type": None},
            {"id": "2", "memory_type": ""},
        ])

        assert await storage.get_type_counts() == {"untyped": 2}

    @pytest.mark.asyncio
    async def test_soft_deleted_rows_are_not_counted(self):
        storage = _make_storage()
        storage._drain_rows = AsyncMock(return_value=[
                {"id": "1", "memory_type": "note", "metadata": "{}"},
                {
                    "id": "2",
                    "memory_type": "note",
                    "metadata": '{"deleted_at": 123}',
                },
        ])

        assert await storage.get_type_counts() == {"note": 1}


# -- count_untagged_memories / delete_untagged_memories -----------------------


class TestUntaggedMemories:

    @pytest.mark.asyncio
    async def test_count_uses_server_side_filter(self):
        storage = _make_storage()
        storage._drain_rows = AsyncMock(return_value=[
            {"id": f"h{i}", "metadata": "{}"} for i in range(7)
        ])

        assert await storage.count_untagged_memories() == 7

        args = storage._drain_rows.await_args.args
        # Tags are stored with sentinel commas, so both forms mean "no tags".
        assert args[0] == 'tags == "" or tags == ","'
        assert args[1] == ["id", "metadata"]

    @pytest.mark.asyncio
    async def test_count_excludes_soft_deleted_rows(self):
        storage = _make_storage()
        storage._drain_rows = AsyncMock(return_value=[
            {"id": "live", "metadata": "{}"},
            {"id": "deleted", "metadata": '{"deleted_at": 123}'},
        ])

        assert await storage.count_untagged_memories() == 1

    @pytest.mark.asyncio
    async def test_count_returns_zero_on_error(self):
        storage = _make_storage()
        storage._drain_rows = AsyncMock(side_effect=RuntimeError("boom"))

        assert await storage.count_untagged_memories() == 0

    @pytest.mark.asyncio
    async def test_delete_matches_same_predicate(self):
        storage = _make_storage()
        storage._delete_matching = AsyncMock(return_value=(3, "Successfully deleted 3"))

        count, _message = await storage.delete_untagged_memories()

        assert count == 3
        assert storage._delete_matching.await_args.args[0] == 'tags == "" or tags == ","'
        assert storage._delete_matching.await_args.kwargs["require_live"] is True

    @pytest.mark.asyncio
    async def test_delete_chunks_large_result_sets(self):
        storage = _make_storage()
        storage._collect_hashes = AsyncMock(
            return_value=[f"h{i}" for i in range(storage._GET_BY_ID_CHUNK + 1)]
        )

        count, hashes = await storage._delete_matching_parts(
            'tags == "" or tags == ","',
            "",
            require_live=True,
        )

        assert count == storage._GET_BY_ID_CHUNK + 1
        assert len(hashes) == count
        assert storage._call_client.await_count == 2

    @pytest.mark.asyncio
    async def test_delete_by_tags_chunks_large_result_sets(self):
        storage = _make_storage()
        storage._collect_hashes = AsyncMock(
            return_value=[f"h{i}" for i in range(storage._GET_BY_ID_CHUNK + 1)]
        )

        count, _message, hashes = await storage.delete_by_tags(["shared"])

        assert count == storage._GET_BY_ID_CHUNK + 1
        assert len(hashes) == count
        assert storage._call_client.await_count == 2

    @pytest.mark.asyncio
    async def test_delete_by_tags_reports_partial_failure(self):
        storage = _make_storage()
        all_hashes = [f"h{i}" for i in range(storage._GET_BY_ID_CHUNK + 1)]
        storage._collect_hashes = AsyncMock(return_value=all_hashes)
        storage._call_client = AsyncMock(
            side_effect=[None, RuntimeError("second chunk failed")]
        )

        with pytest.raises(MilvusDeleteError) as exc_info:
            await storage.delete_by_tags(["shared"])

        assert exc_info.value.deleted_count == storage._GET_BY_ID_CHUNK
        assert exc_info.value.deleted_hashes == all_hashes[:storage._GET_BY_ID_CHUNK]
        assert "Deleted 500 of 501 memories" in str(exc_info.value)
        assert "second chunk failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_delete_by_tags_failure_is_not_reported_as_no_matches(self):
        storage = _make_storage()
        storage._collect_hashes = AsyncMock(return_value=["only"])
        storage._call_client = AsyncMock(side_effect=RuntimeError("first chunk failed"))

        with pytest.raises(MilvusDeleteError) as exc_info:
            await storage.delete_by_tags(["shared"])

        assert exc_info.value.deleted_count == 0
        assert exc_info.value.deleted_hashes == []
        assert "first chunk failed" in str(exc_info.value)


class TestDrainRowsLock:

    @pytest.mark.asyncio
    async def test_full_iterator_scan_holds_write_lock(self):
        storage = _make_storage()
        storage._write_lock = asyncio.Lock()
        storage._call_client = MilvusMemoryStorage._call_client.__get__(
            storage, MilvusMemoryStorage,
        )
        scan_started = threading.Event()
        release_scan = threading.Event()
        delete_started = threading.Event()

        class Iterator:
            def __init__(self):
                self.first = True

            def next(self):
                if self.first:
                    self.first = False
                    scan_started.set()
                    assert release_scan.wait(timeout=3)
                    return [{"id": "h1"}]
                return []

            def close(self):
                pass

        class Client:
            def query_iterator(self, **_kwargs):
                return Iterator()

            def delete(self, **_kwargs):
                delete_started.set()

        storage.client = Client()

        scan = asyncio.create_task(storage._drain_rows("", ["id"]))
        assert await asyncio.to_thread(scan_started.wait, 3)
        competitor = asyncio.create_task(storage._call_client(
            "delete", collection_name=storage.collection_name, ids=["h1"],
        ))
        await asyncio.sleep(0.1)
        assert not delete_started.is_set()

        release_scan.set()
        assert await scan == [{"id": "h1"}]
        await competitor
        assert delete_started.is_set()


# -- update_memory_versioned --------------------------------------------------


class TestUpdateMemoryVersioned:

    @pytest.mark.asyncio
    async def test_stores_new_version_and_supersedes_old(self):
        storage = _make_storage()
        original = _memory("oldhash", "v1", tags=["a"], memory_type="note")
        storage.get_by_hash = AsyncMock(return_value=original)
        storage.store = AsyncMock(return_value=(True, "ok"))
        storage.update_memory_metadata = AsyncMock(return_value=(True, "ok"))

        ok, message, new_hash = await storage.update_memory_versioned(
            "oldhash", "v2", reason="corrected",
        )

        assert ok is True, message
        assert new_hash and new_hash != "oldhash"

        # Old memory is annotated, not overwritten.
        meta_args = storage.update_memory_metadata.await_args
        assert meta_args.args[0] == "oldhash"
        assert meta_args.args[1]["metadata"]["superseded_by"] == new_hash
        assert meta_args.args[1]["metadata"]["evolution_reason"] == "corrected"
        assert meta_args.kwargs["preserve_timestamps"] is True

    @pytest.mark.asyncio
    async def test_inherits_tags_and_type_when_not_overridden(self):
        storage = _make_storage()
        storage.get_by_hash = AsyncMock(
            return_value=_memory("oldhash", "v1", tags=["x", "y"], memory_type="decision")
        )
        storage.store = AsyncMock(return_value=(True, "ok"))
        storage.update_memory_metadata = AsyncMock(return_value=(True, "ok"))

        await storage.update_memory_versioned("oldhash", "v2")

        stored: Memory = storage.store.await_args.args[0]
        assert stored.tags == ["x", "y"]
        assert stored.memory_type == "decision"

    @pytest.mark.asyncio
    async def test_explicit_overrides_win(self):
        storage = _make_storage()
        storage.get_by_hash = AsyncMock(
            return_value=_memory("oldhash", "v1", tags=["x"], memory_type="note")
        )
        storage.store = AsyncMock(return_value=(True, "ok"))
        storage.update_memory_metadata = AsyncMock(return_value=(True, "ok"))

        await storage.update_memory_versioned(
            "oldhash", "v2", new_tags=["z"], new_memory_type="decision",
        )

        stored: Memory = storage.store.await_args.args[0]
        assert stored.tags == ["z"]
        assert stored.memory_type == "decision"

    @pytest.mark.asyncio
    async def test_missing_memory_reports_failure(self):
        storage = _make_storage()
        storage.get_by_hash = AsyncMock(return_value=None)

        ok, message, new_hash = await storage.update_memory_versioned("nope", "v2")

        assert ok is False
        assert new_hash is None
        assert "not found" in message

    @pytest.mark.asyncio
    async def test_old_memory_untouched_when_store_fails(self):
        storage = _make_storage()
        storage.get_by_hash = AsyncMock(return_value=_memory("oldhash", "v1"))
        storage.store = AsyncMock(return_value=(False, "duplicate"))
        storage.update_memory_metadata = AsyncMock(return_value=(True, "ok"))

        ok, _, new_hash = await storage.update_memory_versioned("oldhash", "v2")

        assert ok is False
        assert new_hash is None
        storage.update_memory_metadata.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_failed_lineage_write_rolls_back_new_version(self):
        storage = _make_storage()
        # The second lookup represents the old delete() cleanup path: Milvus
        # Lite can acknowledge the store before the new row is visible, so the
        # rollback must not depend on get_by_hash() seeing it.
        storage.get_by_hash = AsyncMock(side_effect=[
            _memory("oldhash", "v1"),
            None,
        ])
        storage.store = AsyncMock(return_value=(True, "ok"))
        storage.update_memory_metadata = AsyncMock(
            return_value=(False, "metadata upsert failed")
        )

        ok, message, new_hash = await storage.update_memory_versioned(
            "oldhash", "v2",
        )

        assert ok is False
        assert new_hash is None
        assert "Failed to link new version" in message
        cleanup_hash = storage.store.await_args.args[0].content_hash
        storage._call_client.assert_awaited_once_with(
            "delete",
            collection_name=storage.collection_name,
            ids=[cleanup_hash],
        )

    @pytest.mark.asyncio
    async def test_rollback_failure_is_reported(self):
        storage = _make_storage()
        storage.get_by_hash = AsyncMock(return_value=_memory("oldhash", "v1"))
        storage.store = AsyncMock(return_value=(True, "ok"))
        storage.update_memory_metadata = AsyncMock(
            return_value=(False, "metadata upsert failed")
        )
        storage._call_client = AsyncMock(side_effect=RuntimeError("delete unavailable"))

        ok, message, new_hash = await storage.update_memory_versioned(
            "oldhash", "v2",
        )

        assert ok is False
        assert new_hash is None
        assert "rollback may have left an orphan" in message

    @pytest.mark.asyncio
    async def test_already_superseded_memory_is_rejected(self):
        storage = _make_storage()
        storage.get_by_hash = AsyncMock(return_value=_memory(
            "oldhash",
            "v1",
            metadata={"superseded_by": "existing-newer-version"},
        ))
        storage.store = AsyncMock(return_value=(True, "ok"))

        ok, message, new_hash = await storage.update_memory_versioned(
            "oldhash", "v2",
        )

        assert ok is False
        assert new_hash is None
        assert "no longer current" in message
        storage.store.assert_not_awaited()


# -- get_stats / lite_db_path -------------------------------------------------


class TestStatsAndPaths:

    @pytest.mark.asyncio
    async def test_stats_include_storage_backend_key(self):
        """Analytics reads "storage_backend"; without it the UI showed "unknown"."""
        storage = _make_storage()
        storage._call_client = AsyncMock(return_value=[{"count(*)": 0}])
        storage.get_all_tags = AsyncMock(return_value=["a"])

        stats = await storage.get_stats()

        assert stats["storage_backend"] == "milvus"
        assert stats["backend"] == "milvus"

    @pytest.mark.asyncio
    async def test_stats_report_size_for_milvus_lite(self, tmp_path):
        db_file = tmp_path / "milvus.db"
        db_file.write_bytes(b"x" * 2048)

        storage = _make_storage(uri=str(db_file))
        storage._call_client = AsyncMock(return_value=[{"count(*)": 0}])
        storage.get_all_tags = AsyncMock(return_value=[])

        stats = await storage.get_stats()

        assert stats["database_size_bytes"] == 2048
        assert stats["database_size_mb"] == round(2048 / (1024 * 1024), 2)

    @pytest.mark.asyncio
    async def test_stats_omit_size_for_remote_milvus(self):
        """A server/Zilliz deployment has no local file; report nothing, not 0."""
        storage = _make_storage(uri="https://in01-abc.zillizcloud.com:19530")
        storage._call_client = AsyncMock(return_value=[{"count(*)": 0}])
        storage.get_all_tags = AsyncMock(return_value=[])

        stats = await storage.get_stats()

        assert "database_size_mb" not in stats
        assert "database_size_bytes" not in stats

    def test_lite_db_path_local_vs_remote(self):
        assert _make_storage("./milvus.db").lite_db_path.endswith("milvus.db")
        assert _make_storage("http://localhost:19530").lite_db_path is None
        assert _make_storage("https://x.zillizcloud.com").lite_db_path is None

    def test_no_db_path_attribute(self):
        """db_path would make SQLite duck-typing misfire across the codebase."""
        assert not hasattr(_make_storage(), "db_path")
