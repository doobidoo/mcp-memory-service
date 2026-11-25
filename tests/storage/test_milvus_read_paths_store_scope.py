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

"""Milvus read paths confine to the claimed store (#1958).

#1962 taught the hash-addressed operations to confine and left
`get_by_hash_is_store_scoped` at `False`, because the read paths still ignored
`store`: a `read:<store>` credential could have searched, listed, counted and
recalled across every partition. Eleven methods accepted `store` and dropped it,
several with a comment saying so.

This pins the confinement at the layer each method drives — the filter
expression it hands to the query helper — so a later refactor cannot quietly
stop passing it. The legacy case is the other half: a collection without the
`store` field refuses a named store instead of serving every partition, which is
what lets the capability flag be `True`.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from mcp_memory_service.storage.milvus import MilvusMemoryStorage

STORE = "alpha"
OTHER = "beta"


def _make_storage(has_store: bool = True) -> MilvusMemoryStorage:
    """A MilvusMemoryStorage with __init__ skipped and the client mocked."""
    storage = MilvusMemoryStorage.__new__(MilvusMemoryStorage)
    storage.uri = "./milvus.db"
    storage.collection_name = "unit_test_collection"
    storage.embedding_dimension = 4
    storage.embedding_model_name = "test-model"
    storage.embedding_model = MagicMock()
    storage._initialized = True
    storage.client = MagicMock()
    storage._has_content_lower = True
    storage._has_bm25 = False
    storage._has_access_collection = False
    storage._has_store = has_store
    storage._legacy_store_warned = False
    storage._graph_has_store = None
    storage.semantic_dedup_enabled = False
    storage._generate_embedding = MagicMock(return_value=[0.1, 0.2, 0.3, 0.4])
    storage._detect_conflicts = AsyncMock(return_value=[])
    storage._delete_graph_edges = AsyncMock()
    return storage


def _captured_filter(call: Any) -> str:
    """The filter expression a mocked helper was called with."""
    kwargs = call.kwargs or {}
    if "filter_expr" in kwargs:
        return kwargs["filter_expr"]
    if "store_filter" in kwargs:
        return kwargs["store_filter"] or ""
    args = call.args or ()
    return next((a for a in args if isinstance(a, str)), "")


class TestFlagIsClaimed:
    def test_milvus_claims_store_scoping(self):
        """Only safe because every read path confines or refuses below."""
        assert MilvusMemoryStorage.get_by_hash_is_store_scoped is True


class TestQueryFiltersCarryTheStore:
    async def test_get_by_exact_content(self):
        storage = _make_storage()
        storage._query_memories = AsyncMock(return_value=[])

        await storage.get_by_exact_content("needle", store=STORE)

        assert f'store == "{STORE}"' in _captured_filter(storage._query_memories.call_args)

    async def test_get_memories_by_time_range(self):
        storage = _make_storage()
        storage._query_memories = AsyncMock(return_value=[])

        await storage.get_memories_by_time_range(1.0, 2.0, store=STORE)

        assert f'store == "{STORE}"' in _captured_filter(storage._query_memories.call_args)

    async def test_get_all_memories(self):
        storage = _make_storage()
        storage._query_memories = AsyncMock(return_value=[])

        await storage.get_all_memories(store=STORE)

        assert f'store == "{STORE}"' in _captured_filter(storage._query_memories.call_args)

    async def test_search_memories(self):
        storage = _make_storage()
        storage._query_memories = AsyncMock(return_value=[])
        storage._drain_rows = AsyncMock(return_value=[])

        # Without a query the scan branch runs and hands its filter to
        # _query_memories(filter_expr=...).
        await storage.search_memories(query=None, store=STORE)

        assert f'store == "{STORE}"' in _captured_filter(storage._query_memories.call_args)

    async def test_get_memory_timestamps(self):
        storage = _make_storage()
        storage._drain_rows = AsyncMock(return_value=[])

        await storage.get_memory_timestamps(store=STORE)

        assert f'store == "{STORE}"' in _captured_filter(storage._drain_rows.call_args)

    async def test_get_all_tags_with_counts(self):
        storage = _make_storage()
        storage._scan_tags = AsyncMock(return_value=[])

        await storage.get_all_tags_with_counts(store=STORE)

        assert f'store == "{STORE}"' in (storage._scan_tags.call_args.args[0] or "")

    async def test_delete_by_tags(self):
        storage = _make_storage()
        storage._collect_hashes = AsyncMock(return_value=[])

        await storage.delete_by_tags(["t"], store=STORE)

        assert f'store == "{STORE}"' in _captured_filter(storage._collect_hashes.call_args)

    async def test_retrieve(self):
        storage = _make_storage()
        storage._embed_query = MagicMock(return_value=[0.1, 0.2, 0.3, 0.4])
        storage._run_search = AsyncMock(return_value=[])

        await storage.retrieve("query", store=STORE)

        assert f'store == "{STORE}"' in _captured_filter(storage._run_search.call_args)

    async def test_recall(self):
        storage = _make_storage()
        storage._recall_active_time_window = AsyncMock(return_value=[])

        await storage.recall(store=STORE)

        call = storage._recall_active_time_window.call_args
        assert f'store == "{STORE}"' in (call.kwargs["store_filter"] or "")


class TestCountAndGraph:
    async def test_count_all_memories(self):
        storage = _make_storage()
        storage._drain_rows = AsyncMock(return_value=[])
        storage._count_stale_memories = AsyncMock(return_value=0)

        await storage.count_all_memories(stale_days=1, store=STORE)

        call = storage._count_stale_memories.call_args
        assert f'store == "{STORE}"' in (call.args[0] or "")

    async def test_get_graph_visualization_data(self):
        storage = _make_storage()
        storage._graph_has_store = True
        storage._drain_all_graph_edges = AsyncMock(return_value=[])

        await storage.get_graph_visualization_data(store=STORE)

        call = storage._drain_all_graph_edges.call_args
        assert f'store == "{STORE}"' in (call.kwargs.get("filter_expr") or "")

    async def test_graph_view_refuses_when_the_edges_carry_no_store(self):
        storage = _make_storage()
        storage._graph_has_store = False
        storage._drain_all_graph_edges = AsyncMock(return_value=[])

        result = await storage.get_graph_visualization_data(store=STORE)

        assert result["nodes"] == []
        storage._drain_all_graph_edges.assert_not_called()


class TestLegacyCollectionFailsClosed:
    @pytest.mark.parametrize(
        "call",
        [
            lambda s: s.get_by_exact_content("needle", store=OTHER),
            lambda s: s.get_memories_by_time_range(1.0, 2.0, store=OTHER),
            lambda s: s.get_all_memories(store=OTHER),
            lambda s: s.get_memory_timestamps(store=OTHER),
            lambda s: s.get_all_tags_with_counts(store=OTHER),
            lambda s: s.delete_by_tags(["t"], store=OTHER),
            lambda s: s.retrieve("query", store=OTHER),
            lambda s: s.recall(store=OTHER),
            lambda s: s.get_graph_visualization_data(store=OTHER),
        ],
    )
    async def test_a_named_store_is_refused_without_the_field(self, call):
        storage = _make_storage(has_store=False)
        storage._drain_rows = AsyncMock(return_value=[])
        storage._query_memories = AsyncMock(return_value=[])
        storage._scan_tags = AsyncMock(return_value=[])
        storage._collect_hashes = AsyncMock(return_value=[])
        storage._drain_all_graph_edges = AsyncMock(return_value=[])
        storage._embed_query = MagicMock(return_value=[0.1, 0.2, 0.3, 0.4])
        storage._run_search = AsyncMock(return_value=[])

        result = await call(storage)

        if isinstance(result, dict):
            assert result["nodes"] == []
        elif isinstance(result, tuple):
            assert result[0] == 0
        else:
            assert result in ([], 0), result
        storage._drain_rows.assert_not_called()
        storage._query_memories.assert_not_called()
        storage._scan_tags.assert_not_called()
        storage._collect_hashes.assert_not_called()
        storage._run_search.assert_not_called()

    async def test_unscoped_reads_stay_unscoped(self):
        storage = _make_storage()
        storage._query_memories = AsyncMock(return_value=[])

        await storage.get_all_memories(store="all")

        assert "store ==" not in _captured_filter(storage._query_memories.call_args)


class TestRecallWindowSplitting:
    """The split helpers keep the newest-first guarantee when ranges saturate."""

    @staticmethod
    def _memory(index: int):
        from mcp_memory_service.models.memory import Memory

        memory = Memory(content="x", content_hash=f"h{index}")
        memory.created_at = float(index)
        return memory

    def test_remember_live_keeps_only_the_recallable_and_trims_to_newest(self):
        storage = _make_storage()
        storage._is_recallable_memory = staticmethod(lambda m: m.created_at != 0.0)
        candidates: dict = {}
        memories = [self._memory(i) for i in range(6)]

        storage._remember_live(candidates, memories, n_results=1, batch_size=1)

        # h0 is not recallable; the pool is trimmed to the newest single result.
        assert set(candidates) == {"h5"}
