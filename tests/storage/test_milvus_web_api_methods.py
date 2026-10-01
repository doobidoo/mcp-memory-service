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

"""Milvus overrides for the storage methods the web API calls unguarded (#213).

web/api/memories.py::get_tags called ``storage.get_all_tags_with_counts()``
without a hasattr guard and turned the resulting AttributeError into HTTP 501,
which is what the Browse tab showed on a Milvus deployment. ``recall`` and
``get_largest_memories`` were missing from the same backend and are called just
as unguarded from web/api/search.py and web/api/analytics.py.

Deliberately no ``pytest.importorskip("pymilvus")``: the other Milvus test
modules skip themselves out of existence on a runner without the optional
Milvus extras, which is every CI runner we have (ci.yml installs
``.[dev,sqlite]``). storage/milvus.py imports fine without pymilvus, so these
tests mock the client instead and actually run.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

import mcp_memory_service.storage.milvus as milvus_module
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.milvus import MilvusMemoryStorage


def _make_storage() -> MilvusMemoryStorage:
    """Return a MilvusMemoryStorage with __init__ skipped and the client mocked."""
    storage = MilvusMemoryStorage.__new__(MilvusMemoryStorage)
    storage.collection_name = "unit_test_collection"
    storage.embedding_dimension = 4
    storage.embedding_model_name = "test-model"
    storage.embedding_model = MagicMock()
    storage._initialized = True
    storage.client = MagicMock()
    storage._has_content_lower = True
    storage._has_bm25 = False
    storage._lock = None
    storage._call_client = AsyncMock()
    storage._generate_embedding = MagicMock(return_value=[0.1, 0.2, 0.3, 0.4])
    return storage


def _tag_rows(*tag_strings: str) -> List[Dict[str, Any]]:
    return [{"tags": t} for t in tag_strings]


def _entity(
    content_hash: str,
    content: str,
    created_at: Optional[float] = None,
    tags: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    now = time.time()
    return {
        "id": content_hash,
        "content": content,
        "tags": tags,
        "memory_type": "note",
        "metadata": json.dumps(metadata or {}),
        "created_at": created_at if created_at is not None else now,
        "updated_at": created_at if created_at is not None else now,
        "created_at_iso": None,
        "updated_at_iso": None,
    }


# -- get_all_tags_with_counts (the #213 symptom) -------------------------------


class TestGetAllTagsWithCounts:

    @pytest.mark.asyncio
    async def test_counts_occurrences_across_memories(self):
        storage = _make_storage()
        storage._call_client = AsyncMock(
            return_value=_tag_rows(",alpha,beta,", ",alpha,", ",beta,gamma,")
        )

        result = await storage.get_all_tags_with_counts()

        assert result == [
            {"tag": "alpha", "count": 2},
            {"tag": "beta", "count": 2},
            {"tag": "gamma", "count": 1},
        ]

    @pytest.mark.asyncio
    async def test_sorted_by_count_desc_then_tag(self):
        """The /api/tags contract documents descending count order."""
        storage = _make_storage()
        storage._call_client = AsyncMock(
            return_value=_tag_rows(",zulu,", ",alpha,", ",alpha,", ",alpha,", ",mike,", ",mike,")
        )

        result = await storage.get_all_tags_with_counts()

        assert [item["tag"] for item in result] == ["alpha", "mike", "zulu"]
        assert [item["count"] for item in result] == [3, 2, 1]

    @pytest.mark.asyncio
    async def test_tag_set_matches_get_all_tags(self):
        """Both readers must see the same tags — they share one row scan."""
        rows = _tag_rows(",alpha,beta,", ",beta,", "", None)
        storage = _make_storage()
        storage._call_client = AsyncMock(return_value=rows)

        with_counts = await storage.get_all_tags_with_counts()
        plain = await storage.get_all_tags()

        assert sorted(item["tag"] for item in with_counts) == plain

    @pytest.mark.asyncio
    async def test_empty_collection_returns_empty_list(self):
        storage = _make_storage()
        storage._call_client = AsyncMock(return_value=[])

        assert await storage.get_all_tags_with_counts() == []

    @pytest.mark.asyncio
    async def test_query_failure_returns_empty_list(self):
        """A backend error must not surface as a 500 on the Browse tab."""
        storage = _make_storage()
        storage._call_client = AsyncMock(side_effect=RuntimeError("milvus down"))

        assert await storage.get_all_tags_with_counts() == []

    @pytest.mark.asyncio
    async def test_uninitialized_returns_empty_list(self):
        storage = _make_storage()
        storage._initialized = False

        assert await storage.get_all_tags_with_counts() == []


# -- get_largest_memories (analytics.py, unguarded) ---------------------------


class TestGetLargestMemories:

    @pytest.mark.asyncio
    async def test_returns_longest_content_first(self):
        storage = _make_storage()
        storage._iterate_all_rows = AsyncMock(return_value=[
            _entity("h1", "short"),
            _entity("h2", "x" * 100),
            _entity("h3", "medium content"),
        ])

        result = await storage.get_largest_memories(n=2)

        assert [m.content_hash for m in result] == ["h2", "h3"]

    @pytest.mark.asyncio
    async def test_respects_n(self):
        storage = _make_storage()
        storage._iterate_all_rows = AsyncMock(return_value=[
            _entity(f"h{i}", "x" * i) for i in range(1, 20)
        ])

        assert len(await storage.get_largest_memories(n=5)) == 5

    @pytest.mark.asyncio
    async def test_uninitialized_returns_empty_list(self):
        storage = _make_storage()
        storage._initialized = False

        assert await storage.get_largest_memories() == []


# -- recall (search.py, unguarded) --------------------------------------------


class TestRecall:

    @pytest.mark.asyncio
    async def test_time_only_recall_filters_and_orders_by_recency(self):
        storage = _make_storage()
        captured: Dict[str, Any] = {}

        async def _query_window(filter_expr, limit, offset=0):
            captured["filter"] = filter_expr
            captured["limit"] = limit
            rows = [
                Memory(content="old memory", content_hash="old", created_at=100.0),
                Memory(content="new memory", content_hash="new", created_at=300.0),
            ]
            return rows, len(rows)

        storage._query_time_window = AsyncMock(side_effect=_query_window)

        results = await storage.recall(start_timestamp=50.0, end_timestamp=400.0)

        assert [r.memory.content_hash for r in results] == ["new", "old"]
        assert "created_at >= 50.0" in captured["filter"]
        assert "created_at <= 400.0" in captured["filter"]
        assert captured["limit"] == 100

    @pytest.mark.asyncio
    async def test_time_only_recall_respects_n_results(self):
        storage = _make_storage()
        captured: Dict[str, Any] = {}

        async def _query_window(filter_expr, limit, offset=0):
            captured["limit"] = limit
            rows = [
                Memory(content=f"memory {i}", content_hash=f"h{i}", created_at=float(i))
                for i in range(3)
            ]
            return rows, len(rows)

        storage._query_time_window = AsyncMock(side_effect=_query_window)

        results = await storage.recall(n_results=3)

        assert len(results) == 3
        assert captured["limit"] == 100

    @pytest.mark.asyncio
    async def test_time_only_recall_finds_newest_beyond_first_query_page(self):
        storage = _make_storage()
        memories = [
            Memory(content=f"memory {i}", content_hash=f"h{i}", created_at=float(i))
            for i in range(120)
        ]

        async def _query_window(filter_expr, limit, offset=0):
            bounds = [
                float(value)
                for value in re.findall(
                    r"created_at [<>=]+ ([0-9.eE+-]+)", filter_expr
                )
            ]
            lower, upper = min(bounds), max(bounds)
            matches = [
                memory
                for memory in memories
                if lower <= (memory.created_at or 0.0) <= upper
            ]
            page = matches[offset:offset + limit]
            return page, len(page)

        storage._query_time_window = AsyncMock(side_effect=_query_window)

        results = await storage.recall(
            start_timestamp=0.0, end_timestamp=119.0, n_results=3
        )

        assert [result.memory.content_hash for result in results] == ["h119", "h118", "h117"]

    @pytest.mark.asyncio
    async def test_time_only_recall_propagates_query_failure(self):
        storage = _make_storage()
        storage._query_time_window = AsyncMock(side_effect=RuntimeError("milvus down"))

        with pytest.raises(RuntimeError, match="milvus down"):
            await storage.recall(n_results=1)

    @pytest.mark.asyncio
    async def test_semantic_recall_applies_the_time_window_to_the_search(self):
        storage = _make_storage()
        captured: Dict[str, Any] = {}

        async def _run_search(embedding, filter_expr, fetch_n):
            captured["filter"] = filter_expr
            return [{
                "id": "h1",
                "distance": 0.9,
                "entity": _entity("h1", "hit", created_at=200.0),
                **_entity("h1", "hit", created_at=200.0),
            }]

        storage._run_search = AsyncMock(side_effect=_run_search)

        results = await storage.recall(
            query="anything", n_results=5, start_timestamp=100.0, end_timestamp=300.0
        )

        assert [r.memory.content_hash for r in results] == ["h1"]
        assert "created_at >= 100.0" in captured["filter"]
        assert "created_at <= 300.0" in captured["filter"]

    @pytest.mark.asyncio
    async def test_semantic_recall_without_a_window_passes_an_empty_filter(self):
        storage = _make_storage()
        captured: Dict[str, Any] = {}

        async def _run_search(embedding, filter_expr, fetch_n):
            captured["filter"] = filter_expr
            return []

        storage._run_search = AsyncMock(side_effect=_run_search)

        await storage.recall(query="anything")

        assert captured["filter"] == ""

    @pytest.mark.asyncio
    async def test_embedding_failure_returns_empty_list(self):
        storage = _make_storage()
        storage._embed_query = MagicMock(return_value=None)

        assert await storage.recall(query="anything") == []

    @pytest.mark.asyncio
    async def test_uninitialized_returns_empty_list(self):
        storage = _make_storage()
        storage._initialized = False

        assert await storage.recall(query="anything") == []


# -- the returned shape is what the API layer serializes ----------------------


@pytest.mark.asyncio
async def test_tags_with_counts_shape_matches_the_api_response_model():
    """web/api/memories.py builds TagResponse(tag=..., count=...) from each item."""
    storage = _make_storage()
    storage._call_client = AsyncMock(return_value=_tag_rows(",alpha,"))

    item = (await storage.get_all_tags_with_counts())[0]

    assert set(item) == {"tag", "count"}
    assert isinstance(item["tag"], str)
    assert isinstance(item["count"], int)


@pytest.mark.asyncio
async def test_largest_memories_returns_memory_objects():
    storage = _make_storage()
    storage._iterate_all_rows = AsyncMock(return_value=[_entity("h1", "content")])

    result = await storage.get_largest_memories(n=1)

    assert isinstance(result[0], Memory)


class TestRecallSupersededBacklog:
    """recall() must filter superseded rows *before* applying the limit.

    A fixed over-fetch multiple only moves the threshold: if the newest or
    best-ranked ``k * n_results`` candidates are all superseded, the caller still
    gets fewer results than requested — or none — while live memories sit beyond
    that window. Both branches must keep scanning until they have n_results live
    rows or the collection is genuinely exhausted.
    """

    N_RESULTS = 3
    SUPERSEDED_TOTAL = 40  # greater than 3x, 4x and 8x of N_RESULTS

    @pytest.mark.asyncio
    async def test_semantic_recall_past_a_superseded_backlog(self):
        storage = _make_storage()
        superseded_total = self.SUPERSEDED_TOTAL
        n_results = self.N_RESULTS

        async def _run_search(embedding, filter_expr, fetch_n):
            hits = []
            for i in range(min(fetch_n, superseded_total)):
                ent = _entity(
                    f"stale-{i}", f"obsolete {i}", metadata={"superseded_by": "current"}
                )
                hits.append({"id": ent["id"], "distance": 0.9, "entity": ent, **ent})
            if fetch_n > superseded_total:
                for i in range(n_results):
                    ent = _entity(f"live-{i}", f"live {i}")
                    hits.append({"id": ent["id"], "distance": 0.1, "entity": ent, **ent})
            return hits

        storage._run_search = AsyncMock(side_effect=_run_search)

        results = await storage.recall(query="probe", n_results=n_results)

        assert [r.memory.content_hash for r in results] == [
            "live-0",
            "live-1",
            "live-2",
        ], "a superseded backlog longer than any fixed multiple starved the result"

    @pytest.mark.asyncio
    async def test_time_only_recall_past_a_superseded_backlog(self):
        storage = _make_storage()
        superseded_total = self.SUPERSEDED_TOTAL
        n_results = self.N_RESULTS

        # Newest rows are all superseded; the live ones are older.
        rows = [
            _entity(
                f"stale-{i}",
                f"obsolete {i}",
                created_at=float(1000 - i),
                metadata={"superseded_by": "current"},
            )
            for i in range(superseded_total)
        ] + [
            _entity(f"live-{i}", f"live {i}", created_at=float(500 - i))
            for i in range(n_results)
        ]
        memories = [
            Memory(
                content=row["content"],
                content_hash=row["id"],
                metadata=json.loads(row["metadata"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

        async def _query_window(filter_expr, limit, offset=0):
            page = memories[:limit]
            return page, len(page)

        storage._query_time_window = AsyncMock(side_effect=_query_window)
        storage._iterate_all_rows = AsyncMock(
            side_effect=AssertionError("bounded recall must not full-scan")
        )

        results = await storage.recall(query=None, n_results=n_results)

        assert [r.memory.content_hash for r in results] == [
            "live-0",
            "live-1",
            "live-2",
        ], "the time-only branch limited before filtering out superseded rows"

    @pytest.mark.asyncio
    async def test_time_only_recall_handles_dense_superseded_timestamp(self):
        storage = _make_storage()
        n_results = self.N_RESULTS
        stale = [
            Memory(
                content=f"obsolete {i}",
                content_hash=f"stale-{i}",
                metadata={"superseded_by": "current"},
                created_at=1000.0,
            )
            for i in range(100)
        ]
        live = [
            Memory(
                content=f"live {i}",
                content_hash=f"live-{i}",
                created_at=float(500 - i),
            )
            for i in range(n_results)
        ]
        exact_pages = []

        async def _query_window(filter_expr, limit, offset=0):
            bounds = [
                float(value)
                for value in re.findall(
                    r"created_at [<>=]+ ([0-9.eE+-]+)", filter_expr
                )
            ]
            lower, upper = min(bounds), max(bounds)
            matches = [
                memory
                for memory in stale + live
                if lower <= (memory.created_at or 0.0) <= upper
            ]
            page = matches[offset:offset + limit]
            return page, len(page)

        async def _batches(filter_expr, batch_size):
            exact_pages.append((filter_expr, batch_size))
            yield stale, len(stale)

        storage._query_time_window = AsyncMock(side_effect=_query_window)
        storage._iter_time_window_batches = _batches
        storage._iterate_all_rows = AsyncMock(
            side_effect=AssertionError("same-timestamp backlog must not full-scan")
        )

        results = await storage.recall(
            start_timestamp=0.0, end_timestamp=1000.0, n_results=n_results
        )

        assert [result.memory.content_hash for result in results] == [
            "live-0",
            "live-1",
            "live-2",
        ]
        assert len(exact_pages) == 1
        assert "created_at >= 1000.0" in exact_pages[0][0]
        assert "created_at <= 1000.0" in exact_pages[0][0]
        assert exact_pages[0][1] == 100
        storage._iterate_all_rows.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dense_timestamp_beyond_query_cap_uses_paged_scan(self, monkeypatch):
        """A saturated timestamp must page without the scalar-query offset cap."""
        monkeypatch.setattr(milvus_module, "_MILVUS_MAX_LIMIT", 100)
        storage = _make_storage()
        stale = [
            Memory(
                content=f"obsolete {i}",
                content_hash=f"stale-{i}",
                metadata={"superseded_by": "current"},
                created_at=1000.0,
            )
            for i in range(150)
        ]
        live = [
            Memory(content=f"live {i}", content_hash=f"live-{i}", created_at=500.0 - i)
            for i in range(3)
        ]
        pages = []

        async def _query_window(filter_expr, limit, offset=0):
            bounds = [
                float(value)
                for value in re.findall(
                    r"created_at [<>=]+ ([0-9.eE+-]+)", filter_expr
                )
            ]
            if min(bounds) == max(bounds) == 1000.0:
                raise AssertionError("exact timestamp must use the iterator path")
            if max(bounds) < 1000.0:
                page = live[offset:offset + limit]
                return page, len(page)
            page = stale[offset:offset + limit]
            return page, len(page)

        async def _batches(filter_expr, batch_size):
            pages.extend([stale[:batch_size], stale[batch_size:]])
            yield stale[:batch_size], batch_size
            yield stale[batch_size:], len(stale) - batch_size

        storage._query_time_window = AsyncMock(side_effect=_query_window)
        storage._iter_time_window_batches = _batches
        storage._iterate_all_rows = AsyncMock(
            side_effect=AssertionError("exact-timestamp fallback must be paged")
        )

        results = await storage.recall(
            start_timestamp=0.0, end_timestamp=1000.0, n_results=3
        )

        assert [result.memory.content_hash for result in results] == [
            "live-0",
            "live-1",
            "live-2",
        ]
        assert [len(page) for page in pages] == [100, 50]
        storage._iterate_all_rows.assert_not_awaited()


class TestRecallSoftDeletedRows:
    @pytest.mark.asyncio
    async def test_time_only_recall_hides_soft_deleted(self):
        storage = _make_storage()
        rows = [
            Memory(
                content="deleted",
                content_hash="deleted",
                metadata={"deleted_at": "2026-10-01T00:00:00Z"},
                created_at=1000.0,
            ),
            Memory(content="live", content_hash="live", created_at=999.0),
        ]

        async def _query_window(filter_expr, limit, offset=0):
            page = rows[offset:offset + limit]
            return page, len(page)

        storage._query_time_window = AsyncMock(side_effect=_query_window)

        results = await storage.recall(n_results=2)

        assert [result.memory.content_hash for result in results] == ["live"]

    @pytest.mark.asyncio
    async def test_semantic_recall_hides_soft_deleted(self):
        storage = _make_storage()
        deleted = _entity(
            "deleted",
            "deleted",
            created_at=1000.0,
            metadata={"deleted_at": "2026-10-01T00:00:00Z"},
        )
        live = _entity("live", "live", created_at=999.0)

        async def _run_search(embedding, filter_expr, fetch_n):
            return [
                {"id": deleted["id"], "distance": 0.9, "entity": deleted, **deleted},
                {"id": live["id"], "distance": 0.8, "entity": live, **live},
            ]

        storage._run_search = AsyncMock(side_effect=_run_search)

        results = await storage.recall(query="probe", n_results=1)

        assert [result.memory.content_hash for result in results] == ["live"]


class TestPagedTimeWindowIterator:
    @pytest.mark.asyncio
    async def test_restarts_iterator_from_same_checkpoint_after_channel_death(self):
        storage = _make_storage()
        storage._is_lite = True
        storage._write_lock = asyncio.Lock()

        dead_client = MagicMock()
        dead_iterator = MagicMock()
        dead_iterator.next.side_effect = ValueError(
            "Cannot invoke RPC on closed channel!"
        )
        dead_client.query_iterator.return_value = dead_iterator

        recovered_client = MagicMock()
        recovered_iterator = MagicMock()
        recovered_iterator.next.side_effect = [
            [_entity("h1", "one", created_at=1.0)],
            [],
        ]
        recovered_client.query_iterator.return_value = recovered_iterator

        storage.client = dead_client
        storage._reconnect_lite_client = MagicMock(
            side_effect=lambda: setattr(storage, "client", recovered_client)
        )
        storage._call_client = MilvusMemoryStorage._call_client.__get__(
            storage, MilvusMemoryStorage
        )

        pages = [
            page
            async for page in storage._iter_time_window_batches(
                "created_at >= 0.0", 1
            )
        ]

        assert [(len(memories), raw_count) for memories, raw_count in pages] == [(1, 1)]
        assert pages[0][0][0].content_hash == "h1"
        dead_client.query_iterator.assert_called_once()
        recovered_client.query_iterator.assert_called_once()
        first_call = dead_client.query_iterator.call_args.kwargs
        second_call = recovered_client.query_iterator.call_args.kwargs
        assert first_call["iterator_cp_file"] == second_call["iterator_cp_file"]
        recovered_iterator.close.assert_called_once_with()
        storage._reconnect_lite_client.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_yields_raw_page_count_when_row_conversion_fails(self):
        storage = _make_storage()
        storage._write_lock = asyncio.Lock()
        storage._entity_to_memory = MagicMock(
            side_effect=[
                None,
                Memory(content="one", content_hash="h1", created_at=1.0),
            ]
        )
        iterator = MagicMock()
        iterator.next.side_effect = [[{"id": "bad"}, _entity("h1", "one")], []]
        storage._call_client = AsyncMock(return_value=iterator)

        pages = [
            page
            async for page in storage._iter_time_window_batches(
                "created_at >= 0.0", 2
            )
        ]

        assert len(pages) == 1
        memories, raw_count = pages[0]
        assert [memory.content_hash for memory in memories] == ["h1"]
        assert raw_count == 2
        iterator.close.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_recall_continues_after_unparseable_full_page(self):
        storage = _make_storage()
        stale = [
            Memory(
                content=f"obsolete {i}",
                content_hash=f"stale-{i}",
                metadata={"superseded_by": "current"},
                created_at=1000.0,
            )
            for i in range(100)
        ]
        live = Memory(content="live", content_hash="live", created_at=1000.0)

        async def _batches(filter_expr, batch_size):
            yield [], batch_size
            yield [live], 1

        storage._query_time_window = AsyncMock(return_value=(stale, len(stale)))
        storage._iter_time_window_batches = _batches

        results = await storage.recall(
            start_timestamp=1000.0, end_timestamp=1000.0, n_results=1
        )

        assert [result.memory.content_hash for result in results] == ["live"]
