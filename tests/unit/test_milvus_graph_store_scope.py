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

"""The Milvus association collection carries the store its edges belong to (#2119).

Unlike SQLite, the graph collection is separate from the memories, so the store
has to travel with the edge (#2118 got away with a join). Nobody who writes an
edge knows the store: `Memory` has no such attribute and the call sites hold only
the hash. The backend therefore resolves it from the paired memory collection —
the same authority SQLite reads through `memories.store`.

A collection created before the field existed has nothing to confine on, so a
named store is refused there instead of being served unscoped, and writes keep
their pre-#2119 shape.

Mocked-client style of test_milvus_store_scope.py, so these run without
pymilvus; test_milvus_graph_store_scope_e2e.py runs the same contract on Lite.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from mcp_memory_service.storage.milvus_graph import (
    MilvusGraphStorage,
    _named_store_refused,
)

STORE_A = "alpha"
STORE_B = "beta"

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def _make_graph(has_store: bool = True) -> MilvusGraphStorage:
    """A MilvusGraphStorage with __init__ skipped and both collections mocked."""
    graph = MilvusGraphStorage.__new__(MilvusGraphStorage)
    graph.uri = "./milvus.db"
    graph.token = None
    graph._memory_collection = "unit_memories"
    graph.collection_name = "unit_memories_graph"
    graph.client = MagicMock()
    graph._lock = asyncio.Lock()
    graph._has_store = has_store
    graph._store_scope_warned = False
    return graph


def _memory_rows(store: str) -> list[dict[str, Any]]:
    return [{"id": HASH_A, "store": store}]


class _FakeSchema:
    def __init__(self) -> None:
        self.fields: list[dict[str, Any]] = []

    def add_field(self, **kwargs: Any) -> None:
        self.fields.append(kwargs)


class _FakeIndexParams:
    def __init__(self) -> None:
        self.indexes: list[dict[str, Any]] = []

    def add_index(self, **kwargs: Any) -> None:
        self.indexes.append(kwargs)


class _SchemaClient:
    """Records the schema and index parameters _ensure_collection builds."""

    def __init__(self, existing: bool) -> None:
        self._existing = existing
        self.schema = _FakeSchema()
        self.index_params = _FakeIndexParams()
        self.created: dict[str, Any] = {}

    def has_collection(self, collection_name: str) -> bool:
        return self._existing

    def describe_collection(self, collection_name: str) -> dict[str, Any]:
        return {"fields": [{"name": "id"}, {"name": "source_hash"}]}

    def create_schema(self, **kwargs: Any) -> _FakeSchema:
        return self.schema

    def prepare_index_params(self) -> _FakeIndexParams:
        return self.index_params

    def create_collection(self, **kwargs: Any) -> None:
        self.created = kwargs


class _StubDataType:
    """Stand-in for pymilvus.DataType, which is None without pymilvus installed."""

    VARCHAR = "VARCHAR"
    FLOAT = "FLOAT"
    DOUBLE = "DOUBLE"
    FLOAT_VECTOR = "FLOAT_VECTOR"


@pytest.fixture()
def stub_datatype(monkeypatch):
    from mcp_memory_service.storage import milvus_graph as module

    monkeypatch.setattr(module, "DataType", _StubDataType)
    return _StubDataType


class TestCollectionSchema:
    def test_new_collection_declares_and_indexes_the_store(self, stub_datatype):
        graph = _make_graph(has_store=False)
        graph.client = _SchemaClient(existing=False)

        graph._ensure_collection()

        names = [field["field_name"] for field in graph.client.schema.fields]
        assert "store" in names
        store_field = next(
            f for f in graph.client.schema.fields if f["field_name"] == "store"
        )
        assert store_field["max_length"] == 64
        assert store_field["default_value"] == "default"
        assert {"field_name": "store", "index_type": "INVERTED"} in graph.client.index_params.indexes
        assert graph._has_store is True

    def test_existing_collection_without_the_field_is_recorded_as_legacy(self, stub_datatype):
        graph = _make_graph(has_store=True)
        graph.client = _SchemaClient(existing=True)

        graph._ensure_collection()

        assert graph._has_store is False

    def test_existing_collection_with_the_field_is_recorded(self, stub_datatype):
        graph = _make_graph(has_store=False)
        client = _SchemaClient(existing=True)
        client.describe_collection = lambda collection_name: {
            "fields": [{"name": "id"}, {"name": "store"}]
        }
        graph.client = client

        graph._ensure_collection()

        assert graph._has_store is True


class TestNamedStoreRefusal:
    def test_refused_on_a_legacy_collection(self):
        assert _named_store_refused(_make_graph(has_store=False), STORE_A, "list_entities") is True

    def test_allowed_on_a_confining_collection(self):
        assert _named_store_refused(_make_graph(has_store=True), STORE_A, "list_entities") is False

    def test_unscoped_is_never_refused(self):
        graph = _make_graph(has_store=False)
        assert _named_store_refused(graph, None, "list_entities") is False
        assert _named_store_refused(graph, "all", "list_entities") is False


class TestMemoryStoreResolution:
    async def test_reads_the_store_from_the_memory_collection(self):
        graph = _make_graph()
        graph.client.get.return_value = _memory_rows(STORE_A)

        assert await graph._memory_store(HASH_A) == STORE_A
        graph.client.get.assert_called_once_with(
            collection_name="unit_memories",
            ids=[HASH_A],
            output_fields=["store"],
        )

    async def test_unknown_memory_returns_none(self):
        graph = _make_graph()
        graph.client.get.return_value = []

        assert await graph._memory_store(HASH_A) is None

    async def test_pre_1962_memory_collection_reports_the_default_store(self):
        """A memory collection without the field holds only default memories."""
        graph = _make_graph()

        def _get(**kwargs):
            if "output_fields" in kwargs:
                raise RuntimeError("field store not exist")
            return [{"id": HASH_A}]

        graph.client.get.side_effect = _get

        assert await graph._memory_store(HASH_A) == "default"


class TestEntityLinkWritesTheStore:
    async def test_stamps_the_resolved_store(self):
        graph = _make_graph()
        graph.client.get.return_value = _memory_rows(STORE_A)

        assert await graph.store_entity_link(HASH_A, "FastAPI", "technology") is True

        rows = graph.client.upsert.call_args.kwargs["data"]
        assert rows[0]["store"] == STORE_A
        assert rows[0]["source_hash"] == HASH_A

    async def test_refuses_a_memory_it_cannot_place(self):
        graph = _make_graph()
        graph.client.get.return_value = []

        assert await graph.store_entity_link(HASH_A, "FastAPI", "technology") is False
        graph.client.upsert.assert_not_called()

    async def test_legacy_collection_keeps_the_old_row_shape(self):
        graph = _make_graph(has_store=False)

        assert await graph.store_entity_link(HASH_A, "FastAPI", "technology") is True

        rows = graph.client.upsert.call_args.kwargs["data"]
        assert "store" not in rows[0]
        graph.client.get.assert_not_called()


class TestAssociationWritesTheStore:
    async def test_symmetric_pair_carries_the_store_on_both_rows(self):
        graph = _make_graph()
        graph.client.get.side_effect = [
            _memory_rows(STORE_A), _memory_rows(STORE_A),
        ]

        assert await graph.store_association(
            HASH_A, HASH_B, 0.9, ["semantic"], relationship_type="related",
        ) is True

        rows = graph.client.upsert.call_args.kwargs["data"]
        assert [row["store"] for row in rows] == [STORE_A, STORE_A]

    async def test_cross_store_pair_is_refused(self):
        graph = _make_graph()
        graph.client.get.side_effect = [
            _memory_rows(STORE_A), _memory_rows(STORE_B),
        ]

        assert await graph.store_association(
            HASH_A, HASH_B, 0.9, ["semantic"], relationship_type="related",
        ) is False
        graph.client.upsert.assert_not_called()

    async def test_unknown_endpoint_is_refused(self):
        graph = _make_graph()
        graph.client.get.side_effect = [_memory_rows(STORE_A), []]

        assert await graph.store_association(
            HASH_A, HASH_C, 0.9, ["semantic"], relationship_type="related",
        ) is False
        graph.client.upsert.assert_not_called()


class TestReadersCarryTheStore:
    async def test_find_memories_by_entity_scopes_the_query(self):
        graph = _make_graph()
        graph._query_entity_edges = AsyncMock(return_value=[])

        await graph.find_memories_by_entity("FastAPI", store=STORE_A)

        extra = graph._query_entity_edges.call_args.kwargs["extra_filter"]
        assert f'store == "{STORE_A}"' in extra
        assert "target_hash" in extra

    async def test_unscoped_read_carries_no_store_filter(self):
        graph = _make_graph()
        graph._query_entity_edges = AsyncMock(return_value=[])

        await graph.find_memories_by_entity("FastAPI")

        extra = graph._query_entity_edges.call_args.kwargs["extra_filter"]
        assert extra is not None
        assert "store ==" not in extra

    async def test_list_entities_scopes_the_query(self):
        graph = _make_graph()
        graph._query_entity_edges = AsyncMock(return_value=[])

        await graph.list_entities(store=STORE_A)

        extra = graph._query_entity_edges.call_args.kwargs["extra_filter"]
        assert f'store == "{STORE_A}"' in extra

    async def test_common_neighbors_scopes_both_hops(self):
        graph = _make_graph()
        graph._query_edges_both = AsyncMock(return_value=[])

        await graph.common_neighbors(HASH_A, store=STORE_A)

        assert graph._query_edges_both.call_args.kwargs["store"] == STORE_A

    async def test_legacy_collection_refuses_a_named_store(self):
        graph = _make_graph(has_store=False)
        graph._query_entity_edges = AsyncMock(return_value=[])

        assert await graph.find_memories_by_entity("FastAPI", store=STORE_A) == []
        assert await graph.list_entities(store=STORE_A) == []
        assert await graph.get_entity_profile("FastAPI", store=STORE_A) == {}
        assert await graph.get_entities_for_memory(HASH_A, store=STORE_A) == []
        assert await graph.common_neighbors(HASH_A, store=STORE_A) == []
        graph._query_entity_edges.assert_not_called()
