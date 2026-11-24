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

"""End-to-end store confinement for the Milvus graph, on Milvus Lite (#2119).

The mocked-client cases live in tests/unit/test_milvus_graph_store_scope.py and
run everywhere. This one exercises the real query expressions, the real
`default_value` behaviour of the `store` field and the real `get` against a
memory collection, so it runs only in the Milvus CI jobs.
"""

import os
import shutil
import tempfile

import pytest

pymilvus = pytest.importorskip("pymilvus", reason="pymilvus required for the Milvus graph e2e")

from pymilvus import DataType, MilvusClient

from mcp_memory_service.storage.milvus_graph import MilvusGraphStorage

STORE_A = "alpha"
STORE_B = "beta"

HASH_A1 = "a1" * 32
HASH_A2 = "a2" * 32
HASH_B1 = "b1" * 32

MEMORY_COLLECTION = "e2e_memories"
_DUMMY_VEC = [0.0]


def _create_memory_collection(uri: str) -> None:
    """The smallest memory collection the graph needs: id, store, one vector."""
    client = MilvusClient(uri=uri)
    try:
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(
            field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=64
        )
        schema.add_field(
            field_name="store", datatype=DataType.VARCHAR, max_length=64,
            default_value="default",
        )
        schema.add_field(field_name="_vec", datatype=DataType.FLOAT_VECTOR, dim=1)
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="_vec", index_type="AUTOINDEX", metric_type="L2"
        )
        client.create_collection(
            collection_name=MEMORY_COLLECTION,
            schema=schema,
            index_params=index_params,
        )
        client.insert(
            collection_name=MEMORY_COLLECTION,
            data=[
                {"id": HASH_A1, "store": STORE_A, "_vec": _DUMMY_VEC},
                {"id": HASH_A2, "store": STORE_A, "_vec": _DUMMY_VEC},
                {"id": HASH_B1, "store": STORE_B, "_vec": _DUMMY_VEC},
            ],
        )
    finally:
        client.close()


@pytest.fixture()
def tmp_dir():
    d = tempfile.mkdtemp(prefix="milvus_graph_store_e2e_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture()
async def graph(tmp_dir):
    uri = os.path.join(tmp_dir, "test.db")
    _create_memory_collection(uri)
    gs = MilvusGraphStorage(uri=uri, collection_name=MEMORY_COLLECTION)
    await gs.initialize()
    yield gs
    await gs.close()


class TestStoreScopeOnLite:
    async def test_new_collection_confines(self, graph):
        assert graph._has_store is True

    async def test_entity_links_are_scoped_to_the_store(self, graph):
        assert await graph.store_entity_link(HASH_A1, "Shared", "concept") is True
        assert await graph.store_entity_link(HASH_B1, "Shared", "concept") is True

        assert await graph.find_memories_by_entity("Shared", store=STORE_A) == [HASH_A1]
        assert await graph.find_memories_by_entity("Shared", store=STORE_B) == [HASH_B1]
        assert set(await graph.find_memories_by_entity("Shared")) == {HASH_A1, HASH_B1}

        assert await graph.get_entities_for_memory(HASH_B1, store=STORE_A) == []
        assert await graph.get_entities_for_memory(HASH_B1, store=STORE_B) == ["Shared"]

        profile = await graph.get_entity_profile("Shared", store=STORE_A)
        assert profile["entity_name"] == "Shared"
        assert profile["memory_count"] == 1
        assert profile["entity_types"] == ["concept"]

    async def test_associations_stay_inside_one_store(self, graph):
        assert await graph.store_association(
            HASH_A1, HASH_A2, 0.9, ["semantic"], relationship_type="related",
        ) is True
        # Cross-store pairs are refused, the write-side form of the both-ends
        # rule #2121 applies on read.
        assert await graph.store_association(
            HASH_A1, HASH_B1, 0.9, ["semantic"], relationship_type="related",
        ) is False

        assert HASH_A2 in {c for c, _s, _d in await graph.common_neighbors(HASH_A1)}
        assert HASH_B1 not in {c for c, _s, _d in await graph.common_neighbors(HASH_A1)}

    async def test_unknown_memory_is_refused(self, graph):
        assert await graph.store_entity_link("f" * 64, "Ghost", "concept") is False
