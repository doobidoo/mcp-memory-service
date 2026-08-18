# Copyright 2024 Heinrich Krupp
# Licensed under the Apache License, Version 2.0

"""
Entity-layer and reasoning-helper tests for MilvusGraphStorage.

These cover the methods memory_explore / memory_detail / memory_graph depend
on, which previously existed only on the SQLite GraphStorage. Uses Milvus Lite
(in-process, file-backed) so the assertions run against real Milvus queries.
"""

import os
import shutil
import tempfile

import pytest

pymilvus = pytest.importorskip("pymilvus", reason="pymilvus required for Milvus graph tests")

from mcp_memory_service.storage.milvus_graph import MilvusGraphStorage, _entity_key


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64


@pytest.fixture()
def tmp_dir():
    d = tempfile.mkdtemp(prefix="milvus_entity_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture()
async def graph(tmp_dir):
    uri = os.path.join(tmp_dir, "test.db")
    gs = MilvusGraphStorage(uri=uri, collection_name="test_mem")
    await gs.initialize()
    yield gs
    await gs.close()


# ---------------------------------------------------------------------------
# Entity key derivation
# ---------------------------------------------------------------------------

class TestEntityKey:
    def test_fits_varchar_64(self):
        """Long entity names must still produce a key inside VARCHAR(64)."""
        key = _entity_key("x" * 500)
        assert len(key) <= 64

    def test_deterministic_and_case_insensitive(self):
        assert _entity_key("FastAPI") == _entity_key("fastapi")
        assert _entity_key("  fastapi  ") == _entity_key("fastapi")

    def test_distinct_names_distinct_keys(self):
        assert _entity_key("fastapi") != _entity_key("django")

    def test_disjoint_from_content_hashes(self):
        """Entity keys must never collide with a real 64-char content hash."""
        assert _entity_key("anything").startswith("ent:")


# ---------------------------------------------------------------------------
# store_entity_link / find_memories_by_entity
# ---------------------------------------------------------------------------

class TestStoreEntityLink:
    async def test_store_and_find(self, graph):
        assert await graph.store_entity_link(HASH_A, "fastapi", "technology")
        assert await graph.find_memories_by_entity("fastapi") == [HASH_A]

    async def test_idempotent_relink(self, graph):
        """Re-linking the same pair upserts rather than duplicating."""
        await graph.store_entity_link(HASH_A, "fastapi", "technology")
        await graph.store_entity_link(HASH_A, "fastapi", "technology")
        assert await graph.find_memories_by_entity("fastapi") == [HASH_A]
        profile = await graph.get_entity_profile("fastapi")
        assert profile["memory_count"] == 1

    async def test_multiple_memories_one_entity(self, graph):
        for h in (HASH_A, HASH_B, HASH_C):
            await graph.store_entity_link(h, "fastapi", "technology")
        found = await graph.find_memories_by_entity("fastapi")
        assert set(found) == {HASH_A, HASH_B, HASH_C}

    async def test_respects_limit(self, graph):
        for h in (HASH_A, HASH_B, HASH_C):
            await graph.store_entity_link(h, "fastapi", "technology")
        assert len(await graph.find_memories_by_entity("fastapi", limit=2)) == 2

    async def test_long_entity_name_roundtrips(self, graph):
        """The surface form survives even though target_hash is hashed."""
        name = "a very long entity name " * 10
        assert await graph.store_entity_link(HASH_A, name, "concept")
        assert await graph.find_memories_by_entity(name) == [HASH_A]
        assert await graph.get_entities_for_memory(HASH_A) == [name]

    async def test_rejects_empty_input(self, graph):
        assert not await graph.store_entity_link("", "fastapi", "technology")
        assert not await graph.store_entity_link(HASH_A, "", "technology")

    async def test_unknown_entity_returns_empty(self, graph):
        assert await graph.find_memories_by_entity("nope") == []
        assert await graph.get_entity_profile("nope") == {}


# ---------------------------------------------------------------------------
# list_entities / get_entities_for_memory / get_entity_profile
# ---------------------------------------------------------------------------

class TestEntityQueries:
    async def test_list_entities_counts_and_orders(self, graph):
        await graph.store_entity_link(HASH_A, "fastapi", "technology")
        await graph.store_entity_link(HASH_B, "fastapi", "technology")
        await graph.store_entity_link(HASH_C, "django", "technology")

        entities = await graph.list_entities()
        names = [e["entity_name"] for e in entities]
        assert names == ["fastapi", "django"]  # ordered by count desc
        assert entities[0]["count"] == 2
        assert entities[1]["count"] == 1
        assert entities[0]["last_activity"] > 0

    async def test_list_entities_respects_limit(self, graph):
        for i, name in enumerate(["a", "b", "c", "d"]):
            await graph.store_entity_link(HASH_A, name, "concept")
        assert len(await graph.list_entities(limit=2)) == 2

    async def test_list_entities_empty(self, graph):
        assert await graph.list_entities() == []

    async def test_get_entities_for_memory(self, graph):
        await graph.store_entity_link(HASH_A, "fastapi", "technology")
        await graph.store_entity_link(HASH_A, "pytest", "tool")
        await graph.store_entity_link(HASH_B, "django", "technology")

        assert set(await graph.get_entities_for_memory(HASH_A)) == {"fastapi", "pytest"}
        assert await graph.get_entities_for_memory(HASH_B) == ["django"]
        assert await graph.get_entities_for_memory(HASH_C) == []

    async def test_entity_profile_shape(self, graph):
        await graph.store_entity_link(HASH_A, "fastapi", "technology")
        await graph.store_entity_link(HASH_B, "fastapi", "framework")

        profile = await graph.get_entity_profile("fastapi")
        assert profile["entity_name"] == "fastapi"
        assert profile["memory_count"] == 2
        assert set(profile["entity_types"]) == {"technology", "framework"}
        assert profile["last_activity"] > 0

    async def test_entity_edges_excluded_from_association_queries(self, graph):
        """has_entity edges must not masquerade as memory associations."""
        await graph.store_entity_link(HASH_A, "fastapi", "technology")
        await graph.store_association(HASH_A, HASH_B, 0.9, ["semantic"])

        types = await graph.get_relationship_types(HASH_A)
        assert types.get("has_entity") == 1
        assert types.get("related") == 1


# ---------------------------------------------------------------------------
# transitive_closure
# ---------------------------------------------------------------------------

class TestTransitiveClosure:
    async def test_two_hop_inferred(self, graph):
        """A causes B causes C yields an inferred A->C at distance 2."""
        await graph.store_association(HASH_A, HASH_B, 0.9, ["semantic"], relationship_type="causes")
        await graph.store_association(HASH_B, HASH_C, 0.9, ["semantic"], relationship_type="causes")

        result = await graph.transitive_closure("causes", max_hops=2)
        assert (HASH_A, HASH_C, 2) in result

    async def test_direct_edges_excluded(self, graph):
        """An existing direct edge suppresses the inferred pair."""
        await graph.store_association(HASH_A, HASH_B, 0.9, ["semantic"], relationship_type="causes")
        await graph.store_association(HASH_B, HASH_C, 0.9, ["semantic"], relationship_type="causes")
        await graph.store_association(HASH_A, HASH_C, 0.9, ["semantic"], relationship_type="causes")

        result = await graph.transitive_closure("causes", max_hops=2)
        assert not [r for r in result if r[0] == HASH_A and r[1] == HASH_C]

    async def test_three_hop_within_max(self, graph):
        await graph.store_association(HASH_A, HASH_B, 0.9, ["semantic"], relationship_type="causes")
        await graph.store_association(HASH_B, HASH_C, 0.9, ["semantic"], relationship_type="causes")
        await graph.store_association(HASH_C, HASH_D, 0.9, ["semantic"], relationship_type="causes")

        assert (HASH_A, HASH_D, 3) in await graph.transitive_closure("causes", max_hops=3)
        # max_hops=2 must not reach D
        assert (HASH_A, HASH_D, 3) not in await graph.transitive_closure("causes", max_hops=2)

    async def test_other_relationship_types_ignored(self, graph):
        await graph.store_association(HASH_A, HASH_B, 0.9, ["semantic"], relationship_type="causes")
        await graph.store_association(HASH_B, HASH_C, 0.9, ["semantic"], relationship_type="fixes")
        assert await graph.transitive_closure("causes", max_hops=2) == []

    async def test_empty_graph(self, graph):
        assert await graph.transitive_closure("causes", max_hops=2) == []

    async def test_cycle_terminates(self, graph):
        """A cycle must not loop forever."""
        await graph.store_association(HASH_A, HASH_B, 0.9, ["semantic"], relationship_type="causes")
        await graph.store_association(HASH_B, HASH_C, 0.9, ["semantic"], relationship_type="causes")
        await graph.store_association(HASH_C, HASH_A, 0.9, ["semantic"], relationship_type="causes")
        result = await graph.transitive_closure("causes", max_hops=4)
        assert all(src != tgt for src, tgt, _ in result)


# ---------------------------------------------------------------------------
# common_neighbors
# ---------------------------------------------------------------------------

class TestCommonNeighbors:
    async def test_shared_neighbor_found(self, graph):
        """A-B and C-B means C is a 2-hop candidate for A via shared B."""
        await graph.store_association(HASH_A, HASH_B, 0.9, ["semantic"])
        await graph.store_association(HASH_C, HASH_B, 0.9, ["semantic"])

        result = await graph.common_neighbors(HASH_A)
        candidates = [r[0] for r in result]
        assert HASH_C in candidates

    async def test_directly_connected_excluded(self, graph):
        await graph.store_association(HASH_A, HASH_B, 0.9, ["semantic"])
        await graph.store_association(HASH_C, HASH_B, 0.9, ["semantic"])
        await graph.store_association(HASH_A, HASH_C, 0.9, ["semantic"])

        assert HASH_C not in [r[0] for r in await graph.common_neighbors(HASH_A)]

    async def test_self_excluded(self, graph):
        await graph.store_association(HASH_A, HASH_B, 0.9, ["semantic"])
        await graph.store_association(HASH_C, HASH_B, 0.9, ["semantic"])
        assert HASH_A not in [r[0] for r in await graph.common_neighbors(HASH_A)]

    async def test_source_degree_reported(self, graph):
        await graph.store_association(HASH_A, HASH_B, 0.9, ["semantic"])
        await graph.store_association(HASH_A, HASH_D, 0.9, ["semantic"])
        await graph.store_association(HASH_C, HASH_B, 0.9, ["semantic"])

        result = await graph.common_neighbors(HASH_A)
        assert result
        # 'related' is symmetric, so A-B and A-D each store two rows; degree
        # counts neighbour rows the same way the SQL self-join does.
        assert all(degree > 0 for _, _, degree in result)

    async def test_min_shared_filter(self, graph):
        await graph.store_association(HASH_A, HASH_B, 0.9, ["semantic"])
        await graph.store_association(HASH_C, HASH_B, 0.9, ["semantic"])
        assert await graph.common_neighbors(HASH_A, min_shared=99) == []

    async def test_isolated_node(self, graph):
        assert await graph.common_neighbors(HASH_A) == []

    async def test_empty_hash(self, graph):
        assert await graph.common_neighbors("") == []
