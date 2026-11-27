"""Store confinement for the entity graph (#2118).

The entity graph carries no store of its own: entity links are rows in
``memory_graph`` with ``relationship_type='has_entity'``, and the table shares
its SQLite file with ``memories``. Before the fix every entity read in
``GraphStorage`` was global, so a credential scoped to one store could reach
entity names through another store's memories. ``get_entity_detail`` confined
``chunks`` by post-filtering ``get_by_hash``, but ``_discover_related_entities``
walked ``common_neighbors`` unfiltered and returned ``related_entities`` from
every partition.

These tests are red before the fix and green after it. They seed two stores with
an entity that exists in both (``Shared``) plus one entity each that exists only
in that store, then assert that a store-scoped read never crosses over while the
unscoped read stays exactly as wide as before.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from mcp_memory_service.server.handlers.graph import (
    _discover_related_entities,
    get_entity_detail,
)
from mcp_memory_service.storage.graph import GraphStorage

STORE_A = "alpha"
STORE_B = "beta"

MEM_A1 = "a1" * 32
MEM_A2 = "a2" * 32
MEM_B1 = "b1" * 32
MEM_B2 = "b2" * 32
# A pair of in-store memories that a foreign memory associates with on both
# sides, plus a soft-deleted memory that still carries an entity link.
MEM_X1 = "x1" * 32
MEM_X2 = "x2" * 32
MEM_Y1 = "y1" * 32
MEM_D1 = "d1" * 32


def _seed_database(db_path: Path) -> None:
    """Two stores, two memories each, and the entity links that connect them.

    ``Shared`` is carried by memories in both stores, so a global read counts it
    more often than a scoped read. ``OnlyB`` is reachable from ``MEM_A1`` only
    through ``MEM_B1`` — that is the hop the old code allowed.
    """
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "CREATE TABLE memories ("
            "content_hash TEXT PRIMARY KEY, store TEXT NOT NULL, deleted_at REAL)"
        )
        conn.execute(
            """
            CREATE TABLE memory_graph (
                source_hash TEXT NOT NULL,
                target_hash TEXT NOT NULL,
                similarity REAL NOT NULL,
                connection_types TEXT NOT NULL,
                metadata TEXT,
                created_at REAL NOT NULL,
                relationship_type TEXT NOT NULL DEFAULT 'related',
                PRIMARY KEY (source_hash, target_hash)
            )
            """
        )
        conn.executemany(
            "INSERT INTO memories (content_hash, store, deleted_at) VALUES (?, ?, ?)",
            [
                (MEM_A1, STORE_A, None),
                (MEM_A2, STORE_A, None),
                (MEM_B1, STORE_B, None),
                (MEM_B2, STORE_B, None),
                (MEM_X1, STORE_A, None),
                (MEM_X2, STORE_A, None),
                (MEM_Y1, STORE_B, None),
                # Soft-deleted: its edges must not count for a scoped read.
                (MEM_D1, STORE_A, 1234.0),
            ],
        )
        # Entity links: source is the memory, target is the entity name.
        conn.executemany(
            "INSERT INTO memory_graph (source_hash, target_hash, similarity,"
            " connection_types, metadata, created_at, relationship_type)"
            " VALUES (?, ?, 1.0, '[\"entity\"]', ?, 1.0, 'has_entity')",
            [
                (MEM_A1, "Shared", json.dumps({"entity_type": "concept"})),
                (MEM_A2, "Shared", json.dumps({"entity_type": "concept"})),
                (MEM_A2, "OnlyA", json.dumps({"entity_type": "concept"})),
                (MEM_B1, "Shared", json.dumps({"entity_type": "concept"})),
                (MEM_B1, "OnlyB", json.dumps({"entity_type": "concept"})),
                (MEM_B2, "Shared", json.dumps({"entity_type": "concept"})),
                (MEM_D1, "Ghost", json.dumps({"entity_type": "concept"})),
            ],
        )
        # A foreign memory (Y1, store beta) associated with both X1 and X2 in
        # store alpha. Symmetric "related" edges are stored in both directions.
        conn.executemany(
            "INSERT INTO memory_graph (source_hash, target_hash, similarity,"
            " connection_types, metadata, created_at, relationship_type)"
            " VALUES (?, ?, 0.8, '[\"semantic\"]', '{}', 2.0, 'related')",
            [
                (MEM_X1, MEM_Y1),
                (MEM_Y1, MEM_X1),
                (MEM_X2, MEM_Y1),
                (MEM_Y1, MEM_X2),
            ],
        )
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
async def graph(tmp_path):
    db_path = tmp_path / "graph.db"
    _seed_database(db_path)
    storage = GraphStorage(str(db_path))
    try:
        yield storage
    finally:
        await storage.close()


class _Memory:
    def __init__(self, content: str):
        self.content = content
        self.quality_score = None


class _Storage:
    """Just enough storage for ``get_entity_detail`` to hydrate chunks."""

    get_by_hash_is_store_scoped = True

    def __init__(self, contents: dict[str, str]):
        self._contents = contents
        self.lookup_stores = []

    async def get_by_hash(self, content_hash, store=None):
        self.lookup_stores.append(store)
        if content_hash in self._contents:
            return _Memory(self._contents[content_hash])
        return None


class TestEntityReadsAreStoreScoped:
    async def test_list_entities_counts_only_the_claimed_store(self, graph):
        entities = {e["entity_name"]: e["count"] for e in await graph.list_entities(store=STORE_A)}

        # Both alpha memories carry "Shared"; only one carries "OnlyA".
        assert entities == {"Shared": 2, "OnlyA": 1}

    async def test_list_entities_unscoped_still_spans_every_store(self, graph):
        entities = {e["entity_name"]: e["count"] for e in await graph.list_entities()}

        assert entities["Shared"] == 4
        assert entities["OnlyA"] == 1
        assert entities["OnlyB"] == 1

    async def test_find_memories_by_entity_stays_in_store(self, graph):
        hashes = await graph.find_memories_by_entity("Shared", store=STORE_A)

        assert set(hashes) == {MEM_A1, MEM_A2}

    async def test_get_entities_for_memory_of_another_store_is_empty(self, graph):
        assert await graph.get_entities_for_memory(MEM_B1, store=STORE_A) == []
        assert sorted(await graph.get_entities_for_memory(MEM_B1)) == ["OnlyB", "Shared"]

    async def test_entity_profile_of_another_store_is_empty(self, graph):
        assert await graph.get_entity_profile("OnlyB", store=STORE_A) == {}

        profile = await graph.get_entity_profile("OnlyB", store=STORE_B)
        assert profile["memory_count"] == 1

    async def test_common_neighbors_does_not_cross_stores(self, graph):
        scoped = await graph.common_neighbors(MEM_A1, store=STORE_A)
        assert MEM_B1 not in {candidate for candidate, _shared, _deg in scoped}

        # The expansion itself is unchanged: unscoped, MEM_B1 is a candidate
        # through the shared "Shared" entity.
        unscoped = await graph.common_neighbors(MEM_A1)
        assert MEM_B1 in {candidate for candidate, _shared, _deg in unscoped}

    async def test_cross_store_association_does_not_bridge_two_in_store_memories(self, graph):
        """Both endpoints of an association have to be in the store.

        Y1 lives in store beta and associates with X1 and X2 in alpha. Confining
        only the source hash would let X2 surface as related to X1 through Y1.
        """
        scoped = await graph.common_neighbors(MEM_X1, store=STORE_A)

        assert MEM_X2 not in {candidate for candidate, _shared, _deg in scoped}
        # Control: unscoped, the foreign memory is the bridge.
        unscoped = await graph.common_neighbors(MEM_X1)
        assert MEM_X2 in {candidate for candidate, _shared, _deg in unscoped}

    async def test_soft_deleted_memory_does_not_count(self, graph):
        """A deleted memory's entity link is not part of a scoped read."""
        assert await graph.find_memories_by_entity("Ghost", store=STORE_A) == []
        assert await graph.get_entities_for_memory(MEM_D1, store=STORE_A) == []
        assert await graph.get_entity_profile("Ghost", store=STORE_A) == {}

        # Control: the unscoped path reads only memory_graph and still reports
        # it. That gap is pre-existing and out of scope here.
        assert await graph.find_memories_by_entity("Ghost") == [MEM_D1]

    async def test_related_entities_do_not_leak_across_stores(self, graph):
        scoped = await _discover_related_entities(graph, [MEM_A1], 1, store=STORE_A)
        scoped_names = {entity["name"] for entity in scoped}

        assert scoped_names == {"Shared", "OnlyA"}

    async def test_related_entities_unscoped_reach_every_store(self, graph):
        """The pre-#2118 behaviour, kept as the control case."""
        unscoped = await _discover_related_entities(graph, [MEM_A1], 1)
        names = {entity["name"] for entity in unscoped}

        assert "OnlyB" in names


class TestEntityDetailIsStoreScoped:
    async def test_chunks_and_related_entities_stay_in_store(self, graph):
        storage = _Storage({MEM_A1: "alpha one", MEM_A2: "alpha two", MEM_B1: "beta one"})

        detail = await get_entity_detail(
            storage,
            graph,
            entity_id="shared",
            store=STORE_A,
            entity_name="Shared",
        )

        assert {chunk["hash"] for chunk in detail["chunks"]} == {MEM_A1, MEM_A2}
        assert {entity["name"] for entity in detail["related_entities"]} <= {"Shared", "OnlyA"}

    async def test_chunk_rows_are_read_with_the_claimed_store(self, graph):
        storage = _Storage({MEM_A1: "alpha one", MEM_A2: "alpha two"})

        await get_entity_detail(
            storage, graph, entity_id="shared", store=STORE_A, entity_name="Shared",
        )

        assert storage.lookup_stores and set(storage.lookup_stores) == {STORE_A}

    async def test_unconfining_flag_no_longer_refuses_a_named_store(self, graph):
        """The fail-closed ValueError is gone (#2120): per-store credentials are
        gated at the transport, and the backend confines or refuses the read."""
        storage = _Storage({MEM_A1: "alpha one", MEM_A2: "alpha two"})
        storage.get_by_hash_is_store_scoped = False

        detail = await get_entity_detail(
            storage, graph, entity_id="shared", store=STORE_A, entity_name="Shared",
        )

        assert {chunk["hash"] for chunk in detail["chunks"]} == {MEM_A1, MEM_A2}

    async def test_detail_of_an_entity_only_in_another_store_is_empty(self, graph):
        storage = _Storage({MEM_B1: "beta one"})

        detail = await get_entity_detail(
            storage,
            graph,
            entity_id="onlyb",
            store=STORE_A,
            entity_name="OnlyB",
        )

        assert detail["chunks"] == []
        assert detail["related_entities"] == []


class TestStorePredicateContract:
    def test_none_and_all_are_unscoped(self):
        for store in (None, "all"):
            assert GraphStorage._store_predicate(store) == ("", [])

    def test_named_store_confines_through_the_source_memory(self):
        fragment, params = GraphStorage._store_predicate(STORE_A)

        assert "source_hash IN" in fragment
        assert "memories" in fragment
        assert "deleted_at IS NULL" in fragment
        assert params == [STORE_A]

    def test_both_endpoints_adds_the_target_check_for_associations(self):
        fragment, params = GraphStorage._store_predicate(STORE_A, both_endpoints=True)

        assert "target_hash IN" in fragment
        # Entity names are targets of has_entity rows and are exempt.
        assert "relationship_type = 'has_entity'" in fragment
        assert params == [STORE_A, STORE_A]


class _FakeGraphStorage:
    """Production-shaped stand-in: an instance that can carry the warn flag."""


class TestMilvusGraphRefusesNamedStore:
    """The Milvus graph accepts `store` for parity (#2118) and refuses it.

    Signature parity with GraphStorage is enforced by
    ``tests/storage/test_backend_store_signature_conformance.py``. Until #2119
    gives the Milvus association collection a store field, the honest answer to
    a named store is an empty result rather than the global one.
    """

    def test_named_store_is_refused_and_unscoped_passes_through(self):
        from mcp_memory_service.storage.milvus_graph import _named_store_refused

        fake = _FakeGraphStorage()
        assert _named_store_refused(fake, STORE_A, "list_entities") is True
        assert _named_store_refused(fake, None, "list_entities") is False
        assert _named_store_refused(fake, "all", "list_entities") is False

    def test_warning_is_emitted_once_per_instance(self):
        from mcp_memory_service.storage.milvus_graph import _named_store_refused

        fake = _FakeGraphStorage()
        assert _named_store_refused(fake, STORE_A, "list_entities") is True
        assert fake._store_scope_warned is True
        # Second call stays quiet but still refuses.
        assert _named_store_refused(fake, STORE_B, "list_entities") is True
