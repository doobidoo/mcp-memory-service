# Copyright 2024 Heinrich Krupp
# Licensed under the Apache License, Version 2.0

"""
Unit tests for MilvusGraphStorage.

Uses Milvus Lite (in-process, file-backed) for real integration testing.
Each test gets a fresh collection via a unique temp-file URI.
"""

import asyncio
import os
import shutil
import tempfile

import pytest

# Skip entire module if pymilvus is not installed
pymilvus = pytest.importorskip("pymilvus", reason="pymilvus required for Milvus graph tests")

from mcp_memory_service.storage.milvus_graph import MilvusGraphStorage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_dir():
    """Create a temporary directory for Milvus Lite databases."""
    d = tempfile.mkdtemp(prefix="milvus_graph_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture()
async def graph(tmp_dir):
    """Provide an initialized MilvusGraphStorage backed by Milvus Lite."""
    uri = os.path.join(tmp_dir, "test.db")
    gs = MilvusGraphStorage(uri=uri, collection_name="test_mem")
    await gs.initialize()
    yield gs
    await gs.close()


# ---------------------------------------------------------------------------
# Collection naming
# ---------------------------------------------------------------------------

class TestCollectionNaming:
    def test_graph_suffix(self):
        """Verify _graph suffix derivation from collection name."""
        gs = MilvusGraphStorage(uri="./unused.db", collection_name="my_memories")
        assert gs.collection_name == "my_memories_graph"

    def test_default_collection_name(self):
        gs = MilvusGraphStorage(uri="./unused.db")
        assert gs.collection_name == "mcp_memory_graph"


# ---------------------------------------------------------------------------
# store_association
# ---------------------------------------------------------------------------

class TestStoreAssociation:
    @pytest.mark.asyncio
    async def test_symmetric_creates_both_directions(self, graph):
        """Symmetric relationship stores A→B and B→A."""
        ok = await graph.store_association(
            "hash_a", "hash_b", 0.8, ["semantic"],
            relationship_type="related",
        )
        assert ok is True

        # Both directions should be retrievable
        assoc_ab = await graph.get_association("hash_a", "hash_b")
        assert assoc_ab is not None
        assert assoc_ab["similarity"] == pytest.approx(0.8, abs=0.01)

        assoc_ba = await graph.get_association("hash_b", "hash_a")
        assert assoc_ba is not None

    @pytest.mark.asyncio
    async def test_asymmetric_creates_forward_only(self, graph):
        """Asymmetric relationship stores only A→B."""
        ok = await graph.store_association(
            "hash_a", "hash_b", 0.7, ["causal"],
            relationship_type="causes",
        )
        assert ok is True

        # Forward direction exists
        assoc = await graph.get_association("hash_a", "hash_b")
        assert assoc is not None

        # get_association checks both directions, so it will find the forward edge
        # But the raw count from source_hash should only show 1 record for hash_a
        count_a = await graph.get_association_count("hash_a")
        count_b = await graph.get_association_count("hash_b")
        assert count_a == 1
        assert count_b == 0  # No reverse edge stored

    @pytest.mark.asyncio
    async def test_self_loop_rejected(self, graph):
        """Self-loop should be rejected."""
        ok = await graph.store_association(
            "hash_a", "hash_a", 0.5, ["self"],
        )
        assert ok is False

    @pytest.mark.asyncio
    async def test_invalid_similarity_below_zero(self, graph):
        """Similarity < 0 should be rejected."""
        ok = await graph.store_association(
            "hash_a", "hash_b", -0.1, ["test"],
        )
        assert ok is False

    @pytest.mark.asyncio
    async def test_invalid_similarity_above_one(self, graph):
        """Similarity > 1 should be rejected."""
        ok = await graph.store_association(
            "hash_a", "hash_b", 1.5, ["test"],
        )
        assert ok is False

    @pytest.mark.asyncio
    async def test_empty_hash_rejected(self, graph):
        """Empty hash strings should be rejected."""
        assert await graph.store_association("", "hash_b", 0.5, ["test"]) is False
        assert await graph.store_association("hash_a", "", 0.5, ["test"]) is False

    @pytest.mark.asyncio
    async def test_upsert_overwrites(self, graph):
        """Storing same edge twice should overwrite with new data."""
        await graph.store_association(
            "hash_a", "hash_b", 0.5, ["v1"],
            relationship_type="related",
        )
        await graph.store_association(
            "hash_a", "hash_b", 0.9, ["v2"],
            relationship_type="related",
        )

        assoc = await graph.get_association("hash_a", "hash_b")
        assert assoc is not None
        assert assoc["similarity"] == pytest.approx(0.9, abs=0.01)


# ---------------------------------------------------------------------------
# find_connected
# ---------------------------------------------------------------------------

class TestFindConnected:
    @pytest.mark.asyncio
    async def test_empty_graph(self, graph):
        """Empty graph returns empty list."""
        result = await graph.find_connected("nonexistent")
        assert result == []

    @pytest.mark.asyncio
    async def test_direct_neighbors(self, graph):
        """Find direct neighbors (1 hop)."""
        await graph.store_association("A", "B", 0.8, ["s"], relationship_type="related")
        await graph.store_association("A", "C", 0.7, ["s"], relationship_type="related")

        connected = await graph.find_connected("A", max_hops=1)
        hashes = {h for h, _ in connected}
        assert "B" in hashes
        assert "C" in hashes
        assert all(d == 1 for _, d in connected)

    @pytest.mark.asyncio
    async def test_multi_hop(self, graph):
        """Find nodes 2 hops away."""
        await graph.store_association("A", "B", 0.8, ["s"], relationship_type="related")
        await graph.store_association("B", "C", 0.7, ["s"], relationship_type="related")

        connected = await graph.find_connected("A", max_hops=2)
        hashes = {h for h, _ in connected}
        assert "B" in hashes
        assert "C" in hashes

    @pytest.mark.asyncio
    async def test_direction_outgoing(self, graph):
        """Outgoing direction only follows source→target."""
        await graph.store_association("A", "B", 0.8, ["s"], relationship_type="causes")

        out = await graph.find_connected("A", max_hops=1, direction="outgoing")
        assert len(out) == 1
        assert out[0][0] == "B"

        # B has no outgoing edges
        out_b = await graph.find_connected("B", max_hops=1, direction="outgoing")
        assert len(out_b) == 0

    @pytest.mark.asyncio
    async def test_direction_incoming(self, graph):
        """Incoming direction only follows target→source."""
        await graph.store_association("A", "B", 0.8, ["s"], relationship_type="causes")

        inc = await graph.find_connected("B", max_hops=1, direction="incoming")
        assert len(inc) == 1
        assert inc[0][0] == "A"


# ---------------------------------------------------------------------------
# shortest_path
# ---------------------------------------------------------------------------

class TestShortestPath:
    @pytest.mark.asyncio
    async def test_trivial_path(self, graph):
        """Path from A to A is [A]."""
        path = await graph.shortest_path("A", "A")
        assert path == ["A"]

    @pytest.mark.asyncio
    async def test_direct_path(self, graph):
        """Direct edge A→B gives path [A, B]."""
        await graph.store_association("A", "B", 0.8, ["s"], relationship_type="related")
        path = await graph.shortest_path("A", "B")
        assert path == ["A", "B"]

    @pytest.mark.asyncio
    async def test_two_hop_path(self, graph):
        """Path through intermediate node."""
        await graph.store_association("A", "B", 0.8, ["s"], relationship_type="related")
        await graph.store_association("B", "C", 0.7, ["s"], relationship_type="related")
        path = await graph.shortest_path("A", "C")
        assert path is not None
        assert path[0] == "A"
        assert path[-1] == "C"
        assert len(path) <= 3

    @pytest.mark.asyncio
    async def test_no_path(self, graph):
        """No path between disconnected nodes."""
        await graph.store_association("A", "B", 0.8, ["s"], relationship_type="related")
        path = await graph.shortest_path("A", "Z")
        assert path is None


# ---------------------------------------------------------------------------
# get_subgraph
# ---------------------------------------------------------------------------

class TestGetSubgraph:
    @pytest.mark.asyncio
    async def test_empty_graph(self, graph):
        """Subgraph of empty graph has center node only."""
        sg = await graph.get_subgraph("A")
        assert "A" in sg["nodes"]
        assert sg["edges"] == []

    @pytest.mark.asyncio
    async def test_subgraph_structure(self, graph):
        """Subgraph includes nodes and edges within radius."""
        await graph.store_association("A", "B", 0.8, ["s"], relationship_type="related")
        await graph.store_association("B", "C", 0.7, ["s"], relationship_type="related")

        sg = await graph.get_subgraph("A", radius=2)
        assert "A" in sg["nodes"]
        assert "B" in sg["nodes"]
        assert "C" in sg["nodes"]
        assert len(sg["edges"]) >= 1

    @pytest.mark.asyncio
    async def test_edge_deduplication(self, graph):
        """Edges should be deduplicated (no A-B and B-A duplicates)."""
        await graph.store_association("A", "B", 0.8, ["s"], relationship_type="related")

        sg = await graph.get_subgraph("A", radius=1)
        # Should have exactly 1 edge despite bidirectional storage
        ab_edges = [
            e for e in sg["edges"]
            if {e["source"], e["target"]} == {"A", "B"}
        ]
        assert len(ab_edges) == 1


# ---------------------------------------------------------------------------
# delete_association
# ---------------------------------------------------------------------------

class TestDeleteAssociation:
    @pytest.mark.asyncio
    async def test_delete_symmetric(self, graph):
        """Deleting symmetric edge removes both directions."""
        await graph.store_association("A", "B", 0.8, ["s"], relationship_type="related")

        ok = await graph.delete_association("A", "B")
        assert ok is True

        assert await graph.get_association("A", "B") is None
        assert await graph.get_association("B", "A") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, graph):
        """Deleting nonexistent edge returns True (no-op delete)."""
        ok = await graph.delete_association("X", "Y")
        assert ok is True  # Milvus delete is idempotent


# ---------------------------------------------------------------------------
# get_relationship_types
# ---------------------------------------------------------------------------

class TestGetRelationshipTypes:
    @pytest.mark.asyncio
    async def test_multiple_types(self, graph):
        """Count relationship types for a memory."""
        await graph.store_association("A", "B", 0.8, ["s"], relationship_type="related")
        await graph.store_association("A", "C", 0.7, ["s"], relationship_type="causes")
        await graph.store_association("A", "D", 0.6, ["s"], relationship_type="causes")

        types = await graph.get_relationship_types("A")
        assert types.get("related", 0) >= 1
        assert types.get("causes", 0) >= 2


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_operations_before_init(self):
        """Operations on uninitialized storage return empty/failure values."""
        gs = MilvusGraphStorage(uri="./unused.db", collection_name="test")
        # client is None — all operations should return gracefully
        assert await gs.store_association("A", "B", 0.5, ["s"]) is False
        assert await gs.find_connected("A") == []
        assert await gs.shortest_path("A", "B") is None
        assert await gs.get_subgraph("A") == {"nodes": [], "edges": []}
        assert await gs.get_association("A", "B") is None
        assert await gs.delete_association("A", "B") is False
        assert await gs.get_association_count("A") == 0
        assert await gs.get_relationship_types("A") == {}
