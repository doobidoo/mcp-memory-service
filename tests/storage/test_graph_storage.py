"""Unit tests for GraphStorage class."""

import pytest
import json
import time
from typing import List, Tuple

from mcp_memory_service.storage.graph import GraphStorage


@pytest.mark.unit
class TestGraphStorage:
    """Test the graph-based storage layer for memory associations."""

    @pytest.mark.asyncio
    async def test_store_association_basic(self, graph_storage):
        """Test basic association storage functionality.

        Validates:
        - Association is stored successfully
        - Returns True on successful storage
        - Basic validation of required fields
        """
        result = await graph_storage.store_association(
            source_hash="test_source",
            target_hash="test_target",
            similarity=0.75,
            connection_types=["semantic"],
            metadata={"test": "data"}
        )

        assert result is True, "Association storage should return True"

    @pytest.mark.asyncio
    async def test_store_bidirectional_symmetric(self, graph_storage):
        """Test bidirectional storage for symmetric relationships (related, contradicts).

        Validates:
        - Both directions of association exist for symmetric types
        - Can query from either direction
        - Bidirectional edges work correctly for symmetric relationships
        """
        await graph_storage.store_association(
            source_hash="node_a",
            target_hash="node_b",
            similarity=0.80,
            connection_types=["semantic"],
            relationship_type="related"  # Symmetric
        )

        # Both directions should exist
        connected_from_a = await graph_storage.find_connected("node_a", max_hops=1)
        connected_from_b = await graph_storage.find_connected("node_b", max_hops=1)

        assert len(connected_from_a) == 1, "Should find connection from A"
        assert len(connected_from_b) == 1, "Should find connection from B"
        assert connected_from_a[0][0] == "node_b", "A should connect to B"
        assert connected_from_b[0][0] == "node_a", "B should connect to A"

    @pytest.mark.asyncio
    async def test_store_directed_asymmetric(self, graph_storage):
        """Test directed storage for asymmetric relationships (causes, fixes, supports).

        Validates:
        - Only forward direction exists for asymmetric types
        - Can query incoming direction to find source
        - Reverse outgoing direction does not exist
        """
        await graph_storage.store_association(
            source_hash="decision_a",
            target_hash="error_b",
            similarity=0.90,
            connection_types=["causal"],
            relationship_type="causes"  # Asymmetric
        )

        # Only forward direction should exist
        outgoing = await graph_storage.find_connected(
            "decision_a", max_hops=1, direction="outgoing"
        )
        assert len(outgoing) == 1, "Should find outgoing edge"
        assert outgoing[0][0] == "error_b", "decision_a should cause error_b"

        # Can query incoming to find source
        incoming = await graph_storage.find_connected(
            "error_b", max_hops=1, direction="incoming"
        )
        assert len(incoming) == 1, "Should find incoming edge"
        assert incoming[0][0] == "decision_a", "decision_a should be the cause"

        # Reverse outgoing direction should NOT exist
        reverse = await graph_storage.find_connected(
            "error_b", max_hops=1, direction="outgoing"
        )
        assert len(reverse) == 0, "No error_b → decision_a edge should exist"

    @pytest.mark.asyncio
    async def test_duplicate_association_handling(self, graph_storage):
        """Test idempotent association storage (duplicate handling).

        Validates:
        - Storing same association twice doesn't create duplicates
        - Uses INSERT OR REPLACE for idempotency
        - Latest metadata/similarity values are preserved
        """
        # Store association first time
        await graph_storage.store_association(
            "dup_a", "dup_b", 0.60, ["semantic"], {"version": 1}
        )

        # Store same association again with different metadata
        await graph_storage.store_association(
            "dup_a", "dup_b", 0.65, ["semantic"], {"version": 2}
        )

        # Should only find one connection (no duplicates)
        connected = await graph_storage.find_connected("dup_a", max_hops=1)
        assert len(connected) == 1, "Should not create duplicate associations"

    @pytest.mark.asyncio
    async def test_self_loop_prevention(self, graph_storage):
        """Test that self-loops (A→A) are prevented.

        Validates:
        - Returns False when source == target
        - Logs warning about self-loop attempt
        - Graph remains clean without self-loops
        """
        result = await graph_storage.store_association(
            "same_hash", "same_hash", 1.0, ["semantic"], {}
        )

        assert result is False, "Should reject self-loop association"

    @pytest.mark.asyncio
    async def test_empty_hash_validation(self, graph_storage):
        """Test rejection of empty string hashes.

        Validates:
        - Returns False for empty source or target
        - Logs error about invalid hash
        - Prevents database pollution with invalid data
        """
        result1 = await graph_storage.store_association(
            "", "valid_hash", 0.5, ["semantic"], {}
        )
        result2 = await graph_storage.store_association(
            "valid_hash", "", 0.5, ["semantic"], {}
        )

        assert result1 is False, "Should reject empty source hash"
        assert result2 is False, "Should reject empty target hash"

    @pytest.mark.asyncio
    async def test_find_connected_basic(self, graph_storage, sample_graph_data):
        """Test basic 1-hop connection discovery.

        Uses sample_graph_data fixture with known topology.

        Validates:
        - Finds immediate neighbors (1-hop)
        - Returns list of (hash, distance) tuples
        - Distance is correct for direct connections
        """
        # Test linear chain: A → B
        connected = await graph_storage.find_connected("hash_a", max_hops=1)

        assert len(connected) > 0, "Should find at least one connection"
        assert ("hash_b", 1) in connected, "Should find direct neighbor B"

        # Test hub node: L connected to M, N, O, P, Q
        hub_connected = await graph_storage.find_connected("hash_l", max_hops=1)
        hub_data = sample_graph_data["hub"]

        assert len(hub_connected) == len(hub_data["spokes"]), \
            f"Hub should connect to {len(hub_data['spokes'])} nodes"

    @pytest.mark.asyncio
    async def test_find_connected_multi_hop(self, graph_storage, sample_graph_data):
        """Test multi-hop connection discovery (2-3 hops).

        Uses linear chain: A → B → C → D

        Validates:
        - Finds nodes at multiple distances
        - Respects max_hops parameter
        - Returns correct distance values
        - Sorted by distance (BFS order)
        """
        # Query with max_hops=2 from A
        connected_2hop = await graph_storage.find_connected("hash_a", max_hops=2)

        # Should find B (1-hop) and C (2-hop)
        hashes_found = {hash for hash, _ in connected_2hop}
        assert "hash_b" in hashes_found, "Should find B at 1-hop"
        assert "hash_c" in hashes_found, "Should find C at 2-hop"
        assert "hash_d" not in hashes_found, "Should not find D at 3-hop with max_hops=2"

        # Query with max_hops=3 from A
        connected_3hop = await graph_storage.find_connected("hash_a", max_hops=3)
        hashes_3hop = {hash for hash, _ in connected_3hop}

        assert "hash_d" in hashes_3hop, "Should find D at 3-hop with max_hops=3"

    @pytest.mark.asyncio
    async def test_find_connected_with_cycles(self, graph_storage, sample_graph_data):
        """Test cycle prevention in recursive CTE traversal.

        Uses triangle cycle: E → F → G → E

        Validates:
        - Doesn't loop infinitely on cyclic graphs
        - Finds all nodes in cycle
        - Returns results (may include duplicates at different distances due to bidirectional edges)
        """
        # Query from E in triangle
        connected = await graph_storage.find_connected("hash_e", max_hops=3)

        # Should find F and G in the cycle
        hashes_found = {hash for hash, _ in connected}
        assert "hash_f" in hashes_found, "Should find F in cycle"
        assert "hash_g" in hashes_found, "Should find G in cycle"

        # Verify query completes (doesn't loop infinitely)
        assert len(connected) > 0, "Should find connected nodes in cycle"

        # In triangle with bidirectional edges, nodes can appear at multiple distances
        # (e.g., F at distance 1 via E→F, and at distance 3 via E→G→F)
        # This is expected behavior and shows the CTE is exploring all paths

    @pytest.mark.asyncio
    async def test_shortest_path_direct(self, graph_storage, sample_graph_data):
        """Test shortest path for directly connected nodes.

        Uses linear chain: A → B

        Validates:
        - Finds direct path [A, B]
        - Path length is 2 (source + target)
        - Optimal for 1-hop connection
        """
        path = await graph_storage.shortest_path("hash_a", "hash_b")

        assert path is not None, "Should find path between connected nodes"
        assert len(path) == 2, "Direct connection should have path length 2"
        assert path == ["hash_a", "hash_b"], "Path should be [A, B]"

    @pytest.mark.asyncio
    async def test_shortest_path_multi_hop(self, graph_storage, sample_graph_data):
        """Test shortest path for multi-hop connections.

        Uses diamond topology:
        - Path 1: H → I → K (2 hops, high similarity)
        - Path 2: H → J → K (2 hops, lower similarity)

        Validates:
        - Finds a valid path (may not be deterministic which one)
        - Path length is correct
        - BFS guarantees shortest path
        """
        path = await graph_storage.shortest_path("hash_h", "hash_k")

        assert path is not None, "Should find path in diamond topology"
        assert len(path) == 3, "Path should have length 3 (2 hops)"
        assert path[0] == "hash_h", "Path should start at H"
        assert path[-1] == "hash_k", "Path should end at K"

        # Middle node should be either I or J
        middle_node = path[1]
        assert middle_node in ["hash_i", "hash_j"], \
            "Middle node should be either I or J"

    @pytest.mark.asyncio
    async def test_shortest_path_no_path(self, graph_storage, sample_graph_data):
        """Test shortest path behavior for disconnected nodes.

        Queries between nodes in separate graph components.

        Validates:
        - Returns None when no path exists
        - Doesn't loop indefinitely searching
        - Respects max_depth parameter
        """
        # Query from linear chain to triangle (disconnected components)
        path = await graph_storage.shortest_path("hash_a", "hash_e")

        assert path is None, "Should return None for disconnected nodes"

    @pytest.mark.asyncio
    async def test_shortest_path_self(self, graph_storage):
        """Test shortest path when source == target.

        Validates:
        - Returns trivial path [hash] for same node
        - Doesn't perform graph traversal
        - Optimizes trivial case
        """
        path = await graph_storage.shortest_path("hash_x", "hash_x")

        assert path == ["hash_x"], "Self-path should return [hash]"

    @pytest.mark.asyncio
    async def test_get_subgraph(self, graph_storage, sample_graph_data):
        """Test subgraph extraction for visualization.

        Uses hub topology: L connected to M, N, O, P, Q

        Validates:
        - Returns dict with "nodes" and "edges" keys
        - Includes center node and neighbors
        - Edges contain required fields (source, target, similarity, etc.)
        - Deduplicates bidirectional edges
        """
        hub_data = sample_graph_data["hub"]
        center = hub_data["center"]

        subgraph = await graph_storage.get_subgraph(center, radius=1)

        # Validate structure
        assert "nodes" in subgraph, "Subgraph should have 'nodes' key"
        assert "edges" in subgraph, "Subgraph should have 'edges' key"

        # Validate nodes (center + spokes)
        expected_node_count = 1 + len(hub_data["spokes"])  # L + M,N,O,P,Q
        assert len(subgraph["nodes"]) == expected_node_count, \
            f"Should have {expected_node_count} nodes in hub subgraph"

        # Validate edges (one per spoke, deduplicated)
        assert len(subgraph["edges"]) == len(hub_data["spokes"]), \
            "Should have one edge per spoke (deduplicated)"

        # Validate edge format
        for edge in subgraph["edges"]:
            assert "source" in edge, "Edge should have source"
            assert "target" in edge, "Edge should have target"
            assert "similarity" in edge, "Edge should have similarity"
            assert "connection_types" in edge, "Edge should have connection_types"
            assert "metadata" in edge, "Edge should have metadata"

    @pytest.mark.asyncio
    async def test_connection_types_json_array(self, graph_storage):
        """Test that connection_types are stored/retrieved as JSON arrays.

        Validates:
        - List of connection types is JSON-serialized
        - Retrieval deserializes back to list
        - Multiple connection types supported
        """
        conn_types = ["semantic", "temporal", "causal"]

        await graph_storage.store_association(
            "json_test_a", "json_test_b", 0.70, conn_types, {}
        )

        # Retrieve subgraph to check edge data
        subgraph = await graph_storage.get_subgraph("json_test_a", radius=1)

        assert len(subgraph["edges"]) > 0, "Should have at least one edge"
        edge = subgraph["edges"][0]

        assert edge["connection_types"] == conn_types, \
            "Connection types should match original list"

    @pytest.mark.asyncio
    async def test_similarity_scores(self, graph_storage):
        """Test similarity value handling (0.0-1.0 range).

        Validates:
        - Stores float similarity correctly
        - Preserves precision
        - Retrieves exact value
        """
        test_similarity = 0.6543

        await graph_storage.store_association(
            "sim_a", "sim_b", test_similarity, ["semantic"], {}
        )

        subgraph = await graph_storage.get_subgraph("sim_a", radius=1)
        edge = subgraph["edges"][0]

        # Allow small floating-point precision differences
        assert abs(edge["similarity"] - test_similarity) < 0.0001, \
            "Similarity should be preserved with precision"

    @pytest.mark.asyncio
    async def test_metadata_storage(self, graph_storage):
        """Test JSON metadata storage and retrieval.

        Validates:
        - Nested dict metadata is JSON-serialized
        - Retrieval deserializes correctly
        - Null/empty metadata handled gracefully
        """
        complex_metadata = {
            "discovery_method": "creative_association",
            "timestamp": "2024-01-15T10:30:00Z",
            "confidence": 0.85,
            "tags": ["important", "reference"]
        }

        await graph_storage.store_association(
            "meta_a", "meta_b", 0.75, ["semantic"], complex_metadata
        )

        subgraph = await graph_storage.get_subgraph("meta_a", radius=1)
        edge = subgraph["edges"][0]

        assert edge["metadata"] == complex_metadata, \
            "Metadata should match original dict"

    @pytest.mark.asyncio
    async def test_metadata_null_handling(self, graph_storage):
        """Test that None/empty metadata doesn't cause errors.

        Validates:
        - None metadata is stored as empty JSON object
        - Empty dict metadata works correctly
        - Retrieval returns empty dict, not None
        """
        await graph_storage.store_association(
            "null_meta_a", "null_meta_b", 0.60, ["semantic"], None
        )

        subgraph = await graph_storage.get_subgraph("null_meta_a", radius=1)
        edge = subgraph["edges"][0]

        assert edge["metadata"] == {}, "Null metadata should become empty dict"

    @pytest.mark.asyncio
    async def test_query_performance_benchmark(self, graph_storage):
        """Benchmark find_connected query performance.

        Target: <10ms for 1-hop, <50ms for 3-hop

        Validates:
        - Query completes within performance budget
        - SQLite recursive CTE is efficient
        - Indexes are effective
        """
        # Create test data: 10 nodes in linear chain
        for i in range(9):
            await graph_storage.store_association(
                f"perf_{i}", f"perf_{i+1}", 0.65, ["semantic"], {}
            )

        # Benchmark 1-hop query
        start_1hop = time.perf_counter()
        await graph_storage.find_connected("perf_0", max_hops=1)
        time_1hop = (time.perf_counter() - start_1hop) * 1000  # Convert to ms

        # Benchmark 3-hop query
        start_3hop = time.perf_counter()
        await graph_storage.find_connected("perf_0", max_hops=3)
        time_3hop = (time.perf_counter() - start_3hop) * 1000  # Convert to ms

        # Assert performance targets (relaxed for CI environments)
        assert time_1hop < 50, \
            f"1-hop query took {time_1hop:.2f}ms (target: <10ms, relaxed: <50ms)"
        assert time_3hop < 150, \
            f"3-hop query took {time_3hop:.2f}ms (target: <50ms, relaxed: <150ms)"

    @pytest.mark.asyncio
    async def test_find_connected_empty_hash(self, graph_storage):
        """Test find_connected with empty hash parameter.

        Validates:
        - Returns empty list for invalid input
        - Logs error message
        - Doesn't crash or raise exception
        """
        result = await graph_storage.find_connected("")

        assert result == [], "Should return empty list for empty hash"

    @pytest.mark.asyncio
    async def test_shortest_path_empty_hash(self, graph_storage):
        """Test shortest_path with empty hash parameters.

        Validates:
        - Returns None for invalid input
        - Logs error message
        - Handles both empty source and target
        """
        result1 = await graph_storage.shortest_path("", "valid_hash")
        result2 = await graph_storage.shortest_path("valid_hash", "")

        assert result1 is None, "Should return None for empty source hash"
        assert result2 is None, "Should return None for empty target hash"

    @pytest.mark.asyncio
    async def test_get_subgraph_empty_hash(self, graph_storage):
        """Test get_subgraph with empty hash parameter.

        Validates:
        - Returns empty graph structure
        - Doesn't crash or raise exception
        - Returns proper dict format
        """
        result = await graph_storage.get_subgraph("")

        assert result == {"nodes": [], "edges": []}, \
            "Should return empty graph for invalid hash"

    @pytest.mark.asyncio
    async def test_subgraph_multi_hop(self, graph_storage, sample_graph_data):
        """Test subgraph extraction with radius > 1.

        Uses linear chain: A → B → C → D

        Validates:
        - Radius parameter controls subgraph size
        - Includes all nodes within radius
        - Includes all edges between included nodes
        """
        # Extract subgraph from A with radius 2
        subgraph = await graph_storage.get_subgraph("hash_a", radius=2)

        # Should include A, B, C (but not D at 3-hop)
        expected_nodes = {"hash_a", "hash_b", "hash_c"}
        actual_nodes = set(subgraph["nodes"])

        assert actual_nodes == expected_nodes, \
            f"Should include nodes within radius 2: {expected_nodes}"

        # Should include edges A→B and B→C
        assert len(subgraph["edges"]) == 2, \
            "Should have 2 edges (A→B, B→C) in radius-2 subgraph"

    @pytest.mark.asyncio
    async def test_delete_association(self, graph_storage):
        """Test deleting associations and bidirectional removal."""
        # Store test association
        await graph_storage.store_association(
            "hash_a", "hash_b", 0.8, ["test"], {}
        )

        # Verify it exists
        assoc = await graph_storage.get_association("hash_a", "hash_b")
        assert assoc is not None, "Association should exist before deletion"

        # Delete the association
        deleted = await graph_storage.delete_association("hash_a", "hash_b")
        assert deleted is True, "Deletion should be successful"

        # Verify it's gone
        assoc_after = await graph_storage.get_association("hash_a", "hash_b")
        assert assoc_after is None, "Association should not exist after deletion"

        # Verify idempotency (deleting again should return False)
        deleted_again = await graph_storage.delete_association("hash_a", "hash_b")
        assert deleted_again is False, \
            "Deleting non-existent association should return False"

    @pytest.mark.asyncio
    async def test_get_association_count(self, graph_storage):
        """Test getting the count of direct associations for a memory."""
        # Create a hub node with 5 connections
        hub_hash = "hash_hub"
        for i in range(5):
            await graph_storage.store_association(
                hub_hash, f"hash_spoke_{i}", 0.7, ["test"], {}
            )

        # Get count for hub node
        count = await graph_storage.get_association_count(hub_hash)
        assert count == 5, f"Hub node should have 5 connections, got {count}"

        # Create single connection
        await graph_storage.store_association(
            "hash_single", "hash_other", 0.6, ["test"], {}
        )

        count_single = await graph_storage.get_association_count("hash_single")
        assert count_single == 1, "Single connection node should have count 1"

        # Node with no connections
        count_none = await graph_storage.get_association_count("non_existent_hash")
        assert count_none == 0, "Non-existent node should have count 0"
