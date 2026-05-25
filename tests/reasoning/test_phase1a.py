"""
Phase 1a tests: transitive closure, abduction, and handler integration.

Covers:
  A. GraphStorage.transitive_closure (15 tests)
  B. GraphStorage.common_neighbors (10 tests)
  C. SemanticReasoner.abduce (15 tests)
  D. Handler integration for abduce / infer / suggest (20 tests)

Total: 60 tests
"""

import json
import os
import sys
import tempfile
import time
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Module loading (matching tests/test_semantic_reasoner.py pattern)
# ---------------------------------------------------------------------------

_tests_dir = Path(__file__).parent.parent.parent  # tests/reasoning -> tests -> repo root
_graph_path = _tests_dir / "src" / "mcp_memory_service" / "storage" / "graph.py"
_reasoning_path = _tests_dir / "src" / "mcp_memory_service" / "reasoning" / "inference.py"
_handler_path = _tests_dir / "src" / "mcp_memory_service" / "server" / "handlers" / "graph.py"

spec = importlib.util.spec_from_file_location("graph_storage_mod", _graph_path)
_graph_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_graph_mod)
GraphStorage = _graph_mod.GraphStorage

spec2 = importlib.util.spec_from_file_location("inference_mod", _reasoning_path)
_inference_mod = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(_inference_mod)
SemanticReasoner = _inference_mod.SemanticReasoner

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
async def storage():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        db_path = f.name
    gs = GraphStorage(db_path)
    conn = await gs._get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_graph (
            source_hash TEXT NOT NULL,
            target_hash TEXT NOT NULL,
            similarity REAL NOT NULL,
            connection_types TEXT NOT NULL,
            metadata TEXT,
            created_at REAL NOT NULL,
            relationship_type TEXT DEFAULT 'related',
            PRIMARY KEY (source_hash, target_hash)
        )
    """)
    conn.commit()
    yield gs
    if gs._connection:
        gs._connection.close()
    os.unlink(db_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _add_edge(gs: GraphStorage, src: str, tgt: str, rel: str) -> None:
    conn = await gs._get_connection()
    conn.execute(
        """
        INSERT OR REPLACE INTO memory_graph
            (source_hash, target_hash, similarity, connection_types, metadata, created_at, relationship_type)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (src, tgt, 0.9, '["semantic"]', None, time.time(), rel),
    )
    conn.commit()


# ===========================================================================
# Section A: GraphStorage.transitive_closure (15 tests)
# ===========================================================================


class TestTransitiveClosure:

    @pytest.mark.asyncio
    async def test_empty_graph_returns_empty(self, storage):
        """A1: empty graph yields no inferred pairs."""
        result = await storage.transitive_closure("causes", max_hops=2)
        assert result == []

    @pytest.mark.asyncio
    async def test_single_direct_edge_no_chain(self, storage):
        """A2: single A->B edge produces no transitive pair (nothing to close)."""
        await _add_edge(storage, "A", "B", "causes")
        result = await storage.transitive_closure("causes", max_hops=2)
        assert result == []

    @pytest.mark.asyncio
    async def test_two_hop_chain(self, storage):
        """A3: A->B->C yields inferred (A, C, 2)."""
        await _add_edge(storage, "A", "B", "causes")
        await _add_edge(storage, "B", "C", "causes")
        result = await storage.transitive_closure("causes", max_hops=2)
        pairs = {(s, t): d for s, t, d in result}
        assert ("A", "C") in pairs
        assert pairs[("A", "C")] == 2

    @pytest.mark.asyncio
    async def test_three_hop_chain_all_inferred(self, storage):
        """A4: A->B->C->D with max_hops=3 finds (A,C,2), (A,D,3), (B,D,2)."""
        await _add_edge(storage, "A", "B", "causes")
        await _add_edge(storage, "B", "C", "causes")
        await _add_edge(storage, "C", "D", "causes")
        result = await storage.transitive_closure("causes", max_hops=3)
        pairs = {(s, t): d for s, t, d in result}
        assert ("A", "C") in pairs and pairs[("A", "C")] == 2
        assert ("A", "D") in pairs and pairs[("A", "D")] == 3
        assert ("B", "D") in pairs and pairs[("B", "D")] == 2

    @pytest.mark.asyncio
    async def test_max_hops_limits_distance(self, storage):
        """A5: max_hops=2 does NOT return a 3-hop pair."""
        await _add_edge(storage, "A", "B", "causes")
        await _add_edge(storage, "B", "C", "causes")
        await _add_edge(storage, "C", "D", "causes")
        result = await storage.transitive_closure("causes", max_hops=2)
        pairs = {(s, t) for s, t, d in result}
        assert ("A", "D") not in pairs

    @pytest.mark.asyncio
    async def test_max_hops_minimum_two(self, storage):
        """A6: passing max_hops=1 still finds 2-hop pairs (minimum is 2)."""
        await _add_edge(storage, "A", "B", "causes")
        await _add_edge(storage, "B", "C", "causes")
        # even if clamped to 2, the 2-hop result should appear
        result = await storage.transitive_closure("causes", max_hops=1)
        pairs = {(s, t) for s, t, d in result}
        assert ("A", "C") in pairs

    @pytest.mark.asyncio
    async def test_wrong_rel_type_empty(self, storage):
        """A7: querying wrong rel_type returns []."""
        await _add_edge(storage, "A", "B", "causes")
        await _add_edge(storage, "B", "C", "causes")
        result = await storage.transitive_closure("fixes", max_hops=2)
        assert result == []

    @pytest.mark.asyncio
    async def test_direct_edges_excluded(self, storage):
        """A8: direct A->C edge is NOT returned as an inferred pair."""
        await _add_edge(storage, "A", "B", "causes")
        await _add_edge(storage, "B", "C", "causes")
        await _add_edge(storage, "A", "C", "causes")  # direct — must be excluded
        result = await storage.transitive_closure("causes", max_hops=2)
        # (A, C) with distance 2 must NOT appear since edge exists directly
        pairs = {(s, t) for s, t, d in result}
        assert ("A", "C") not in pairs

    @pytest.mark.asyncio
    async def test_two_independent_chains(self, storage):
        """A9: two separate chains both produce their own inferred pairs."""
        await _add_edge(storage, "A", "B", "causes")
        await _add_edge(storage, "B", "C", "causes")
        await _add_edge(storage, "X", "Y", "causes")
        await _add_edge(storage, "Y", "Z", "causes")
        result = await storage.transitive_closure("causes", max_hops=2)
        pairs = {(s, t) for s, t, d in result}
        assert ("A", "C") in pairs
        assert ("X", "Z") in pairs

    @pytest.mark.asyncio
    async def test_diamond_graph(self, storage):
        """A10: A->B->D and A->C->D: (A, D, 2) should be inferred."""
        await _add_edge(storage, "A", "B", "causes")
        await _add_edge(storage, "A", "C", "causes")
        await _add_edge(storage, "B", "D", "causes")
        await _add_edge(storage, "C", "D", "causes")
        result = await storage.transitive_closure("causes", max_hops=2)
        pairs = {(s, t) for s, t, d in result}
        assert ("A", "D") in pairs

    @pytest.mark.asyncio
    async def test_cycle_safe(self, storage):
        """A11: A->B->A cycle doesn't infinite-loop."""
        await _add_edge(storage, "A", "B", "causes")
        await _add_edge(storage, "B", "A", "causes")
        # Should complete without error
        result = await storage.transitive_closure("causes", max_hops=2)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_mixed_rel_types_no_bleed(self, storage):
        """A12: causes chain doesn't produce inferred pairs for fixes."""
        await _add_edge(storage, "A", "B", "causes")
        await _add_edge(storage, "B", "C", "causes")
        await _add_edge(storage, "X", "Y", "fixes")
        await _add_edge(storage, "Y", "Z", "fixes")
        causes_result = await storage.transitive_closure("causes", max_hops=2)
        fixes_result = await storage.transitive_closure("fixes", max_hops=2)
        causes_pairs = {(s, t) for s, t, d in causes_result}
        fixes_pairs = {(s, t) for s, t, d in fixes_result}
        assert ("X", "Z") not in causes_pairs
        assert ("A", "C") not in fixes_pairs

    @pytest.mark.asyncio
    async def test_result_is_list_of_3tuples(self, storage):
        """A13: result type is List[Tuple[str, str, int]]."""
        await _add_edge(storage, "A", "B", "causes")
        await _add_edge(storage, "B", "C", "causes")
        result = await storage.transitive_closure("causes", max_hops=2)
        assert isinstance(result, list)
        for item in result:
            assert len(item) == 3
            s, t, d = item
            assert isinstance(s, str)
            assert isinstance(t, str)
            assert isinstance(d, int)

    @pytest.mark.asyncio
    async def test_distance_always_gte_2(self, storage):
        """A14: all distances in result are >= 2 (direct edges excluded)."""
        await _add_edge(storage, "A", "B", "causes")
        await _add_edge(storage, "B", "C", "causes")
        await _add_edge(storage, "C", "D", "causes")
        result = await storage.transitive_closure("causes", max_hops=3)
        for _, _, d in result:
            assert d >= 2

    @pytest.mark.asyncio
    async def test_no_duplicate_pairs(self, storage):
        """A15: no duplicate (source, target) pairs in result."""
        await _add_edge(storage, "A", "B", "causes")
        await _add_edge(storage, "B", "C", "causes")
        await _add_edge(storage, "A", "D", "causes")
        await _add_edge(storage, "D", "C", "causes")
        result = await storage.transitive_closure("causes", max_hops=2)
        pairs = [(s, t) for s, t, d in result]
        assert len(pairs) == len(set(pairs))


# ===========================================================================
# Section B: GraphStorage.common_neighbors (10 tests)
# ===========================================================================


class TestCommonNeighbors:

    @pytest.mark.asyncio
    async def test_isolated_node_returns_empty(self, storage):
        """B1: node with no edges returns []."""
        result = await storage.common_neighbors("isolated", min_shared=1)
        assert result == []

    @pytest.mark.asyncio
    async def test_triangle_suggests_each_other(self, storage):
        """B2: A and B both connected to hub -> each suggested to the other."""
        await _add_edge(storage, "A", "hub", "related")
        await _add_edge(storage, "B", "hub", "related")
        result_a = await storage.common_neighbors("A", min_shared=1)
        candidates = [c for c, _, _ in result_a]
        assert "B" in candidates

    @pytest.mark.asyncio
    async def test_min_shared_filters_low_overlap(self, storage):
        """B3: min_shared=2 filters out candidates sharing only 1 neighbor."""
        await _add_edge(storage, "A", "hub1", "related")
        await _add_edge(storage, "B", "hub1", "related")
        # B shares only 1 neighbor with A — should be excluded at min_shared=2
        result = await storage.common_neighbors("A", min_shared=2)
        candidates = [c for c, _, _ in result]
        assert "B" not in candidates

    @pytest.mark.asyncio
    async def test_already_connected_excluded(self, storage):
        """B4: direct neighbors of source are excluded from suggestions."""
        await _add_edge(storage, "A", "B", "related")
        await _add_edge(storage, "A", "hub", "related")
        await _add_edge(storage, "B", "hub", "related")
        # B is a direct neighbor of A — should not be suggested
        result = await storage.common_neighbors("A", min_shared=1)
        candidates = [c for c, _, _ in result]
        assert "B" not in candidates

    @pytest.mark.asyncio
    async def test_source_degree_non_negative(self, storage):
        """B5: source_degree field is a non-negative integer."""
        await _add_edge(storage, "A", "hub", "related")
        await _add_edge(storage, "B", "hub", "related")
        result = await storage.common_neighbors("A", min_shared=1)
        for _, _, source_degree in result:
            assert isinstance(source_degree, int)
            assert source_degree >= 0

    @pytest.mark.asyncio
    async def test_shared_count_positive(self, storage):
        """B6: shared_count is a positive integer."""
        await _add_edge(storage, "A", "hub", "related")
        await _add_edge(storage, "B", "hub", "related")
        result = await storage.common_neighbors("A", min_shared=1)
        for _, shared_count, _ in result:
            assert isinstance(shared_count, int)
            assert shared_count > 0

    @pytest.mark.asyncio
    async def test_bidirectional_counts_as_neighbor(self, storage):
        """B7: both A->hub and hub->A count for neighborhood."""
        await _add_edge(storage, "hub", "A", "related")
        await _add_edge(storage, "hub", "B", "related")
        result = await storage.common_neighbors("A", min_shared=1)
        # A and B share hub as a common neighbor via outgoing edges from hub
        candidates = [c for c, _, _ in result]
        assert "B" in candidates

    @pytest.mark.asyncio
    async def test_limit_ten_results(self, storage):
        """B8: at most 10 candidates returned even if more exist."""
        hub = "hub"
        source = "source"
        await _add_edge(storage, source, hub, "related")
        for i in range(15):
            await _add_edge(storage, f"cand{i}", hub, "related")
        result = await storage.common_neighbors(source, min_shared=1)
        assert len(result) <= 10

    @pytest.mark.asyncio
    async def test_result_is_list_of_3tuples(self, storage):
        """B9: result is List[Tuple[str, int, int]]."""
        await _add_edge(storage, "A", "hub", "related")
        await _add_edge(storage, "B", "hub", "related")
        result = await storage.common_neighbors("A", min_shared=1)
        assert isinstance(result, list)
        for item in result:
            assert len(item) == 3
            cand, shared, degree = item
            assert isinstance(cand, str)
            assert isinstance(shared, int)
            assert isinstance(degree, int)

    @pytest.mark.asyncio
    async def test_empty_graph_returns_empty(self, storage):
        """B10: empty graph returns []."""
        result = await storage.common_neighbors("anything", min_shared=1)
        assert result == []


# ===========================================================================
# Section C: SemanticReasoner.abduce (15 tests)
# ===========================================================================


class TestAbduce:

    @pytest.mark.asyncio
    async def test_empty_graph_returns_empty(self, storage):
        """C1: empty graph -> []."""
        reasoner = SemanticReasoner(storage)
        result = await reasoner.abduce("obs", "causes", max_hops=1)
        assert result == []

    @pytest.mark.asyncio
    async def test_direct_incoming_causes_edge(self, storage):
        """C2: A->obs via causes -> [{antecedent: A, distance: 1}]."""
        await _add_edge(storage, "A", "obs", "causes")
        reasoner = SemanticReasoner(storage)
        result = await reasoner.abduce("obs", "causes", max_hops=1)
        assert len(result) == 1
        assert result[0]["antecedent"] == "A"
        assert result[0]["distance"] == 1

    @pytest.mark.asyncio
    async def test_no_incoming_edges_empty(self, storage):
        """C3: obs has only outgoing edges -> []."""
        await _add_edge(storage, "obs", "downstream", "causes")
        reasoner = SemanticReasoner(storage)
        result = await reasoner.abduce("obs", "causes", max_hops=1)
        assert result == []

    @pytest.mark.asyncio
    async def test_wrong_rel_type_empty(self, storage):
        """C4: A->obs via causes, querying fixes -> []."""
        await _add_edge(storage, "A", "obs", "causes")
        reasoner = SemanticReasoner(storage)
        result = await reasoner.abduce("obs", "fixes", max_hops=1)
        assert result == []

    @pytest.mark.asyncio
    async def test_max_hops_1_only_direct(self, storage):
        """C5: max_hops=1 doesn't return 2-hop antecedent."""
        await _add_edge(storage, "A", "B", "causes")
        await _add_edge(storage, "B", "obs", "causes")
        reasoner = SemanticReasoner(storage)
        result = await reasoner.abduce("obs", "causes", max_hops=1)
        antecedents = [r["antecedent"] for r in result]
        assert "A" not in antecedents
        assert "B" in antecedents

    @pytest.mark.asyncio
    async def test_max_hops_2_returns_two_levels(self, storage):
        """C6: A->B->obs with max_hops=2 returns B(dist 1) and A(dist 2)."""
        await _add_edge(storage, "A", "B", "causes")
        await _add_edge(storage, "B", "obs", "causes")
        reasoner = SemanticReasoner(storage)
        result = await reasoner.abduce("obs", "causes", max_hops=2)
        antecedents = {r["antecedent"]: r["distance"] for r in result}
        assert antecedents.get("B") == 1
        assert antecedents.get("A") == 2

    @pytest.mark.asyncio
    async def test_results_sorted_by_distance_ascending(self, storage):
        """C7: results sorted distance ascending (closest first)."""
        await _add_edge(storage, "A", "B", "causes")
        await _add_edge(storage, "B", "obs", "causes")
        reasoner = SemanticReasoner(storage)
        result = await reasoner.abduce("obs", "causes", max_hops=2)
        distances = [r["distance"] for r in result]
        assert distances == sorted(distances)

    @pytest.mark.asyncio
    async def test_multiple_antecedents_at_distance_1(self, storage):
        """C8: multiple direct antecedents all returned."""
        await _add_edge(storage, "A", "obs", "causes")
        await _add_edge(storage, "B", "obs", "causes")
        await _add_edge(storage, "C", "obs", "causes")
        reasoner = SemanticReasoner(storage)
        result = await reasoner.abduce("obs", "causes", max_hops=1)
        antecedents = {r["antecedent"] for r in result}
        assert antecedents == {"A", "B", "C"}

    @pytest.mark.asyncio
    async def test_distance_key_present(self, storage):
        """C9: every result dict has 'distance' key."""
        await _add_edge(storage, "A", "obs", "causes")
        reasoner = SemanticReasoner(storage)
        result = await reasoner.abduce("obs", "causes", max_hops=1)
        for r in result:
            assert "distance" in r

    @pytest.mark.asyncio
    async def test_antecedent_key_present(self, storage):
        """C10: every result dict has 'antecedent' key."""
        await _add_edge(storage, "A", "obs", "causes")
        reasoner = SemanticReasoner(storage)
        result = await reasoner.abduce("obs", "causes", max_hops=1)
        for r in result:
            assert "antecedent" in r

    @pytest.mark.asyncio
    async def test_max_hops_clamped_to_4(self, storage):
        """C11: max_hops > 4 clamped to 4 (no error raised)."""
        reasoner = SemanticReasoner(storage)
        result = await reasoner.abduce("obs", "causes", max_hops=10)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_max_hops_clamped_to_min_1(self, storage):
        """C12: max_hops=0 clamped to 1 (no error raised)."""
        await _add_edge(storage, "A", "obs", "causes")
        reasoner = SemanticReasoner(storage)
        result = await reasoner.abduce("obs", "causes", max_hops=0)
        assert isinstance(result, list)
        # clamped to 1, direct antecedent still found
        antecedents = [r["antecedent"] for r in result]
        assert "A" in antecedents

    @pytest.mark.asyncio
    async def test_works_for_supports_rel_type(self, storage):
        """C13: works with rel_type='supports'."""
        await _add_edge(storage, "A", "obs", "supports")
        reasoner = SemanticReasoner(storage)
        result = await reasoner.abduce("obs", "supports", max_hops=1)
        assert any(r["antecedent"] == "A" for r in result)

    @pytest.mark.asyncio
    async def test_works_for_related_rel_type(self, storage):
        """C14: works with rel_type='related'."""
        await _add_edge(storage, "A", "obs", "related")
        reasoner = SemanticReasoner(storage)
        result = await reasoner.abduce("obs", "related", max_hops=1)
        assert any(r["antecedent"] == "A" for r in result)

    @pytest.mark.asyncio
    async def test_returns_list_of_dicts_not_tuples(self, storage):
        """C15: return type is List[Dict], not List[Tuple]."""
        await _add_edge(storage, "A", "obs", "causes")
        reasoner = SemanticReasoner(storage)
        result = await reasoner.abduce("obs", "causes", max_hops=1)
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, dict), f"Expected dict, got {type(item)}"


# ===========================================================================
# Section D: Handler integration (20 tests)
# ===========================================================================
# We test handlers by building mock graph / reasoner objects so we don't need
# the full MCP server stack.  Handlers are loaded via importlib the same way.

def _make_graph_mock(find_connected_return=None, transitive_return=None, common_neighbors_return=None):
    """Build an async-capable mock GraphStorage."""
    mock = MagicMock()
    mock.find_connected = AsyncMock(return_value=find_connected_return or [])
    mock.transitive_closure = AsyncMock(return_value=transitive_return or [])
    mock.common_neighbors = AsyncMock(return_value=common_neighbors_return or [])
    mock.shortest_path = AsyncMock(return_value=None)
    return mock


def _parse_text(content_list) -> dict:
    """Parse the text from a handler return value."""
    assert len(content_list) == 1
    return json.loads(content_list[0].text)


# We import the handler functions we need to test.
# Because the handler imports from relative paths (mcp, config, etc.) we need
# to test them via direct mock-patching at the module level.

import importlib.util as _ilu
import types as _types_mod

def _load_handler_module():
    """
    Load graph handler via importlib, stub out unavailable imports.
    """
    # Stub 'mcp' with a minimal types shim
    mcp_stub = _types_mod.ModuleType("mcp")
    mcp_types_stub = _types_mod.ModuleType("mcp.types")

    class _TextContent:
        def __init__(self, type: str, text: str):
            self.type = type
            self.text = text

    mcp_types_stub.TextContent = _TextContent
    mcp_stub.types = mcp_types_stub
    sys.modules.setdefault("mcp", mcp_stub)
    sys.modules.setdefault("mcp.types", mcp_types_stub)

    # Stub the storage + config imports used inside the handler
    config_stub = _types_mod.ModuleType("mcp_memory_service.config")
    config_stub.SQLITE_VEC_PATH = "/tmp/test.db"
    config_stub.STORAGE_BACKEND = "sqlite_vec"
    sys.modules["mcp_memory_service.config"] = config_stub

    graph_storage_stub = _types_mod.ModuleType("mcp_memory_service.storage.graph")
    graph_storage_stub.GraphStorage = GraphStorage
    sys.modules["mcp_memory_service.storage.graph"] = graph_storage_stub

    inference_stub = _types_mod.ModuleType("mcp_memory_service.reasoning.inference")
    inference_stub.SemanticReasoner = SemanticReasoner
    sys.modules["mcp_memory_service.reasoning.inference"] = inference_stub

    # Load actual handler file by executing it in a fresh module namespace
    handler_mod = _types_mod.ModuleType("_handler_under_test")
    handler_mod.__package__ = "mcp_memory_service.server.handlers"
    handler_src = _handler_path.read_text()

    # Replace relative imports with already-stubbed absolute ones
    handler_src = handler_src.replace(
        "from mcp import types",
        "from mcp import types",
    )
    handler_src = handler_src.replace(
        "from ...storage.graph import GraphStorage",
        "from mcp_memory_service.storage.graph import GraphStorage",
    )
    handler_src = handler_src.replace(
        "from ...config import SQLITE_VEC_PATH, STORAGE_BACKEND",
        "from mcp_memory_service.config import SQLITE_VEC_PATH, STORAGE_BACKEND",
    )
    handler_src = handler_src.replace(
        "from ...reasoning.inference import SemanticReasoner",
        "from mcp_memory_service.reasoning.inference import SemanticReasoner",
    )
    exec(compile(handler_src, str(_handler_path), "exec"), handler_mod.__dict__)
    return handler_mod


_handler = _load_handler_module()
handle_abduce = _handler.handle_abduce
handle_infer = _handler.handle_infer
handle_suggest = _handler.handle_suggest
handle_memory_graph = _handler.handle_memory_graph


class TestHandlerIntegration:

    # --- handle_abduce ---

    @pytest.mark.asyncio
    async def test_abduce_missing_hash_returns_error(self):
        """D1: missing hash -> error JSON with antecedents=[]."""
        with patch.object(_handler, "get_graph_storage", new=AsyncMock(return_value=_make_graph_mock())):
            result = await handle_abduce({})
        data = _parse_text(result)
        assert data["success"] is False
        assert data["antecedents"] == []
        assert "hash" in data["error"].lower() or "missing" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_abduce_graph_none_returns_failure(self):
        """D2: graph=None -> success=False."""
        with patch.object(_handler, "get_graph_storage", new=AsyncMock(return_value=None)):
            result = await handle_abduce({"hash": "obs"})
        data = _parse_text(result)
        assert data["success"] is False
        assert data["antecedents"] == []

    @pytest.mark.asyncio
    async def test_abduce_success_true(self):
        """D3: valid call returns success=True."""
        mock_gs = _make_graph_mock(find_connected_return=[("A", 1)])
        with patch.object(_handler, "get_graph_storage", new=AsyncMock(return_value=mock_gs)):
            result = await handle_abduce({"hash": "obs", "rel_type": "causes"})
        data = _parse_text(result)
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_abduce_count_equals_len_antecedents(self):
        """D4: count == len(antecedents)."""
        mock_gs = _make_graph_mock(find_connected_return=[("A", 1), ("B", 2)])
        with patch.object(_handler, "get_graph_storage", new=AsyncMock(return_value=mock_gs)):
            result = await handle_abduce({"hash": "obs", "rel_type": "causes", "max_hops": 2})
        data = _parse_text(result)
        assert data["count"] == len(data["antecedents"])

    @pytest.mark.asyncio
    async def test_abduce_empty_result_count_zero(self):
        """D5: empty antecedents -> count=0."""
        mock_gs = _make_graph_mock(find_connected_return=[])
        with patch.object(_handler, "get_graph_storage", new=AsyncMock(return_value=mock_gs)):
            result = await handle_abduce({"hash": "obs"})
        data = _parse_text(result)
        assert data["count"] == 0

    @pytest.mark.asyncio
    async def test_abduce_default_rel_type_is_causes(self):
        """D6: default rel_type is 'causes'."""
        mock_gs = _make_graph_mock()
        with patch.object(_handler, "get_graph_storage", new=AsyncMock(return_value=mock_gs)):
            await handle_abduce({"hash": "obs"})
        # find_connected called with relationship_type="causes"
        call_kwargs = mock_gs.find_connected.call_args
        assert call_kwargs.kwargs.get("relationship_type") == "causes"

    @pytest.mark.asyncio
    async def test_abduce_max_hops_passed_through(self):
        """D7: max_hops parameter passed to reasoner."""
        mock_gs = _make_graph_mock()
        with patch.object(_handler, "get_graph_storage", new=AsyncMock(return_value=mock_gs)):
            await handle_abduce({"hash": "obs", "rel_type": "causes", "max_hops": 3})
        call_kwargs = mock_gs.find_connected.call_args
        # max_hops clamped to 3
        assert call_kwargs.kwargs.get("max_hops") == 3

    @pytest.mark.asyncio
    async def test_abduce_rel_type_passed_through(self):
        """D8: rel_type parameter passed through to graph call."""
        mock_gs = _make_graph_mock()
        with patch.object(_handler, "get_graph_storage", new=AsyncMock(return_value=mock_gs)):
            await handle_abduce({"hash": "obs", "rel_type": "supports"})
        call_kwargs = mock_gs.find_connected.call_args
        assert call_kwargs.kwargs.get("relationship_type") == "supports"

    @pytest.mark.asyncio
    async def test_handle_memory_graph_routes_abduce(self):
        """D9: handle_memory_graph with action='abduce' routes correctly."""
        mock_gs = _make_graph_mock(find_connected_return=[("A", 1)])
        with patch.object(_handler, "get_graph_storage", new=AsyncMock(return_value=mock_gs)):
            result = await handle_memory_graph(None, {"action": "abduce", "hash": "obs"})
        data = _parse_text(result)
        assert data["success"] is True
        assert "antecedents" in data

    @pytest.mark.asyncio
    async def test_handle_memory_graph_abduce_missing_hash(self):
        """D10: action='abduce' without hash returns error."""
        mock_gs = _make_graph_mock()
        with patch.object(_handler, "get_graph_storage", new=AsyncMock(return_value=mock_gs)):
            result = await handle_memory_graph(None, {"action": "abduce"})
        data = _parse_text(result)
        assert data["success"] is False

    # --- handle_infer ---

    @pytest.mark.asyncio
    async def test_infer_count_equals_len_inferred(self):
        """D11: count == len(inferred)."""
        tc_return = [("A", "C", 2), ("B", "D", 2)]
        mock_gs = _make_graph_mock(transitive_return=tc_return)
        with patch.object(_handler, "get_graph_storage", new=AsyncMock(return_value=mock_gs)):
            result = await handle_infer({"rel_type": "causes", "max_hops": 2})
        data = _parse_text(result)
        assert data["count"] == len(data["inferred"])

    @pytest.mark.asyncio
    async def test_infer_items_have_required_keys(self):
        """D12: inferred items have source, target, distance keys."""
        mock_gs = _make_graph_mock(transitive_return=[("A", "C", 2)])
        with patch.object(_handler, "get_graph_storage", new=AsyncMock(return_value=mock_gs)):
            result = await handle_infer({"rel_type": "causes"})
        data = _parse_text(result)
        for item in data["inferred"]:
            assert "source" in item
            assert "target" in item
            assert "distance" in item

    @pytest.mark.asyncio
    async def test_infer_max_hops_passed(self):
        """D13: max_hops=3 is passed to transitive_closure."""
        mock_gs = _make_graph_mock(transitive_return=[])
        with patch.object(_handler, "get_graph_storage", new=AsyncMock(return_value=mock_gs)):
            await handle_infer({"rel_type": "causes", "max_hops": 3})
        mock_gs.transitive_closure.assert_called_once_with("causes", 3)

    @pytest.mark.asyncio
    async def test_infer_default_rel_type_related(self):
        """D14: default rel_type for infer is 'related'."""
        mock_gs = _make_graph_mock(transitive_return=[])
        with patch.object(_handler, "get_graph_storage", new=AsyncMock(return_value=mock_gs)):
            await handle_infer({})
        call_args = mock_gs.transitive_closure.call_args
        assert call_args[0][0] == "related"

    # --- handle_suggest ---

    @pytest.mark.asyncio
    async def test_suggest_count_equals_len_suggestions(self):
        """D15: count == len(suggestions)."""
        cn_return = [("B", 2, 3)]
        mock_gs = _make_graph_mock(common_neighbors_return=cn_return)
        with patch.object(_handler, "get_graph_storage", new=AsyncMock(return_value=mock_gs)):
            result = await handle_suggest({"hash": "A"})
        data = _parse_text(result)
        assert data["count"] == len(data["suggestions"])

    @pytest.mark.asyncio
    async def test_suggest_items_have_required_keys(self):
        """D16: suggestion items have target, type, confidence keys."""
        cn_return = [("B", 2, 3)]
        mock_gs = _make_graph_mock(common_neighbors_return=cn_return)
        with patch.object(_handler, "get_graph_storage", new=AsyncMock(return_value=mock_gs)):
            result = await handle_suggest({"hash": "A"})
        data = _parse_text(result)
        for item in data["suggestions"]:
            assert "target" in item
            assert "type" in item
            assert "confidence" in item

    @pytest.mark.asyncio
    async def test_suggest_missing_hash_returns_error(self):
        """D17: missing hash -> error JSON."""
        mock_gs = _make_graph_mock()
        with patch.object(_handler, "get_graph_storage", new=AsyncMock(return_value=mock_gs)):
            result = await handle_suggest({})
        data = _parse_text(result)
        assert data["success"] is False
        assert "suggestions" in data

    @pytest.mark.asyncio
    async def test_abduce_in_valid_actions(self):
        """D18: 'abduce' is in the valid_actions list."""
        import inspect
        src = inspect.getsource(handle_memory_graph)
        assert '"abduce"' in src or "'abduce'" in src

    @pytest.mark.asyncio
    async def test_invalid_action_error_mentions_abduce(self):
        """D19: invalid action error message lists abduce among valid options."""
        result = await handle_memory_graph(None, {"action": "nonexistent_action"})
        text = result[0].text
        assert "abduce" in text

    @pytest.mark.asyncio
    async def test_abduce_two_hop_mock_both_distances(self):
        """D20: 2-hop mock returns antecedents at distance 1 and 2."""
        mock_gs = _make_graph_mock(find_connected_return=[("B", 1), ("A", 2)])
        with patch.object(_handler, "get_graph_storage", new=AsyncMock(return_value=mock_gs)):
            result = await handle_abduce({"hash": "obs", "rel_type": "causes", "max_hops": 2})
        data = _parse_text(result)
        distances = {item["distance"] for item in data["antecedents"]}
        assert 1 in distances
        assert 2 in distances
