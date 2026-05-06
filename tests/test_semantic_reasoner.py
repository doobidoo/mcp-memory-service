"""
Unit tests for SemanticReasoner

Tests the lightweight reasoning engine capabilities.
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
import importlib.util

# Load modules directly
graph_path = Path(__file__).parent.parent / "src" / "mcp_memory_service" / "storage" / "graph.py"
spec = importlib.util.spec_from_file_location("graph", graph_path)
graph_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(graph_module)

reasoning_path = Path(__file__).parent.parent / "src" / "mcp_memory_service" / "reasoning" / "inference.py"
spec2 = importlib.util.spec_from_file_location("inference", reasoning_path)
reasoning_module = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(reasoning_module)

GraphStorage = graph_module.GraphStorage
SemanticReasoner = reasoning_module.SemanticReasoner


@pytest.fixture
async def setup_graph():
    """Create graph storage with sample data"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
        db_path = tmp.name

    storage = GraphStorage(db_path)

    # Initialize schema
    conn = await storage._get_connection()
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

    # Add sample data for testing
    # Error1 is caused by Decision1, fixed by Decision2
    await storage.store_association("decision1", "error1", 0.9, ["causal"], relationship_type="causes")
    await storage.store_association("decision2", "error1", 0.9, ["remediation"], relationship_type="fixes")

    # Learning1 contradicts Learning2
    await storage.store_association("learning1", "learning2", 0.8, ["semantic"], relationship_type="contradicts")

    # Decision3 supports Decision4
    await storage.store_association("decision3", "decision4", 0.85, ["semantic"], relationship_type="supports")

    yield storage

    # Cleanup
    if storage._connection:
        storage._connection.close()
    os.unlink(db_path)


class TestSemanticReasonerValidation:
    """Tests for SemanticReasoner input validation"""

    def test_raises_on_none_graph_storage(self):
        """Should raise ValueError when graph_storage is None"""
        with pytest.raises(ValueError, match="graph_storage cannot be None"):
            SemanticReasoner(None)

    def test_raises_on_missing_find_connected_method(self):
        """Should raise ValueError when graph_storage lacks find_connected method"""
        class InvalidStorage:
            def shortest_path(self):
                pass

        with pytest.raises(ValueError, match="graph_storage must have find_connected method"):
            SemanticReasoner(InvalidStorage())

    def test_raises_on_missing_shortest_path_method(self):
        """Should raise ValueError when graph_storage lacks shortest_path method"""
        class InvalidStorage:
            def find_connected(self):
                pass

        with pytest.raises(ValueError, match="graph_storage must have shortest_path method"):
            SemanticReasoner(InvalidStorage())

    def test_accepts_valid_graph_storage(self, setup_graph):
        """Should accept graph_storage with required methods"""
        storage = setup_graph
        reasoner = SemanticReasoner(storage)
        assert reasoner.graph is storage


class TestBurst43DetectContradictions:
    """Tests for Burst 4.3: detect_contradictions"""

    @pytest.mark.asyncio
    async def test_finds_contradicting_memories(self, setup_graph):
        """Should find memories with contradicts relationship"""
        storage = setup_graph
        reasoner = SemanticReasoner(storage)

        contradictions = await reasoner.detect_contradictions("learning1")
        assert "learning2" in contradictions

    @pytest.mark.asyncio
    async def test_empty_list_for_no_contradictions(self, setup_graph):
        """Should return empty list when no contradictions exist"""
        storage = setup_graph
        reasoner = SemanticReasoner(storage)

        contradictions = await reasoner.detect_contradictions("error1")
        assert contradictions == []


class TestBurst44FindFixes:
    """Tests for Burst 4.4: find_fixes"""

    @pytest.mark.asyncio
    async def test_finds_fixing_memories(self, setup_graph):
        """Should find memories that fix an error"""
        storage = setup_graph
        reasoner = SemanticReasoner(storage)

        fixes = await reasoner.find_fixes("error1")
        assert "decision2" in fixes

    @pytest.mark.asyncio
    async def test_empty_list_for_no_fixes(self, setup_graph):
        """Should return empty list when no fixes exist"""
        storage = setup_graph
        reasoner = SemanticReasoner(storage)

        fixes = await reasoner.find_fixes("decision1")
        assert fixes == []


class TestBurst45FindCauses:
    """Tests for Burst 4.5: find_causes"""

    @pytest.mark.asyncio
    async def test_finds_causing_memories(self, setup_graph):
        """Should find memories that caused an error"""
        storage = setup_graph
        reasoner = SemanticReasoner(storage)

        causes = await reasoner.find_causes("error1")
        assert "decision1" in causes

    @pytest.mark.asyncio
    async def test_empty_list_for_no_causes(self, setup_graph):
        """Should return empty list when no causes exist"""
        storage = setup_graph
        reasoner = SemanticReasoner(storage)

        causes = await reasoner.find_causes("learning1")
        assert causes == []


class TestBurst46AbstractToConcept:
    """Tests for Burst 4.6: abstract_to_concept"""

    @pytest.mark.asyncio
    async def test_returns_none_placeholder(self, setup_graph):
        """Should return None as placeholder (to be integrated later)"""
        storage = setup_graph
        reasoner = SemanticReasoner(storage)

        parent = await reasoner.abstract_to_concept("decision1")
        assert parent is None


class TestBurst47InferTransitive:
    """Tests for Burst 4.7: infer_transitive"""

    @pytest.mark.asyncio
    async def test_finds_transitive_chain(self, setup_graph):
        """Should find A→C when A→B→C exists but not A→C directly"""
        storage = setup_graph

        # Add chain: chain_a → chain_b → chain_c (via "causes")
        await storage.store_association("chain_a", "chain_b", 0.9, ["causal"], relationship_type="causes")
        await storage.store_association("chain_b", "chain_c", 0.9, ["causal"], relationship_type="causes")

        reasoner = SemanticReasoner(storage)
        inferred = await reasoner.infer_transitive("causes", max_hops=2)

        # Should find chain_a → chain_c at distance 2
        assert any(src == "chain_a" and tgt == "chain_c" and dist == 2
                   for src, tgt, dist in inferred)

    @pytest.mark.asyncio
    async def test_excludes_direct_edges(self, setup_graph):
        """Should NOT report pairs that already have a direct edge"""
        storage = setup_graph

        # decision1 → error1 is direct, should not appear in inferred
        reasoner = SemanticReasoner(storage)
        inferred = await reasoner.infer_transitive("causes", max_hops=3)

        assert not any(src == "decision1" and tgt == "error1"
                       for src, tgt, _ in inferred)

    @pytest.mark.asyncio
    async def test_empty_for_unknown_rel_type(self, setup_graph):
        """Should return empty for relationship type with no edges"""
        storage = setup_graph
        reasoner = SemanticReasoner(storage)

        inferred = await reasoner.infer_transitive("nonexistent_type", max_hops=2)
        assert inferred == []


class TestBurst48SuggestRelationships:
    """Tests for Burst 4.8: suggest_relationships"""

    @pytest.mark.asyncio
    async def test_suggests_based_on_shared_neighbors(self, setup_graph):
        """Should suggest relationships when memories share neighbors"""
        storage = setup_graph

        # Create a triangle-like structure:
        # node_a → hub1, node_a → hub2
        # node_b → hub1, node_b → hub2
        # node_a and node_b share 2 neighbors → should be suggested
        await storage.store_association("node_a", "hub1", 0.8, ["semantic"], relationship_type="related")
        await storage.store_association("node_a", "hub2", 0.8, ["semantic"], relationship_type="related")
        await storage.store_association("node_b", "hub1", 0.8, ["semantic"], relationship_type="related")
        await storage.store_association("node_b", "hub2", 0.8, ["semantic"], relationship_type="related")

        reasoner = SemanticReasoner(storage)
        suggestions = await reasoner.suggest_relationships("node_a")

        # node_b shares 2 neighbors with node_a
        assert any(s["target"] == "node_b" for s in suggestions)
        if suggestions:
            assert "confidence" in suggestions[0]
            assert "shared_neighbors" in suggestions[0]

    @pytest.mark.asyncio
    async def test_empty_for_isolated_node(self, setup_graph):
        """Should return empty for a node with no connections"""
        storage = setup_graph
        reasoner = SemanticReasoner(storage)

        suggestions = await reasoner.suggest_relationships("isolated_hash")
        assert suggestions == []


class TestSemanticReasonerIntegration:
    """Integration tests for complete reasoning workflows"""

    @pytest.mark.asyncio
    async def test_causal_chain_reasoning(self, setup_graph):
        """Should handle complete causal reasoning workflow"""
        storage = setup_graph
        reasoner = SemanticReasoner(storage)

        # Given an error, find what caused it and what fixed it
        causes = await reasoner.find_causes("error1")
        fixes = await reasoner.find_fixes("error1")

        assert "decision1" in causes
        assert "decision2" in fixes
        assert len(causes) == 1
        assert len(fixes) == 1
