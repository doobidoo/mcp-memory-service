"""
RED tests for GRAPH PROXIMITY injection signal (learning-loop).

Feature: the context injection should boost beliefs that share entities with
the query task, creating better thematic coherence. A belief about "MIR deployment"
should rank higher when the task is "troubleshoot MIR API issues" because they
share the entity "MIR", even if their term overlap is similar to a generic belief.

Requirements from RFC:
- R1: New signal in _apply_rerank_a weighted sum
- R2: Additive term (never negative penalty) 
- R3: Zero regression when flag off (MCP_INJECT_GRAPH_PROXIMITY)
- R4: Breakdown dict gains 'graph_proximity' key without losing existing keys
- R5: Neutral behavior (== flag-off) when no entities resolved
- R7: Only active when rerank-A is also enabled (guarded)

These tests will FAIL until the feature is implemented.
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

try:
    import sqlite_vec  # noqa: F401
    SQLITE_VEC_AVAILABLE = True
except ImportError:
    SQLITE_VEC_AVAILABLE = False

from mcp_memory_service.storage.context_injection import _apply_rerank_a, memory_context

if SQLITE_VEC_AVAILABLE:
    from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage

pytestmark = pytest.mark.skipif(not SQLITE_VEC_AVAILABLE, reason="sqlite-vec not available")

_TASK = "troubleshoot MIR API connectivity issues"


@pytest.fixture
async def real_storage():
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test.db")
    storage = SqliteVecMemoryStorage(db_path)
    await storage.initialize()
    yield storage
    await storage.close()
    try:
        os.remove(db_path)
        os.rmdir(temp_dir)
    except (OSError, FileNotFoundError):
        pass


async def _insert_belief(storage, bhash, content, confidence, derived_from=None):
    now = datetime.now(timezone.utc).isoformat()
    storage.conn.execute(
        """
        INSERT INTO beliefs (belief_hash, content, confidence, status, created_at, updated_at, derived_from)
        VALUES (?, ?, ?, 'active', ?, ?, ?)
        """,
        (bhash, content, confidence, now, now, json.dumps(derived_from or [])),
    )
    storage.conn.commit()


@pytest.fixture
async def storage_with_themed_beliefs(real_storage):
    """Two beliefs with similar relevance: one MIR-themed, one generic."""
    await _insert_belief(
        real_storage, "b_mir_themed",
        "MIR deployment uses PostgreSQL database connections for data access",
        0.75, derived_from=["mir1", "mir2"]
    )
    await _insert_belief(
        real_storage, "b_generic", 
        "API connectivity troubleshooting requires checking network configuration",
        0.77, derived_from=["gen1", "gen2", "gen3"]
    )
    return real_storage


@pytest.mark.asyncio
async def test_zero_regression_when_graph_proximity_flag_off(storage_with_themed_beliefs):
    """CA1/R3: With MCP_INJECT_GRAPH_PROXIMITY off, _apply_rerank_a produces a
    _rank_a_score byte-identical to omitting the graph term entirely. Tested on the
    PURE function (deterministic) rather than memory_context end-to-end, because
    memory_context's freshness term depends on datetime.now() and is not stable
    across two calls — comparing it to itself would be a flaky non-test.

    Baseline: call _apply_rerank_a with the current 2-arg signature. Then (once the
    feature lands) call it with graph_proximity omitted/empty and assert identical
    scores. Until then this pins the invariant the implementation MUST preserve.
    """
    os.environ.pop("MCP_INJECT_GRAPH_PROXIMITY", None)

    # A fixed, deterministic scored pool (no datetime, no DB).
    def _pool():
        return [
            {"belief_hash": "b_themed", "relevance": 0.46, "confidence": 0.68,
             "freshness_score": 0.03, "derived_from": ["m1", "m2"], "utility_prior": 0.5},
            {"belief_hash": "b_generic", "relevance": 0.45, "confidence": 0.98,
             "freshness_score": 0.03, "derived_from": ["g1", "g2", "g3"], "utility_prior": 0.98},
        ]

    gen = {"b_themed": 0.1, "b_generic": 0.9}

    baseline = _pool()
    _apply_rerank_a(baseline, gen)
    baseline_scores = {s["belief_hash"]: s["_rank_a_score"] for s in baseline}

    # With the flag off, the graph term must not exist. Once _apply_rerank_a accepts an
    # optional graph_proximity, omitting it (or passing {}) MUST reproduce baseline.
    with_off = _pool()
    try:
        _apply_rerank_a(with_off, gen, {})  # 3-arg form, empty proximity
    except TypeError:
        # Feature not implemented yet (RED): the 2-arg form is the baseline itself.
        _apply_rerank_a(with_off, gen)
    off_scores = {s["belief_hash"]: s["_rank_a_score"] for s in with_off}

    for bh, score in off_scores.items():
        assert score == baseline_scores[bh], (
            f"Score regression with graph flag off: {bh} {baseline_scores[bh]} -> {score}"
        )
    for s in with_off:
        assert "graph_proximity" not in s.get("rerank_a", {}), (
            "rerank_a breakdown must not contain graph_proximity when flag is off"
        )

    # Stronger anchor (addresses G5 finding): even with a NON-EMPTY graph_proximity, the
    # default weight (MCP_INJECT_RERANK_A_W_GRAPH unset -> 0.0) must neutralize the term,
    # so the score stays byte-identical to the baseline. This proves the R4 invariant is
    # the DEFAULT WEIGHT, not merely an empty dict.
    os.environ.pop("MCP_INJECT_RERANK_A_W_GRAPH", None)
    with_prox_zero_weight = _pool()
    _apply_rerank_a(with_prox_zero_weight, gen, {"b_themed": 1.0, "b_generic": 1.0})
    for s in with_prox_zero_weight:
        assert s["_rank_a_score"] == baseline_scores[s["belief_hash"]], (
            f"Default graph weight must be 0 (no effect): {s['belief_hash']} "
            f"{baseline_scores[s['belief_hash']]} -> {s['_rank_a_score']}"
        )


@pytest.mark.asyncio
async def test_graph_proximity_boosts_themed_belief_when_on():
    """R2/R4: With flag ON and graph_proximity provided, belief with proximity>0 
    gets higher score than same belief with proximity=0. Tests additive behavior 
    and breakdown dict gains 'graph_proximity' key."""
    # This test will FAIL because the API doesn't exist yet
    os.environ["MCP_INJECT_GRAPH_PROXIMITY"] = "true"
    os.environ["MCP_INJECT_RERANK_A"] = "true"
    # The graph weight defaults to 0.0 (R4 invariant: off = no effect). To exercise the
    # additive term we must set the weight explicitly inside the test.
    os.environ["MCP_INJECT_RERANK_A_W_GRAPH"] = "0.5"
    try:
        # Mock scored items with identical base metrics
        scored_base = [
            {
                "belief_hash": "b_with_proximity",
                "relevance": 0.8,
                "freshness_score": 0.6,
                "derived_from": ["s1", "s2"]
            },
            {
                "belief_hash": "b_without_proximity", 
                "relevance": 0.8,
                "freshness_score": 0.6,
                "derived_from": ["s1", "s2"]
            }
        ]
        
        # Mock graph_proximity dict - this API doesn't exist yet, so will fail
        graph_proximity = {
            "b_with_proximity": 0.7,  # High proximity - shares entities with task
            "b_without_proximity": 0.0  # No shared entities
        }
        
        generality = {"b_with_proximity": 0.1, "b_without_proximity": 0.1}
        
        # This call will FAIL because _apply_rerank_a doesn't accept 3rd parameter yet
        _apply_rerank_a(scored_base, generality, graph_proximity)
        
        # Verify the proximity signal boosted the score (additive)
        with_score = scored_base[0]["_rank_a_score"]
        without_score = scored_base[1]["_rank_a_score"]
        assert with_score > without_score, (
            f"Belief with graph_proximity=0.7 must score higher than proximity=0.0: "
            f"{with_score} vs {without_score}"
        )
        
        # Verify breakdown dict gains graph_proximity without losing existing keys
        breakdown_with = scored_base[0]["rerank_a"]
        breakdown_without = scored_base[1]["rerank_a"]
        
        # Must have graph_proximity key
        assert "graph_proximity" in breakdown_with, "rerank_a must contain graph_proximity"
        assert "graph_proximity" in breakdown_without, "rerank_a must contain graph_proximity"
        
        # Must keep existing keys
        expected_keys = {"relevance", "advantage", "conf_saturated", "freshness", 
                        "generality", "util_surplus", "util_mult", "score", "graph_proximity"}
        assert set(breakdown_with.keys()) == expected_keys, (
            f"Breakdown missing keys: {expected_keys - set(breakdown_with.keys())}"
        )
        
        # Proximity values should match input
        assert breakdown_with["graph_proximity"] == 0.7
        assert breakdown_without["graph_proximity"] == 0.0
        
    finally:
        os.environ.pop("MCP_INJECT_GRAPH_PROXIMITY", None)
        os.environ.pop("MCP_INJECT_RERANK_A", None)
        os.environ.pop("MCP_INJECT_RERANK_A_W_GRAPH", None)


@pytest.mark.asyncio
async def test_themed_belief_wins_with_graph_proximity_on(storage_with_themed_beliefs):
    """CA2: Thematic scenario - belief sharing entities with query should rank higher
    when graph proximity is enabled, lower when disabled."""
    # Mock entity extraction to return entities for the task
    mock_entities = ["MIR", "API", "connectivity"]
    
    # Create a mock graph storage object
    mock_graph = type('MockGraph', (), {
        'find_memories_by_entity': None
    })()
    
    async def mock_find_side_effect(entity, limit=20):
        if entity == "MIR":
            return ["mir1", "mir2"]  # Memory hashes that match b_mir_themed's derived_from
        return []
    
    mock_graph.find_memories_by_entity = mock_find_side_effect

    # Mock that MIR-themed belief shares "MIR" entity, generic doesn't.
    # Patch get_graph_storage (the real resolution path, Issue #219) — NOT storage.graph,
    # which no storage class ever sets. This exercises the production code path.
    async def _fake_get_graph_storage():
        return mock_graph

    with patch('mcp_memory_service.reasoning.entities.EntityExtractor.extract_entities') as mock_extract, \
         patch('mcp_memory_service.server.handlers.graph.get_graph_storage', _fake_get_graph_storage):

        # Mock entity extraction from task
        mock_extract.return_value = [type('Entity', (), {'name': e, 'type': 'CONCEPT'}) for e in mock_entities]

        # Test with graph proximity ON - MIR themed should win
        os.environ["MCP_INJECT_GRAPH_PROXIMITY"] = "true"
        os.environ["MCP_INJECT_RERANK_A"] = "true"
        os.environ["MCP_INJECT_RERANK_A_W_GRAPH"] = "0.5"
        
        try:
            result_on = await memory_context(storage_with_themed_beliefs, task=_TASK, limit=10)
            hashes_on = [b["belief_hash"] for b in result_on["beliefs"]]
            
            # Test with graph proximity OFF - generic might win due to higher confidence
            os.environ["MCP_INJECT_GRAPH_PROXIMITY"] = "false"
            _result_off = await memory_context(storage_with_themed_beliefs, task=_TASK, limit=10)
            
            # With proximity ON, themed belief should rank higher
            mir_pos_on = hashes_on.index("b_mir_themed")
            gen_pos_on = hashes_on.index("b_generic")
            assert mir_pos_on < gen_pos_on, (
                f"With graph proximity ON, MIR-themed belief should rank higher: "
                f"positions {mir_pos_on} vs {gen_pos_on} in {hashes_on}"
            )
            
        finally:
            os.environ.pop("MCP_INJECT_GRAPH_PROXIMITY", None)
            os.environ.pop("MCP_INJECT_RERANK_A", None)
            os.environ.pop("MCP_INJECT_RERANK_A_W_GRAPH", None)


@pytest.mark.asyncio 
async def test_neutral_when_no_entities_resolved():
    """R5: When no entities are resolved (empty dict or all proximity=0), 
    no belief is penalized - scores should equal flag-off behavior."""
    # Mock scored items
    scored_items = [
        {"belief_hash": "b1", "relevance": 0.8, "freshness_score": 0.6, "derived_from": ["s1"]},
        {"belief_hash": "b2", "relevance": 0.6, "freshness_score": 0.8, "derived_from": ["s2"]}
    ]
    generality = {"b1": 0.1, "b2": 0.2}
    
    # Test with empty proximity dict (no entities resolved)
    scored_with_empty = [item.copy() for item in scored_items]
    scored_without = [item.copy() for item in scored_items] 
    
    # This will FAIL - function signature doesn't exist yet
    _apply_rerank_a(scored_with_empty, generality, {})  # Empty proximity dict
    _apply_rerank_a(scored_without, generality)  # No proximity parameter
    
    # Scores should be identical when no entities resolved
    for i in range(len(scored_items)):
        assert scored_with_empty[i]["_rank_a_score"] == scored_without[i]["_rank_a_score"], (
            f"Scores diverged with empty proximity dict: item {i}"
        )


@pytest.mark.asyncio
async def test_graph_proximity_only_active_when_rerank_a_enabled():
    """R7: Graph proximity signal only acts when rerank-A guard is enabled.
    If rerank-A is off, graph proximity doesn't alter fallback sorting."""
    # This test will FAIL because MCP_INJECT_GRAPH_PROXIMITY flag doesn't exist yet
    os.environ["MCP_INJECT_GRAPH_PROXIMITY"] = "true"  
    os.environ["MCP_INJECT_RERANK_A"] = "false"  # Rerank A is OFF
    
    try:
        # Mock storage with beliefs
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test.db")
        storage = SqliteVecMemoryStorage(db_path)
        await storage.initialize()
        
        await _insert_belief(storage, "b1", "test belief one", 0.9, ["s1"])
        await _insert_belief(storage, "b2", "test belief two", 0.7, ["s2"]) 
        
        with patch('mcp_memory_service.reasoning.entities.EntityExtractor.extract_entities') as mock_extract:
            # Mock entities detected
            mock_extract.return_value = [type('Entity', (), {'name': 'test', 'type': 'CONCEPT'})]
            
            # Should fall back to confidence ordering since rerank-A is off
            result = await memory_context(storage, task="test query", limit=10)
            beliefs = result["beliefs"]
            
            # Verify rerank_a breakdown is NOT present (rerank-A is off)
            for belief in beliefs:
                assert "rerank_a" not in belief, (
                    "rerank_a breakdown should not exist when MCP_INJECT_RERANK_A=false"
                )
            
            # Should order by confidence (b1=0.9 before b2=0.7)
            hashes = [b["belief_hash"] for b in beliefs]
            assert hashes.index("b1") < hashes.index("b2"), (
                "Should fall back to confidence ordering when rerank-A disabled"
            )
        
        await storage.close()
        
    finally:
        os.environ.pop("MCP_INJECT_GRAPH_PROXIMITY", None)
        os.environ.pop("MCP_INJECT_RERANK_A", None)


@pytest.mark.asyncio
async def test_graph_proximity_flag_function_exists():
    """Test that _get_graph_proximity_enabled() function follows same pattern as rerank_a."""
    # This will FAIL because the function doesn't exist yet
    from mcp_memory_service.storage.context_injection import (
        _get_graph_proximity_enabled,
    )
    
    # Test flag off
    os.environ.pop("MCP_INJECT_GRAPH_PROXIMITY", None)
    assert not _get_graph_proximity_enabled(), "Should be False when unset"
    
    # Test flag on values
    for val in ["true", "TRUE", "1", "yes", "YES"]:
        os.environ["MCP_INJECT_GRAPH_PROXIMITY"] = val
        assert _get_graph_proximity_enabled(), f"Should be True for '{val}'"
        
    # Test flag off values  
    for val in ["false", "FALSE", "0", "no", "NO", "anything"]:
        os.environ["MCP_INJECT_GRAPH_PROXIMITY"] = val
        assert not _get_graph_proximity_enabled(), f"Should be False for '{val}'"
        
    os.environ.pop("MCP_INJECT_GRAPH_PROXIMITY", None)