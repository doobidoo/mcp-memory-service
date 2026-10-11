"""
RED tests for fechamento learning-loop A↔D — PARTE 1+2 (failing until FASE 5).

PARTE 1 — Laço A↔D: _apply_rerank_a must consume utility_prior from belief metadata.
Today the scored formula is: score = w_rel*rel + w_adv*adv + w_conf*conf + w_fresh*fresh - w_gen*gen.
MISSING: + w_util * util where util comes from beliefs.metadata.utility_prior (written by D1).

PARTE 2 — D2 refinement: belief_service._beta_utility must handle never-used as neutral,
asymmetric weights for neg vs pos signals, and clamp per-episode updates.

Both parts RED until the implementation is done.
"""

import json
import os
import tempfile
from datetime import datetime
from typing import Dict, List, Any

import pytest

try:
    import sqlite_vec  # noqa: F401
    SQLITE_VEC_AVAILABLE = True
except ImportError:
    SQLITE_VEC_AVAILABLE = False

if SQLITE_VEC_AVAILABLE:
    from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
    from mcp_memory_service.storage.context_injection import _apply_rerank_a
    from mcp_memory_service.consolidation.belief_service import BeliefService

pytestmark = pytest.mark.skipif(not SQLITE_VEC_AVAILABLE, reason="sqlite-vec not available")


@pytest.fixture
async def storage():
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test.db")
    storage = SqliteVecMemoryStorage(db_path)
    await storage.initialize()
    yield storage
    await storage.close()


async def _seed_belief_with_utility(storage, belief_hash: str, confidence: float, utility_prior: float):
    """Seed a belief with utility_prior in metadata."""
    metadata = {"utility_prior": utility_prior}
    storage.conn.execute(
        "INSERT INTO beliefs (belief_hash, content, confidence, status, created_at, updated_at, "
        "derived_from, metadata) VALUES (?, ?, ?, 'active', '2026-01-01', '2026-01-01', "
        "'[\"m1\", \"m2\"]', ?)",
        (belief_hash, f"content for {belief_hash}", confidence, json.dumps(metadata))
    )
    storage.conn.commit()


class TestFrente1LacoAD:
    """RED tests for PARTE 1 — utility_prior integration in _apply_rerank_a."""

    @pytest.mark.asyncio
    async def test_utility_prior_influences_ranking_order(self, storage):
        """RED até FASE 5: Beliefs with higher utility_prior rank above lower ones with same rel/adv/conf."""
        # RED até FASE 5
        
        # Setup: two beliefs with identical relevance/advantage/confidence but different utility_prior
        scored = [
            {
                "relevance": 0.8,
                "derived_from": ["m1", "m2"],  # support=2 for identical conf_sat
                "belief_hash": "belief_high_util",
                "freshness_score": 0.5,
                "utility_prior": 0.9,  # HIGH utility
            },
            {
                "relevance": 0.8,
                "derived_from": ["m3", "m4"],  # support=2 for identical conf_sat  
                "belief_hash": "belief_low_util",
                "freshness_score": 0.5,
                "utility_prior": 0.3,  # LOW utility
            }
        ]
        
        generality = {"belief_high_util": 0.1, "belief_low_util": 0.1}  # Same generality
        
        # Apply rerank_a (this will fail because utility_prior is not used yet)
        _apply_rerank_a(scored, generality)
        
        high_score = scored[0]["_rank_a_score"]
        low_score = scored[1]["_rank_a_score"]
        
        # EXPECTED: high utility belief should score higher
        # REALITY: will fail because utility_prior is not in the formula yet
        assert high_score > low_score, f"High utility belief should rank above low utility: {high_score} vs {low_score}"

    @pytest.mark.asyncio
    async def test_missing_utility_prior_uses_neutral(self, storage):
        """RED até FASE 5: Beliefs without utility_prior should use neutral 0.5, not be penalized."""
        # RED até FASE 5
        
        scored = [
            {
                "relevance": 0.7,
                "derived_from": ["m1"],
                "belief_hash": "belief_no_util",
                "freshness_score": 0.3,
                # NO utility_prior field - should default to neutral 0.5
            },
            {
                "relevance": 0.7,
                "derived_from": ["m2"], 
                "belief_hash": "belief_with_util",
                "freshness_score": 0.3,
                "utility_prior": 0.5,  # Explicit neutral
            }
        ]
        
        generality = {"belief_no_util": 0.2, "belief_with_util": 0.2}
        
        _apply_rerank_a(scored, generality)
        
        # EXPECTED: both should have same score (missing treated as neutral 0.5)
        # REALITY: will fail because utility_prior is not handled yet
        score_no_util = scored[0]["_rank_a_score"]
        score_with_util = scored[1]["_rank_a_score"]
        
        assert abs(score_no_util - score_with_util) < 0.001, f"Missing utility_prior should be neutral: {score_no_util} vs {score_with_util}"

    @pytest.mark.asyncio  
    async def test_rerank_a_breakdown_includes_utility(self, storage):
        """RED até FASE 5: The rerank_a breakdown must include 'utility' field."""
        # RED até FASE 5
        
        scored = [
            {
                "relevance": 0.6,
                "derived_from": ["m1"],
                "belief_hash": "test_belief", 
                "freshness_score": 0.4,
                "utility_prior": 0.8,
            }
        ]
        
        generality = {"test_belief": 0.15}
        
        _apply_rerank_a(scored, generality)
        
        breakdown = scored[0]["rerank_a"]
        
        # EXPECTED: breakdown should include 'utility' field
        # REALITY: will fail because 'utility' is not added to breakdown yet
        assert "utility" in breakdown, f"rerank_a breakdown missing utility field: {breakdown.keys()}"
        assert isinstance(breakdown["utility"], (int, float)), f"utility should be numeric: {breakdown['utility']}"

    @pytest.mark.asyncio
    async def test_utility_weight_configurable_via_env(self, storage):
        """RED até FASE 5: Weight should be configurable via MCP_INJECT_RERANK_A_W_UTIL."""
        # RED até FASE 5
        
        # Set custom utility weight
        os.environ["MCP_INJECT_RERANK_A_W_UTIL"] = "0.35"
        
        try:
            scored = [
                {
                    "relevance": 0.5,
                    "derived_from": ["m1"],
                    "belief_hash": "test_belief",
                    "freshness_score": 0.2, 
                    "utility_prior": 1.0,  # Max utility
                }
            ]
            
            generality = {"test_belief": 0.0}
            
            _apply_rerank_a(scored, generality)
            
            # EXPECTED: The utility component should reflect the custom weight
            # REALITY: will fail because MCP_INJECT_RERANK_A_W_UTIL is not implemented yet
            breakdown = scored[0]["rerank_a"]
            expected_utility_contribution = 0.35 * 1.0  # weight * utility_prior
            
            # This will fail because the whole utility system is not implemented
            assert "utility" in breakdown, "Utility field missing from breakdown"
            
        finally:
            os.environ.pop("MCP_INJECT_RERANK_A_W_UTIL", None)


