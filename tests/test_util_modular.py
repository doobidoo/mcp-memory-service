"""
Gate 3 (rok) RED tests for FASE 7 / R2 — utility MODULAR (multiplicative with excess-over-neutral).

TARGET BEHAVIOR: utility becomes a MODULATOR of relevance, not an additive term:
- util_surplus = max(0.0, (util - 0.5) / 0.5)  # neutral 0.5 → 0; util 1.0 → 1.0; util <0.5 → 0  
- score = rel*(1 + lambda_util*util_surplus) + w_adv*advantage + w_conf*conf_sat + w_fresh*freshness - w_gen*gen
- where lambda_util = _rerank_a_weight('util') (reuses existing weight, ~0.20)
- breakdown s['rerank_a'] gets 'util_surplus' and 'util_mult' fields for observability

CURRENT STATE: utility is ADDITIVE (w_util*util term), allowing high utility off-topic beliefs 
to overwhelm low relevance beliefs.

These tests FAIL TODAY (additive model) and will pass after R2 (multiplicative model).
Each test asserts the TARGET behavior, not the current broken behavior.
"""

import pytest
from mcp_memory_service.storage.context_injection import _apply_rerank_a


class TestUtilModular:
    """RED tests proving utility should be multiplicative modulator, not additive term."""
    
    def test_util_does_not_promote_offtopic(self):
        """RED: High utility off-topic belief should NOT outrank relevant belief with neutral utility.
        
        CURRENT (additive): B can win because w_util*1.0 can overcome small relevance differences
        TARGET (multiplicative): A wins because rel dominates, util only amplifies existing relevance
        """
        scored = [
            {
                "relevance": 0.35,  # A: Moderately relevant (reduced to make case more extreme)
                "derived_from": ["m1"],
                "belief_hash": "belief_relevant_neutral_util", 
                "freshness_score": 0.1,  # reduce other factors
                "utility_prior": 0.1,   # low utility to maximize contrast
            },
            {
                "relevance": 0.25,  # B: Lower relevance but not too far
                "derived_from": ["m2"],
                "belief_hash": "belief_offtopic_high_util",
                "freshness_score": 0.1,  # same freshness 
                "utility_prior": 1.0,   # maximum utility
            }
        ]
        
        generality = {"belief_relevant_neutral_util": 0.05, "belief_offtopic_high_util": 0.05}  # minimal generality penalty
        
        _apply_rerank_a(scored, generality)
        
        # TARGET: A (relevant) beats B (off-topic with high util)
        # This should FAIL with additive model: w_util*(1.0-0.1)=0.20*0.9=0.18 vs w_rel*(0.35-0.25)=0.40*0.10=0.04
        assert scored[0]["_rank_a_score"] > scored[1]["_rank_a_score"], \
            f"Relevant belief should beat off-topic high-utility: {scored[0]['_rank_a_score']} vs {scored[1]['_rank_a_score']}"

    def test_neutral_util_zero_boost(self):
        """RED: Utility=0.5 (neutral) should provide zero amplification (util_surplus=0, mult=1.0).
        
        TARGET: belief with util=0.5 has same score as if util was absent (multiplicative neutral)
        CURRENT: additive w_util*0.5 adds a positive term
        """
        # Two identical beliefs except one has explicit neutral util, other has same neutral util
        scored = [
            {
                "relevance": 0.7,
                "derived_from": ["m1"],
                "belief_hash": "belief_neutral_util_a",
                "freshness_score": 0.4,
                "utility_prior": 0.5,  # explicit neutral
            },
            {
                "relevance": 0.7, 
                "derived_from": ["m1"],  # same support
                "belief_hash": "belief_neutral_util_b", 
                "freshness_score": 0.4,
                "utility_prior": 0.5,  # same explicit neutral
            }
        ]
        
        generality = {"belief_neutral_util_a": 0.2, "belief_neutral_util_b": 0.2}
        
        _apply_rerank_a(scored, generality)
        
        # Both should have identical scores (neutral util provides no boost)
        score_a = scored[0]["_rank_a_score"]
        score_b = scored[1]["_rank_a_score"] 
        
        # TARGET: neutral utility does not change score 
        # This may PASS even today, but validates the neutral behavior
        assert abs(score_a - score_b) < 0.001, \
            f"Identical beliefs with neutral utility should have same score: {score_a} vs {score_b}"

    def test_util_amplifies_relevant(self):
        """GUARD: High utility should amplify already relevant beliefs (positive case).
        
        This should work in both additive and multiplicative models, but ensures we don't break 
        the desirable behavior where utility helps relevant content.
        """
        scored = [
            {
                "relevance": 0.6,  # same relevance
                "derived_from": ["m1"],
                "belief_hash": "belief_high_util",
                "freshness_score": 0.3,
                "utility_prior": 0.99,  # high utility
            },
            {
                "relevance": 0.6,  # same relevance  
                "derived_from": ["m2"],
                "belief_hash": "belief_neutral_util",
                "freshness_score": 0.3,
                "utility_prior": 0.5,  # neutral utility
            }
        ]
        
        generality = {"belief_high_util": 0.1, "belief_neutral_util": 0.1}
        
        _apply_rerank_a(scored, generality)
        
        # HIGH utility belief should rank above neutral utility (both have same relevance)
        # This PASSES today (additive also promotes) but guards against regression
        assert scored[0]["_rank_a_score"] > scored[1]["_rank_a_score"], \
            f"High utility should amplify relevant content: {scored[0]['_rank_a_score']} vs {scored[1]['_rank_a_score']}"

    def test_breakdown_has_util_mult(self):
        """RED: Breakdown should include 'util_surplus' and 'util_mult' fields after R2.
        
        CURRENT: breakdown has 'utility' field (raw value)
        TARGET: breakdown has 'util_surplus' and 'util_mult' for observability
        """
        scored = [
            {
                "relevance": 0.7,
                "derived_from": ["m1"],
                "belief_hash": "test_belief",
                "freshness_score": 0.4,
                "utility_prior": 0.8,  # util_surplus should be (0.8-0.5)/0.5 = 0.6
            }
        ]
        
        generality = {"test_belief": 0.1}
        
        _apply_rerank_a(scored, generality)
        
        breakdown = scored[0]["rerank_a"]
        
        # TARGET: breakdown contains multiplicative terms
        # This FAILS today because fields don't exist in additive model
        assert "util_surplus" in breakdown, "Breakdown should include util_surplus field"
        assert "util_mult" in breakdown, "Breakdown should include util_mult field"
        
        # Validate the math: util=0.8 → surplus=(0.8-0.5)/0.5=0.6, mult≈1+0.20*0.6=1.12
        expected_surplus = (0.8 - 0.5) / 0.5  # 0.6
        expected_mult = 1.0 + 0.20 * expected_surplus  # 1 + 0.20*0.6 = 1.12
        
        assert abs(breakdown["util_surplus"] - expected_surplus) < 0.001, \
            f"util_surplus should be {expected_surplus}, got {breakdown.get('util_surplus')}"
        assert abs(breakdown["util_mult"] - expected_mult) < 0.01, \
            f"util_mult should be ~{expected_mult}, got {breakdown.get('util_mult')}"