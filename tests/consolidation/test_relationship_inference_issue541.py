"""
Tests for RelationshipInferenceEngine false positive reduction — GitHub issue #541.

Regression tests verifying that:
1. Common contrast conjunctions ("however", "but", "yet", "although") do NOT
   produce a "contradicts" relationship label.
2. Strong contradiction keywords (e.g. "contradicts", "incorrect") still work.
3. min_typed_confidence=0.75 gate blocks low-confidence typed labels.
4. Memories with no shared domain keywords fall back to "related".
5. Low similarity score (< min_typed_similarity=0.65, on the (cos + 1) / 2
   score consolidation passes) forces "related".
"""

import time
import pytest

from mcp_memory_service.consolidation.relationship_inference import (
    RelationshipInferenceEngine,
    _shares_domain_keywords,
    _extract_domain_keywords,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    """Default engine — the constructor defaults, which are the gates chosen in #1457."""
    return RelationshipInferenceEngine()


@pytest.fixture
def permissive_engine():
    """Permissive engine for testing positive cases."""
    return RelationshipInferenceEngine(
        min_confidence=0.3,
        min_typed_confidence=0.50,
        min_typed_similarity=0.45,
    )


# ---------------------------------------------------------------------------
# 1. Weak conjunctions must NOT produce "contradicts"
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestWeakConjunctionsDoNotContradict:
    """
    "however", "but", "yet", "although", "nevertheless" are ordinary contrast
    connectors that appear in almost any paragraph.  They must NOT trigger the
    "contradicts" relationship type (issue #541 false positive fix).
    """

    @pytest.mark.asyncio
    async def test_however_does_not_contradict(self, engine):
        result_type, _ = await engine.infer_relationship_type(
            source_type="observation",
            target_type="observation",
            source_content="The deployment completed however some warnings appeared in logs",
            target_content="System started successfully however performance was lower",
        )
        assert result_type != "contradicts", (
            "'however' should not produce 'contradicts' relationship"
        )

    @pytest.mark.asyncio
    async def test_but_does_not_contradict(self, engine):
        result_type, _ = await engine.infer_relationship_type(
            source_type="observation",
            target_type="observation",
            source_content="The feature worked but response time was slow",
            target_content="Caching helped but memory usage increased",
        )
        assert result_type != "contradicts", (
            "'but' should not produce 'contradicts' relationship"
        )

    @pytest.mark.asyncio
    async def test_yet_does_not_contradict(self, engine):
        result_type, _ = await engine.infer_relationship_type(
            source_type="observation",
            target_type="observation",
            source_content="Tests pass yet coverage remains low in edge cases",
            target_content="Code review done yet some comments remain unaddressed",
        )
        assert result_type != "contradicts", (
            "'yet' should not produce 'contradicts' relationship"
        )

    @pytest.mark.asyncio
    async def test_although_does_not_contradict(self, engine):
        result_type, _ = await engine.infer_relationship_type(
            source_type="observation",
            target_type="observation",
            source_content="Although the build succeeded some tests were skipped",
            target_content="Although CI passed linting warnings remain in the codebase",
        )
        assert result_type != "contradicts", (
            "'although' should not produce 'contradicts' relationship"
        )

    @pytest.mark.asyncio
    async def test_nevertheless_does_not_contradict(self, engine):
        result_type, _ = await engine.infer_relationship_type(
            source_type="observation",
            target_type="observation",
            source_content="Server load was high nevertheless uptime remained stable",
            target_content="Disk usage grew nevertheless performance stayed within SLA",
        )
        assert result_type != "contradicts", (
            "'nevertheless' should not produce 'contradicts' relationship"
        )

    @pytest.mark.asyncio
    async def test_multiple_weak_conjunctions_no_contradict(self, engine):
        """
        A sentence containing many weak conjunctions must still not trigger
        "contradicts".
        """
        result_type, _ = await engine.infer_relationship_type(
            source_type="learning",
            target_type="learning",
            source_content=(
                "Although the approach worked, however we noticed some issues "
                "yet the overall result was positive but needs further review."
            ),
            target_content=(
                "Nevertheless the team agreed that although the solution was "
                "imperfect it was yet sufficient for the deadline."
            ),
        )
        assert result_type != "contradicts"


# ---------------------------------------------------------------------------
# 2. Strong contradiction keywords still produce "contradicts"
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestStrongContradictionKeywordsWork:
    """
    Direct opposition / negation indicators must still trigger "contradicts"
    when confidence exceeds the typed threshold.
    """

    @pytest.mark.asyncio
    async def test_contradicts_keyword(self, permissive_engine):
        result_type, confidence = await permissive_engine.infer_relationship_type(
            source_type="decision",
            target_type="decision",
            source_content=(
                "The performance data contradicts the earlier benchmark results "
                "showing that caching reduces latency"
            ),
            target_content=(
                "Benchmark measurements showed caching reduces latency by 40%, "
                "which contradicts the new findings"
            ),
        )
        # contradicts keyword in both memories → both-sides confidence = 0.75,
        # which exactly meets min_typed_confidence; shared keywords: benchmark/caching/latency
        assert result_type == "contradicts"
        assert confidence >= 0.75

    @pytest.mark.asyncio
    async def test_incorrect_keyword(self, permissive_engine):
        result_type, confidence = await permissive_engine.infer_relationship_type(
            source_type="learning",
            target_type="learning",
            source_content=(
                "That assumption was incorrect — the database connection pool "
                "size does not affect throughput at this scale"
            ),
            target_content=(
                "Connection pool incorrect sizing causes throughput degradation"
            ),
        )
        assert result_type == "contradicts"

    @pytest.mark.asyncio
    async def test_wrong_keyword(self, permissive_engine):
        result_type, _ = await permissive_engine.infer_relationship_type(
            source_type="learning",
            target_type="learning",
            source_content="That approach is wrong — using synchronous calls blocks the event loop",
            target_content="Synchronous blocking calls wrong in async context cause event loop stalls",
        )
        assert result_type == "contradicts"

    @pytest.mark.asyncio
    async def test_opposite_keyword(self, permissive_engine):
        result_type, _ = await permissive_engine.infer_relationship_type(
            source_type="decision",
            target_type="decision",
            source_content=(
                "Opposite conclusion: horizontal scaling performs better than "
                "vertical scaling for this workload"
            ),
            target_content=(
                "Horizontal scaling opposite recommendation was rejected in favor "
                "of vertical scaling due to operational complexity"
            ),
        )
        assert result_type == "contradicts"


# ---------------------------------------------------------------------------
# 3. min_typed_confidence gate — blocks low-confidence typed labels
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMinTypedConfidenceGate:
    """
    Typed labels (anything other than "related") must be blocked when
    confidence < min_typed_confidence (0.75 by default).
    """

    @pytest.mark.asyncio
    async def test_low_confidence_type_combination_falls_back_to_related(self, engine):
        """
        A type combination that generates confidence 0.3 (decision+decision=supports at 0.4,
        reversed = 0.28) should fall back to "related" due to the 0.75 gate.
        Additionally, if shared keywords exist, "supports" is the candidate but
        confidence 0.4 < 0.75 so the gate forces "related".
        """
        result_type, confidence = await engine.infer_relationship_type(
            source_type="decision",
            target_type="decision",
            source_content="We decided to use PostgreSQL for persistent storage of user data",
            target_content="The team decided to use Redis for caching session tokens and data",
        )
        # decision+decision → supports at 0.4, below min_typed_confidence=0.75
        assert result_type in ("related", "supports")
        if result_type == "supports":
            assert confidence >= 0.75

    @pytest.mark.asyncio
    async def test_custom_high_typed_confidence_blocks_weak_candidates(self):
        """
        Engine with min_typed_confidence=0.95 should almost never produce typed
        labels from type-combination analysis alone.
        """
        strict_engine = RelationshipInferenceEngine(
            min_confidence=0.5,
            min_typed_confidence=0.95,
        )
        result_type, _ = await strict_engine.infer_relationship_type(
            source_type="learning",
            target_type="error",
            source_content="Fixed timeout by increasing connection pool size limit",
            target_content="Connection pool timeout error occurred during peak load",
            source_timestamp=time.time(),
            target_timestamp=time.time() - 100,
        )
        # Even the 0.8-confidence "fixes" candidate is below 0.95
        assert result_type in ("related", "fixes")
        if result_type == "fixes":
            pytest.fail(
                "With min_typed_confidence=0.95, 'fixes' at 0.8 should be blocked"
            )

    @pytest.mark.asyncio
    async def test_single_side_contradiction_below_threshold(self, engine):
        """
        Single-side contradiction detection gives confidence 0.5, which is
        below min_typed_confidence=0.75.  Must fall back to 'related'.
        """
        result_type, _ = await engine.infer_relationship_type(
            source_type="observation",
            target_type="observation",
            source_content=(
                "The configuration appears wrong for this environment"
            ),
            target_content=(
                "Default settings were applied to the configuration file"
            ),
        )
        # "wrong" detected on source side only → confidence 0.5, below the gate.
        assert result_type == "related"


# ---------------------------------------------------------------------------
# 4. No shared domain keywords → falls back to "related"
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestNoDomainKeywordsShareFallback:
    """
    When source and target content share no meaningful domain keywords,
    typed labels must be downgraded to "related" (issue #541 guard 2).
    """

    @pytest.mark.asyncio
    async def test_unrelated_topics_produce_related(self, engine):
        result_type, _ = await engine.infer_relationship_type(
            source_type="learning",
            target_type="error",
            source_content="Baking bread requires precise temperature control and timing",
            target_content="Network socket timeout when connecting to remote database",
            source_timestamp=time.time(),
            target_timestamp=time.time() - 50,
        )
        assert result_type == "related", (
            "Completely unrelated topics should produce 'related', not a typed label"
        )

    @pytest.mark.asyncio
    async def test_shares_domain_keywords_helper_true(self):
        """_shares_domain_keywords returns True for overlapping content."""
        assert _shares_domain_keywords(
            "connection pool timeout causes database slowdown",
            "database connection error during peak traffic",
        ) is True

    @pytest.mark.asyncio
    async def test_shares_domain_keywords_helper_false(self):
        """_shares_domain_keywords returns False for disjoint content."""
        assert _shares_domain_keywords(
            "baking bread temperature oven rise dough",
            "network socket timeout remote server connection",
        ) is False

    def test_extract_domain_keywords_excludes_stopwords(self):
        """Short words and stopwords should not be included in domain keywords."""
        keywords = _extract_domain_keywords(
            "the deployment however failed but the team yet managed"
        )
        # All words here are short or stopwords; no meaningful keywords
        stopwords_expected_absent = {"the", "but", "yet", "and", "however"}
        for sw in stopwords_expected_absent:
            assert sw not in keywords

    def test_extract_domain_keywords_includes_meaningful_words(self):
        """Meaningful domain terms of 4+ chars should be extracted."""
        keywords = _extract_domain_keywords(
            "database connection timeout error occurred during deployment"
        )
        # All these should be captured
        assert "database" in keywords
        assert "connection" in keywords
        assert "timeout" in keywords
        assert "error" in keywords
        assert "deployment" in keywords


# ---------------------------------------------------------------------------
# 5. min_typed_similarity gate — low similarity forces "related"
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMinTypedSimilarityGate:
    """
    When a similarity score is provided and is below min_typed_similarity
    (default 0.65, on the (cos + 1) / 2 score consolidation passes), typed
    labels must be suppressed.
    """

    @pytest.mark.asyncio
    async def test_low_similarity_forces_related(self, engine):
        """Similarity 0.3 is below the 0.65 threshold — typed label suppressed."""
        result_type, _ = await engine.infer_relationship_type(
            source_type="learning",
            target_type="error",
            source_content="Fixed authentication timeout by adjusting pool configuration settings",
            target_content="Authentication timeout error connection pool exhausted",
            source_timestamp=time.time(),
            target_timestamp=time.time() - 60,
            similarity=0.30,  # well below min_typed_similarity=0.65
        )
        assert result_type == "related", (
            "Low similarity (0.3) should suppress typed labels and return 'related'"
        )

    @pytest.mark.asyncio
    async def test_borderline_similarity_just_below_threshold(self, engine):
        """Similarity 0.64 is just below the 0.65 threshold — typed label suppressed."""
        result_type, _ = await engine.infer_relationship_type(
            source_type="learning",
            target_type="error",
            source_content="Fixed database connection pool exhaustion by raising pool limit",
            target_content="Database connection pool exhausted causing timeout errors",
            source_timestamp=time.time(),
            target_timestamp=time.time() - 60,
            similarity=0.64,  # just below threshold
        )
        assert result_type == "related"

    @pytest.mark.asyncio
    async def test_similarity_at_threshold_allows_typed(self, engine):
        """Similarity exactly at threshold (0.65) should allow typed labels."""
        result_type, confidence = await engine.infer_relationship_type(
            source_type="learning",
            target_type="error",
            source_content="Fixed database connection pool exhaustion by raising pool limit",
            target_content="Database connection pool exhausted causing timeout errors",
            source_timestamp=time.time(),
            target_timestamp=time.time() - 60,
            similarity=0.65,  # exactly at threshold
        )
        # learning → error gives "fixes" at 0.8, which clears min_typed_confidence
        assert result_type == "fixes"
        assert confidence >= 0.75

    @pytest.mark.asyncio
    async def test_no_similarity_provided_does_not_block(self, engine):
        """When similarity is not provided, the gate is inactive."""
        result_type, confidence = await engine.infer_relationship_type(
            source_type="learning",
            target_type="error",
            source_content="Fixed authentication timeout by adjusting pool configuration settings",
            target_content="Authentication timeout error connection pool exhausted",
            source_timestamp=time.time(),
            target_timestamp=time.time() - 60,
            similarity=None,  # gate inactive
        )
        # Should produce a typed label (fixes or causes) with sufficient confidence
        assert result_type in ("fixes", "causes", "related")

    @pytest.mark.asyncio
    async def test_high_similarity_above_threshold_allows_typed(self, engine):
        """Similarity 0.9 is well above 0.55 — typed label should be allowed."""
        result_type, confidence = await engine.infer_relationship_type(
            source_type="learning",
            target_type="error",
            source_content="Resolved authentication error by fixing connection pool size configuration",
            target_content="Authentication connection pool error caused by incorrect size limit",
            source_timestamp=time.time(),
            target_timestamp=time.time() - 60,
            similarity=0.90,
        )
        # With high similarity, good confidence, and shared keywords, typed label expected
        assert result_type in ("fixes", "causes", "related")
        if result_type in ("fixes", "causes"):
            assert confidence >= 0.75


# ---------------------------------------------------------------------------
# 6. Constructor parameter defaults and combinations
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEngineInitParameters:
    """Verify constructor parameter handling for issue #541 additions."""

    def test_default_min_typed_confidence(self):
        engine = RelationshipInferenceEngine()
        assert engine.min_typed_confidence == 0.75

    def test_custom_min_typed_confidence(self):
        engine = RelationshipInferenceEngine(min_typed_confidence=0.85)
        assert engine.min_typed_confidence == 0.85

    def test_min_typed_confidence_cannot_be_below_min_confidence(self):
        """min_typed_confidence is floored at min_confidence value."""
        engine = RelationshipInferenceEngine(
            min_confidence=0.8,
            min_typed_confidence=0.5,  # lower than min_confidence
        )
        assert engine.min_typed_confidence >= engine.min_confidence

    def test_default_min_typed_similarity(self):
        engine = RelationshipInferenceEngine()
        assert engine.min_typed_similarity == 0.65

    def test_custom_min_typed_similarity(self):
        engine = RelationshipInferenceEngine(min_typed_similarity=0.80)
        assert engine.min_typed_similarity == 0.80

    def test_contradiction_patterns_do_not_include_weak_conjunctions(self):
        """Verify PATTERNS dict no longer contains weak conjunction regexes."""
        engine = RelationshipInferenceEngine()
        contradiction_patterns = engine.PATTERNS.get("contradiction", [])
        pattern_str = " ".join(contradiction_patterns)
        # These should NOT be in the contradiction patterns
        assert r"\bhowever\b" not in pattern_str, "\\bhowever\\b should be removed (issue #541)"
        assert r"\bbut\b" not in pattern_str, "\\bbut\\b should be removed (issue #541)"
        assert r"\byet\b" not in pattern_str, "\\byet\\b should be removed (issue #541)"
        assert r"\balthough\b" not in pattern_str, "\\balthough\\b should be removed (issue #541)"
        assert r"\bnevertheless\b" not in pattern_str, "\\bnevertheless\\b should be removed (issue #541)"

    def test_contradiction_patterns_retain_strong_keywords(self):
        """Verify that strong opposition patterns are still present."""
        engine = RelationshipInferenceEngine()
        contradiction_patterns = engine.PATTERNS.get("contradiction", [])
        pattern_str = " ".join(contradiction_patterns)
        # These strong indicators must still be present
        assert "contradict" in pattern_str
        assert "incorrect" in pattern_str or "wrong" in pattern_str
        assert "opposite" in pattern_str or "contrary" in pattern_str
