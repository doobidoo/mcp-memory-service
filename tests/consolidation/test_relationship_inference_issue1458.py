"""
Issue #1458: relationship inference must honor the ontology's valid_patterns.

RELATIONSHIPS[...]["valid_patterns"] in models/ontology.py declares which
parent-type pairs a relationship may connect. Before the fix nothing read it:
type_patterns could emit pairs outside it (and the non-type "uses"), and the
content analyzer emitted "fixes" for any target whose type string contains
"error".
"""

import itertools

import pytest

from mcp_memory_service.consolidation.relationship_inference import RelationshipInferenceEngine
from mcp_memory_service.models.ontology import RELATIONSHIPS, is_allowed_pair

BASE_TYPES = ["observation", "decision", "learning", "error", "pattern"]


@pytest.fixture
def engine():
    """Thresholds at zero, so the pair rules are the only thing that can reject a label."""
    return RelationshipInferenceEngine(min_confidence=0.0, min_typed_confidence=0.0)


class TestIsAllowedPair:
    def test_listed_pair_is_allowed(self):
        assert is_allowed_pair("fixes", "learning", "error")

    def test_unlisted_pair_is_rejected(self):
        assert not is_allowed_pair("fixes", "observation", "error")
        assert not is_allowed_pair("fixes", "error", "learning")

    def test_any_is_a_wildcard(self):
        assert is_allowed_pair("follows", "error", "pattern")
        assert is_allowed_pair("related", None, None)

    def test_unknown_relationship_is_rejected(self):
        assert not is_allowed_pair("uses", "decision", "error")


class TestInference:
    @pytest.mark.asyncio
    async def test_learning_still_fixes_error(self, engine):
        rel_type, _ = await engine.infer_relationship_type(
            source_type="learning/insight",
            target_type="error/bug",
            source_content="Fixed the database deadlock by reordering locks",
            target_content="Database deadlock on transaction commit",
        )
        assert rel_type == "fixes"

    @pytest.mark.asyncio
    async def test_note_cannot_fix_error(self, engine):
        """note resolves to observation; observation -> error allows causes, not fixes."""
        rel_type, _ = await engine.infer_relationship_type(
            source_type="note",
            target_type="error",
            source_content="Fixed the database deadlock by reordering locks",
            target_content="Database deadlock on transaction commit",
        )
        assert rel_type != "fixes"

    @pytest.mark.asyncio
    async def test_observation_cannot_fix_observation(self, engine):
        """The content analyzer matches "error" in the type string; the parent is still observation.

        The disallowed fixes/causes candidates must be dropped before ranking, so the
        allowed follows candidate wins instead of everything collapsing to related.
        """
        rel_type, _ = await engine.infer_relationship_type(
            source_type="observation",
            target_type="observation/error_log",
            source_content="Fixed the database deadlock, which caused the timeouts",
            target_content="Database deadlock on transaction commit",
        )
        assert rel_type == "follows"


class TestTypePatterns:
    @pytest.mark.parametrize("source,target", itertools.product(BASE_TYPES, repeat=2))
    def test_type_candidates_respect_valid_patterns(self, engine, source, target):
        for rel_type, _ in engine._analyze_type_combination(source, target):
            assert rel_type in RELATIONSHIPS, f"{source} -> {target}: unknown type {rel_type}"
            assert is_allowed_pair(rel_type, source, target), f"{source} -> {target}: {rel_type}"
