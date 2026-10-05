"""Regression tests for ontology pair rules in relationship inference (issue #1458)."""

import pytest

from mcp_memory_service.consolidation.relationship_inference import RelationshipInferenceEngine
from mcp_memory_service.models.ontology import is_allowed_pair


@pytest.fixture
def engine():
    return RelationshipInferenceEngine(
        min_confidence=0.5,
        min_typed_confidence=0.5,
    )


class TestAllowedPairs:
    def test_fixes_allows_learning_to_error(self):
        assert is_allowed_pair("fixes", "learning", "error") is True

    def test_fixes_rejects_observation_to_observation(self):
        assert is_allowed_pair("fixes", "observation", "observation") is False

    def test_fixes_rejects_note_parent_observation_to_error(self):
        assert is_allowed_pair("fixes", "observation", "error") is False

    def test_unknown_relationship_is_rejected(self):
        assert is_allowed_pair("uses", "decision", "error") is False

    def test_any_wildcard_allows_follows(self):
        assert is_allowed_pair("follows", "decision", "error") is True


class TestInferenceRespectsValidPatterns:
    @pytest.mark.asyncio
    async def test_learning_to_error_can_still_fix(self, engine):
        rel_type, confidence = await engine.infer_relationship_type(
            source_type="learning/insight",
            target_type="error/bug",
            source_content="Fixed authentication timeout by adjusting configuration",
            target_content="Authentication error: Request timeout after 30 seconds",
        )
        assert rel_type == "fixes"
        assert confidence >= 0.5

    @pytest.mark.asyncio
    async def test_note_to_error_does_not_produce_fixes(self, engine):
        rel_type, _confidence = await engine.infer_relationship_type(
            source_type="note",
            target_type="error",
            source_content="Fixed authentication timeout by adjusting configuration",
            target_content="Authentication error: Request timeout after 30 seconds",
        )
        assert rel_type != "fixes"

    @pytest.mark.asyncio
    async def test_observation_to_observation_does_not_produce_fixes(self, engine):
        rel_type, _confidence = await engine.infer_relationship_type(
            source_type="observation",
            target_type="observation",
            source_content="Fixed authentication timeout by adjusting configuration",
            target_content="Authentication error: Request timeout after 30 seconds",
        )
        assert rel_type != "fixes"

    @pytest.mark.asyncio
    async def test_decision_to_error_does_not_emit_uses(self, engine):
        rel_type, _confidence = await engine.infer_relationship_type(
            source_type="decision",
            target_type="error",
            source_content="Retry policy for authentication timeout",
            target_content="Authentication error: Request timeout after 30 seconds",
        )
        assert rel_type != "uses"
        assert rel_type in {"related", "causes", "follows"}

    def test_type_table_never_emits_unknown_relationship(self, engine):
        candidates = engine._analyze_type_combination("decision", "error")
        assert all(rel_type != "uses" for rel_type, _confidence in candidates)
        assert all(
            is_allowed_pair(rel_type, "decision", "error")
            for rel_type, _confidence in candidates
        )
