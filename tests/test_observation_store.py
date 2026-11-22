"""Tests for §1 Observation Store — new observation subtypes.

Validates that user_correction, tool_outcome, and preference_signal
are recognized as valid observation subtypes for the §2 Belief Store
weighting system.
"""

import pytest

from mcp_memory_service.models.ontology import (
    TAXONOMY,
    validate_memory_type,
)


class TestObservationSubtypes:
    """§1: New observation subtypes for belief derivation."""

    @pytest.mark.parametrize("subtype", [
        "user_correction",
        "tool_outcome",
        "preference_signal",
    ])
    def test_new_subtypes_are_valid_observation_types(self, subtype):
        """Each new subtype should be a valid observation subtype."""
        assert subtype in TAXONOMY["observation"]

    @pytest.mark.parametrize("subtype", [
        "user_correction",
        "tool_outcome",
        "preference_signal",
    ])
    def test_new_subtypes_validate_correctly(self, subtype):
        """validate_memory_type should accept new subtypes."""
        assert validate_memory_type(subtype) is True

    def test_observation_subtypes_count(self):
        """Observation should have 14 subtypes (10 original + 4 new)."""
        assert len(TAXONOMY["observation"]) == 14

    def test_existing_subtypes_preserved(self):
        """Original observation subtypes should still be present."""
        original = {
            "note", "reference", "code_edit", "command",
            "conversation", "conversation_turn", "session",
            "document", "search",
        }
        for subtype in original:
            assert subtype in TAXONOMY["observation"], f"{subtype} missing"
