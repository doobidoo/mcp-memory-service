"""Tests for fact mutability classification (RFC #1008 §5)."""

import pytest
from mcp_memory_service.reasoning.mutability import classify_mutability, contradiction_action


class TestMutabilityClassifier:
    def test_import(self):
        assert classify_mutability is not None

    def test_version_is_volatile(self):
        assert classify_mutability("mcp-memory-service version is 10.66.1") == "volatile"

    def test_date_reference_is_volatile(self):
        assert classify_mutability("Deployed on 2026-05-28 to production") == "volatile"

    def test_currently_is_volatile(self):
        assert classify_mutability("The service is currently running on port 3202") == "volatile"

    def test_definition_is_stable(self):
        assert classify_mutability("Python uses indentation to define code blocks") == "stable"

    def test_session_context_is_ephemeral(self):
        assert classify_mutability("Working on branch feat/xyz in this session") == "ephemeral"

    def test_right_now_is_ephemeral(self):
        assert classify_mutability("I'm fixing this right now") == "ephemeral"

    def test_unknown_defaults_to_stable(self):
        assert classify_mutability("The sky is blue on clear days") == "stable"


class TestMutabilityInContradiction:
    def test_volatile_conflict_is_supersede(self):
        assert contradiction_action("volatile", "volatile") == "supersede"

    def test_stable_conflict_is_flag(self):
        assert contradiction_action("stable", "stable") == "flag"

    def test_ephemeral_never_flags(self):
        assert contradiction_action("ephemeral", "stable") == "ignore"

    def test_mixed_volatile_stable_is_supersede(self):
        assert contradiction_action("stable", "volatile") == "supersede"

    def test_invalid_mutability_raises(self):
        with pytest.raises(ValueError, match="Invalid mutability"):
            contradiction_action("unknown", "stable")
