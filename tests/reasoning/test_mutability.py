"""Tests for fact mutability classification (RFC #1008 §5)."""

import pytest


class TestMutabilityClassifier:
    def test_import(self):
        from mcp_memory_service.reasoning.mutability import classify_mutability
        assert classify_mutability is not None

    def test_version_is_volatile(self):
        from mcp_memory_service.reasoning.mutability import classify_mutability
        assert classify_mutability("mcp-memory-service version is 10.66.1") == "volatile"

    def test_date_reference_is_volatile(self):
        from mcp_memory_service.reasoning.mutability import classify_mutability
        assert classify_mutability("The service is currently running on port 3202") == "volatile"

    def test_definition_is_stable(self):
        from mcp_memory_service.reasoning.mutability import classify_mutability
        assert classify_mutability("Python uses indentation to define code blocks") == "stable"

    def test_session_context_is_ephemeral(self):
        from mcp_memory_service.reasoning.mutability import classify_mutability
        assert classify_mutability("Working on branch feat/xyz in this session") == "ephemeral"

    def test_unknown_defaults_to_stable(self):
        from mcp_memory_service.reasoning.mutability import classify_mutability
        assert classify_mutability("The sky is blue on clear days") == "stable"


class TestMutabilityInContradiction:
    def test_volatile_conflict_is_supersede(self):
        from mcp_memory_service.reasoning.mutability import contradiction_action
        assert contradiction_action("volatile", "volatile") == "supersede"

    def test_stable_conflict_is_flag(self):
        from mcp_memory_service.reasoning.mutability import contradiction_action
        assert contradiction_action("stable", "stable") == "flag"

    def test_ephemeral_never_flags(self):
        from mcp_memory_service.reasoning.mutability import contradiction_action
        assert contradiction_action("ephemeral", "stable") == "ignore"

    def test_mixed_volatile_stable_is_supersede(self):
        from mcp_memory_service.reasoning.mutability import contradiction_action
        assert contradiction_action("stable", "volatile") == "supersede"
