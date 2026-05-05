"""
Unit tests for Memory Type Ontology

Tests the formal ontology layer for memory classification and type validation.
"""

import sys
from pathlib import Path
import importlib.util
import pytest

# Load ontology module directly without importing the package
ontology_path = Path(__file__).parent.parent / "src" / "mcp_memory_service" / "models" / "ontology.py"
spec = importlib.util.spec_from_file_location("ontology", ontology_path)
ontology = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ontology)

BaseMemoryType = ontology.BaseMemoryType
TAXONOMY = ontology.TAXONOMY
RELATIONSHIPS = ontology.RELATIONSHIPS
validate_memory_type = ontology.validate_memory_type
get_parent_type = ontology.get_parent_type
get_all_types = ontology.get_all_types
validate_relationship = ontology.validate_relationship
is_symmetric_relationship = ontology.is_symmetric_relationship
MemoryTypeOntology = ontology.MemoryTypeOntology


class TestBurst11BaseMemoryTypes:
    """Tests for Burst 1.1: Base Memory Types Enum"""

    def test_enum_has_exactly_twelve_base_types(self):
        """Base type enum should have exactly 12 types (5 original + 7 new)"""
        assert len(BaseMemoryType) == 12

    def test_each_type_is_valid_string_constant(self):
        """Each base type should be a valid string constant"""
        expected_types = {
            # Software Development (original 5)
            "observation",
            "decision",
            "learning",
            "error",
            "pattern",
            # Project Management - Agile (2 new)
            "planning",
            "ceremony",
            # Project Management - Traditional (2 new)
            "milestone",
            "stakeholder",
            # General Knowledge Work (3 new)
            "meeting",
            "research",
            "communication"
        }
        actual_types = {member.value for member in BaseMemoryType}
        assert actual_types == expected_types

    def test_base_types_are_lowercase(self):
        """All base type values should be lowercase for consistency"""
        for member in BaseMemoryType:
            assert member.value.islower()

    def test_base_types_accessible_as_enum_members(self):
        """Base types should be accessible as enum members"""
        # Original types
        assert BaseMemoryType.OBSERVATION.value == "observation"
        assert BaseMemoryType.DECISION.value == "decision"
        assert BaseMemoryType.LEARNING.value == "learning"
        assert BaseMemoryType.ERROR.value == "error"
        assert BaseMemoryType.PATTERN.value == "pattern"
        # New types
        assert BaseMemoryType.PLANNING.value == "planning"
        assert BaseMemoryType.CEREMONY.value == "ceremony"
        assert BaseMemoryType.MILESTONE.value == "milestone"
        assert BaseMemoryType.STAKEHOLDER.value == "stakeholder"
        assert BaseMemoryType.MEETING.value == "meeting"
        assert BaseMemoryType.RESEARCH.value == "research"
        assert BaseMemoryType.COMMUNICATION.value == "communication"


class TestBurst12TaxonomyHierarchy:
    """Tests for Burst 1.2: Taxonomy Hierarchy Dictionary"""

    def test_each_base_type_has_at_least_two_subtypes(self):
        """Each base type should have at least 2 subtypes for meaningful classification"""
        for base_type in BaseMemoryType:
            subtypes = TAXONOMY.get(base_type.value, [])
            assert len(subtypes) >= 2, f"{base_type.value} has {len(subtypes)} subtypes, expected >= 2"

    def test_all_subtypes_are_unique_across_taxonomy(self):
        """No subtype should appear under multiple base types"""
        all_subtypes = []
        for base_type, subtypes in TAXONOMY.items():
            all_subtypes.extend(subtypes)

        # Check for duplicates
        assert len(all_subtypes) == len(set(all_subtypes)), \
            f"Found duplicate subtypes: {[s for s in all_subtypes if all_subtypes.count(s) > 1]}"

    def test_taxonomy_covers_all_base_types(self):
        """TAXONOMY dict should have entries for all base types"""
        base_type_values = {member.value for member in BaseMemoryType}
        taxonomy_keys = set(TAXONOMY.keys())
        assert base_type_values == taxonomy_keys

    def test_all_subtypes_are_lowercase_with_underscores(self):
        """Subtypes should follow snake_case naming convention"""
        for base_type, subtypes in TAXONOMY.items():
            for subtype in subtypes:
                assert subtype.islower() or '_' in subtype, \
                    f"Subtype '{subtype}' should be lowercase with underscores"
                assert ' ' not in subtype, f"Subtype '{subtype}' should not contain spaces"


class TestBurst13RelationshipTypes:
    """Tests for Burst 1.3: Relationship Types Definition"""

    def test_six_relationship_types_defined(self):
        """Should have exactly 6 relationship types"""
        expected_types = {"causes", "fixes", "contradicts", "supports", "follows", "related"}
        actual_types = set(RELATIONSHIPS.keys())
        assert actual_types == expected_types

    def test_each_type_has_description(self):
        """Each relationship type should have a description"""
        for rel_type, config in RELATIONSHIPS.items():
            assert "description" in config, f"{rel_type} missing description"
            assert len(config["description"]) > 0, f"{rel_type} has empty description"

    def test_each_type_has_valid_patterns(self):
        """Each relationship type should define valid source→target patterns"""
        for rel_type, config in RELATIONSHIPS.items():
            assert "valid_patterns" in config, f"{rel_type} missing valid_patterns"
            assert len(config["valid_patterns"]) > 0, f"{rel_type} has no valid patterns"

    def test_related_type_accepts_any_pattern(self):
        """The 'related' type should accept any→any pattern (default fallback)"""
        related_patterns = RELATIONSHIPS["related"]["valid_patterns"]
        assert "any → any" in related_patterns, "related type should accept any→any pattern"


class TestBurst14MemoryTypeValidation:
    """Tests for Burst 1.4: Memory Type Validation"""

    def test_base_types_validate_correctly(self):
        """All base types should validate as True"""
        assert validate_memory_type("observation") is True
        assert validate_memory_type("decision") is True
        assert validate_memory_type("learning") is True
        assert validate_memory_type("error") is True
        assert validate_memory_type("pattern") is True

    def test_subtypes_validate_correctly(self):
        """All subtypes should validate as True"""
        # Test a few subtypes from each category
        assert validate_memory_type("code_edit") is True  # observation subtype
        assert validate_memory_type("architecture") is True  # decision subtype
        assert validate_memory_type("insight") is True  # learning subtype
        assert validate_memory_type("bug") is True  # error subtype
        assert validate_memory_type("code_smell") is True  # pattern subtype

    def test_invalid_types_return_false(self):
        """Invalid types should return False"""
        assert validate_memory_type("invalid_type") is False
        assert validate_memory_type("not_a_type") is False
        assert validate_memory_type("") is False
        assert validate_memory_type("ObSeRvAtIoN") is False  # Case sensitive


class TestBurst15GetParentType:
    """Tests for Burst 1.5: Get Parent Type Function"""

    def test_subtype_returns_correct_parent(self):
        """Subtypes should return their parent base type"""
        assert get_parent_type("code_edit") == "observation"
        assert get_parent_type("architecture") == "decision"
        assert get_parent_type("insight") == "learning"
        assert get_parent_type("bug") == "error"
        assert get_parent_type("code_smell") == "pattern"

    def test_base_type_returns_itself(self):
        """Base types should return themselves (they are their own parent)"""
        assert get_parent_type("observation") == "observation"
        assert get_parent_type("decision") == "decision"
        assert get_parent_type("learning") == "learning"
        assert get_parent_type("error") == "error"
        assert get_parent_type("pattern") == "pattern"

    def test_invalid_type_returns_none(self):
        """Invalid types should return None"""
        assert get_parent_type("invalid_type") is None
        assert get_parent_type("not_a_type") is None
        assert get_parent_type("") is None


class TestBurst16GetAllTypes:
    """Tests for Burst 1.6: Get All Types Function"""

    def test_returns_correct_count(self):
        """Should return the correct total count of base and subtypes"""
        all_types = get_all_types()
        expected_count = len(BaseMemoryType) + sum(len(subtypes) for subtypes in TAXONOMY.values())
        assert len(all_types) == expected_count

    def test_no_duplicates_in_list(self):
        """Should not have duplicate types in the list"""
        all_types = get_all_types()
        assert len(all_types) == len(set(all_types))

    def test_includes_base_types(self):
        """Should include all base types"""
        all_types = get_all_types()
        assert "observation" in all_types
        assert "decision" in all_types
        assert "learning" in all_types
        assert "error" in all_types
        assert "pattern" in all_types

    def test_includes_subtypes(self):
        """Should include subtypes from all categories"""
        all_types = get_all_types()
        assert "code_edit" in all_types
        assert "architecture" in all_types
        assert "insight" in all_types
        assert "bug" in all_types
        assert "code_smell" in all_types

    def test_document_types_are_valid(self):
        """Document-related types should be valid observation subtypes"""
        # Validate that new document types are recognized
        assert validate_memory_type("document") is True
        assert validate_memory_type("note") is True
        assert validate_memory_type("reference") is True

        # Validate parent type hierarchy
        assert get_parent_type("document") == "observation"
        assert get_parent_type("note") == "observation"
        assert get_parent_type("reference") == "observation"

        # Validate they appear in the full type list
        all_types = get_all_types()
        assert "document" in all_types
        assert "note" in all_types
        assert "reference" in all_types


class TestBurst17RelationshipTypeValidation:
    """Tests for Burst 1.7: Relationship Type Validation"""

    def test_valid_types_return_true(self):
        """All 6 defined relationship types should validate as True"""
        assert validate_relationship("causes") is True
        assert validate_relationship("fixes") is True
        assert validate_relationship("contradicts") is True
        assert validate_relationship("supports") is True
        assert validate_relationship("follows") is True
        assert validate_relationship("related") is True

    def test_invalid_types_return_false(self):
        """Invalid relationship types should return False"""
        assert validate_relationship("invalid_relationship") is False
        assert validate_relationship("not_a_rel") is False
        assert validate_relationship("") is False


class TestBurst18OntologyClassIntegration:
    """Tests for Burst 1.8: Ontology Class Integration"""

    def test_class_methods_accessible(self):
        """All ontology methods should be accessible via the class"""
        # Test that class has all expected methods
        assert hasattr(MemoryTypeOntology, 'validate_memory_type')
        assert hasattr(MemoryTypeOntology, 'get_parent_type')
        assert hasattr(MemoryTypeOntology, 'get_all_types')
        assert hasattr(MemoryTypeOntology, 'validate_relationship')

    def test_validate_memory_type_via_class(self):
        """validate_memory_type should work via class method"""
        assert MemoryTypeOntology.validate_memory_type("observation") is True
        assert MemoryTypeOntology.validate_memory_type("code_edit") is True
        assert MemoryTypeOntology.validate_memory_type("invalid") is False

    def test_get_parent_type_via_class(self):
        """get_parent_type should work via class method"""
        assert MemoryTypeOntology.get_parent_type("code_edit") == "observation"
        assert MemoryTypeOntology.get_parent_type("observation") == "observation"
        assert MemoryTypeOntology.get_parent_type("invalid") is None

    def test_get_all_types_via_class(self):
        """get_all_types should work via class method"""
        all_types = MemoryTypeOntology.get_all_types()
        expected_count = len(BaseMemoryType) + sum(len(subtypes) for subtypes in TAXONOMY.values())
        assert len(all_types) == expected_count
        assert "observation" in all_types
        assert "code_edit" in all_types

    def test_validate_relationship_via_class(self):
        """validate_relationship should work via class method"""
        assert MemoryTypeOntology.validate_relationship("causes") is True
        assert MemoryTypeOntology.validate_relationship("invalid") is False


class TestBurst19SymmetricRelationshipClassification:
    """Tests for is_symmetric_relationship() - PR #348"""

    def test_is_symmetric_relationship_symmetric_types(self):
        """Test symmetric relationship types return True"""
        assert is_symmetric_relationship("related") is True
        assert is_symmetric_relationship("contradicts") is True

    def test_is_symmetric_relationship_asymmetric_types(self):
        """Test asymmetric relationship types return False"""
        assert is_symmetric_relationship("causes") is False
        assert is_symmetric_relationship("fixes") is False
        assert is_symmetric_relationship("supports") is False
        assert is_symmetric_relationship("follows") is False

    def test_is_symmetric_relationship_invalid_type(self):
        """Test invalid relationship type raises ValueError"""
        with pytest.raises(ValueError, match="Invalid relationship type"):
            is_symmetric_relationship("invalid_type")

    def test_is_symmetric_relationship_via_class(self):
        """Test is_symmetric_relationship works via class method"""
        assert MemoryTypeOntology.is_symmetric_relationship("related") is True
        assert MemoryTypeOntology.is_symmetric_relationship("causes") is False

        with pytest.raises(ValueError):
            MemoryTypeOntology.is_symmetric_relationship("invalid")


class TestBurst20NewProjectManagementTypes:
    """Test new project management base types (Issue #464)."""

    def test_new_base_types_exist(self):
        """Verify all 7 new base types are defined."""
        new_base_types = [
            "planning", "ceremony", "milestone", "stakeholder",
            "meeting", "research", "communication"
        ]

        all_types = get_all_types()
        for base_type in new_base_types:
            assert base_type in all_types, f"New base type '{base_type}' missing"

    def test_planning_subtypes(self):
        """Verify planning subtypes."""
        expected = [
            "sprint_goal", "backlog_item", "story_point_estimate",
            "velocity", "retrospective", "standup_note", "acceptance_criteria"
        ]

        all_types = get_all_types()
        for subtype in expected:
            assert subtype in all_types
            assert get_parent_type(subtype) == "planning"

    def test_ceremony_subtypes(self):
        """Verify ceremony subtypes."""
        expected = [
            "sprint_review", "sprint_planning", "daily_standup",
            "retrospective_action", "demo_feedback"
        ]

        all_types = get_all_types()
        for subtype in expected:
            assert subtype in all_types
            assert get_parent_type(subtype) == "ceremony"

    def test_milestone_subtypes(self):
        """Verify milestone subtypes."""
        expected = [
            "deliverable", "dependency", "risk", "constraint", "assumption", "deadline"
        ]

        all_types = get_all_types()
        for subtype in expected:
            assert subtype in all_types
            assert get_parent_type(subtype) == "milestone"

    def test_stakeholder_subtypes(self):
        """Verify stakeholder subtypes."""
        expected = [
            "requirement", "feedback", "escalation", "approval",
            "change_request", "status_update"
        ]

        all_types = get_all_types()
        for subtype in expected:
            assert subtype in all_types
            assert get_parent_type(subtype) == "stakeholder"

    def test_meeting_subtypes(self):
        """Verify meeting subtypes."""
        expected = ["action_item", "attendee_note", "agenda_item", "follow_up", "minutes"]

        all_types = get_all_types()
        for subtype in expected:
            assert subtype in all_types
            assert get_parent_type(subtype) == "meeting"

    def test_research_subtypes(self):
        """Verify research subtypes."""
        expected = ["finding", "comparison", "recommendation", "source", "hypothesis"]

        all_types = get_all_types()
        for subtype in expected:
            assert subtype in all_types
            assert get_parent_type(subtype) == "research"

    def test_communication_subtypes(self):
        """Verify communication subtypes."""
        expected = ["email_summary", "chat_summary", "announcement", "request", "response"]

        all_types = get_all_types()
        for subtype in expected:
            assert subtype in all_types
            assert get_parent_type(subtype) == "communication"

    def test_total_type_count(self):
        """Verify total type count.

        Update the expected value whenever the type ontology in
        src/mcp_memory_service/models/ontology.py gains or loses a type.
        """
        all_types = get_all_types()
        assert len(all_types) == 78, f"Expected 78 types, got {len(all_types)}"


class TestBurst21CustomMemoryTypeConfiguration:
    """Test configurable memory types via MCP_CUSTOM_MEMORY_TYPES."""

    def setup_method(self):
        """Clear caches and environment before each test."""
        import os
        # Remove any custom types from environment to ensure clean state
        os.environ.pop('MCP_CUSTOM_MEMORY_TYPES', None)
        ontology.clear_ontology_caches()

    def teardown_method(self):
        """Clear caches and environment after each test."""
        import os
        # Remove any custom types from environment
        os.environ.pop('MCP_CUSTOM_MEMORY_TYPES', None)
        ontology.clear_ontology_caches()

    def test_load_custom_types_json(self, monkeypatch):
        """Test loading custom types from JSON environment variable."""
        import json

        # Set custom types
        custom_config = json.dumps({
            "legal": ["contract", "clause", "obligation"],
            "sales": ["opportunity", "objection", "competitor"]
        })
        monkeypatch.setenv('MCP_CUSTOM_MEMORY_TYPES', custom_config)

        # Clear all caches to force reload
        ontology.clear_ontology_caches()

        # Verify custom types loaded
        all_types = ontology.get_all_types()
        assert "legal" in all_types
        assert "contract" in all_types
        assert "sales" in all_types
        assert "opportunity" in all_types

        # Verify parent relationships
        assert ontology.get_parent_type("contract") == "legal"
        assert ontology.get_parent_type("opportunity") == "sales"

    def test_merge_with_builtin_types(self, monkeypatch):
        """Test that custom types merge with (not replace) built-in types."""
        import json

        custom_config = json.dumps({"meeting": ["board_meeting"]})
        monkeypatch.setenv('MCP_CUSTOM_MEMORY_TYPES', custom_config)

        # Clear caches to force reload
        ontology.clear_ontology_caches()

        all_types = ontology.get_all_types()

        # Built-in meeting subtypes should still exist
        assert "action_item" in all_types
        assert "follow_up" in all_types

        # Custom subtype should be added
        assert "board_meeting" in all_types

    def test_invalid_json_graceful_fallback(self, monkeypatch):
        """Test graceful fallback on invalid JSON."""
        monkeypatch.setenv('MCP_CUSTOM_MEMORY_TYPES', '{invalid json}')

        # Clear caches to force reload
        ontology.clear_ontology_caches()

        # Should fall back to built-in types
        all_types = ontology.get_all_types()
        assert "observation" in all_types  # Built-in type still works

    def test_custom_base_type_with_empty_subtypes(self, monkeypatch):
        """Issue #842: `{"foo": []}` must register `foo` as a base type.

        The MCP coercion warning explicitly tells users to register custom
        types with `'{"<type>": []}'`. Empty subtype lists were silently
        dropped, so the bare custom type never reached the validator and
        kept getting coerced to 'observation'.
        """
        import json

        monkeypatch.setenv('MCP_CUSTOM_MEMORY_TYPES', json.dumps({"foo": []}))
        ontology.clear_ontology_caches()

        all_types = ontology.get_all_types()
        assert "foo" in all_types
        assert ontology.validate_memory_type("foo") is True
        # Bare custom base types map to themselves
        assert ontology.get_parent_type("foo") == "foo"

    def test_custom_type_with_only_invalid_subtypes_still_registers_base(self, monkeypatch):
        """If every subtype is filtered out, the base type must still register."""
        import json

        # `@@bad` fails the `replace('_','').isalnum()` check
        monkeypatch.setenv('MCP_CUSTOM_MEMORY_TYPES', json.dumps({"foo": ["@@bad"]}))
        ontology.clear_ontology_caches()

        all_types = ontology.get_all_types()
        assert "foo" in all_types
        assert "@@bad" not in all_types

    def test_no_custom_types_default_behavior(self, monkeypatch):
        """Test default behavior when no custom types configured."""
        import os

        # Ensure env var is not set - must be done BEFORE clearing caches
        # Use both monkeypatch AND direct removal to ensure clean state
        monkeypatch.delenv('MCP_CUSTOM_MEMORY_TYPES', raising=False)
        os.environ.pop('MCP_CUSTOM_MEMORY_TYPES', None)

        # Now clear caches to force reload with no custom types
        ontology.clear_ontology_caches()

        all_types = ontology.get_all_types()

        # Should have at least 75 built-in types (might have test pollution from previous tests)
        # The key thing is that custom types from test_load_custom_types_json are NOT present
        assert len(all_types) >= 75, f"Expected at least 75 types, got {len(all_types)}"

        # Verify that the custom types from other tests in this class are NOT present
        # (These are the custom types that should only appear when MCP_CUSTOM_MEMORY_TYPES is set)
        assert "legal" not in all_types, "Custom base type 'legal' should not be present"
        assert "contract" not in all_types, "Custom subtype 'contract' should not be present"
        assert "sales" not in all_types, "Custom base type 'sales' should not be present"
        assert "opportunity" not in all_types, "Custom subtype 'opportunity' should not be present"

        # Verify standard types work
        assert "observation" in all_types
        assert "planning" in all_types
        assert "meeting" in all_types
