"""
Unit tests for Memory Type Ontology

Tests the formal ontology layer for memory classification and type validation.
"""

import sys
from pathlib import Path
import importlib.util

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
MemoryTypeOntology = ontology.MemoryTypeOntology


class TestBurst11BaseMemoryTypes:
    """Tests for Burst 1.1: Base Memory Types Enum"""

    def test_enum_has_exactly_five_base_types(self):
        """Base type enum should have exactly 5 types"""
        assert len(BaseMemoryType) == 5

    def test_each_type_is_valid_string_constant(self):
        """Each base type should be a valid string constant"""
        expected_types = {
            "observation",
            "decision",
            "learning",
            "error",
            "pattern"
        }
        actual_types = {member.value for member in BaseMemoryType}
        assert actual_types == expected_types

    def test_base_types_are_lowercase(self):
        """All base type values should be lowercase for consistency"""
        for member in BaseMemoryType:
            assert member.value.islower()

    def test_base_types_accessible_as_enum_members(self):
        """Base types should be accessible as enum members"""
        assert BaseMemoryType.OBSERVATION.value == "observation"
        assert BaseMemoryType.DECISION.value == "decision"
        assert BaseMemoryType.LEARNING.value == "learning"
        assert BaseMemoryType.ERROR.value == "error"
        assert BaseMemoryType.PATTERN.value == "pattern"


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
        """Should return 5 base types + 21 subtypes = 26 total"""
        all_types = get_all_types()
        assert len(all_types) == 26

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
        assert len(all_types) == 26
        assert "observation" in all_types
        assert "code_edit" in all_types

    def test_validate_relationship_via_class(self):
        """validate_relationship should work via class method"""
        assert MemoryTypeOntology.validate_relationship("causes") is True
        assert MemoryTypeOntology.validate_relationship("invalid") is False
