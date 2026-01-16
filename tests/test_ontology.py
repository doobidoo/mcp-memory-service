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
