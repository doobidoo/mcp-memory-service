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
