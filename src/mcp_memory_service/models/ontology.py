"""
Formal Memory Type Ontology for Knowledge Graph

Provides controlled vocabulary and type hierarchy for semantic memory classification.
Part of Phase 0: Ontology Foundation (Knowledge Graph Evolution).

Usage:
    from mcp_memory_service.models.ontology import MemoryTypeOntology

    # Validate memory type
    is_valid = MemoryTypeOntology.validate_memory_type("observation")

    # Get parent type
    parent = MemoryTypeOntology.get_parent_type("code_edit")  # Returns "observation"
"""

from enum import Enum
from typing import Dict, List, Optional, Final


class BaseMemoryType(str, Enum):
    """
    Base memory types forming the top-level ontology.

    These are the fundamental categories that all memories belong to.
    Each base type can have multiple subtypes for finer-grained classification.
    """
    OBSERVATION = "observation"
    DECISION = "decision"
    LEARNING = "learning"
    ERROR = "error"
    PATTERN = "pattern"


# Taxonomy hierarchy: base types → subtypes
TAXONOMY: Final[Dict[str, List[str]]] = {
    "observation": [
        "code_edit",
        "file_access",
        "search",
        "command",
        "conversation"
    ],
    "decision": [
        "architecture",
        "tool_choice",
        "approach",
        "configuration"
    ],
    "learning": [
        "insight",
        "best_practice",
        "anti_pattern",
        "gotcha"
    ],
    "error": [
        "bug",
        "failure",
        "exception",
        "timeout"
    ],
    "pattern": [
        "recurring_issue",
        "code_smell",
        "design_pattern",
        "workflow"
    ]
}


# Relationship types with valid source → target patterns
RELATIONSHIPS: Final[Dict[str, Dict[str, List[str]]]] = {
    "causes": {
        "description": "A causes B (causal relationship)",
        "valid_patterns": ["observation → error", "decision → observation", "error → error"]
    },
    "fixes": {
        "description": "A fixes B (remediation relationship)",
        "valid_patterns": ["decision → error", "learning → error", "pattern → error"]
    },
    "contradicts": {
        "description": "A contradicts B (conflict relationship)",
        "valid_patterns": ["decision → decision", "learning → learning", "observation → observation"]
    },
    "supports": {
        "description": "A supports B (reinforcement relationship)",
        "valid_patterns": ["learning → decision", "observation → learning", "pattern → learning"]
    },
    "follows": {
        "description": "A follows B (temporal/sequential relationship)",
        "valid_patterns": ["observation → observation", "decision → decision", "any → any"]
    },
    "related": {
        "description": "A is related to B (generic association)",
        "valid_patterns": ["any → any"]
    }
}


def validate_memory_type(memory_type: str) -> bool:
    """
    Validate if a memory type is in the ontology (base type or subtype).

    Args:
        memory_type: The type string to validate

    Returns:
        True if the type is valid (exists in base types or subtypes), False otherwise

    Examples:
        >>> validate_memory_type("observation")  # Base type
        True
        >>> validate_memory_type("code_edit")    # Subtype
        True
        >>> validate_memory_type("invalid")
        False
    """
    # Check if it's a base type
    base_types = {member.value for member in BaseMemoryType}
    if memory_type in base_types:
        return True

    # Check if it's a subtype
    all_subtypes = []
    for subtypes in TAXONOMY.values():
        all_subtypes.extend(subtypes)

    return memory_type in all_subtypes


def get_parent_type(subtype: str) -> Optional[str]:
    """
    Get the parent base type for a subtype. Returns itself if already a base type.

    Args:
        subtype: The subtype (or base type) to look up

    Returns:
        Parent base type string, or None if subtype is invalid

    Examples:
        >>> get_parent_type("code_edit")  # Subtype
        'observation'
        >>> get_parent_type("observation")  # Base type returns itself
        'observation'
        >>> get_parent_type("invalid")
        None
    """
    # Check if it's already a base type
    base_types = {member.value for member in BaseMemoryType}
    if subtype in base_types:
        return subtype

    # Look up parent from taxonomy
    for base_type, subtypes in TAXONOMY.items():
        if subtype in subtypes:
            return base_type

    return None


def get_all_types() -> List[str]:
    """
    Get flattened list of all valid memory types (base + subtypes).

    Returns:
        List of all type strings in the ontology

    Examples:
        >>> types = get_all_types()
        >>> "observation" in types
        True
        >>> "code_edit" in types
        True
        >>> len(types)  # 5 base + 21 subtypes
        26
    """
    # Get all base types
    all_types = [member.value for member in BaseMemoryType]

    # Add all subtypes
    for subtypes in TAXONOMY.values():
        all_types.extend(subtypes)

    return all_types
