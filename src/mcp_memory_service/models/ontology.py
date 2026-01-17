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


# Module-level caches for performance optimization
_ALL_TYPES_CACHE: Optional[List[str]] = None
_PARENT_TYPE_MAP_CACHE: Optional[Dict[str, str]] = None
_BASE_TYPES_CACHE: Optional[set] = None


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
    global _BASE_TYPES_CACHE

    # Initialize base types cache on first access
    if _BASE_TYPES_CACHE is None:
        _BASE_TYPES_CACHE = {member.value for member in BaseMemoryType}

    # Check if it's a base type
    if memory_type in _BASE_TYPES_CACHE:
        return True

    # Check if it's a subtype - leverage get_all_types cache
    all_types = get_all_types()
    return memory_type in all_types


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
    global _PARENT_TYPE_MAP_CACHE, _BASE_TYPES_CACHE

    # Initialize caches on first access
    if _BASE_TYPES_CACHE is None:
        _BASE_TYPES_CACHE = {member.value for member in BaseMemoryType}

    if _PARENT_TYPE_MAP_CACHE is None:
        # Build reverse lookup map: subtype → parent
        _PARENT_TYPE_MAP_CACHE = {}

        # Base types map to themselves
        for base_type in _BASE_TYPES_CACHE:
            _PARENT_TYPE_MAP_CACHE[base_type] = base_type

        # Subtypes map to their parent
        for base_type, subtypes in TAXONOMY.items():
            for st in subtypes:
                _PARENT_TYPE_MAP_CACHE[st] = base_type

    # Return cached lookup result
    return _PARENT_TYPE_MAP_CACHE.get(subtype)


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
    global _ALL_TYPES_CACHE

    # Initialize cache on first access
    if _ALL_TYPES_CACHE is None:
        # Get all base types
        all_types = [member.value for member in BaseMemoryType]

        # Add all subtypes
        for subtypes in TAXONOMY.values():
            all_types.extend(subtypes)

        _ALL_TYPES_CACHE = all_types

    # Return copy to prevent mutation of cached list
    return _ALL_TYPES_CACHE.copy()


def validate_relationship(rel_type: str) -> bool:
    """
    Validate if a relationship type is in the allowed list.

    Args:
        rel_type: The relationship type string to validate

    Returns:
        True if relationship type is valid, False otherwise

    Examples:
        >>> validate_relationship("causes")
        True
        >>> validate_relationship("fixes")
        True
        >>> validate_relationship("invalid_rel")
        False
    """
    return rel_type in RELATIONSHIPS


class MemoryTypeOntology:
    """
    Memory Type Ontology - Formal type system for knowledge graph classification.

    Provides controlled vocabulary, validation, and type hierarchy management
    for semantic memory classification. All methods are class methods for
    stateless, pure functional access to the ontology.

    Usage:
        # Validate memory type
        if MemoryTypeOntology.validate_memory_type("observation"):
            print("Valid type")

        # Get parent type
        parent = MemoryTypeOntology.get_parent_type("code_edit")  # → "observation"

        # Get all types
        all_types = MemoryTypeOntology.get_all_types()

        # Validate relationship
        if MemoryTypeOntology.validate_relationship("causes"):
            print("Valid relationship")
    """

    @classmethod
    def validate_memory_type(cls, memory_type: str) -> bool:
        """Validate if memory type exists in ontology."""
        return validate_memory_type(memory_type)

    @classmethod
    def get_parent_type(cls, subtype: str) -> Optional[str]:
        """Get parent base type for a subtype."""
        return get_parent_type(subtype)

    @classmethod
    def get_all_types(cls) -> List[str]:
        """Get flattened list of all types."""
        return get_all_types()

    @classmethod
    def validate_relationship(cls, rel_type: str) -> bool:
        """Validate if relationship type is valid."""
        return validate_relationship(rel_type)
