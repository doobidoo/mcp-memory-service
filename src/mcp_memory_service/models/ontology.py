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

import json
import logging
import os
import re
import time
from enum import Enum
from typing import Dict, List, Optional, Final, Tuple


# Module-level caches for performance optimization
_ALL_TYPES_CACHE: Optional[List[str]] = None
_PARENT_TYPE_MAP_CACHE: Optional[Dict[str, str]] = None
_BASE_TYPES_CACHE: Optional[set] = None
_MERGED_TAXONOMY_CACHE: Optional[Dict[str, List[str]]] = None
_ALLOWED_PAIRS_CACHE: Optional[Dict[str, set]] = None


def clear_ontology_caches():
    """Clear all ontology caches. Useful for testing and dynamic configuration changes."""
    global _ALL_TYPES_CACHE, _PARENT_TYPE_MAP_CACHE, _BASE_TYPES_CACHE, _MERGED_TAXONOMY_CACHE
    global _ALLOWED_PAIRS_CACHE
    _ALL_TYPES_CACHE = None
    _PARENT_TYPE_MAP_CACHE = None
    _BASE_TYPES_CACHE = None
    _MERGED_TAXONOMY_CACHE = None
    _ALLOWED_PAIRS_CACHE = None


# DB overlay (#1897). The sqlite_vec storage registers a source with two
# methods, read_ontology_version() and read_memory_type_overlay(). Readers poll
# the version and reload the overlay when it moved, so a write by the HTTP
# server reaches the MCP server sharing the same database file.
_OVERLAY_SOURCE = None
_OVERLAY_ENTRIES: List[Tuple[str, Optional[str]]] = []
_OVERLAY_VERSION: Optional[int] = None
_OVERLAY_CHECKED_AT = float("-inf")
# ponytail: polled at most once per second, so a write in another process can
# take that long to show here. Lower it, or push a notification, if that matters.
_OVERLAY_POLL_SECONDS = 1.0


def set_overlay_source(source) -> None:
    """Register the DB overlay source. The last storage to register wins."""
    global _OVERLAY_SOURCE, _OVERLAY_ENTRIES, _OVERLAY_VERSION
    _OVERLAY_SOURCE = source
    _OVERLAY_ENTRIES = []
    _OVERLAY_VERSION = None
    invalidate_overlay()


def clear_overlay_source(source) -> None:
    """Unregister ``source`` if it is still the registered one."""
    if source is not None and source is _OVERLAY_SOURCE:
        set_overlay_source(None)


def invalidate_overlay() -> None:
    """Make the next read check the overlay version (after a write in this process)."""
    global _OVERLAY_CHECKED_AT
    _OVERLAY_CHECKED_AT = float("-inf")
    clear_ontology_caches()


def _refresh_overlay() -> None:
    """Reload the overlay when the stored ontology_version has moved."""
    global _OVERLAY_ENTRIES, _OVERLAY_VERSION, _OVERLAY_CHECKED_AT
    source = _OVERLAY_SOURCE
    now = time.monotonic()
    if source is None or now - _OVERLAY_CHECKED_AT < _OVERLAY_POLL_SECONDS:
        return
    _OVERLAY_CHECKED_AT = now
    try:
        version = source.read_ontology_version()
        if version == _OVERLAY_VERSION:
            return
        entries = list(source.read_memory_type_overlay())
    except Exception as e:  # keep the last overlay we had; retry on the next poll
        safe_error = str(e).replace('\r', '').replace('\n', '').replace('\x1b', '')
        logging.getLogger(__name__).warning("Could not read the ontology overlay: %s", safe_error)
        return
    _OVERLAY_ENTRIES = entries
    _OVERLAY_VERSION = version
    clear_ontology_caches()


def _merge_overlay(merged: Dict[str, List[str]]) -> None:
    """Add the DB overlay to ``merged`` in place. Additive only.

    A name that already exists is skipped, so the overlay can neither shadow
    nor re-parent a built-in or env type. A subtype whose parent is not a base
    type here (removed, or never known on this node) is skipped too.
    """
    known = set(merged).union(*merged.values())
    # Base types first, so a subtype can hang off a base type from the overlay.
    for key, parent in sorted(_OVERLAY_ENTRIES, key=lambda entry: entry[1] is not None):
        if key in known or not _is_valid_custom_type_name(key):
            continue
        if parent is None:
            merged[key] = []
        elif parent in merged:
            merged[parent].append(key)
        else:
            continue
        known.add(key)


class BaseMemoryType(str, Enum):
    """
    Base memory types forming the top-level ontology.

    These are the fundamental categories that all memories belong to.
    Each base type can have multiple subtypes for finer-grained classification.
    """
    # Software Development (original 5 types)
    OBSERVATION = "observation"
    DECISION = "decision"
    LEARNING = "learning"
    ERROR = "error"
    PATTERN = "pattern"

    # Project Management - Agile (2 new types)
    PLANNING = "planning"
    CEREMONY = "ceremony"

    # Project Management - Traditional (2 new types)
    MILESTONE = "milestone"
    STAKEHOLDER = "stakeholder"

    # General Knowledge Work (3 new types)
    MEETING = "meeting"
    RESEARCH = "research"
    COMMUNICATION = "communication"


# Taxonomy hierarchy: base types → subtypes
TAXONOMY: Final[Dict[str, List[str]]] = {
    # Software Development (26 types)
    "observation": [
        "code_edit",
        "file_access",
        "search",
        "command",
        "conversation",
        "conversation_turn",
        "session",
        "document",
        "note",
        "reference",
        "user_correction",
        "tool_outcome",
        "preference_signal",
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
        "timeout",
        "mistake"
    ],
    "pattern": [
        "recurring_issue",
        "code_smell",
        "design_pattern",
        "workflow"
    ],

    # Project Management - Agile (12 types)
    "planning": [
        "sprint_goal",
        "backlog_item",
        "story_point_estimate",
        "velocity",
        "retrospective",
        "standup_note",
        "acceptance_criteria"
    ],
    "ceremony": [
        "sprint_review",
        "sprint_planning",
        "daily_standup",
        "retrospective_action",
        "demo_feedback"
    ],

    # Project Management - Traditional (12 types)
    "milestone": [
        "deliverable",
        "dependency",
        "risk",
        "constraint",
        "assumption",
        "deadline"
    ],
    "stakeholder": [
        "requirement",
        "feedback",
        "escalation",
        "approval",
        "change_request",
        "status_update"
    ],

    # General Knowledge Work (18 types)
    "meeting": [
        "action_item",
        "attendee_note",
        "agenda_item",
        "follow_up",
        "minutes"
    ],
    "research": [
        "finding",
        "comparison",
        "recommendation",
        "source",
        "hypothesis"
    ],
    "communication": [
        "email_summary",
        "chat_summary",
        "announcement",
        "request",
        "response"
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
    },
    "derived_from": {
        "description": "Target was extracted or synthesized from source",
        "valid_patterns": ["any → any"]
    },
    "shares_entity": {
        "description": "A and B share a common entity (auto-linked)",
        "valid_patterns": ["any → any"]
    }
}

# Symmetric relationships (bidirectional semantics)
SYMMETRIC_RELATIONSHIPS: Final[set] = {"related", "contradicts", "shares_entity"}


def canonicalize_memory_type(memory_type: str) -> str:
    """Return the canonical form of a memory type: stripped and lowercased.

    Type names are labels, not Python identifiers, and callers disagree on
    capitalization — the bundled hooks send ``Decision`` while the REST and MCP
    examples use ``decision``. Registration already lowercased custom base
    types while validation compared the raw string, so a capitalized type could
    never be registered at all (issue #176). Canonicalizing on both sides keeps
    them the same type.
    """
    if not isinstance(memory_type, str):
        return memory_type
    return memory_type.strip().lower()


# Custom base type and subtype names: lowercase letters, digits, underscore and
# hyphen, starting with a letter. `str.isidentifier()` used to guard this, which
# rejected every hyphenated name (`session-summary`) even though nothing here
# needs a valid Python identifier.
_CUSTOM_TYPE_NAME_RE = re.compile(r'^[a-z][a-z0-9_-]*$')


def _is_valid_custom_type_name(name: str) -> bool:
    """Check a canonicalized custom type name against the allowed character set."""
    return isinstance(name, str) and bool(_CUSTOM_TYPE_NAME_RE.match(name))


def _load_custom_types_from_config() -> Dict[str, List[str]]:
    """Load custom memory types from MCP_CUSTOM_MEMORY_TYPES environment variable.

    Expected format: JSON dict mapping base types to list of subtypes
    Example: {"planning": ["sprint_goal", "backlog_item"], "meeting": ["action_item"]}

    Returns:
        Dict mapping base type names to lists of subtype names
    """
    # Try to import project logger, fall back to standard logging if not available
    try:
        from mcp_memory_service.logger import logger
    except ImportError:
        logger = logging.getLogger(__name__)

    # Type names come from an environment variable and end up in log lines, so
    # they are sanitized before logging. The fallback mirrors compat's stripping
    # for the standalone module load used by tests/test_ontology.py.
    try:
        from mcp_memory_service.compat import _sanitize_log_value
    except ImportError:
        def _sanitize_log_value(value):
            return re.sub(r'[\r\n\x1b]', '', str(value))

    custom_types_json = os.getenv('MCP_CUSTOM_MEMORY_TYPES')
    if not custom_types_json:
        return {}

    try:
        custom_types = json.loads(custom_types_json)

        # Validation
        if not isinstance(custom_types, dict):
            logger.error("MCP_CUSTOM_MEMORY_TYPES must be a JSON object/dict")
            return {}

        validated_types = {}
        for raw_base_type, subtypes in custom_types.items():
            # Validate base type name (canonicalized, so `Decision` and
            # `decision` register as the same type — issue #176)
            base_type = canonicalize_memory_type(raw_base_type)
            safe_name = _sanitize_log_value(raw_base_type)
            if not _is_valid_custom_type_name(base_type):
                logger.warning(
                    f"Invalid base type name '{safe_name}', skipping. Names may "
                    f"contain lowercase letters, digits, underscore and hyphen, "
                    f"and must start with a letter."
                )
                continue

            # Validate subtypes
            if not isinstance(subtypes, list):
                logger.warning("Subtypes for '%s' must be a list, skipping", safe_name)
                continue

            valid_subtypes = [
                canonicalize_memory_type(st) for st in subtypes
                if _is_valid_custom_type_name(canonicalize_memory_type(st))
            ]

            if base_type in TAXONOMY:
                logger.warning(
                    f"Custom memory type '{safe_name}' matches the built-in type "
                    f"'{base_type}'; its subtypes are merged into the built-in type. "
                    f"Registering it is not needed — validation is case-insensitive."
                )

            # Register the base type even when no (valid) subtypes were provided
            # so that callers can use `{"foo": []}` to add a bare custom type.
            # Issue #842: empty subtype lists were silently dropped.
            if subtypes and not valid_subtypes:
                logger.warning(
                    f"All subtypes for custom memory type '{base_type}' were invalid; "
                    f"registering base type with no subtypes"
                )
            validated_types[base_type] = valid_subtypes
            logger.info(
                f"Loaded custom memory type '{base_type}' with {len(valid_subtypes)} subtypes"
            )

        return validated_types

    except json.JSONDecodeError as e:
        logger.error("Failed to parse MCP_CUSTOM_MEMORY_TYPES: %s", _sanitize_log_value(e))
        return {}


def _get_merged_taxonomy() -> Dict[str, List[str]]:
    """Get merged taxonomy combining built-in and custom types.

    Returns:
        Dict mapping all base types to their subtypes
    """
    global _MERGED_TAXONOMY_CACHE

    # Return cached version if available
    if _MERGED_TAXONOMY_CACHE is not None:
        return _MERGED_TAXONOMY_CACHE

    # Try to import project logger, fall back to standard logging if not available
    try:
        from mcp_memory_service.logger import logger
    except ImportError:
        logger = logging.getLogger(__name__)

    # Start with built-in taxonomy. Copy the subtype lists too: extending them in
    # place below would otherwise write custom subtypes into TAXONOMY itself.
    merged = {base: list(subtypes) for base, subtypes in TAXONOMY.items()}

    # Load and merge custom types
    custom_types = _load_custom_types_from_config()
    for base_type, subtypes in custom_types.items():
        if base_type in merged:
            # Merge subtypes, avoiding duplicates
            existing = set(merged[base_type])
            new_subtypes = [st for st in subtypes if st not in existing]
            merged[base_type].extend(new_subtypes)
            logger.info("Extended '%s' with %d custom subtypes", base_type, len(new_subtypes))
        else:
            # New base type
            merged[base_type] = subtypes
            logger.info("Added new custom base type '%s' with %d subtypes", base_type, len(subtypes))

    # Then the DB overlay (#1897): built-in, then env, then DB.
    _merge_overlay(merged)

    # Cache the merged taxonomy
    _MERGED_TAXONOMY_CACHE = merged

    return _MERGED_TAXONOMY_CACHE


def get_base_types() -> set:
    """
    Get the set of all base (top-level) memory types.

    Returns:
        Set of base type strings (keys of the merged taxonomy)

    Examples:
        >>> base = get_base_types()
        >>> "observation" in base
        True
        >>> "code_edit" in base  # subtype, not a base type
        False
    """
    global _BASE_TYPES_CACHE

    _refresh_overlay()
    if _BASE_TYPES_CACHE is not None:
        return _BASE_TYPES_CACHE.copy()

    taxonomy = _get_merged_taxonomy()
    _BASE_TYPES_CACHE = set(taxonomy.keys())
    return _BASE_TYPES_CACHE.copy()


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
    # Use get_all_types which includes both built-in and custom types.
    # Compare canonicalized, so `Decision` and `decision` are the same type
    # (issue #176) — the taxonomy itself is stored canonicalized.
    all_types = get_all_types()
    return canonicalize_memory_type(memory_type) in all_types


def coerce_memory_type(memory_type: Optional[str]) -> Optional[str]:
    """Return the canonical type for a new write, or ``observation`` if unknown.

    Call this where a caller-supplied type enters a NEW memory (store paths).
    ``Memory`` itself no longer coerces: hydrating a stored row on a node that
    lacks a custom type must keep the type, or the next update writes
    ``observation`` back over it.
    """
    if memory_type is None:
        return None
    if validate_memory_type(memory_type):
        return canonicalize_memory_type(memory_type)
    # Inline replace() barriers: CodeQL does not follow the try-import/fallback
    # sanitizer used elsewhere in this module (alert 641).
    safe_type = str(memory_type).replace('\r', '').replace('\n', '').replace('\x1b', '')
    logging.getLogger(__name__).warning(
        "Invalid memory_type '%s'. Valid types: %s... Defaulting to 'observation'.",
        safe_type,
        ', '.join(get_all_types()[:5]),
    )
    return "observation"


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
    global _PARENT_TYPE_MAP_CACHE

    _refresh_overlay()
    if _PARENT_TYPE_MAP_CACHE is None:
        # Build reverse lookup map: subtype → parent
        _PARENT_TYPE_MAP_CACHE = {}

        # Use merged taxonomy instead of TAXONOMY
        taxonomy = _get_merged_taxonomy()

        # Base types (both built-in and custom) map to themselves
        for base_type in taxonomy.keys():
            _PARENT_TYPE_MAP_CACHE[base_type] = base_type

        # Subtypes map to their parent
        for base_type, subtypes in taxonomy.items():
            for st in subtypes:
                _PARENT_TYPE_MAP_CACHE[st] = base_type

    # Return cached lookup result
    return _PARENT_TYPE_MAP_CACHE.get(canonicalize_memory_type(subtype))


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
        >>> len(types)  # 12 base + 63 subtypes (75 built-in, more with custom types)
        75
    """
    global _ALL_TYPES_CACHE

    _refresh_overlay()
    # Initialize cache on first access
    if _ALL_TYPES_CACHE is None:
        # Use merged taxonomy instead of TAXONOMY
        taxonomy = _get_merged_taxonomy()

        # Get all base types from merged taxonomy (includes custom base types)
        all_types = list(taxonomy.keys())

        # Add all subtypes
        for subtypes in taxonomy.values():
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


def is_allowed_pair(rel_type: str, source_parent: Optional[str], target_parent: Optional[str]) -> bool:
    """
    Check a relationship against the source → target pairs in its valid_patterns.

    Parents are base types (resolve subtypes first). "any" matches every parent,
    including an unknown one (None); a named type never matches None.

    Examples:
        >>> is_allowed_pair("fixes", "learning", "error")
        True
        >>> is_allowed_pair("fixes", "observation", "error")
        False
        >>> is_allowed_pair("related", None, None)
        True
    """
    global _ALLOWED_PAIRS_CACHE

    if _ALLOWED_PAIRS_CACHE is None:
        _ALLOWED_PAIRS_CACHE = {
            name: {tuple(part.strip() for part in pattern.split("→")) for pattern in spec["valid_patterns"]}
            for name, spec in RELATIONSHIPS.items()
        }

    pairs = _ALLOWED_PAIRS_CACHE.get(rel_type, set())
    return any(
        src in ("any", source_parent) and tgt in ("any", target_parent)
        for src, tgt in pairs
    )


def is_symmetric_relationship(rel_type: str) -> bool:
    """
    Determine if a relationship type is symmetric (bidirectional).

    Symmetric relationships work the same in both directions:
    - related: A related to B implies B related to A
    - contradicts: A contradicts B implies B contradicts A

    Asymmetric relationships have directionality:
    - causes: A causes B does NOT imply B causes A
    - fixes: A fixes B does NOT imply B fixes A
    - supports: A supports B does NOT imply B supports A
    - follows: A follows B does NOT imply B follows A

    Args:
        rel_type: The relationship type string to check

    Returns:
        True if symmetric (bidirectional storage correct), False if asymmetric

    Examples:
        >>> is_symmetric_relationship("related")
        True
        >>> is_symmetric_relationship("causes")
        False
        >>> is_symmetric_relationship("contradicts")
        True

    Raises:
        ValueError: If rel_type is not a valid relationship type
    """
    # Validate input first
    if not validate_relationship(rel_type):
        raise ValueError(f"Invalid relationship type: '{rel_type}'")

    return rel_type in SYMMETRIC_RELATIONSHIPS


class MemoryTypeOntology:
    """
    Memory Type Ontology - Formal type system for knowledge graph classification.

    Provides controlled vocabulary, validation, and type hierarchy management
    for semantic memory classification. All methods are class methods for
    stateless, pure functional access to the ontology.

    Usage:
        # Validate memory type
        is_valid = MemoryTypeOntology.validate_memory_type("observation")  # → True

        # Get parent type
        parent = MemoryTypeOntology.get_parent_type("code_edit")  # → "observation"

        # Get all types
        all_types = MemoryTypeOntology.get_all_types()

        # Validate relationship
        is_valid = MemoryTypeOntology.validate_relationship("causes")  # → True
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
    def get_base_types(cls) -> set:
        """Get set of all base (top-level) memory types."""
        return get_base_types()

    @classmethod
    def validate_relationship(cls, rel_type: str) -> bool:
        """Validate if relationship type is valid."""
        return validate_relationship(rel_type)

    @classmethod
    def is_symmetric_relationship(cls, rel_type: str) -> bool:
        """Check if relationship type is symmetric (bidirectional)."""
        return is_symmetric_relationship(rel_type)
