"""
Tag Taxonomy with Namespaces

Provides structured tag organization using namespace prefixes for better
categorization and filtering. Part of Phase 0: Ontology Foundation.

Namespace format: "namespace:value" (e.g., "q:high", "proj:authentication")

Usage:
    from mcp_memory_service.models.tag_taxonomy import TagTaxonomy

    # Parse tag
    namespace, value = TagTaxonomy.parse_tag("q:high")  # ("q:", "high")

    # Add namespace
    tag = TagTaxonomy.add_namespace("high", "q:")  # "q:high"
"""

import logging
import os
import re
from typing import Final, Tuple, Optional, List, Set

# No package imports: tests load this module standalone (see tests/test_tag_taxonomy.py).
logger = logging.getLogger(__name__)


# Namespace constants - each ends with ":" for easy concatenation
NAMESPACE_SYSTEM: Final[str] = "sys:"       # System-generated tags
NAMESPACE_QUALITY: Final[str] = "q:"        # Quality scores (q:high, q:medium, q:low)
NAMESPACE_PROJECT: Final[str] = "proj:"     # Project/repository context
NAMESPACE_TOPIC: Final[str] = "topic:"      # Subject matter topics
NAMESPACE_TEMPORAL: Final[str] = "t:"       # Time-based tags (t:2024-01, t:sprint-3)
NAMESPACE_USER: Final[str] = "user:"        # User-defined custom tags
NAMESPACE_AGENT: Final[str] = "agent:"      # Agent identity tags (agent:<id>, crew:<name>)
# Namespaces the server's own code writes
NAMESPACE_HARVEST: Final[str] = "harvest:"        # Session harvest (harvest:<type>, harvest:method:<m>)
NAMESPACE_SOURCE: Final[str] = "source:"          # Origin of a memory (source:<hostname>, source:<parent hash>)
NAMESPACE_CONFLICT: Final[str] = "conflict:"      # Conflict state (conflict:unresolved)
NAMESPACE_CONFIDENCE: Final[str] = "confidence:"  # Confidence stamp from session analysis (confidence:<0-100>)

# Operator-registered namespaces, comma-separated: MCP_CUSTOM_TAG_NAMESPACES="project,cleanup"
CUSTOM_NAMESPACES_ENV: Final[str] = "MCP_CUSTOM_TAG_NAMESPACES"
_NAMESPACE_NAME = re.compile(r"^[a-z0-9_-]+$")


def parse_tag(tag: str) -> Tuple[Optional[str], str]:
    """
    Parse a tag into namespace and value components.

    Args:
        tag: The tag string to parse

    Returns:
        Tuple of (namespace, value) if namespaced, or (None, tag) for legacy tags

    Examples:
        >>> parse_tag("q:high")
        ("q:", "high")
        >>> parse_tag("legacy-tag")
        (None, "legacy-tag")
        >>> parse_tag("topic:authentication")
        ("topic:", "authentication")
    """
    if ":" in tag:
        parts = tag.split(":", 1)  # Split on first colon only
        namespace = parts[0] + ":"
        value = parts[1]
        return (namespace, value)
    else:
        # Legacy tag without namespace
        return (None, tag)


# Valid namespaces list for validation
VALID_NAMESPACES: Final[List[str]] = [
    NAMESPACE_SYSTEM,
    NAMESPACE_QUALITY,
    NAMESPACE_PROJECT,
    NAMESPACE_TOPIC,
    NAMESPACE_TEMPORAL,
    NAMESPACE_USER,
    NAMESPACE_AGENT,
    NAMESPACE_HARVEST,
    NAMESPACE_SOURCE,
    NAMESPACE_CONFLICT,
    NAMESPACE_CONFIDENCE,
]


def parse_custom_namespaces(raw: Optional[str]) -> Set[str]:
    """Namespaces from a comma-separated list such as ``"project, cleanup:"``.

    Names are lowercased and given a trailing ``:``; an entry that is not a plain
    name (letters, digits, ``_``, ``-``) is skipped with a warning.
    """
    found: Set[str] = set()
    for entry in (raw or "").split(","):
        name = entry.strip().lower().removesuffix(":")
        if not name:
            continue
        if _NAMESPACE_NAME.match(name):
            found.add(f"{name}:")
        else:
            # %r escapes newlines and control characters in the operator's value
            logger.warning("Ignoring invalid tag namespace in %s: %r", CUSTOM_NAMESPACES_ENV, entry)
    return found


_env_namespaces_registered = False


def ensure_env_namespaces() -> None:
    """Register MCP_CUSTOM_TAG_NAMESPACES once, on first use.

    Read lazily rather than at import: importing the package loads Memory, and
    with it this module, before config.base loads .env, so a value set only in
    .env would be missed.
    """
    global _env_namespaces_registered
    if not _env_namespaces_registered:
        _env_namespaces_registered = True
        register_custom_namespaces(os.getenv(CUSTOM_NAMESPACES_ENV))


def register_custom_namespaces(raw: Optional[str]) -> Set[str]:
    """Add namespaces to the valid set at runtime; returns the ones added."""
    added = parse_custom_namespaces(raw) - set(VALID_NAMESPACES)
    VALID_NAMESPACES.extend(sorted(added))
    TagTaxonomy.VALID_NAMESPACES.update(added)
    return added


def validate_tag(tag: str) -> bool:
    """
    Validate if a tag has a valid namespace or is a legacy tag.

    Args:
        tag: The tag string to validate

    Returns:
        True if tag has valid namespace OR is legacy format, False otherwise

    Examples:
        >>> validate_tag("q:high")  # Valid namespace
        True
        >>> validate_tag("legacy-tag")  # Legacy format (no namespace)
        True
        >>> validate_tag("invalid:tag")  # Invalid namespace
        False
    """
    ensure_env_namespaces()
    namespace, _ = parse_tag(tag)

    # Legacy tags (no namespace) are valid for backward compatibility
    if namespace is None:
        return True

    # Check if namespace is in valid list
    return namespace in VALID_NAMESPACES


def add_namespace(value: str, namespace: str) -> str:
    """
    Add namespace prefix to a value, creating a namespaced tag.

    Strips existing namespace from value if present before adding new one.

    Args:
        value: The tag value
        namespace: The namespace to add (should include trailing ":")

    Returns:
        Formatted "namespace:value" tag

    Examples:
        >>> add_namespace("high", "q:")
        "q:high"
        >>> add_namespace("q:high", "proj:")  # Strips existing namespace
        "proj:high"
    """
    # Strip existing namespace if present
    _, clean_value = parse_tag(value)
    return f"{namespace}{clean_value}"


def filter_by_namespace(tags: List[str], namespace: str) -> List[str]:
    """
    Filter tags to only those matching the specified namespace.

    Args:
        tags: List of tags to filter
        namespace: The namespace to match (should include trailing ":")

    Returns:
        List of tags with matching namespace

    Examples:
        >>> filter_by_namespace(["q:high", "proj:auth", "q:medium"], "q:")
        ["q:high", "q:medium"]
        >>> filter_by_namespace(["legacy", "q:high"], "q:")
        ["q:high"]
    """
    filtered = []
    for tag in tags:
        tag_namespace, _ = parse_tag(tag)
        if tag_namespace == namespace:
            filtered.append(tag)
    return filtered


class TagTaxonomy:
    """
    Tag Taxonomy - Namespace-based tag organization system.

    Provides structured tag management with namespace prefixes for better
    categorization and filtering. All methods are class methods for stateless access.

    Usage:
        # Parse tag
        namespace, value = TagTaxonomy.parse_tag("q:high")

        # Validate tag
        is_valid = TagTaxonomy.validate_tag("q:high")  # True

        # Add namespace
        tag = TagTaxonomy.add_namespace("high", "q:")

        # Filter by namespace
        quality_tags = TagTaxonomy.filter_by_namespace(tags, "q:")
    """

    # Expose valid namespaces for efficient external access (same members as the
    # module-level list, including operator-registered ones)
    VALID_NAMESPACES: Final[Set[str]] = set(VALID_NAMESPACES)

    @classmethod
    def parse_tag(cls, tag: str) -> Tuple[Optional[str], str]:
        """Parse tag into namespace and value."""
        return parse_tag(tag)

    @classmethod
    def is_valid_namespace(cls, namespace: str) -> bool:
        """Whether a parsed namespace (with trailing ':') is registered."""
        ensure_env_namespaces()
        return namespace in cls.VALID_NAMESPACES

    @classmethod
    def validate_tag(cls, tag: str) -> bool:
        """Validate if tag has valid namespace or is legacy format."""
        return validate_tag(tag)

    @classmethod
    def add_namespace(cls, value: str, namespace: str) -> str:
        """Add namespace prefix to value."""
        return add_namespace(value, namespace)

    @classmethod
    def filter_by_namespace(cls, tags: List[str], namespace: str) -> List[str]:
        """Filter tags to only those matching namespace."""
        return filter_by_namespace(tags, namespace)

