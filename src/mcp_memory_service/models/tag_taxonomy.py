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

from typing import Final, Tuple, Optional, List


# Namespace constants - each ends with ":" for easy concatenation
NAMESPACE_SYSTEM: Final[str] = "sys:"       # System-generated tags
NAMESPACE_QUALITY: Final[str] = "q:"        # Quality scores (q:high, q:medium, q:low)
NAMESPACE_PROJECT: Final[str] = "proj:"     # Project/repository context
NAMESPACE_TOPIC: Final[str] = "topic:"      # Subject matter topics
NAMESPACE_TEMPORAL: Final[str] = "t:"       # Time-based tags (t:2024-01, t:sprint-3)
NAMESPACE_USER: Final[str] = "user:"        # User-defined custom tags


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
    NAMESPACE_USER
]


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
    namespace, _ = parse_tag(tag)

    # Legacy tags (no namespace) are valid for backward compatibility
    if namespace is None:
        return True

    # Check if namespace is in valid list
    return namespace in VALID_NAMESPACES
