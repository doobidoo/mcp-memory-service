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

from typing import Final


# Namespace constants - each ends with ":" for easy concatenation
NAMESPACE_SYSTEM: Final[str] = "sys:"       # System-generated tags
NAMESPACE_QUALITY: Final[str] = "q:"        # Quality scores (q:high, q:medium, q:low)
NAMESPACE_PROJECT: Final[str] = "proj:"     # Project/repository context
NAMESPACE_TOPIC: Final[str] = "topic:"      # Subject matter topics
NAMESPACE_TEMPORAL: Final[str] = "t:"       # Time-based tags (t:2024-01, t:sprint-3)
NAMESPACE_USER: Final[str] = "user:"        # User-defined custom tags
