"""Ontology relationship pair allow-list (issue #1458).

``RELATIONSHIPS[*][\"valid_patterns\"]`` is the only source of truth for which
parent memory types a relationship may connect. ``any`` is a wildcard.
"""

from typing import Dict, List, Optional

from mcp_memory_service.models._ontology_data import (
    RELATIONSHIPS,
    canonicalize_memory_type,
)

_VALID_PAIR_CACHE: Optional[Dict[str, List[tuple]]] = None


def clear_pair_cache() -> None:
    """Drop the parsed valid_patterns cache."""
    global _VALID_PAIR_CACHE
    _VALID_PAIR_CACHE = None


def _valid_relationship_pairs() -> Dict[str, List[tuple]]:
    """Parse valid_patterns once. Each pattern is \"<source> \u2192 <target>\"."""
    global _VALID_PAIR_CACHE
    if _VALID_PAIR_CACHE is None:
        parsed: Dict[str, List[tuple]] = {}
        for rel_type, spec in RELATIONSHIPS.items():
            pairs = []
            for pattern in spec.get("valid_patterns", []):
                parts = [part.strip() for part in pattern.replace("->", "\u2192").split("\u2192")]
                if len(parts) != 2 or not parts[0] or not parts[1]:
                    continue
                pairs.append((parts[0], parts[1]))
            parsed[rel_type] = pairs
        _VALID_PAIR_CACHE = parsed
    return _VALID_PAIR_CACHE


def is_allowed_pair(
    rel_type: str,
    source_parent: Optional[str],
    target_parent: Optional[str],
) -> bool:
    """Return whether rel_type may connect these parent memory types.

    Parents should already be resolved (learning/insight counts as learning).
    ``any`` matches every parent, including a missing one. Unknown relationship
    types are never allowed.
    """
    patterns = _valid_relationship_pairs().get(rel_type)
    if not patterns:
        return False
    source = canonicalize_memory_type(source_parent) if source_parent else None
    target = canonicalize_memory_type(target_parent) if target_parent else None
    for left, right in patterns:
        left_ok = left == "any" or (source is not None and source == left)
        right_ok = right == "any" or (target is not None and target == right)
        if left_ok and right_ok:
            return True
    return False
