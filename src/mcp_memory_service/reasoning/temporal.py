"""Temporal edges for graph associations (RFC #1008 §4).

Adds valid_from/valid_until to associations for point-in-time queries.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class TemporalEdge:
    source: str
    target: str
    valid_from: Optional[float] = None
    valid_until: Optional[float] = None

    def __post_init__(self):
        if self.valid_from is not None and self.valid_until is not None:
            if self.valid_from > self.valid_until:
                raise ValueError("valid_from cannot be greater than valid_until")


async def store_temporal_association(
    graph, source_hash: str, target_hash: str,
    similarity: float, connection_types: List[str],
    relationship_type: str = "related",
    valid_from: Optional[float] = None,
    valid_until: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Store association with temporal bounds in metadata."""
    if valid_from is not None and valid_until is not None and valid_from > valid_until:
        raise ValueError("valid_from cannot be greater than valid_until")
    meta = dict(metadata) if metadata else {}
    if valid_from is not None:
        meta["valid_from"] = valid_from
    if valid_until is not None:
        meta["valid_until"] = valid_until
    return await graph.store_association(
        source_hash=source_hash, target_hash=target_hash,
        similarity=similarity, connection_types=connection_types,
        relationship_type=relationship_type, metadata=meta,
    )


def filter_temporal_edges(edges: List[TemporalEdge], as_of: Optional[float] = None) -> List[TemporalEdge]:
    """Filter edges to those valid at as_of timestamp."""
    if as_of is None:
        return edges
    result = []
    for e in edges:
        if e.valid_from is not None and e.valid_from > as_of:
            continue
        if e.valid_until is not None and e.valid_until < as_of:
            continue
        result.append(e)
    return result


def classify_temporal_relationship(edge_a: TemporalEdge, edge_b: TemporalEdge) -> str:
    """Classify if two edges represent evolution or contradiction."""
    if edge_a.valid_until is not None and edge_b.valid_from is not None:
        if edge_a.valid_until <= edge_b.valid_from:
            return "evolution"
    return "contradiction"
