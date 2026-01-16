"""
Memory Association Models

Dataclasses for typed memory associations in the knowledge graph.
Part of Phase 0: Ontology Foundation (Component 3).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class TypedAssociation:
    """
    Typed memory association with semantic relationship type.

    Represents a directed edge in the knowledge graph with explicit
    relationship semantics (causes, fixes, supports, etc.).

    Attributes:
        source_hash: Source memory content hash
        target_hash: Target memory content hash
        similarity: Similarity score (0.0-1.0)
        connection_types: List of connection types (semantic, temporal, etc.)
        relationship_type: Semantic relationship type (default: "related")
        metadata: Optional metadata dict
        created_at: Unix timestamp

    Example:
        assoc = TypedAssociation(
            source_hash="abc123",
            target_hash="def456",
            similarity=0.85,
            connection_types=["semantic"],
            relationship_type="causes"
        )
    """
    source_hash: str
    target_hash: str
    similarity: float
    connection_types: List[str]
    relationship_type: str = "related"
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[float] = None

    def __post_init__(self):
        """Validate association data after initialization."""
        # Validate hashes
        if not self.source_hash or not self.target_hash:
            raise ValueError("source_hash and target_hash must be non-empty")

        if self.source_hash == self.target_hash:
            raise ValueError("Cannot create self-loop association")

        # Validate similarity
        if not (0.0 <= self.similarity <= 1.0):
            raise ValueError(f"similarity must be in range [0.0, 1.0], got {self.similarity}")

        # Ensure connection_types is a list
        if not isinstance(self.connection_types, list):
            raise ValueError("connection_types must be a list")

        # Validate relationship_type is a string
        if not isinstance(self.relationship_type, str):
            raise ValueError("relationship_type must be a string")
