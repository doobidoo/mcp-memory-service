"""
Lightweight reasoning engine for knowledge graph inference.

Provides semantic reasoning capabilities including contradiction detection,
causal inference, and relationship suggestions.

Copyright (c) 2024 MCP Memory Service
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SemanticReasoner:
    """
    Lightweight reasoning engine for knowledge graph inference.

    Provides semantic reasoning capabilities including contradiction detection,
    causal inference, and relationship suggestions.
    """

    def __init__(self, graph_storage):
        """
        Initialize with graph storage dependency.

        Args:
            graph_storage: GraphStorage instance for relationship queries

        Raises:
            ValueError: If graph_storage is None or missing required methods
        """
        if graph_storage is None:
            raise ValueError("graph_storage cannot be None")
        if not hasattr(graph_storage, 'find_connected'):
            raise ValueError("graph_storage must have find_connected method")
        if not hasattr(graph_storage, 'shortest_path'):
            raise ValueError("graph_storage must have shortest_path method")
        self.graph = graph_storage

    async def _get_connected(self, hash: str, rel_type: str, direction: str = "both") -> List[str]:
        """
        Helper to fetch memories connected via specific relationship type.

        Args:
            hash: Source memory hash
            rel_type: Relationship type to filter
            direction: Direction to traverse ("outgoing", "incoming", "both")

        Returns:
            List of connected memory hashes
        """
        try:
            # Use graph.find_connected with relationship_type filter
            connected = await self.graph.find_connected(
                memory_hash=hash,
                relationship_type=rel_type,
                direction=direction,
                max_hops=1
            )
            # Return only the memory hashes (strip distance info)
            # connected is List[Tuple[str, int]]
            return [mem_hash for mem_hash, distance in connected]
        except Exception as e:
            logger.error(f"Failed to get connected memories: {e}")
            return []

    async def detect_contradictions(self, hash: str) -> List[str]:
        """
        Find memories that contradict the given memory.

        Args:
            hash: Memory content hash

        Returns:
            List of contradicting memory hashes

        Example:
            >>> contradictions = await reasoner.detect_contradictions("abc123")
            ["def456", "ghi789"]
        """
        try:
            # Find memories with "contradicts" relationship
            return await self._get_connected(hash, "contradicts")
        except Exception as e:
            logger.error(f"Failed to detect contradictions for {hash}: {e}")
            return []

    async def find_fixes(self, error_hash: str) -> List[str]:
        """
        Find memories that fix the given error.

        Args:
            error_hash: Error memory content hash

        Returns:
            List of fixing decision/learning memory hashes

        Example:
            >>> fixes = await reasoner.find_fixes("error_hash")
            ["decision_hash", "learning_hash"]
        """
        try:
            # Find memories with "fixes" relationship pointing TO this error
            # Use direction="incoming" to find sources that fix this target
            return await self._get_connected(error_hash, "fixes", direction="incoming")
        except Exception as e:
            logger.error(f"Failed to find fixes for {error_hash}: {e}")
            return []

    async def find_causes(self, error_hash: str) -> List[str]:
        """
        Find memories that caused the given error (backward traversal).

        Args:
            error_hash: Error memory content hash

        Returns:
            List of causing memory hashes

        Example:
            >>> causes = await reasoner.find_causes("error_hash")
            ["observation_hash", "decision_hash"]
        """
        try:
            # Traverse backward via "causes" relationships
            # Use direction="incoming" to find what caused this error
            return await self._get_connected(error_hash, "causes", direction="incoming")
        except Exception as e:
            logger.error(f"Failed to find causes for {error_hash}: {e}")
            return []

    async def abstract_to_concept(self, hash: str) -> Optional[str]:
        """
        Get parent base type for a memory's subtype.

        Requires memory metadata with memory_type field.
        For now, return None as placeholder (will be integrated later).

        Args:
            hash: Memory content hash

        Returns:
            Parent type string or None

        Example:
            >>> parent = await reasoner.abstract_to_concept("hash")
            "observation"  # if memory type was "code_edit"
        """
        # Placeholder: Will be integrated with memory storage in Integration phase
        # For now, return None
        return None

    async def infer_transitive(
        self,
        rel_type: str,
        max_hops: int = 2
    ) -> List[Tuple[str, str, int]]:
        """
        Find transitive relationships (A→B→C implies A→C).

        Args:
            rel_type: Relationship type to traverse
            max_hops: Maximum hops for transitive closure

        Returns:
            List of (source, target, distance) tuples for inferred relationships

        Example:
            >>> inferred = await reasoner.infer_transitive("causes", max_hops=2)
            [("hash1", "hash3", 2), ...]  # hash1→hash2→hash3
        """
        # This is a placeholder - full implementation would:
        # 1. Get all edges of rel_type
        # 2. Build graph structure
        # 3. Find paths of length 2+ (transitive closure)
        # 4. Return inferred edges
        # For now, return empty list
        return []

    async def suggest_relationships(self, hash: str) -> List[Dict[str, Any]]:
        """
        Suggest potential relationships for a memory.

        Uses semantic similarity to suggest typed relationships.

        Args:
            hash: Memory content hash

        Returns:
            List of suggested relationships with confidence scores

        Example:
            >>> suggestions = await reasoner.suggest_relationships("hash1")
            [{"target": "hash2", "type": "supports", "confidence": 0.85}]
        """
        # Placeholder - will be enhanced with semantic analysis
        # For now, return empty list
        return []
