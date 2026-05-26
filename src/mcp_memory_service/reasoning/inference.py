"""
Lightweight reasoning engine for knowledge graph inference.

Provides semantic reasoning capabilities including contradiction detection,
causal inference, and relationship suggestions.

Copyright (c) 2024 MCP Memory Service
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


TRAVERSABLE_EDGE_TYPES = {'relates_to', 'superseded_by', 'causes', 'fixes', 'related'}
NON_TRAVERSABLE = {'contradicts', 'contradicted_by'}


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

    async def _get_connected(self, hash: str, rel_type: str, direction: str = "both", max_hops: int = 1) -> List[str]:
        """
        Helper to fetch memories connected via specific relationship type.

        Args:
            hash: Source memory hash
            rel_type: Relationship type to filter
            direction: Direction to traverse ("outgoing", "incoming", "both")
            max_hops: Maximum traversal depth (default 1 = direct connections only)

        Returns:
            List of connected memory hashes
        """
        try:
            # Use graph.find_connected with relationship_type filter
            connected = await self.graph.find_connected(
                memory_hash=hash,
                relationship_type=rel_type,
                direction=direction,
                max_hops=max_hops
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
        max_hops: int = 2,
        decay_factor: float = 1.0
    ) -> List[Tuple[str, str, int, float]]:
        """
        Find transitive relationships (A→B→C implies A→C) with decay by distance.

        Args:
            rel_type: Relationship type to traverse (must be in TRAVERSABLE_EDGE_TYPES)
            max_hops: Maximum hops for transitive closure (2-4)
            decay_factor: Base decay multiplier (weight = decay_factor / distance)

        Returns:
            List of (source, target, distance, weight) tuples.

        Raises:
            ValueError: If rel_type is non-traversable
        """
        # Defensive coercion (MCP handler may pass strings/None)
        max_hops = max(1, min(int(max_hops or 2), 4))
        decay_factor = float(decay_factor or 1.0)

        if rel_type in NON_TRAVERSABLE:
            raise ValueError(
                f"Edge type '{rel_type}' is non-traversable. "
                f"Non-traversable types: {sorted(NON_TRAVERSABLE)}"
            )
        if rel_type not in TRAVERSABLE_EDGE_TYPES:
            logger.warning(f"Edge type '{rel_type}' not in TRAVERSABLE_EDGE_TYPES, proceeding anyway")
        if not hasattr(self.graph, 'transitive_closure'):
            logger.warning("GraphStorage does not support transitive_closure")
            return []
        try:
            results = await self.graph.transitive_closure(rel_type, max_hops)
            return [
                (src, tgt, dist, decay_factor / max(dist, 1))
                for src, tgt, dist in results
            ]
        except Exception as e:
            logger.error(f"Failed to infer transitive relationships: {e}")
            return []

    async def abduct(self, effect_hash: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """
        Abductive reasoning: given an effect, find probable causes.

        Strategy:
        1. Find all memories connected to effect_hash via 'causes' or 'fixes' edges (incoming)
        2. For each cause, find what OTHER effects it caused (siblings)
        3. If multiple effects share the same cause pattern, boost confidence
        4. Return ranked list of probable causes with confidence

        Args:
            effect_hash: The observed effect memory hash
            max_depth: Maximum depth for cause traversal (1-4, default 2)

        Returns:
            List of {cause_hash, confidence, evidence_count, shared_effects}
        """
        # Defensive coercion
        max_depth = max(1, min(int(max_depth or 2), 4))

        try:
            # Step 1: Find incoming causes/fixes (using max_depth for traversal)
            causes = await self._get_connected(effect_hash, "causes", direction="incoming", max_hops=max_depth)
            fixes = await self._get_connected(effect_hash, "fixes", direction="incoming", max_hops=max_depth)
            all_causes = sorted(set(causes + fixes))

            if not all_causes:
                return []

            # Step 2 & 3: For each cause, find sibling effects and compute confidence
            results = []
            for cause_hash in all_causes:
                # Find other effects this cause produced (outgoing causes/fixes)
                siblings_causes = await self._get_connected(cause_hash, "causes", direction="outgoing")
                siblings_fixes = await self._get_connected(cause_hash, "fixes", direction="outgoing")
                shared_effects = list(set(siblings_causes + siblings_fixes) - {effect_hash})

                evidence_count = 1 + len(shared_effects)
                # Confidence: base 0.5, boosted by shared effects (capped at 1.0)
                confidence = min(1.0, 0.5 + 0.1 * len(shared_effects))

                results.append({
                    "cause_hash": cause_hash,
                    "confidence": round(confidence, 3),
                    "evidence_count": evidence_count,
                    "shared_effects": shared_effects[:10],
                    "total_shared_effects": len(shared_effects),
                })

            results.sort(key=lambda x: x["confidence"], reverse=True)
            return results

        except Exception as e:
            logger.error(f"Failed abductive reasoning for {effect_hash}: {e}")
            return []

    async def suggest_relationships(self, hash: str) -> List[Dict[str, Any]]:
        """
        Suggest potential relationships for a memory based on shared neighbors.

        Delegates to GraphStorage.common_neighbors which uses a single SQL query
        with self-join. Confidence = shared_count / degree(source).

        Args:
            hash: Memory content hash

        Returns:
            List of suggested relationships with confidence scores, sorted by confidence.

        Example:
            >>> suggestions = await reasoner.suggest_relationships("hash1")
            [{"target": "hash2", "type": "related", "confidence": 0.85}]
        """
        if not hasattr(self.graph, 'common_neighbors'):
            logger.warning("GraphStorage does not support common_neighbors")
            return []
        try:
            candidates = await self.graph.common_neighbors(hash, min_shared=1)
            if not candidates:
                return []

            suggestions = []
            for target, shared_count, source_degree in candidates:
                confidence = shared_count / max(source_degree, 1)
                if confidence >= 0.3:
                    suggestions.append({
                        "target": target,
                        "type": "related",
                        "confidence": round(confidence, 3),
                        "shared_neighbors": shared_count,
                    })

            suggestions.sort(key=lambda x: x["confidence"], reverse=True)
            return suggestions[:10]

        except Exception as e:
            logger.error(f"Failed to suggest relationships for {hash}: {e}")
            return []
