#!/usr/bin/env python3
"""
Memory Relationship Type Inference Engine

Analyzes memory pairs to determine appropriate relationship types based on:
1. Memory type combinations (e.g., decision + error → "fixes")
2. Content analysis (e.g., "fixed", "resolved" → "fixes")
3. Temporal patterns (sequential creation → "follows")
4. Semantic contradictions (opposing content → "contradicts")

This enables rich knowledge graphs with meaningful relationship types beyond "related".

Fixes for GitHub issue #541:
- Raised similarity threshold for typed labels (min_typed_similarity=0.65)
- Tightened contradiction detection: removed weak conjunctions ("but", "yet",
  "although", "however", "nevertheless") — only strong opposition indicators kept
- Added min_typed_confidence parameter (default 0.75) for non-"related" edges
- Added cross-content domain-keyword verification before assigning typed labels
"""

import re
import logging
from typing import Tuple, Optional, List, Set

from ..models.ontology import (
    get_parent_type,
    validate_relationship,
)

logger = logging.getLogger(__name__)

# Common English stopwords excluded from domain-keyword matching
_STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "not", "no", "nor",
    "so", "yet", "both", "either", "neither", "just", "also", "as", "if",
    "this", "that", "these", "those", "it", "its", "we", "our", "they",
    "their", "he", "she", "his", "her", "i", "my", "me", "you", "your",
    "up", "out", "about", "after", "before", "then", "than", "when",
    "which", "who", "what", "where", "how", "all", "some", "any", "more",
    "most", "such", "into", "through", "during", "however", "although",
    "though", "because", "since", "while", "each", "few", "further",
    "once", "only", "same", "other", "new", "own", "very", "too",
}


def _extract_domain_keywords(text: str) -> Set[str]:
    """
    Extract meaningful domain keywords from text, excluding stopwords.

    Returns a set of lowercased words that are at least 4 characters long
    and not in the stopword list.
    """
    # Split on non-alphanumeric characters (handles hyphens, underscores, etc.)
    words = re.findall(r"[a-z][a-z0-9_-]{2,}", text.lower())
    return {w for w in words if w not in _STOPWORDS}


def _shares_domain_keywords(source_lower: str, target_lower: str) -> bool:
    """
    Return True if source and target share at least one meaningful domain keyword.

    Used as a cross-content guard before assigning typed relationship labels.
    """
    source_kw = _extract_domain_keywords(source_lower)
    target_kw = _extract_domain_keywords(target_lower)
    return bool(source_kw & target_kw)


class RelationshipInferenceEngine:
    """
    Analyzes memory pairs to determine appropriate relationship types.

    Uses multi-factor analysis:
    1. Memory type hierarchy compatibility
    2. Content semantic analysis
    3. Temporal relationship detection
    4. Contradiction detection

    Constructor parameters
    ---------------------
    min_confidence : float
        Minimum confidence for ``related`` edges (default 0.6). Typed edges
        additionally require ``min_typed_confidence``.
    min_typed_confidence : float
        Minimum confidence for non-``related`` relationship types such as
        ``fixes``, ``causes``, ``supports``, ``contradicts`` (default 0.75).
        Issue #541 fix: separates thresholds so "related" is easy to assign
        but typed labels require stronger evidence.
    min_typed_similarity : float
        Minimum cosine similarity required before assigning any typed label
        (default 0.65). When similarity is below this threshold the result
        falls back to "related". Only applies when ``similarity`` is provided
        to ``infer_relationship_type``. Issue #541 fix.
    """

    # Content patterns for relationship detection
    PATTERNS = {
        "causation": [
            r"\bcaused?\b",
            r"\blead\s+to\b",
            r"\bresulted\s+in\b",
            r"\btriggered\b",
            r"\bgenerated\b",
        ],
        "resolution": [
            r"\bfixed?\b",
            r"\bresolve[ds]?\b",
            r"\bcorrected?\b",
            r"\bpatched?\b",
            r"\brepaired\b",
            r"\bhealed\b",
        ],
        "support": [
            r"\bsupports?\b",
            r"\benables?\b",
            r"\bfacilitate[ds]?\b",
            r"\bhelps?\b",
            r"\baccompany\b",
        ],
        # Issue #541 fix: removed weak conjunctions ("but", "yet", "although",
        # "however", "nevertheless") — they fire on virtually any sentence that
        # uses contrast connectors, producing near-100% false positive rate.
        # Only strong, direct opposition / negation indicators are kept.
        "contradiction": [
            r"\bcontradict[ds]?\b",
            r"\bconflict[ds]?\b",
            r"\bdisagree[ds]?\b",
            r"\boppose[sd]?\b",
            r"\bincorrect\b",
            r"\bwrong\b",
            r"\bnot\s+true\b",
            r"\buntrue\b",
            r"\bfalsely?\b",
            r"\bcontrary\b",
            r"\bopposite\b",
            r"\bnever\b",
        ],
    }

    def __init__(
        self,
        min_confidence: float = 0.6,
        min_typed_confidence: float = 0.75,
        min_typed_similarity: float = 0.65,
        typed_edges_enabled: bool = True,
    ):
        """
        Initialize the inference engine.

        Args:
            min_confidence: Minimum confidence score (0.0-1.0) to assign
                any relationship type (including ``related``).
            min_typed_confidence: Minimum confidence required specifically for
                typed labels (``fixes``, ``causes``, ``supports``,
                ``contradicts``, ``follows``). Must be >= min_confidence.
                Defaults to 0.75 (issue #541).
            min_typed_similarity: When a cosine similarity is supplied, typed
                labels are only assigned when similarity >= this threshold.
                Defaults to 0.65 (issue #541).
            typed_edges_enabled: When False, all inferred relationships are
                returned as "related" regardless of analysis results.
                Set via MCP_TYPED_EDGES_ENABLED=false to opt out of typed
                edge inference (issue #546).
        """
        self.min_confidence = min_confidence
        self.min_typed_confidence = max(min_typed_confidence, min_confidence)
        self.min_typed_similarity = min_typed_similarity
        self.typed_edges_enabled = typed_edges_enabled

    async def infer_relationship_type(
        self,
        source_type: Optional[str],
        target_type: Optional[str],
        source_content: str,
        target_content: str,
        source_timestamp: Optional[float] = None,
        target_timestamp: Optional[float] = None,
        source_tags: Optional[List[str]] = None,
        target_tags: Optional[List[str]] = None,
        similarity: Optional[float] = None,
    ) -> Tuple[str, float]:
        """
        Infer relationship type between two memories.

        Args:
            source_type: Memory type of source memory
            target_type: Memory type of target memory
            source_content: Content of source memory
            target_content: Content of target memory
            source_timestamp: Creation timestamp of source memory (epoch)
            target_timestamp: Creation timestamp of target memory (epoch)
            source_tags: Tags on source memory
            target_tags: Tags on target memory
            similarity: Optional cosine similarity score between memory
                embeddings. When provided, typed labels are suppressed if
                similarity < min_typed_similarity (issue #541).

        Returns:
            Tuple of (relationship_type, confidence_score)

        Examples:
            >>> engine = RelationshipInferenceEngine()
            >>> result = await engine.infer_relationship_type(
            ...     source_type="learning/insight",
            ...     target_type="error/bug",
            ...     source_content="Fixed authentication issue...",
            ...     target_content="Authentication failed with timeout...",
            ...     source_timestamp=1234567890.0,
            ...     target_timestamp=1234560000.0
            ... )
            >>> result
            ("fixes", 0.85)
        """
        # Issue #546: opt-out flag — skip all typed-label analysis
        if not self.typed_edges_enabled:
            return ("related", 0.0)

        source_tags = source_tags or []
        target_tags = target_tags or []
        source_lower = source_content.lower()
        target_lower = target_content.lower()

        # Issue #541 fix: check whether memories share domain vocabulary.
        # Typed labels require topical relevance; "related" does not.
        has_shared_keywords = _shares_domain_keywords(source_lower, target_lower)

        # Issue #541 fix: if a similarity score is supplied, typed labels
        # require similarity >= min_typed_similarity.
        similarity_allows_typed = (
            similarity is None or similarity >= self.min_typed_similarity
        )

        # Collect relationship type candidates with confidence scores
        candidates = []

        # 1. Analyze memory type combinations
        type_score = self._analyze_type_combination(source_type, target_type)
        if type_score:
            candidates.extend(type_score)

        # 2. Analyze content semantic patterns
        content_score = self._analyze_content_semantics(
            source_lower, target_lower, source_type, target_type
        )
        if content_score:
            candidates.extend(content_score)

        # 3. Analyze temporal relationships
        if source_timestamp and target_timestamp:
            temporal_score = self._analyze_temporal_relationship(
                source_timestamp, target_timestamp, source_type, target_type
            )
            if temporal_score:
                candidates.extend(temporal_score)

        # 4. Analyze contradictions
        contradiction_score = self._analyze_contradictions(
            source_lower,
            target_lower,
            source_type,
            target_type,
            source_tags,
            target_tags,
        )
        if contradiction_score:
            candidates.extend(contradiction_score)

        # Select highest-confidence candidate
        if not candidates:
            logger.debug(
                "No confident relationship type found, defaulting to 'related'"
            )
            return ("related", 0.0)

        # Sort by confidence descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_type, best_confidence = candidates[0]

        # Issue #541 fix: apply the stricter typed-label threshold and guards
        # before the generic min_confidence check.
        if best_type != "related":
            # Guard 1: similarity threshold
            if not similarity_allows_typed:
                logger.debug(
                    f"Similarity {similarity:.3f} below typed threshold "
                    f"{self.min_typed_similarity}, downgrading to 'related'"
                )
                return ("related", best_confidence)

            # Guard 2: shared domain keywords (cross-content verification)
            if not has_shared_keywords:
                logger.debug(
                    "No shared domain keywords between memories, "
                    "downgrading typed label to 'related'"
                )
                return ("related", best_confidence)

            # Guard 3: typed confidence threshold
            if best_confidence < self.min_typed_confidence:
                logger.debug(
                    f"Typed confidence {best_confidence:.2f} below typed threshold "
                    f"{self.min_typed_confidence}, downgrading to 'related'"
                )
                return ("related", best_confidence)

        # Apply generic minimum confidence threshold
        if best_confidence < self.min_confidence:
            logger.debug(
                f"Confidence {best_confidence:.2f} below threshold {self.min_confidence}, "
                f"defaulting to 'related'"
            )
            return ("related", best_confidence)

        # Validate relationship type
        if not validate_relationship(best_type):
            logger.warning(f"Invalid relationship type inferred: {best_type}")
            return ("related", best_confidence)

        logger.debug(
            f"Inferred relationship type '{best_type}' with confidence {best_confidence:.2f}"
        )
        return (best_type, best_confidence)

    @staticmethod
    def _resolve_parent_type(memory_type: str) -> Optional[str]:
        """
        Resolve the base/parent type from a memory type string.

        Handles both plain subtype notation ("insight") and slash notation
        ("learning/insight") used in practice.  For slash notation the first
        segment is returned if it is a valid base type; otherwise the full
        type string is looked up via the ontology helper.

        Args:
            memory_type: Memory type string (e.g. "learning/insight",
                "error/bug", "observation", "insight")

        Returns:
            Base type string (e.g. "learning", "error") or None if unknown.
        """
        if not memory_type:
            return None

        # Handle slash-notation: "learning/insight" → first segment is parent
        if "/" in memory_type:
            first_segment = memory_type.split("/", 1)[0]
            parent = get_parent_type(first_segment)
            if parent is not None:
                return parent
            # Fall through to ontology lookup of the full string
        return get_parent_type(memory_type)

    def _analyze_type_combination(
        self, source_type: Optional[str], target_type: Optional[str]
    ) -> List[Tuple[str, float]]:
        """
        Analyze memory type combinations to determine relationship type.

        Uses ontology-defined patterns:
        - decision → error → uses/causes
        - learning → error → fixes
        - decision → decision → supports/contradicts
        - etc.

        Issue #541 fix: confidence values here feed into the typed-label gate
        (min_typed_confidence=0.75), so only combinations that were already
        at or above 0.75 in the original code will pass through unchanged.
        Lower-confidence combinations will be caught by the gate and fall
        back to "related".

        Returns:
            List of (relationship_type, confidence_score) candidates
        """
        if not source_type or not target_type:
            return []

        candidates = []
        source_parent = self._resolve_parent_type(source_type)
        target_parent = self._resolve_parent_type(target_type)

        # High-confidence patterns from ontology
        type_patterns = {
            ("decision", "error"): ("uses", 0.7),
            ("learning", "error"): ("fixes", 0.8),
            ("pattern", "error"): ("fixes", 0.75),
            ("learning", "decision"): ("supports", 0.6),
            ("observation", "learning"): ("supports", 0.5),
            ("pattern", "learning"): ("supports", 0.6),
            ("decision", "decision"): ("supports", 0.4),
            ("learning", "learning"): ("supports", 0.3),
            ("error", "error"): ("causes", 0.6),
            ("observation", "observation"): ("follows", 0.3),
        }

        # Check both (source, target) and (target, source) directions
        for (type1, type2), (rel_type, confidence) in type_patterns.items():
            if source_parent == type1 and target_parent == type2:
                candidates.append((rel_type, confidence))
            elif source_parent == type2 and target_parent == type1:
                # Reverse relationship - adjust confidence down
                candidates.append((rel_type, confidence * 0.7))

        return candidates

    def _analyze_content_semantics(
        self,
        source_lower: str,
        target_lower: str,
        source_type: Optional[str],
        target_type: Optional[str],
    ) -> List[Tuple[str, float]]:
        """
        Analyze content for semantic relationship patterns.

        Issue #541 fix: cross-content domain-keyword verification is now
        enforced upstream in ``infer_relationship_type``, so this method no
        longer needs to gate on it. The per-type checks (e.g. "error" in
        target_type) remain as an additional signal that combines with the
        upstream guard.

        Returns:
            List of (relationship_type, confidence_score) candidates
        """
        candidates = []

        # Check for resolution/fix patterns
        resolution_patterns = self.PATTERNS["resolution"]
        resolution_count = sum(
            1 for pattern in resolution_patterns if re.search(pattern, source_lower)
        )
        if resolution_count > 0:
            # If source contains resolution keywords and target is an error
            if target_type and "error" in target_type:
                confidence = min(0.9, 0.5 + (resolution_count * 0.1))
                candidates.append(("fixes", confidence))

        # Check for causation patterns
        causation_patterns = self.PATTERNS["causation"]
        causation_count = sum(
            1 for pattern in causation_patterns if re.search(pattern, source_lower)
        )
        if causation_count > 0:
            # If source contains causation keywords
            if target_type and "error" in target_type:
                confidence = min(0.8, 0.5 + (causation_count * 0.1))
                candidates.append(("causes", confidence))

        # Check for support patterns
        support_patterns = self.PATTERNS["support"]
        support_count = sum(
            1 for pattern in support_patterns if re.search(pattern, source_lower)
        )
        if support_count > 0:
            # If source contains support keywords
            if target_type and "decision" in target_type:
                confidence = min(0.75, 0.4 + (support_count * 0.1))
                candidates.append(("supports", confidence))

        return candidates

    def _analyze_temporal_relationship(
        self,
        source_timestamp: float,
        target_timestamp: float,
        source_type: Optional[str],
        target_type: Optional[str],
    ) -> List[Tuple[str, float]]:
        """
        Analyze temporal relationship between memories.

        Returns:
            List of (relationship_type, confidence_score) candidates
        """
        candidates = []

        # Calculate time difference
        time_diff = abs(source_timestamp - target_timestamp)

        # If created within 1 hour and same parent type
        if time_diff < 3600:
            source_parent = self._resolve_parent_type(source_type) if source_type else None
            target_parent = self._resolve_parent_type(target_type) if target_type else None

            if source_parent == target_parent:
                confidence = 0.4
                candidates.append(("follows", confidence))

        # If source created after target and describes fixing
        if (
            source_timestamp > target_timestamp
            and source_type
            and "learning" in source_type
        ):
            if target_type and "error" in target_type:
                confidence = 0.6
                candidates.append(("fixes", confidence))

        return candidates

    def _analyze_contradictions(
        self,
        source_lower: str,
        target_lower: str,
        source_type: Optional[str],
        target_type: Optional[str],
        source_tags: List[str],
        target_tags: List[str],
    ) -> List[Tuple[str, float]]:
        """
        Analyze potential contradictions between memories.

        Issue #541 fix:
        - Removed weak contrast conjunctions from PATTERNS["contradiction"]
          (see class-level docstring).
        - Minimum confidence for single-side detection raised from 0.4 to 0.5.
        - Both-side detection raised from 0.7 to 0.75 to meet
          min_typed_confidence gate.

        Returns:
            List of (relationship_type, confidence_score) candidates
        """
        candidates = []

        # Check for contradiction keywords
        contradiction_patterns = self.PATTERNS["contradiction"]
        source_contradictions = sum(
            1 for pattern in contradiction_patterns if re.search(pattern, source_lower)
        )
        target_contradictions = sum(
            1 for pattern in contradiction_patterns if re.search(pattern, target_lower)
        )

        if source_contradictions > 0 or target_contradictions > 0:
            # High confidence if both have contradiction patterns
            if source_contradictions > 0 and target_contradictions > 0:
                # Issue #541: raised from 0.7 → 0.75 to satisfy min_typed_confidence
                confidence = 0.75
            else:
                # Issue #541: raised from 0.4 → 0.5 (single-side is weaker evidence)
                confidence = 0.5

            candidates.append(("contradicts", confidence))

        return candidates


# Example usage
async def test_inference():
    """Test relationship type inference."""
    engine = RelationshipInferenceEngine(min_confidence=0.5)

    # Test 1: Learning fixes Error
    rel_type, confidence = await engine.infer_relationship_type(
        source_type="learning/insight",
        target_type="error/bug",
        source_content="Fixed authentication timeout by adjusting configuration",
        target_content="Authentication error: Request timeout after 30 seconds",
        source_timestamp=1234567890.0,
        target_timestamp=1234560000.0,
    )
    print(f"Test 1: {rel_type} (confidence: {confidence:.2f})")
    # Expected: "fixes" with high confidence

    # Test 2: Decision causes Error
    rel_type, confidence = await engine.infer_relationship_type(
        source_type="decision/architecture",
        target_type="error/bug",
        source_content="Chose to use HTTP instead of HTTPS for testing",
        target_content="HTTP connection refused: Port 8000 not responding",
        source_timestamp=1234567890.0,
        target_timestamp=1234567800.0,
    )
    print(f"Test 2: {rel_type} (confidence: {confidence:.2f})")
    # Expected: "causes" with moderate-high confidence

    # Test 3: Two decisions supporting each other
    rel_type, confidence = await engine.infer_relationship_type(
        source_type="decision/architecture",
        target_type="decision/tool_choice",
        source_content="Decided to use FastAPI for HTTP server",
        target_content="Chose ONNX for embeddings due to lightweight requirements",
        source_timestamp=1234567890.0,
        target_timestamp=1234567900.0,
    )
    print(f"Test 3: {rel_type} (confidence: {confidence:.2f})")
    # Expected: "supports" with moderate confidence

    # Test 4: Sequential observations
    rel_type, confidence = await engine.infer_relationship_type(
        source_type="observation",
        target_type="observation",
        source_content="Deployed version 9.0.0",
        target_content="Checked deployment status",
        source_timestamp=1234567890.0,
        target_timestamp=1234567950.0,
    )
    print(f"Test 4: {rel_type} (confidence: {confidence:.2f})")
    # Expected: "follows" with low-moderate confidence

    # Test 5: Default - no clear pattern
    rel_type, confidence = await engine.infer_relationship_type(
        source_type="observation",
        target_type="observation",
        source_content="Meeting notes about Q1 planning",
        target_content="Team lunch at Italian restaurant",
        source_timestamp=1234567890.0,
        target_timestamp=1234567000.0,
    )
    print(f"Test 5: {rel_type} (confidence: {confidence:.2f})")
    # Expected: "related" with low confidence


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_inference())
