"""
Implicit signals-based quality scoring.
Uses usage patterns (access count, recency, ranking) to score memory quality.
"""

import logging
import time
from typing import Optional
from ..models.memory import Memory

logger = logging.getLogger(__name__)


class ImplicitSignalsEvaluator:
    """
    Evaluates memory quality based on implicit usage signals.
    This is the fallback tier that always works without external dependencies.
    """

    def __init__(self):
        """Initialize implicit signals evaluator."""
        self.access_frequency_weight = 0.4
        self.recency_weight = 0.3
        self.ranking_weight = 0.3

    def evaluate_quality(self, memory: Memory, query: Optional[str] = None) -> float:
        """
        Evaluate memory quality based on implicit signals.

        Args:
            memory: Memory object to evaluate
            query: Optional query context (not used in implicit signals)

        Returns:
            Quality score between 0.0 and 1.0
        """
        # Extract signals from metadata
        access_count = memory.metadata.get('access_count', 0)
        last_accessed_at = memory.metadata.get('last_accessed_at')
        avg_ranking = memory.metadata.get('avg_ranking', 0.5)  # 0.0 = top result, 1.0 = bottom

        # Calculate access frequency score (normalized)
        # Use logarithmic scaling to prevent very popular items from dominating
        import math
        access_score = min(1.0, math.log(access_count + 1) / math.log(100))  # Normalize to 0-1

        # Calculate recency score
        recency_score = self._calculate_recency_score(last_accessed_at)

        # Calculate ranking score (invert so higher rank = higher score)
        ranking_score = 1.0 - avg_ranking

        # Combine scores with weights
        quality_score = (
            self.access_frequency_weight * access_score +
            self.recency_weight * recency_score +
            self.ranking_weight * ranking_score
        )

        return max(0.0, min(1.0, quality_score))

    def _calculate_recency_score(self, last_accessed_at: Optional[float]) -> float:
        """
        Calculate recency score based on last access time.

        Args:
            last_accessed_at: Unix timestamp of last access

        Returns:
            Recency score between 0.0 and 1.0 (1.0 = accessed very recently)
        """
        if last_accessed_at is None:
            # Never accessed, return low but non-zero score
            return 0.1

        now = time.time()
        time_since_access = now - last_accessed_at

        # Use exponential decay
        # After 30 days, score is ~0.1
        # After 7 days, score is ~0.5
        # After 1 day, score is ~0.8
        import math
        decay_rate = 0.1  # Adjust this to control decay speed
        recency_score = math.exp(-decay_rate * time_since_access / (24 * 3600))  # Convert to days

        return max(0.0, min(1.0, recency_score))

    def update_ranking_signal(self, memory: Memory, position: int, total_results: int):
        """
        Update the average ranking signal for a memory based on search results.

        Args:
            memory: Memory object to update
            position: Position in search results (0 = top)
            total_results: Total number of results returned
        """
        # Normalize position to 0.0-1.0 range
        normalized_position = position / max(1, total_results - 1) if total_results > 1 else 0.0

        # Update running average
        current_avg = memory.metadata.get('avg_ranking', 0.5)

        # Weighted average favoring recent positions
        alpha = 0.3  # Weight for new observation
        new_avg = alpha * normalized_position + (1 - alpha) * current_avg

        memory.metadata['avg_ranking'] = new_avg

    def get_signal_components(self, memory: Memory) -> dict:
        """
        Get breakdown of implicit signal components for debugging.

        Args:
            memory: Memory object to analyze

        Returns:
            Dictionary with individual signal scores
        """
        access_count = memory.metadata.get('access_count', 0)
        last_accessed_at = memory.metadata.get('last_accessed_at')
        avg_ranking = memory.metadata.get('avg_ranking', 0.5)

        import math
        access_score = min(1.0, math.log(access_count + 1) / math.log(100))
        recency_score = self._calculate_recency_score(last_accessed_at)
        ranking_score = 1.0 - avg_ranking

        return {
            'access_score': access_score,
            'access_count': access_count,
            'recency_score': recency_score,
            'last_accessed_at': last_accessed_at,
            'ranking_score': ranking_score,
            'avg_ranking': avg_ranking,
            'composite_score': self.evaluate_quality(memory)
        }
