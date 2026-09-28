"""
Composite quality scorer that combines AI scores with implicit signals.
Main entry point for quality scoring system.
"""

import logging
import time
from typing import Optional
from .config import QualityConfig
from .ai_evaluator import QualityEvaluator
from .implicit_signals import ImplicitSignalsEvaluator
from ..models.memory import Memory

logger = logging.getLogger(__name__)


class QualityScorer:
    """
    Composite quality scorer that combines AI-based and implicit signal scores.
    """

    def __init__(self, config: Optional[QualityConfig] = None):
        """
        Initialize quality scorer.

        Args:
            config: Quality configuration (defaults to env-based config)
        """
        self.config = config or QualityConfig.from_env()
        self.config.validate()

        self._ai_evaluator = QualityEvaluator(config=self.config)
        self._implicit_evaluator = ImplicitSignalsEvaluator()

    async def calculate_quality_score(
        self,
        memory: Memory,
        query: str,
        ai_score: Optional[float] = None
    ) -> float:
        """
        Calculate composite quality score for a memory.

        Args:
            memory: Memory object to score
            query: Search query for context
            ai_score: Pre-computed AI score (if None, will compute)

        Returns:
            Final composite quality score between 0.0 and 1.0
        """
        # Get AI score if not provided
        if ai_score is None:
            ai_score = await self._ai_evaluator.evaluate_quality(query, memory)

        # Get implicit signals score
        implicit_score = self._implicit_evaluator.evaluate_quality(memory, query)

        # Combine scores based on configuration.
        # These are the store-time knobs (#179): the search-side boost settings
        # used to double as the blend control, so enabling quality-boosted
        # search silently changed what was written to the database.
        if self.config.implicit_blend_enabled and ai_score is not None:
            # Weighted combination of AI and implicit signals
            ai_weight = 1.0 - self.config.implicit_weight
            implicit_weight = self.config.implicit_weight
            composite_score = ai_weight * ai_score + implicit_weight * implicit_score
        elif ai_score is not None:
            # Use AI score only
            composite_score = ai_score
        else:
            # Fall back to implicit signals
            composite_score = implicit_score

        # Update memory metadata with quality information
        self._update_memory_metadata(memory, composite_score, ai_score, implicit_score)

        return composite_score

    def _update_memory_metadata(
        self,
        memory: Memory,
        composite_score: float,
        ai_score: Optional[float],
        implicit_score: float
    ):
        """
        Update memory metadata with quality scoring information.

        Args:
            memory: Memory object to update
            composite_score: Final composite score
            ai_score: AI-based score (if available)
            implicit_score: Implicit signals score
        """
        # Quality model split (#1312): this is the machine origin — write
        # computed_quality, then materialize the effective quality_score so an
        # existing human rating still wins and is not erased by re-scoring.
        from .config import effective_quality
        memory.metadata['computed_quality'] = composite_score
        memory.metadata['quality_score'] = effective_quality(
            computed=composite_score,
            user_rating=memory.metadata.get('user_rating'),
        )

        # Track historical AI scores
        if ai_score is not None:
            ai_scores = memory.metadata.get('ai_scores', [])
            ai_scores.append({
                'score': ai_score,
                'timestamp': time.time(),
                'provider': memory.metadata.get('quality_provider', 'unknown')
            })
            # Keep only last 10 scores to avoid unbounded growth
            memory.metadata['ai_scores'] = ai_scores[-10:]

        # Store component scores for debugging
        # Record the knobs that actually produced this score. 'boost_*' are kept
        # for readers of older rows but now describe search-side reranking only.
        memory.metadata['quality_components'] = {
            'ai_score': ai_score,
            'implicit_score': implicit_score,
            'implicit_blend_enabled': self.config.implicit_blend_enabled,
            'implicit_weight': self.config.implicit_weight,
            'boost_enabled': self.config.boost_enabled,
            'boost_weight': self.config.boost_weight
        }

    async def score_batch(self, memories: list[Memory], query: str) -> list[float]:
        """
        Score a batch of memories efficiently.

        Args:
            memories: List of Memory objects to score
            query: Search query for context

        Returns:
            List of quality scores corresponding to input memories
        """
        scores = []
        for memory in memories:
            score = await self.calculate_quality_score(memory, query)
            scores.append(score)
        return scores

    def get_score_breakdown(self, memory: Memory) -> dict:
        """
        Get detailed breakdown of quality score components.

        Args:
            memory: Memory object to analyze

        Returns:
            Dictionary with score components and metadata
        """
        return {
            'quality_score': memory.metadata.get('quality_score', 0.5),
            'quality_provider': memory.metadata.get('quality_provider', 'unknown'),
            'quality_components': memory.metadata.get('quality_components', {}),
            'ai_scores_history': memory.metadata.get('ai_scores', []),
            'implicit_signals': self._implicit_evaluator.get_signal_components(memory),
            'access_count': memory.metadata.get('access_count', 0),
            'last_accessed_at': memory.metadata.get('last_accessed_at')
        }
