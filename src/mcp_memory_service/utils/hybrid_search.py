"""
Hybrid Search Utilities

Pure functions for combining vector similarity with tag matching
using Reciprocal Rank Fusion (RRF). Enabled by default with adaptive
alpha selection based on corpus characteristics.

Research basis:
- RRF: Standard formula 1/(k+rank) with k=60 for score fusion
- Adaptive alpha: RecSys crossover experiments show exact match wins at small scale
- Recency decay: Exponential decay boosts fresher memories
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import HybridSearchSettings
    from ..models.memory import Memory, MemoryQueryResult

# English stop words - common words that don't carry semantic meaning for tag matching
STOP_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "can",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "my",
        "your",
        "his",
        "her",
        "its",
        "our",
        "their",
        "what",
        "which",
        "who",
        "whom",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "about",
        "into",
        "over",
        "after",
        "before",
        "between",
        "under",
        "again",
        "then",
        "here",
        "there",
        "any",
        "as",
        "if",
        "also",
        "now",
        "up",
        "out",
        "get",
        "got",
    }
)

# Regex for tokenization - split on non-alphanumeric characters
_TOKEN_PATTERN = re.compile(r"[^a-zA-Z0-9]+")


def extract_query_keywords(query: str, existing_tags: set[str] | None = None) -> list[str]:
    """
    Extract potential tag keywords from a search query.

    Algorithm:
        1. Lowercase and tokenize (split on whitespace/punctuation)
        2. Remove stop words
        3. If existing_tags provided, filter to only matching tags
        4. Return unique keywords

    Args:
        query: User's search query
        existing_tags: Set of tags that exist in database (for validation)

    Returns:
        List of normalized keywords that may match tags
    """
    # Tokenize: lowercase and split on non-alphanumeric
    tokens = _TOKEN_PATTERN.split(query.lower())

    # Filter: remove empty strings, stop words, and very short tokens
    keywords = [token for token in tokens if token and token not in STOP_WORDS and len(token) > 1]

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_keywords: list[str] = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique_keywords.append(kw)

    # If existing_tags provided, filter to only matching tags
    if existing_tags is not None:
        # Normalize existing_tags to lowercase for comparison
        existing_lower = {tag.lower() for tag in existing_tags}
        unique_keywords = [kw for kw in unique_keywords if kw in existing_lower]

    return unique_keywords


def rrf_score(rank: int, k: int = 60) -> float:
    """
    Calculate Reciprocal Rank Fusion score.

    Standard RRF formula: 1 / (k + rank)
    k=60 is the standard smoothing constant from literature.

    Args:
        rank: Position in ranked list (1-indexed, so rank 1 = top result)
        k: Smoothing constant (default 60)

    Returns:
        RRF score (higher = better)
    """
    if rank < 1:
        return 0.0
    return 1.0 / (k + rank)


def combine_results_rrf(
    vector_results: list[MemoryQueryResult], tag_matches: list[Memory], alpha: float, k: int = 60
) -> list[tuple[Memory, float, dict]]:
    """
    Combine vector search and tag search results using RRF.

    Formula: final_score = alpha * vector_rrf + (1-alpha) * tag_rrf

    For memories appearing in both lists, scores are summed.

    Args:
        vector_results: Ranked results from semantic search (with similarity scores)
        tag_matches: Memories matching extracted tags (unranked)
        alpha: Weight for vector results (0.0 to 1.0)
        k: RRF smoothing constant

    Returns:
        List of (memory, combined_score, debug_info) tuples, sorted by score desc
    """
    # Build score maps
    scores: dict[str, float] = {}  # content_hash -> combined score
    memories: dict[str, Memory] = {}  # content_hash -> memory object
    debug: dict[str, dict] = {}  # content_hash -> debug info

    # Process vector results (ranked by similarity)
    for rank, result in enumerate(vector_results, start=1):
        content_hash = result.memory.content_hash
        vec_rrf = rrf_score(rank, k)
        vec_contribution = alpha * vec_rrf

        memories[content_hash] = result.memory
        scores[content_hash] = vec_contribution
        debug[content_hash] = {
            "vector_score": result.similarity_score,
            "vector_rank": rank,
            "vector_rrf": vec_rrf,
            "tag_boost": 0.0,
            "tag_matches": [],
        }

    # Process tag matches (treat as equally ranked for RRF purposes)
    # All tag matches get rank=1 contribution (they all matched)
    tag_rrf = rrf_score(1, k)
    tag_contribution = (1.0 - alpha) * tag_rrf

    for memory in tag_matches:
        content_hash = memory.content_hash

        if content_hash in scores:
            # Overlap: add tag contribution to existing score
            scores[content_hash] += tag_contribution
            debug[content_hash]["tag_boost"] = tag_contribution
            debug[content_hash]["tag_matches"].append("matched")
        else:
            # Tag-only result (not in vector results)
            memories[content_hash] = memory
            scores[content_hash] = tag_contribution
            debug[content_hash] = {
                "vector_score": 0.0,
                "vector_rank": 0,
                "vector_rrf": 0.0,
                "tag_boost": tag_contribution,
                "tag_matches": ["matched"],
            }

    # Build final results with debug info
    results: list[tuple[Memory, float, dict]] = []
    for content_hash, score in scores.items():
        info = debug[content_hash]
        info["final_score"] = score
        info["alpha_used"] = alpha
        results.append((memories[content_hash], score, info))

    # Sort by score descending
    results.sort(key=lambda x: x[1], reverse=True)

    return results


def get_adaptive_alpha(
    corpus_size: int,
    matching_tag_count: int,
    config: HybridSearchSettings,
) -> float:
    """
    Calculate adaptive alpha based on corpus size and query characteristics.

    Research basis: RecSys crossover experiments show algorithm effectiveness
    varies by scale - exact match outperforms ML at small scale.

    Logic:
        - corpus < threshold_small (500): alpha = 0.5 (balanced)
        - threshold_small <= corpus < threshold_large (5000): alpha = 0.7 (semantic-biased)
        - corpus >= threshold_large: alpha = 0.8 (strong semantic)
        - If matching_tag_count >= 3: boost tag weight by 1.5x

    Args:
        corpus_size: Total memories in database
        matching_tag_count: How many query terms match existing tags
        config: Hybrid search settings with thresholds

    Returns:
        Alpha value (0.0 to 1.0)
    """
    # If explicit alpha is configured, use it
    if config.hybrid_alpha is not None:
        return config.hybrid_alpha

    # Determine base alpha from corpus size
    if corpus_size < config.adaptive_threshold_small:
        base_alpha = 0.5  # Small corpus: balanced hybrid
    elif corpus_size < config.adaptive_threshold_large:
        base_alpha = 0.7  # Medium corpus: semantic-biased
    else:
        base_alpha = 0.8  # Large corpus: strong semantic

    # Apply tag match boost: if >= 3 tags match, increase tag weight by 1.5x
    # This means reducing alpha to give tags more influence
    if matching_tag_count >= 3:
        # Boost tag weight by 1.5x means multiplying (1-alpha) by 1.5
        # So new_alpha = 1 - 1.5*(1-base_alpha)
        # Example: base_alpha=0.7 -> tag_weight=0.3 -> boosted=0.45 -> new_alpha=0.55
        boosted_tag_weight = 1.5 * (1.0 - base_alpha)
        alpha = max(0.0, 1.0 - boosted_tag_weight)
    else:
        alpha = base_alpha

    return alpha


def apply_recency_decay(
    results: list[tuple[Memory, float, dict]],
    decay_rate: float,
) -> list[tuple[Memory, float, dict]]:
    """
    Apply recency decay to search results.

    Formula: final_score = score * exp(-decay * days_since_update)

    With decay=0.01, half-life is ~70 days (exp(-0.01*70) ≈ 0.5)

    Args:
        results: List of (memory, score, debug_info) tuples
        decay_rate: Decay rate (0 = disabled)

    Returns:
        Results with recency-adjusted scores, re-sorted
    """
    if decay_rate <= 0:
        # Decay disabled - add recency_factor=1.0 to debug and return as-is
        for _memory, _score, info in results:
            info["recency_factor"] = 1.0
        return results

    now = datetime.now(timezone.utc)
    adjusted: list[tuple[Memory, float, dict]] = []

    for memory, score, info in results:
        # Calculate days since last update
        try:
            updated_at = datetime.fromisoformat(memory.updated_at_iso)
            # Normalize to UTC - handle both aware and naive datetimes
            if updated_at.tzinfo is None:
                # Assume naive datetime is UTC
                updated_at = updated_at.replace(tzinfo=timezone.utc)
            days_old = (now - updated_at).total_seconds() / 86400
        except (ValueError, TypeError):
            # If we can't parse the date, assume it's old
            days_old = 365

        # Apply exponential decay
        recency_factor = math.exp(-decay_rate * days_old)
        adjusted_score = score * recency_factor

        # Update debug info
        info["recency_factor"] = recency_factor
        info["days_old"] = days_old
        info["final_score"] = adjusted_score

        adjusted.append((memory, adjusted_score, info))

    # Re-sort by adjusted score
    adjusted.sort(key=lambda x: x[1], reverse=True)

    return adjusted
