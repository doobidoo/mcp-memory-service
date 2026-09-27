"""Utilities for bootstrap profile generation — dedup and ranking."""

import time
from typing import Dict, List, Optional


def _simple_similarity(a: str, b: str) -> float:
    """Simple word-overlap similarity (Jaccard) as fallback when embeddings unavailable."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def _embedding_similarity(texts: List[str]) -> Optional[List[List[float]]]:
    """Compute pairwise cosine similarity using the loaded embedding model.

    Returns NxN similarity matrix, or None if model unavailable.
    Uses cached model singleton to avoid reloading on every call.
    """
    try:
        from ..embeddings import get_onnx_embedding_model
        import numpy as np
        global _EMBED_MODEL
        if "_EMBED_MODEL" not in globals() or _EMBED_MODEL is None:
            _EMBED_MODEL = get_onnx_embedding_model()
        if _EMBED_MODEL is None:
            return None
        embeddings = _EMBED_MODEL.encode(texts)
        sim_matrix = np.dot(embeddings, embeddings.T)
        return sim_matrix.tolist()
    except Exception:
        return None


_EMBED_MODEL = None


def deduplicate_entries(
    entries: List[Dict],
    similarity_threshold: float = 0.80,
    use_embeddings: bool = True,
) -> List[Dict]:
    """Remove semantically similar entries, keeping the one with highest confidence.

    Uses embedding cosine similarity when available (cross-language capable),
    falls back to Jaccard word overlap.
    """
    if not entries or len(entries) <= 1:
        return entries

    # Sort by confidence descending (keep best first)
    sorted_entries = sorted(entries, key=lambda e: e.get("confidence", 0), reverse=True)

    # Try embedding-based dedup (handles cross-language)
    sim_matrix = None
    if use_embeddings and len(sorted_entries) <= 100:  # Cap to avoid OOM
        texts = [e["content"] for e in sorted_entries]
        sim_matrix = _embedding_similarity(texts)

    kept: List[Dict] = []
    kept_indices: List[int] = []
    for i, entry in enumerate(sorted_entries):
        is_duplicate = False
        for ki in kept_indices:
            if sim_matrix is not None:
                sim = sim_matrix[i][ki]
            else:
                sim = _simple_similarity(entry["content"], sorted_entries[ki]["content"])
            if sim >= similarity_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(entry)
            kept_indices.append(i)

    return kept


def rank_entries(entries: List[Dict], now: float = None) -> List[Dict]:
    """Rank entries by composite score: quality × confidence × recency.

    Args:
        entries: List of dicts with 'confidence', 'quality_score', 'created_at' keys.
        now: Current timestamp (defaults to time.time()).

    Returns:
        Entries sorted by composite score (highest first).
    """
    if now is None:
        now = time.time()

    def _recency_score(created_at: float) -> float:
        age_days = (now - created_at) / 86400
        if age_days < 7:
            return 1.0
        elif age_days < 30:
            return 0.5
        else:
            return 0.2

    def _composite_score(entry: Dict) -> float:
        quality = entry.get("quality_score", 0.5)
        confidence = entry.get("confidence", 0.5)
        created_at = entry.get("created_at", 0)
        recency = _recency_score(created_at)
        return quality * 0.4 + confidence * 0.4 + recency * 0.2

    return sorted(entries, key=_composite_score, reverse=True)
