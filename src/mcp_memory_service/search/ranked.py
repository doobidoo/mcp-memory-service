"""Multi-signal search ranking (RFC #1008 §2).

Combines semantic similarity with time decay, access frequency, and quality score.
Building blocks mirror consolidation/decay.py and quality/implicit_signals.py.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..models.memory import Memory


@dataclass(frozen=True)
class RankedSearchWeights:
    """Weights for multi-signal ranking (must sum to 1.0 after normalization)."""

    semantic: float = 0.6
    time_decay: float = 0.2
    access_frequency: float = 0.1
    quality: float = 0.1

    def normalized(self) -> "RankedSearchWeights":
        total = self.semantic + self.time_decay + self.access_frequency + self.quality
        if total <= 0:
            return RankedSearchWeights()
        return RankedSearchWeights(
            semantic=self.semantic / total,
            time_decay=self.time_decay / total,
            access_frequency=self.access_frequency / total,
            quality=self.quality / total,
        )

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "RankedSearchWeights":
        if not data:
            return cls()
        return cls(
            semantic=float(
                data.get("semantic", data.get("w1", cls.semantic))
            ),
            time_decay=float(
                data.get("time_decay", data.get("w2", cls.time_decay))
            ),
            access_frequency=float(
                data.get("access_frequency", data.get("w3", cls.access_frequency))
            ),
            quality=float(
                data.get("quality", data.get("w4", cls.quality))
            ),
        ).normalized()


def time_decayed_confidence(
    memory: Memory,
    now: Optional[float] = None,
) -> float:
    """Time-decayed confidence aligned with sqlite_vec._effective_confidence."""
    decay_window = float(os.environ.get("MEMORY_DECAY_WINDOW_DAYS", "30"))
    ts_now = now or time.time()

    confidence = memory.metadata.get("confidence")
    if confidence is None:
        confidence = memory.credibility

    last_accessed = memory.last_accessed_at or memory.metadata.get("last_accessed")
    created_at = memory.created_at
    reference = last_accessed or created_at or ts_now
    days_since = max(0.0, (ts_now - reference) / 86400.0)
    retention = max(decay_window, 1.0)
    decay = math.exp(-days_since / retention)
    return round(float(confidence or 1.0) * decay, 4)


def normalized_access_score(access_count: int) -> float:
    """Log-normalized access frequency (implicit_signals pattern)."""
    return min(1.0, math.log(access_count + 1) / math.log(100))


def compute_ranked_score(
    semantic_score: float,
    memory: Memory,
    weights: Optional[RankedSearchWeights] = None,
    now: Optional[float] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Compute final ranked score and signal breakdown."""
    w = (weights or RankedSearchWeights()).normalized()
    ts_now = now or time.time()

    semantic = max(0.0, min(1.0, float(semantic_score)))
    decay_score = time_decayed_confidence(memory, now=ts_now)
    access_score = normalized_access_score(memory.access_count)
    quality_score = max(0.0, min(1.0, memory.quality_score or 0.0))

    final = (
        w.semantic * semantic
        + w.time_decay * decay_score
        + w.access_frequency * access_score
        + w.quality * quality_score
    )

    breakdown = {
        "semantic_score": round(semantic, 4),
        "time_decay_score": decay_score,
        "access_score": round(access_score, 4),
        "quality_score": round(quality_score, 4),
        "ranked_score": round(final, 4),
        "weights": {
            "semantic": w.semantic,
            "time_decay": w.time_decay,
            "access_frequency": w.access_frequency,
            "quality": w.quality,
        },
    }
    return final, breakdown


def apply_ranked_rerank(
    candidates: List[Any],
    *,
    weights: Optional[RankedSearchWeights] = None,
    now: Optional[float] = None,
) -> List[Any]:
    """Rerank MemoryQueryResult candidates in place and return sorted list."""
    for result in candidates:
        semantic = result.relevance_score
        final, breakdown = compute_ranked_score(
            semantic,
            result.memory,
            weights=weights,
            now=now,
        )
        if result.debug_info is None:
            result.debug_info = {}
        result.debug_info.update(breakdown)
        result.debug_info["original_semantic_score"] = semantic
        result.debug_info["ranked"] = True
        result.relevance_score = final

    candidates.sort(key=lambda r: r.relevance_score, reverse=True)
    return candidates
