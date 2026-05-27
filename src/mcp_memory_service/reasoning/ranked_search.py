"""Multi-signal search ranking (RFC #1008 §2).

Combines semantic similarity with time decay, access frequency, and quality score.
"""

import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RankedSearchWeights:
    """Weights for multi-signal ranking."""
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
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "RankedSearchWeights":
        if not data:
            return cls()

        def _safe_float(val, default: float) -> float:
            if val is None:
                return default
            try:
                return float(val)
            except (TypeError, ValueError):
                return default

        return cls(
            semantic=_safe_float(data.get("semantic"), 0.6),
            time_decay=_safe_float(data.get("time_decay"), 0.2),
            access_frequency=_safe_float(data.get("access_frequency"), 0.1),
            quality=_safe_float(data.get("quality"), 0.1),
        ).normalized()


def compute_ranked_score(
    semantic_score: float,
    memory,
    weights: Optional[RankedSearchWeights] = None,
    now: Optional[float] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Compute final ranked score and signal breakdown."""
    w = (weights or RankedSearchWeights()).normalized()
    ts_now = now or time.time()

    semantic = max(0.0, min(1.0, float(semantic_score)))

    try:
        decay_window = float(os.environ.get("MEMORY_DECAY_WINDOW_DAYS", "30"))
    except (TypeError, ValueError):
        decay_window = 30.0
    reference = memory.last_accessed_at or memory.created_at or ts_now
    days_since = max(0.0, (ts_now - reference) / 86400.0)
    time_decay_score = math.exp(-days_since / max(decay_window, 1.0))

    access_count = max(0, int(memory.access_count or 0))
    access_score = min(1.0, math.log(access_count + 1) / math.log(100))

    quality_score = max(0.0, min(1.0, float(memory.quality_score or 0.0)))

    final = (
        w.semantic * semantic
        + w.time_decay * time_decay_score
        + w.access_frequency * access_score
        + w.quality * quality_score
    )

    breakdown = {
        "semantic_score": round(semantic, 4),
        "time_decay_score": round(time_decay_score, 4),
        "access_score": round(access_score, 4),
        "quality_score": round(quality_score, 4),
        "ranked_score": round(final, 4),
        "weights": {"semantic": w.semantic, "time_decay": w.time_decay, "access_frequency": w.access_frequency, "quality": w.quality},
    }
    return final, breakdown


def apply_ranked_rerank(
    candidates: List[Any],
    weights: Optional[RankedSearchWeights] = None,
    now: Optional[float] = None,
) -> List[Any]:
    """Rerank MemoryQueryResult candidates by multi-signal score."""
    for result in candidates:
        semantic = result.relevance_score
        final, breakdown = compute_ranked_score(semantic, result.memory, weights=weights, now=now)
        if result.debug_info is None:
            result.debug_info = {}
        result.debug_info.update(breakdown)
        result.debug_info["original_semantic_score"] = semantic
        result.debug_info["ranked"] = True
        result.relevance_score = final

    candidates.sort(key=lambda r: r.relevance_score, reverse=True)
    return candidates
