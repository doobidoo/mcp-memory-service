"""Multi-strategy retrieval with RRF fusion (RFC #1008 §6)."""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def rrf_fuse(ranked_lists: List[List[str]], k: int = 60, limit: Optional[int] = None) -> List[str]:
    """Reciprocal Rank Fusion — merge multiple ranked lists into one."""
    scores: Dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    sorted_items = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    if limit:
        sorted_items = sorted_items[:limit]
    return sorted_items


async def multi_strategy_search(
    storage,
    query: str,
    limit: int = 10,
    strategies: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run multiple search strategies and fuse results via RRF."""
    strategies = strategies or ["semantic"]
    ranked_lists: List[List[str]] = []
    all_memories: Dict[str, Dict] = {}

    for strategy in strategies:
        try:
            if strategy == "semantic":
                result = await storage.search_memories(query=query, mode="semantic", limit=limit * 2)
                memories = result.get("memories", []) if isinstance(result, dict) else []
                hashes = []
                for m in memories:
                    h = m.get("content_hash", "")
                    if h:
                        hashes.append(h)
                        if h not in all_memories:
                            all_memories[h] = m
                ranked_lists.append(hashes)

            elif strategy == "tag" and tags:
                results = await storage.search_by_tag(tags)
                hashes = []
                for m in results:
                    h = m.content_hash if hasattr(m, 'content_hash') else m.get("content_hash", "")
                    if h:
                        hashes.append(h)
                        if h not in all_memories:
                            all_memories[h] = {"content_hash": h, "content": getattr(m, 'content', ''), "tags": getattr(m, 'tags', [])}
                ranked_lists.append(hashes)
        except Exception as e:
            logger.warning(f"Strategy '{strategy}' failed: {e}")
            continue

    if not ranked_lists:
        return {"memories": [], "total": 0, "query": query, "mode": "multi_strategy"}

    fused_hashes = rrf_fuse(ranked_lists, k=60, limit=limit)
    memories = [all_memories[h] for h in fused_hashes if h in all_memories]
    return {"memories": memories, "total": len(memories), "query": query, "mode": "multi_strategy"}
