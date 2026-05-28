"""Multi-strategy retrieval with RRF fusion (RFC #1008 §6)."""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

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


async def _run_semantic(storage, query: str, limit: int) -> Tuple[List[str], Dict[str, Dict]]:
    """Run semantic strategy, return (hashes, memory_map)."""
    result = await storage.search_memories(query=query, mode="semantic", limit=limit * 2)
    memories = result.get("memories", []) if isinstance(result, dict) else []
    hashes = []
    mem_map: Dict[str, Dict] = {}
    for m in memories:
        h = m.get("content_hash", "")
        if h:
            hashes.append(h)
            if h not in mem_map:
                mem_map[h] = m
    return hashes, mem_map


async def _run_tag(storage, tags: List[str], limit: int) -> Tuple[List[str], Dict[str, Dict]]:
    """Run tag strategy, return (hashes, memory_map)."""
    results = await storage.search_by_tag_chronological(tags, limit=limit * 2)
    hashes = []
    mem_map: Dict[str, Dict] = {}
    for m in results:
        h = m.content_hash if hasattr(m, 'content_hash') else m.get("content_hash", "")
        if h:
            hashes.append(h)
            if h not in mem_map:
                mem_map[h] = {"content_hash": h, "content": getattr(m, 'content', ''), "tags": getattr(m, 'tags', [])}
    return hashes, mem_map


async def multi_strategy_search(
    storage,
    query: str,
    limit: int = 10,
    strategies: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run multiple search strategies concurrently and fuse results via RRF."""
    strategies = strategies or ["semantic"]
    tasks = []

    for strategy in strategies:
        if strategy == "semantic":
            tasks.append(_run_semantic(storage, query, limit))
        elif strategy == "tag" and tags:
            tasks.append(_run_tag(storage, tags, limit))

    if not tasks:
        return {"memories": [], "total": 0, "query": query, "mode": "multi_strategy"}

    # Run all strategies concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    ranked_lists: List[List[str]] = []
    all_memories: Dict[str, Dict] = {}

    for result in results:
        if isinstance(result, Exception):
            logger.warning(f"Strategy failed: {result}")
            continue
        hashes, mem_map = result
        if hashes:
            ranked_lists.append(hashes)
            all_memories.update(mem_map)

    if not ranked_lists:
        return {"memories": [], "total": 0, "query": query, "mode": "multi_strategy"}

    fused_hashes = rrf_fuse(ranked_lists, k=60, limit=limit)
    memories = [all_memories[h] for h in fused_hashes if h in all_memories]
    return {"memories": memories, "total": len(memories), "query": query, "mode": "multi_strategy"}
