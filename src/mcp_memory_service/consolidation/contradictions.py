"""Temporal Contradiction Detection — Phase 4 of #732.

Detects when newer memories contradict older ones using embedding similarity
in the 0.4-0.75 band (too similar to be independent, too different to be duplicates).

Output: CONTRADICTED_BY graph edge + superseded_by on the older memory.
Integration: maintain Step 7 + opt-in MCP_CONTRADICTION_ON_STORE=true.
"""

import logging
import os

logger = logging.getLogger(__name__)

# Configuration
CONTRADICTION_ENABLED = os.environ.get("MCP_CONTRADICTION_DETECTION_ENABLED", "false").lower() == "true"
CONTRADICTION_ON_STORE = os.environ.get("MCP_CONTRADICTION_ON_STORE", "false").lower() == "true"
SIMILARITY_MIN = float(os.environ.get("MCP_CONTRADICTION_SIM_MIN", "0.4"))
SIMILARITY_MAX = float(os.environ.get("MCP_CONTRADICTION_SIM_MAX", "0.75"))
KNN_K = int(os.environ.get("MCP_CONTRADICTION_KNN_K", "10"))


async def detect_contradictions(storage, dry_run: bool = True) -> dict:
    """Scan all memories for contradictions using embedding similarity band.

    Returns dict with detected pairs and actions taken.
    """
    if not CONTRADICTION_ENABLED:
        return {"skipped": True, "reason": "MCP_CONTRADICTION_DETECTION_ENABLED=false"}

    results = {
        "pairs_detected": 0,
        "edges_created": 0,
        "superseded_marked": 0,
        "dry_run": dry_run,
        "pairs": [],
    }

    try:
        # Get all memories — returns List[Memory] dataclass instances
        memories = await storage.get_all_memories()

        if not memories:
            return {**results, "message": "No memories to scan"}
        logger.info(f"[contradiction] Scanning {len(memories)} memories for contradictions")

        # For each memory, find KNN neighbors in the similarity band
        for memory in memories:
            content_hash = memory.content_hash
            memory_type = memory.memory_type
            content = memory.content

            if not content or not content_hash:
                continue

            # Skip if already superseded (superseded_by lives in metadata dict)
            if (memory.metadata or {}).get("superseded_by"):
                continue

            # Search for similar memories — returns {"memories": [dict, ...], ...}
            try:
                search_result = await storage.search_memories(
                    query=content,
                    limit=KNN_K,
                )
                similar = search_result.get("memories", []) if isinstance(search_result, dict) else []
            except Exception:
                continue

            if not similar:
                continue

            for candidate in similar:
                # candidates are plain dicts (from Memory.to_dict() + similarity_score key)
                cand_hash = candidate.get("content_hash")
                similarity = candidate.get("similarity_score", 0)

                # Skip self
                if cand_hash == content_hash:
                    continue

                # Only consider the contradiction band
                if similarity < SIMILARITY_MIN or similarity > SIMILARITY_MAX:
                    continue

                # Skip if types differ (None is wildcard — matches any)
                cand_type = candidate.get("type")  # to_dict() uses "type", not "memory_type"
                if memory_type and cand_type and memory_type != cand_type:
                    continue

                # Determine which is older (by created_at float timestamp)
                mem_created = memory.created_at or 0
                cand_created = candidate.get("created_at") or 0

                if mem_created < cand_created:
                    older_hash, newer_hash = content_hash, cand_hash
                else:
                    older_hash, newer_hash = cand_hash, content_hash

                pair = {
                    "older": older_hash[:12],
                    "newer": newer_hash[:12],
                    "similarity": round(similarity, 3),
                }
                results["pairs"].append(pair)
                results["pairs_detected"] += 1

                if not dry_run:
                    # Add graph edge
                    try:
                        await storage.add_graph_edge(older_hash, newer_hash, "CONTRADICTED_BY")
                        results["edges_created"] += 1
                    except Exception as e:
                        logger.warning(f"[contradiction] Failed to add edge: {e}")

                    # Mark older as superseded
                    try:
                        await storage.update_memory_metadata(
                            older_hash,
                            {"superseded_by": newer_hash}
                        )
                        results["superseded_marked"] += 1
                    except Exception as e:
                        logger.warning(f"[contradiction] Failed to mark superseded: {e}")

        logger.info(
            f"[contradiction] Done: {results['pairs_detected']} pairs, "
            f"{results['edges_created']} edges, {results['superseded_marked']} superseded"
        )

    except Exception as e:
        results["error"] = str(e)
        logger.error(f"[contradiction] Error: {e}", exc_info=True)

    return results


async def check_contradiction_on_store(storage, content: str, content_hash: str) -> dict | None:
    """Check if a newly stored memory contradicts existing ones.

    Called during memory_store when MCP_CONTRADICTION_ON_STORE=true.
    Returns contradiction info if found, None otherwise.
    """
    if not CONTRADICTION_ON_STORE:
        return None

    try:
        search_result = await storage.search_memories(query=content, limit=KNN_K)
        similar = search_result.get("memories", []) if isinstance(search_result, dict) else []
        if not similar:
            return None

        for candidate in similar:
            cand_hash = candidate.get("content_hash")
            similarity = candidate.get("similarity_score", 0)

            if cand_hash == content_hash:
                continue

            if SIMILARITY_MIN <= similarity <= SIMILARITY_MAX:
                # Found potential contradiction — mark it
                await storage.add_graph_edge(cand_hash, content_hash, "CONTRADICTED_BY")
                await storage.update_memory_metadata(cand_hash, {"superseded_by": content_hash})

                return {
                    "contradicts": cand_hash[:12],
                    "similarity": round(similarity, 3),
                    "action": "older memory marked as superseded",
                }

    except Exception as e:
        logger.warning(f"[contradiction-on-store] Error: {e}")

    return None
