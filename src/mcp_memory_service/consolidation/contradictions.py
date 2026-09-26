"""Temporal Contradiction Detection — Phase 4 of #732.

Detects when newer memories contradict older ones using embedding similarity
in the 0.4-0.75 band (too similar to be independent, too different to be duplicates).

Output: a ``contradicts`` graph edge plus supersession of the older memory
(via ``storage.mark_superseded_batch``, so it drops out of default retrieval).
Integration: maintain Step 7 + opt-in MCP_CONTRADICTION_ON_STORE=true.
"""

import logging
import os

from .base import is_protected_memory

logger = logging.getLogger(__name__)


def _sanitize_log_value(value: object) -> str:
    """Sanitize a user-provided value for safe inclusion in log messages."""
    return str(value).replace("\n", "\\n").replace("\r", "\\r").replace("\x1b", "\\x1b")

# Configuration
CONTRADICTION_ENABLED = os.environ.get("MCP_CONTRADICTION_DETECTION_ENABLED", "false").lower() == "true"
CONTRADICTION_ON_STORE = os.environ.get("MCP_CONTRADICTION_ON_STORE", "false").lower() == "true"
SIMILARITY_MIN = float(os.environ.get("MCP_CONTRADICTION_SIM_MIN", "0.4"))
SIMILARITY_MAX = float(os.environ.get("MCP_CONTRADICTION_SIM_MAX", "0.75"))
KNN_K = int(os.environ.get("MCP_CONTRADICTION_KNN_K", "10"))

# Errors that mean this module is calling the storage API wrong. They are never
# treated as a per-memory miss: that is how a call to a method that does not
# exist went unnoticed.
_PROGRAMMING_ERRORS = (AttributeError, TypeError, NameError)


async def _store_contradicts_edge(graph, newer_hash: str, older_hash: str, similarity: float) -> bool:
    return await graph.store_association(
        source_hash=newer_hash,
        target_hash=older_hash,
        similarity=similarity,
        connection_types=["contradiction"],
        relationship_type="contradicts",
        metadata={"method": "similarity_band"},
    )


def _types_compatible(memory_type, cand_type) -> bool:
    # None is wildcard — matches any
    return not (memory_type and cand_type and memory_type != cand_type)


def _band_candidates(content_hash: str, memory_type, similar: list):
    """Yield (cand_hash, similarity, cand) for neighbours inside the contradiction band."""
    for candidate in similar:
        # candidates are plain dicts (from Memory.to_dict() + similarity_score key)
        cand_hash = candidate.get("content_hash")
        similarity = candidate.get("similarity_score", 0)

        if not cand_hash or cand_hash == content_hash:
            continue
        if not SIMILARITY_MIN <= similarity <= SIMILARITY_MAX:
            continue
        # to_dict() uses "type", not "memory_type"
        if _types_compatible(memory_type, candidate.get("type")):
            yield cand_hash, similarity, candidate


def _is_scannable(memory, losers: set) -> bool:
    if not memory.content or not memory.content_hash or memory.content_hash in losers:
        return False
    # Backends that keep supersession in metadata (Milvus)
    return not (memory.metadata or {}).get("superseded_by")


def _finds_itself(content_hash: str, similar: list) -> bool:
    # get_all_memories() also returns superseded rows, but search excludes
    # them. A memory that does not find itself is hidden from retrieval and
    # must not be allowed to supersede anything.
    return any(c.get("content_hash") == content_hash for c in similar)


def _order_pair(memory, cand_hash: str, candidate: dict) -> tuple:
    """Return (older_hash, newer_hash) by created_at float timestamp."""
    if (memory.created_at or 0) < (candidate.get("created_at") or 0):
        return memory.content_hash, cand_hash
    return cand_hash, memory.content_hash


async def _loser_protected(older_hash: str, state: dict) -> bool:
    # Same protection forgetting and decay apply. get_all_memories() can be
    # capped (Milvus returns the newest 16,384), so a loser missing from the
    # scan is fetched by hash and the hit cached for the rest of the run. A
    # miss is not cached: Milvus get_by_hash() returns None on a transient
    # backend error, so a later pair retries. One that stays unfound is left
    # alone.
    by_hash = state["by_hash"]
    older = by_hash.get(older_hash)
    if older is None:
        older = await state["storage"].get_by_hash(older_hash)
        if older is not None:
            by_hash[older_hash] = older
    return older is None or is_protected_memory(older)


async def _collect_pairs(memory, similar: list, state: dict, results: dict) -> None:
    """Append this memory's contradiction pairs, each pair and each loser at most once."""
    losers = state["losers"]
    for cand_hash, similarity, candidate in _band_candidates(memory.content_hash, memory.memory_type, similar):
        key = frozenset((memory.content_hash, cand_hash))
        if key in state["seen_pairs"]:
            continue
        state["seen_pairs"].add(key)

        older_hash, newer_hash = _order_pair(memory, cand_hash, candidate)
        # A memory superseded earlier in this run can neither lose again nor win.
        if older_hash in losers or newer_hash in losers:
            continue
        if await _loser_protected(older_hash, state):
            results["protected_skipped"] += 1
            continue
        losers.add(older_hash)
        state["pairs"].append({"older_hash": older_hash, "newer_hash": newer_hash, "similarity": similarity})


async def _scan(storage, memories: list, results: dict) -> list:
    state = {
        "storage": storage,
        "seen_pairs": set(),
        "losers": set(),
        "pairs": [],
        "by_hash": {m.content_hash: m for m in memories if m.content_hash},
    }
    for memory in memories:
        if not _is_scannable(memory, state["losers"]):
            continue
        similar = await _search_neighbours(storage, memory.content, results)
        if similar and _finds_itself(memory.content_hash, similar):
            await _collect_pairs(memory, similar, state, results)
    return state["pairs"]


async def _search_neighbours(storage, content: str, results: dict):
    """KNN search for one memory. Returns the hit list, or None on a transient failure."""
    try:
        search_result = await storage.search_memories(query=content, limit=KNN_K)
    except _PROGRAMMING_ERRORS:
        raise
    except Exception as e:
        results["search_errors"] += 1
        results["last_search_error"] = str(e)
        return None
    return search_result.get("memories", []) if isinstance(search_result, dict) else []


async def _apply(storage, graph, pairs: list, results: dict) -> None:
    """Supersede the older memory of each pair and record a contradicts edge."""
    results["superseded_marked"] = await storage.mark_superseded_batch(
        [(p["newer_hash"], p["older_hash"]) for p in pairs]
    )
    if graph is None:
        return
    for p in pairs:
        if await _store_contradicts_edge(graph, p["newer_hash"], p["older_hash"], p["similarity"]):
            results["edges_created"] += 1
        else:
            results["edge_failures"] += 1


async def detect_contradictions(storage, dry_run: bool = True, graph=None) -> dict:
    """Scan all memories for contradictions using embedding similarity band.

    Args:
        storage: memory storage backend.
        dry_run: report pairs without writing anything.
        graph: graph storage for ``contradicts`` edges (``get_graph_storage()``).
            When None, supersession is still applied but no edges are written.

    Returns dict with detected pairs and actions taken.
    """
    if not CONTRADICTION_ENABLED:
        return {"skipped": True, "reason": "MCP_CONTRADICTION_DETECTION_ENABLED=false"}

    results = {
        "pairs_detected": 0,
        "edges_created": 0,
        "edge_failures": 0,
        "superseded_marked": 0,
        "protected_skipped": 0,
        "dry_run": dry_run,
        "graph_available": graph is not None,
        "search_errors": 0,
        "pairs": [],
    }

    try:
        # Get all memories — returns List[Memory] dataclass instances
        memories = await storage.get_all_memories()

        if not memories:
            return {**results, "message": "No memories to scan"}
        logger.info("[contradiction] Scanning %s memories for contradictions", _sanitize_log_value(len(memories)))

        pairs = await _scan(storage, memories, results)
        results["pairs"] = [
            {"older": p["older_hash"][:12], "newer": p["newer_hash"][:12], "similarity": round(p["similarity"], 3)}
            for p in pairs
        ]
        results["pairs_detected"] = len(pairs)
        if pairs and not dry_run:
            await _apply(storage, graph, pairs, results)

        logger.info(
            "[contradiction] Done: %s pairs, %s edges, %s superseded, %s search errors",
            _sanitize_log_value(results['pairs_detected']),
            _sanitize_log_value(results['edges_created']),
            _sanitize_log_value(results['superseded_marked']),
            _sanitize_log_value(results['search_errors']),
        )

    except Exception as e:
        results["error"] = str(e)
        logger.error(f"[contradiction] Error: {_sanitize_log_value(e)}", exc_info=True)

    return results


async def _supersede_on_store(storage, graph, content_hash: str, cand_hash: str, similarity: float) -> bool:
    """Let the new memory supersede an existing one, unless that one is protected."""
    existing = await storage.get_by_hash(cand_hash)
    if existing is None or is_protected_memory(existing):
        return False
    await storage.mark_superseded_batch([(content_hash, cand_hash)])
    if graph is not None:
        await _store_contradicts_edge(graph, content_hash, cand_hash, similarity)
    return True


async def check_contradiction_on_store(storage, content: str, content_hash: str, graph=None) -> dict | None:
    """Check if a newly stored memory contradicts existing ones.

    Intended for memory_store when MCP_CONTRADICTION_ON_STORE=true.
    Returns contradiction info if found, None otherwise.
    """
    if not CONTRADICTION_ON_STORE:
        return None

    try:
        search_result = await storage.search_memories(query=content, limit=KNN_K)
        similar = search_result.get("memories", []) if isinstance(search_result, dict) else []

        for cand_hash, similarity, _ in _band_candidates(content_hash, None, similar):
            if await _supersede_on_store(storage, graph, content_hash, cand_hash, similarity):
                return {
                    "contradicts": cand_hash[:12],
                    "similarity": round(similarity, 3),
                    "action": "older memory marked as superseded",
                }

    except _PROGRAMMING_ERRORS:
        raise
    except Exception as e:
        logger.warning(f"[contradiction-on-store] Error: {_sanitize_log_value(e)}")

    return None
