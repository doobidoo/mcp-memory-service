"""Memory quarantine — holds memories that contradict active beliefs."""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional, List

from ..compat import _sanitize_log_value

logger = logging.getLogger(__name__)

CONTRADICTION_THRESHOLD = int(os.getenv("MCP_QUARANTINE_CONTRADICTION_THRESHOLD", "3"))

# Minimum NLI confidence for a contradiction to quarantine a memory on store.
# Configurable via MCP_QUARANTINE_NLI_THRESHOLD (issue #1216); read at call time
# so runtime config and tests take effect. The default heuristic backend tops
# out below this (see NLIClassifier.max_achievable_confidence), so the default
# config needs MCP_NLI_BACKEND=cascade or a lowered gate to quarantine on store.
DEFAULT_QUARANTINE_NLI_THRESHOLD = 0.7

# Backends already warned about below (one warning per backend per process).
_gate_warned_backends: set = set()


def _quarantine_nli_threshold() -> float:
    """The on-store quarantine gate, parsed with the repo's fallback pattern.

    An unparsable or out-of-range MCP_QUARANTINE_NLI_THRESHOLD logs an error
    and falls back to the default instead of raising — a raise here would be
    swallowed by the callers' broad handlers and silently disable quarantine.
    """
    from ..config import safe_get_float_env
    return safe_get_float_env(
        "MCP_QUARANTINE_NLI_THRESHOLD", DEFAULT_QUARANTINE_NLI_THRESHOLD,
        min_value=0.0, max_value=1.0,
    )


def _warn_if_gate_unreachable(classifier, threshold: float) -> None:
    """Emit one warning per process if no contradiction could ever meet the gate.

    Makes the previously-silent dead configuration (issue #1216) visible: a gate
    above the active backend's achievable ceiling means quarantine-on-store can
    never fire. Robust to a mocked classifier (a non-numeric ceiling is skipped).
    """
    backend = getattr(classifier, "backend", "?")
    if backend in _gate_warned_backends:
        return
    ceiling = getattr(classifier, "max_achievable_confidence", None)
    if ceiling is None:
        return
    try:
        ceiling = ceiling()
    except Exception:
        return
    if isinstance(ceiling, (int, float)) and not isinstance(ceiling, bool) and threshold > ceiling:
        _gate_warned_backends.add(backend)
        logger.warning(
            "MCP_QUARANTINE_NLI_THRESHOLD=%s exceeds the '%s' NLI backend's maximum "
            "achievable confidence (%s); no contradiction can be quarantined on store "
            "with this configuration. Lower MCP_QUARANTINE_NLI_THRESHOLD to <= %s, or set "
            "MCP_NLI_BACKEND=cascade.",
            threshold, backend, ceiling, ceiling,
        )


async def quarantine_memory(
    storage,
    content_hash: str,
    contradicted_belief_hash: Optional[str],
    reason: str = "",
    contradicted_memory_hash: Optional[str] = None,
) -> dict:
    """Quarantine a memory that contradicts an active belief — or, for a
    value-swap rescued from semantic dedup (issue #1216), another *memory*.

    The two are recorded in distinct fields (``contradicted_belief`` /
    ``contradicted_memory``) so consumers can tell a belief contradiction from
    a memory collision; a memory hash never counts toward a belief's
    contradiction tally.
    """
    try:
        quarantine_meta = {
            "quarantined": True,
            "quarantined_at": datetime.now(timezone.utc).isoformat(),
            "contradicted_belief": contradicted_belief_hash,
            "quarantine_reason": reason,
        }
        result = {"status": "quarantined", "content_hash": content_hash, "belief": contradicted_belief_hash}
        if contradicted_memory_hash:
            quarantine_meta["contradicted_memory"] = contradicted_memory_hash
            result["memory"] = contradicted_memory_hash
        ok, msg = await storage.update_memory_metadata(
            content_hash=content_hash,
            updates={"metadata": quarantine_meta, "tags": ["quarantined"]},
            preserve_timestamps=True,
        )
        if not ok:
            # update_memory_metadata reports ordinary failures by returning
            # (False, message) instead of raising. Treating a non-raising
            # failure as success would report a memory as quarantined while it
            # actually stays dedup-bypassed and active — the very leak #1216
            # asks to close. Propagate the error so _file_contradiction and
            # check_beliefs_on_store see status != "quarantined".
            logger.error("Failed to quarantine memory %s: %s", content_hash[:8], _sanitize_log_value(msg))
            return {"status": "error", "message": msg}
        return result
    except Exception as e:
        logger.error("Failed to quarantine memory: %s", _sanitize_log_value(str(e)))
        return {"status": "error", "message": str(e)}


async def unquarantine_memory(storage, content_hash: str) -> dict:
    """Remove quarantine from a memory."""
    try:
        quarantine_meta = {
            "quarantined": False,
            "unquarantined_at": datetime.now(timezone.utc).isoformat(),
        }
        ok, msg = await storage.update_memory_metadata(
            content_hash=content_hash,
            updates={"metadata": quarantine_meta},
            preserve_timestamps=True,
        )
        if not ok:
            # Same honest-reporting rule as quarantine_memory: a non-raising
            # (False, message) failure must not be reported as unquarantined.
            logger.error("Failed to unquarantine memory %s: %s", content_hash[:8], _sanitize_log_value(msg))
            return {"status": "error", "message": msg}
        return {"status": "unquarantined", "content_hash": content_hash}
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def check_beliefs_on_store(storage, belief_service, content: str, content_hash: str) -> Optional[dict]:
    """Check if new memory contradicts any active belief.

    Called from the on_store path (MCP_NLI_ON_STORE=true).
    """
    from ..reasoning.nli import NLIClassifier

    beliefs = await belief_service.get_beliefs(status="active", min_confidence=0.35)
    if not beliefs:
        return None

    classifier = NLIClassifier(backend="auto")
    threshold = _quarantine_nli_threshold()
    _warn_if_gate_unreachable(classifier, threshold)

    for belief in beliefs[:20]:
        result = await classifier.classify(belief["content"], content)
        if result.label == "contradiction" and result.confidence >= threshold:
            q_result = await quarantine_memory(
                storage, content_hash, belief["belief_hash"],
                reason=f"Contradicts belief: {belief['content'][:100]}",
            )

            contradiction_count = await _count_quarantined_for_belief(storage, belief["belief_hash"])
            if contradiction_count >= CONTRADICTION_THRESHOLD:
                await belief_service.challenge_belief(belief["belief_hash"])
                logger.info(f"Belief {belief['belief_hash'][:8]} challenged after {contradiction_count} contradictions")

            return q_result

    return None


async def _count_quarantined_for_belief(storage, belief_hash: str) -> int:
    """Count memories quarantined due to a specific belief."""
    try:
        results = await storage.search_by_tag(["quarantined"])
        count = 0
        for mem in results:
            meta = mem.metadata if hasattr(mem, "metadata") else {}
            if isinstance(meta, str):
                meta = json.loads(meta) if meta else {}
            if meta.get("contradicted_belief") == belief_hash:
                count += 1
        return count
    except Exception:
        return 0


async def get_quarantined_memories(storage, limit: int = 50) -> List[dict]:
    """List all quarantined memories."""
    try:
        results = await storage.search_by_tag(["quarantined"])
        quarantined = []
        for mem in results[:limit]:
            meta = mem.metadata if hasattr(mem, "metadata") else {}
            if isinstance(meta, str):
                meta = json.loads(meta) if meta else {}
            if meta.get("quarantined"):
                quarantined.append({
                    "content_hash": mem.content_hash,
                    "content": mem.content[:200],
                    "contradicted_belief": meta.get("contradicted_belief"),
                    "contradicted_memory": meta.get("contradicted_memory"),
                    "quarantined_at": meta.get("quarantined_at"),
                    "reason": meta.get("quarantine_reason", ""),
                })
        return quarantined
    except Exception:
        return []
