"""Memory quarantine — holds memories that contradict active beliefs."""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional, List

logger = logging.getLogger(__name__)

CONTRADICTION_THRESHOLD = int(os.getenv("MCP_QUARANTINE_CONTRADICTION_THRESHOLD", "3"))


async def quarantine_memory(storage, content_hash: str, contradicted_belief_hash: str, reason: str = "") -> dict:
    """Quarantine a memory that contradicts an active belief."""
    try:
        quarantine_meta = {
            "quarantined": True,
            "quarantined_at": datetime.now(timezone.utc).isoformat(),
            "contradicted_belief": contradicted_belief_hash,
            "quarantine_reason": reason,
        }
        await storage.update_memory_metadata(
            content_hash=content_hash,
            updates={"metadata": quarantine_meta, "tags": ["quarantined"]},
            preserve_timestamps=True,
        )
        return {"status": "quarantined", "content_hash": content_hash, "belief": contradicted_belief_hash}
    except Exception as e:
        logger.error(f"Failed to quarantine memory: {e}")
        return {"status": "error", "message": str(e)}


async def unquarantine_memory(storage, content_hash: str) -> dict:
    """Remove quarantine from a memory."""
    try:
        quarantine_meta = {
            "quarantined": False,
            "unquarantined_at": datetime.now(timezone.utc).isoformat(),
        }
        await storage.update_memory_metadata(
            content_hash=content_hash,
            updates={"metadata": quarantine_meta},
            preserve_timestamps=True,
        )
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

    for belief in beliefs[:20]:
        result = await classifier.classify(belief["content"], content)
        if result.label == "contradiction" and result.confidence >= 0.7:
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
                    "quarantined_at": meta.get("quarantined_at"),
                    "reason": meta.get("quarantine_reason", ""),
                })
        return quarantined
    except Exception:
        return []
