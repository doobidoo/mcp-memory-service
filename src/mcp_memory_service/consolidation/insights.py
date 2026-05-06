"""Insight Cards generator for memory consolidation (Phase 3, #732).

Generates pattern/trend/gap insights from memory clusters using heuristics only.
Not integrated into the consolidator yet — standalone module for PR #864 hook.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple
import hashlib

from ..models.memory import Memory


@dataclass
class InsightCard:
    """A generated insight derived from memory patterns."""
    title: str
    content: str
    source_hashes: List[str]
    insight_type: str  # 'pattern', 'trend', 'gap'
    confidence: float  # 0.0 - 1.0


class InsightGenerator:
    """Generates Insight Cards from memories using heuristics."""

    MIN_PATTERN_MEMORIES = 3
    MIN_SHARED_TAGS = 2
    RECENT_DAYS = 7
    OLD_DAYS = 30
    GAP_MIN_MEMORIES = 5

    def generate_insights(
        self, memories: List[dict], clusters: List[List[dict]]
    ) -> List[InsightCard]:
        """Generate insights from memories and their clusters.

        Args:
            memories: List of memory dicts (must have 'content_hash', 'tags',
                      'memory_type', 'created_at').
            clusters: List of clusters (each cluster is a list of memory dicts).

        Returns:
            List of InsightCard objects.
        """
        insights: List[InsightCard] = []
        insights.extend(self._detect_patterns(memories))
        insights.extend(self._detect_trends(memories))
        insights.extend(self._detect_gaps(memories))
        return insights

    def _detect_patterns(self, memories: List[dict]) -> List[InsightCard]:
        """Pattern: >=3 memories share 2+ tags → insight about the pattern."""
        # Group memories by tag pairs
        pair_memories: Dict[tuple, List[dict]] = defaultdict(list)
        for mem in memories:
            tags = sorted(set(mem.get("tags", [])))
            for i in range(len(tags)):
                for j in range(i + 1, len(tags)):
                    pair_memories[(tags[i], tags[j])].append(mem)

        insights = []
        for tag_pair, mems in pair_memories.items():
            if len(mems) >= self.MIN_PATTERN_MEMORIES:
                hashes = [m["content_hash"] for m in mems]
                confidence = min(1.0, len(mems) / 10.0)
                insights.append(InsightCard(
                    title=f"Recurring pattern: {tag_pair[0]} + {tag_pair[1]}",
                    content=(
                        f"{len(mems)} memories share tags '{tag_pair[0]}' and "
                        f"'{tag_pair[1]}', indicating a recurring theme."
                    ),
                    source_hashes=hashes,
                    insight_type="pattern",
                    confidence=confidence,
                ))
        return insights

    def _detect_trends(self, memories: List[dict]) -> List[InsightCard]:
        """Trend: recent memories (7d) diverge from older (30d) on same tag."""
        now = datetime.now(timezone.utc).timestamp()
        recent_cutoff = now - (self.RECENT_DAYS * 86400)
        old_cutoff = now - (self.OLD_DAYS * 86400)

        recent_tags: Dict[str, List[dict]] = defaultdict(list)
        old_tags: Dict[str, List[dict]] = defaultdict(list)

        for mem in memories:
            created = mem.get("created_at")
            if created is None:
                continue
            for tag in mem.get("tags", []):
                if created >= recent_cutoff:
                    recent_tags[tag].append(mem)
                elif created <= old_cutoff:
                    old_tags[tag].append(mem)

        insights = []
        for tag in recent_tags:
            if tag not in old_tags:
                continue
            recent_types = set(m.get("memory_type", "") for m in recent_tags[tag])
            old_types = set(m.get("memory_type", "") for m in old_tags[tag])
            if recent_types != old_types and len(recent_tags[tag]) >= 2:
                hashes = (
                    [m["content_hash"] for m in recent_tags[tag]] +
                    [m["content_hash"] for m in old_tags[tag]]
                )
                insights.append(InsightCard(
                    title=f"Trend shift in '{tag}'",
                    content=(
                        f"Recent memories tagged '{tag}' show different type distribution "
                        f"({sorted(recent_types)}) vs older ones ({sorted(old_types)})."
                    ),
                    source_hashes=hashes,
                    insight_type="trend",
                    confidence=0.6,
                ))
        return insights

    def _detect_gaps(self, memories: List[dict]) -> List[InsightCard]:
        """Gap: tag has many memories but none of type 'decision'."""
        tag_mems: Dict[str, List[dict]] = defaultdict(list)
        tag_has_decision: Dict[str, bool] = defaultdict(bool)

        for mem in memories:
            for tag in mem.get("tags", []):
                tag_mems[tag].append(mem)
                if mem.get("memory_type") == "decision":
                    tag_has_decision[tag] = True

        insights = []
        for tag, mems in tag_mems.items():
            if len(mems) >= self.GAP_MIN_MEMORIES and not tag_has_decision[tag]:
                hashes = [m["content_hash"] for m in mems]
                insights.append(InsightCard(
                    title=f"Decision gap: '{tag}'",
                    content=(
                        f"Tag '{tag}' has {len(mems)} memories but no decisions recorded. "
                        f"Consider documenting key decisions for this area."
                    ),
                    source_hashes=hashes,
                    insight_type="gap",
                    confidence=0.5,
                ))
        return insights


async def store_insights(insights: List[InsightCard], storage) -> List[str]:
    """Store InsightCards as memories and create derived_from edges.

    Args:
        insights: List of InsightCard to persist.
        storage: Storage backend with store() and store_association() methods.

    Returns:
        List of content hashes for stored insight memories.
    """
    stored_hashes = []
    for card in insights:
        content_hash = _insight_hash(card)

        # Check deduplication — skip if already stored
        existing = getattr(storage, "memories", {})
        if hasattr(storage, "get_memory_by_hash"):
            try:
                if await storage.get_memory_by_hash(content_hash):
                    continue
            except Exception:
                pass
        elif content_hash in existing:
            continue

        memory = Memory(
            content=f"[{card.insight_type.upper()}] {card.title}\n\n{card.content}",
            content_hash=content_hash,
            tags=["auto-generated", "insight-card", card.insight_type],
            memory_type="insight",
            metadata={
                "source_hashes": card.source_hashes,
                "insight_type": card.insight_type,
                "confidence": card.confidence,
            },
            created_at=datetime.now(timezone.utc).timestamp(),
            created_at_iso=datetime.now(timezone.utc).isoformat(),
        )

        success, _ = await storage.store(memory)
        if not success:
            continue
        stored_hashes.append(content_hash)

        # Create derived_from edges
        if hasattr(storage, "store_association"):
            for src_hash in card.source_hashes:
                try:
                    await storage.store_association(
                        source_hash=src_hash,
                        target_hash=content_hash,
                        similarity=card.confidence,
                        connection_types=["derived_from"],
                        relationship_type="derived_from",
                    )
                except Exception:
                    pass  # Best-effort edge creation

    return stored_hashes


def _insight_hash(card: InsightCard) -> str:
    """Deterministic hash for deduplication."""
    key = f"{card.insight_type}:{card.title}:{sorted(card.source_hashes)}"
    return f"insight_{hashlib.sha256(key.encode()).hexdigest()[:16]}"
