"""Insight Cards generator for memory consolidation (Phase 3, #732).

Generates pattern/trend/gap insights from memory clusters using heuristics only.
Integrated into the maintain cycle (memory_quality action='maintain') as Step 5.
Opt-in via MCP_INSIGHT_CARDS_ENABLED=true environment variable.
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

    MIN_PATTERN_MEMORIES = 5
    MIN_SHARED_TAGS = 3
    RECENT_DAYS = 7
    OLD_DAYS = 30
    GAP_MIN_MEMORIES = 5

    # Tags that are status/metadata markers, not knowledge domains — skip gap detection.
    EXCLUDED_GAP_TAGS: frozenset = frozenset({
        "conflict:unresolved", "automated", "__test__", "temporary",
        "processed", "auto-generated", "insight-card",
    })

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
        # Group memories by tag pairs, then consolidate by source set
        pair_memories: Dict[tuple, List[dict]] = defaultdict(list)
        for mem in memories:
            tags = sorted(set(t for t in (mem.get("tags") or []) if t is not None))
            for i in range(len(tags)):
                for j in range(i + 1, len(tags)):
                    pair_memories[(tags[i], tags[j])].append(mem)

        # Consolidate: group by source memory set to avoid redundant insights
        seen_source_sets: set = set()
        insights = []
        for tag_pair, mems in sorted(pair_memories.items(), key=lambda x: -len(x[1])):
            if len(mems) < self.MIN_PATTERN_MEMORIES:
                continue
            hashes = tuple(sorted(set(m["content_hash"] for m in mems)))
            if hashes in seen_source_sets:
                continue
            seen_source_sets.add(hashes)

            # Find all shared tags for this memory set
            shared_tags = set(mems[0].get("tags", []))
            for m in mems[1:]:
                shared_tags &= set(m.get("tags", []))
            tag_label = " + ".join(sorted(shared_tags)) if shared_tags else f"{tag_pair[0]} + {tag_pair[1]}"

            confidence = min(1.0, len(mems) / 10.0)
            insights.append(InsightCard(
                title=f"Recurring pattern: {tag_label}",
                content=(
                    f"{len(mems)} memories share tags {sorted(shared_tags) if shared_tags else list(tag_pair)}, "
                    f"indicating a recurring theme."
                ),
                source_hashes=list(hashes),
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
            recent_types = set(m.get("memory_type") or "" for m in recent_tags[tag])
            old_types = set(m.get("memory_type") or "" for m in old_tags[tag])
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
            if tag in self.EXCLUDED_GAP_TAGS:
                continue
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

        # Check deduplication via public API
        if hasattr(storage, "get_by_hash"):
            try:
                if await storage.get_by_hash(content_hash):
                    continue
            except Exception:
                pass

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

        # Create derived_from edges (best-effort — non-critical if some fail)
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
                    pass

    return stored_hashes


def _insight_hash(card: InsightCard) -> str:
    """Deterministic hash for deduplication."""
    key = f"{card.insight_type}:{card.title}:{','.join(sorted(card.source_hashes))}"
    return f"insight_{hashlib.sha256(key.encode()).hexdigest()[:16]}"
