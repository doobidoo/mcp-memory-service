"""Insight Cards generator for memory consolidation (Phase 3, #732).

Generates pattern/trend/gap insights from memory clusters using heuristics only.
Integrated into the maintain cycle (memory_quality action='maintain') as Step 5.
Opt-in via MCP_INSIGHT_CARDS_ENABLED=true environment variable.
"""

import hashlib
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Dict

logger = logging.getLogger(__name__)

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
    # Skip gap if this fraction of memories have automated memory_type (metadata heuristic).
    DOMINANT_TYPE_THRESHOLD = 0.9

    # Default tags excluded from gap detection — status/metadata markers, not knowledge domains.
    EXCLUDED_GAP_TAGS: frozenset = frozenset({
        "conflict:unresolved", "automated", "__test__", "temporary",
        "processed", "auto-generated", "insight-card",
    })

    # memory_type values that indicate automated/system-generated content.
    _AUTOMATED_TYPES: frozenset = frozenset({"session", "auto-generated", "temporary"})

    def __init__(self) -> None:
        extra_raw = os.getenv("MCP_INSIGHT_EXCLUDE_TAGS", "")
        extra = {t.strip() for t in extra_raw.split(",") if t.strip()}
        self._excluded_gap_tags: frozenset = self.EXCLUDED_GAP_TAGS | extra

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
            if tag in self._excluded_gap_tags:
                continue
            if len(mems) >= self.GAP_MIN_MEMORIES and not tag_has_decision[tag]:
                if self._is_metadata_tag(tag, mems):
                    continue
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

    def _is_metadata_tag(self, tag: str, mems: List[dict]) -> bool:
        """Heuristic: tag dominated by automated memory_type → likely a status/metadata marker.

        Only triggers when >90% of memories have an automated type (session, auto-generated,
        temporary). Generic types like 'observation' are not treated as automated — a tag
        with all-observation memories may still be a real knowledge domain gap.
        """
        if not mems:
            return False
        types = [m.get("memory_type") or "" for m in mems]
        automated_count = sum(1 for t in types if t in self._AUTOMATED_TYPES)
        return automated_count / len(types) >= self.DOMINANT_TYPE_THRESHOLD


async def store_insights(insights: List[InsightCard], storage) -> List[str]:
    """Store InsightCards as memories and create derived_from edges.

    Acknowledgement flow: if an existing insight card has the 'acknowledged' tag,
    a stable sentinel (hash independent of source_hashes) is stored so the card is
    not regenerated even if the user later deletes the original card.

    Args:
        insights: List of InsightCard to persist.
        storage: Storage backend with store() and store_association() methods.

    Returns:
        List of content hashes for stored insight memories.
    """
    stored_hashes = []
    for card in insights:
        content_hash = _insight_hash(card)
        ack_hash = _insight_ack_hash(card)

        if hasattr(storage, "get_by_hash"):
            try:
                # Acknowledgement sentinel check — persists even after card deletion.
                if await storage.get_by_hash(ack_hash):
                    continue

                existing = await storage.get_by_hash(content_hash)
                if existing:
                    if "acknowledged" in (existing.tags or []):
                        # Sentinel creation has its own error handling — kept separate
                        # so a sentinel failure does not swallow the dedup check result.
                        await _store_ack_sentinel(card, ack_hash, storage)
                    continue
            except Exception:
                # Dedup check failed — proceed to store to avoid silent data loss.
                pass

        now = datetime.now(timezone.utc)
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
            created_at=now.timestamp(),
            created_at_iso=now.isoformat(),
        )

        success, _ = await storage.store(memory)
        if not success:
            continue
        stored_hashes.append(content_hash)

        # Create derived_from edges (best-effort — graph edges are non-critical)
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
                    pass  # graph edges are advisory; insight card is already stored

    return stored_hashes


async def _store_ack_sentinel(card: InsightCard, ack_hash: str, storage) -> None:
    """Persist a lightweight sentinel so an acknowledged insight is never regenerated."""
    now = datetime.now(timezone.utc)
    sentinel = Memory(
        content=f"[ACK] {card.insight_type}:{card.title}",
        content_hash=ack_hash,
        tags=["insight-ack", "acknowledged", card.insight_type],
        memory_type="insight",
        metadata={"ack_for": _insight_hash(card)},
        created_at=now.timestamp(),
        created_at_iso=now.isoformat(),
    )
    try:
        await storage.store(sentinel)
    except Exception as exc:
        # Sentinel is best-effort: if storage fails, the card may regenerate after deletion.
        logger.warning("Failed to store insight ack sentinel for %r: %s", card.title, exc)


def _insight_hash(card: InsightCard) -> str:
    """Deterministic hash for deduplication."""
    key = f"{card.insight_type}:{card.title}:{','.join(sorted(card.source_hashes))}"
    return f"insight_{hashlib.sha256(key.encode()).hexdigest()[:16]}"


def _insight_ack_hash(card: InsightCard) -> str:
    """Stable acknowledgement hash — independent of source_hashes.

    Used so an acknowledged insight is not regenerated even after the card
    itself is deleted from storage.
    """
    key = f"ack:{card.insight_type}:{card.title}"
    return f"insight_ack_{hashlib.sha256(key.encode()).hexdigest()[:16]}"
