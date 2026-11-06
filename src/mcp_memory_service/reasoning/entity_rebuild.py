"""Plan entity-graph replacement before making any destructive writes."""

from collections import Counter, defaultdict, deque
from typing import Any, Callable

from .entities import Entity, EntityExtractor
from .entity_linker import ENTITY_RELATIONSHIPS, EntityLinker, MAX_LINKS_PER_ENTITY


def _tags(value: object) -> list[str]:
    """Accept current list tags and legacy comma-separated metadata tags."""
    if isinstance(value, str):
        return [tag.strip() for tag in value.split(",") if tag.strip()]
    return list(value or [])


def _extraction_metadata(memory: Any) -> dict:
    metadata = dict(memory.metadata or {})
    metadata["tags"] = list(dict.fromkeys([
        *_tags(metadata.get("tags")), *_tags(memory.tags),
    ]))
    return metadata


class EntityRebuildPlan:
    """An in-memory graph sink for the real EntityLinker.

    Entity lookups reproduce the graph stores' newest-first 20-row window,
    including the just-inserted memory. Protected pairs cannot be overwritten
    by symmetric shares_entity writes. Counts are physical edge rows; pairs
    are reported separately.
    """

    def __init__(
        self, snapshot: list[dict], entity_key: Callable[[str], str]
    ) -> None:
        self.entity_key = entity_key
        self.protected = {
            (row["source_hash"], row["target_hash"])
            for row in snapshot
            if row["relationship_type"] not in ENTITY_RELATIONSHIPS
        }
        counts = Counter(row["relationship_type"] for row in snapshot)
        self.before = {kind: counts[kind] for kind in ENTITY_RELATIONSHIPS}
        self.pairs_before = len({
            tuple(sorted((row["source_hash"], row["target_hash"])))
            for row in snapshot if row["relationship_type"] == "shares_entity"
        })
        self.entity_links: dict[tuple[str, str], Entity] = {}
        self.associations: dict[tuple[str, str], dict] = {}
        self.recent: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=MAX_LINKS_PER_ENTITY)
        )
        self.memories_scanned = 0

    def _add_entity(self, memory_hash: str, entity: Entity) -> None:
        key = self.entity_key(entity.name)
        pair = (memory_hash, key)
        if pair in self.protected:
            return
        if pair in self.entity_links:
            # Milvus upserts normalized names; preserve its last surface form.
            self.entity_links[pair] = entity
            return
        if tuple(sorted(pair)) in self.associations:
            return
        self.entity_links[pair] = entity
        self.recent[key].appendleft(memory_hash)

    async def find_memories_by_entity(self, entity_name: str) -> list[str]:
        """Return the same capped candidate window used at store time."""
        return list(self.recent[self.entity_key(entity_name)])

    async def get_association(self, source_hash: str, target_hash: str) -> None:
        """Report no edge: store_association() checks the pair against the snapshot."""

    async def store_association(
        self, source_hash: str, target_hash: str, **edge: Any
    ) -> bool:
        """Record a symmetric pair only when both endpoints are writable."""
        forward, reverse = (source_hash, target_hash), (target_hash, source_hash)
        if any(
            pair in self.protected or pair in self.entity_links
            for pair in (forward, reverse)
        ):
            return False
        pair = tuple(sorted(forward))
        self.associations[pair] = {
            "source_hash": source_hash, "target_hash": target_hash, **edge,
        }
        return True

    async def build(self, memories: list) -> None:
        """Replay all live memories oldest first with current extraction settings."""
        # Reload strictly: the fail-soft cache may omit failed plugins.
        extractor = EntityExtractor(
            domain_extractors=EntityExtractor.load_domain_extractors(strict=True),
            strict=True,
        )
        linker = EntityLinker()
        memories.sort(key=lambda m: (m.created_at or 0, m.content_hash))
        for memory in memories:
            entities = extractor.extract_entities(
                memory.content or "", _extraction_metadata(memory)
            )
            for entity in entities:
                self._add_entity(memory.content_hash, entity)
            await linker.link_by_entities(
                memory.content_hash, [e.name for e in entities], self
            )
            self.memories_scanned += 1

    def describe(self) -> dict:
        """Return the preview without exposing memory contents."""
        return {
            "memories_scanned": self.memories_scanned,
            "edges_before": self.before,
            "edges_after": {
                "has_entity": len(self.entity_links),
                "shares_entity": 2 * len(self.associations),
            },
            "shares_entity_pairs_before": self.pairs_before,
            "shares_entity_pairs_after": len(self.associations),
            "protected_edges": len(self.protected),
        }

    async def apply(self, graph: Any, report: dict) -> None:
        """Replace entity edges, stopping and reporting any failed write."""
        deleted = await graph.delete_edges_by_relationship_types(
            list(ENTITY_RELATIONSHIPS)
        )
        report["edges_deleted"] = deleted
        for (memory_hash, _), entity in self.entity_links.items():
            ok = await graph.store_entity_link(
                memory_hash, entity.name, entity.entity_type
            )
            if not ok:
                raise RuntimeError(
                    "Failed to write has_entity edge; rebuild is incomplete"
                )
            report["edges_created"]["has_entity"] += 1
        for edge in self.associations.values():
            if not await graph.store_association(**edge):
                raise RuntimeError(
                    "Failed to write shares_entity pair; rebuild is incomplete"
                )
            report["edges_created"]["shares_entity"] += 2
