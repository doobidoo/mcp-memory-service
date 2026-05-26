"""Auto-link memories by shared entities.

When a memory is stored and entities are extracted, this module creates
bidirectional 'shares_entity' edges between the new memory and all existing
memories that share the same entity.

Opt-in via MCP_ENTITY_LINKING_ENABLED=true (default: false).
"""

import logging
import os
from typing import List

logger = logging.getLogger(__name__)

# Cap to prevent O(N²) edge explosion for common entities
MAX_LINKS_PER_ENTITY = 20


def is_entity_linking_enabled() -> bool:
    """Check if entity linking is enabled via environment variable."""
    return os.environ.get("MCP_ENTITY_LINKING_ENABLED", "").lower() in ("1", "true", "yes")


class EntityLinker:
    """Auto-creates 'shares_entity' edges between memories with common entities."""

    async def link_by_entities(
        self, memory_hash: str, entities: List[str], graph_storage
    ) -> int:
        """
        For each entity in the new memory, find other memories with the same entity
        and create bidirectional 'shares_entity' edges.

        Args:
            memory_hash: Content hash of the newly stored memory.
            entities: List of entity names extracted from the memory.
            graph_storage: GraphStorage instance with store_association and
                find_memories_by_entity methods.

        Returns:
            Count of new edges created.
        """
        if not entities or not graph_storage:
            return 0

        edges_created = 0
        seen_pairs = set()

        # Deduplicate entities to avoid redundant queries
        unique_entities = list(dict.fromkeys(entities))

        for entity_name in unique_entities:
            try:
                existing_hashes = await graph_storage.find_memories_by_entity(entity_name)
            except Exception as e:
                logger.debug(f"Failed to query entity '{entity_name}': {e}")
                continue

            # Defensive: handle None or non-iterable return
            if not existing_hashes:
                continue

            # Cap links per entity to prevent O(N²) explosion
            linked_count = 0
            for other_hash in existing_hashes:
                if other_hash == memory_hash:
                    continue
                if linked_count >= MAX_LINKS_PER_ENTITY:
                    logger.debug(
                        f"EntityLinker: hit cap ({MAX_LINKS_PER_ENTITY}) for entity '{entity_name}'"
                    )
                    break

                # Canonical pair to avoid duplicates
                pair = tuple(sorted([memory_hash, other_hash]))
                if pair in seen_pairs:
                    continue

                try:
                    ok = await graph_storage.store_association(
                        source_hash=memory_hash,
                        target_hash=other_hash,
                        similarity=1.0,
                        connection_types=["entity"],
                        metadata={"shared_entity": entity_name},
                        relationship_type="shares_entity",
                    )
                    if ok:
                        seen_pairs.add(pair)
                        edges_created += 1
                        linked_count += 1
                except Exception as e:
                    logger.debug(f"Failed to create shares_entity edge: {e}")

        if edges_created:
            logger.info(
                f"EntityLinker: created {edges_created} shares_entity edges for {memory_hash}"
            )
        return edges_created
