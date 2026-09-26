"""Regression tests for consolidation code calling storage methods that don't exist (#1319).

Each call used to fail inside a broad ``except`` (or a ``hasattr`` guard that is
always False), so the feature silently did nothing instead of raising.
"""

import json
import os
import sqlite3
from unittest.mock import AsyncMock

import pytest

from mcp_memory_service.consolidation.belief_service import BeliefService
from mcp_memory_service.consolidation.insights import InsightCard, store_insights
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.cloudflare import CloudflareStorage
from mcp_memory_service.storage.graph import GraphStorage
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash


async def _storage(temp_db_path):
    storage = SqliteVecMemoryStorage(os.path.join(temp_db_path, "dead_calls.db"))
    await storage.initialize()
    return storage


async def _store(storage, text):
    memory = Memory(content=text, content_hash=generate_content_hash(text), tags=["test"])
    ok, msg = await storage.store(memory)
    assert ok, msg
    return memory.content_hash


@pytest.mark.asyncio
async def test_belief_observation_lookup_finds_stored_memories(temp_db_path):
    """_get_observations_by_hashes called storage.get_memory_by_hash(), which no
    backend defines, so every lookup raised, was swallowed, and returned []."""
    storage = await _storage(temp_db_path)
    h = await _store(storage, "The nightly backup runs at 02:00 against the NAS share.")

    observations = await BeliefService(storage)._get_observations_by_hashes([h])

    assert [o["content_hash"] for o in observations] == [h]


@pytest.mark.asyncio
async def test_insight_cards_write_derived_from_edges(temp_db_path):
    """store_insights only wrote edges if the memory storage had
    store_association, which lives on GraphStorage, so no derived_from edge
    was ever written. The graph handle is now passed in explicitly."""
    storage = await _storage(temp_db_path)
    sources = [await _store(storage, f"Deploy note {i}: restart caddy after cert renewal.") for i in range(2)]
    graph = GraphStorage(storage.db_path)
    card = InsightCard(
        title="Caddy restarts follow renewals",
        content="Several notes restart caddy after certificate renewal.",
        source_hashes=sources,
        insight_type="pattern",
        confidence=0.8,
    )

    stored = await store_insights([card], storage, graph=graph)

    assert len(stored) == 1
    with sqlite3.connect(storage.db_path) as conn:
        edges = conn.execute(
            "SELECT source_hash FROM memory_graph WHERE target_hash = ? AND relationship_type = 'derived_from'",
            (stored[0],),
        ).fetchall()
    assert sorted(e[0] for e in edges) == sorted(sources)


@pytest.mark.asyncio
async def test_existing_insight_card_gets_missing_edges(temp_db_path):
    """A card stored before edges worked must get them on a later run: the
    dedup path that skips re-storing an existing card still links it."""
    storage = await _storage(temp_db_path)
    sources = [await _store(storage, f"Backup note {i}: rotate the Storj key yearly.") for i in range(2)]
    card = InsightCard(
        title="Storj keys rotate yearly",
        content="Several notes rotate the Storj key once a year.",
        source_hashes=sources,
        insight_type="pattern",
        confidence=0.7,
    )
    first = await store_insights([card], storage)  # as before the fix: no graph, no edges
    assert len(first) == 1

    await store_insights([card], storage, graph=GraphStorage(storage.db_path))

    with sqlite3.connect(storage.db_path) as conn:
        edges = conn.execute(
            "SELECT source_hash FROM memory_graph WHERE target_hash = ? AND relationship_type = 'derived_from'",
            (first[0],),
        ).fetchall()
    assert sorted(e[0] for e in edges) == sorted(sources)


@pytest.mark.asyncio
async def test_relink_leaves_existing_edge_untouched(temp_db_path):
    """Relinking an existing card must not overwrite the edge it already has.
    The write is an INSERT OR REPLACE, so an unconditional rewrite would reset
    a pre-existing derived_from edge's similarity, metadata and created_at."""
    storage = await _storage(temp_db_path)
    sources = [await _store(storage, f"Deploy note {i}: push Forgejo, then Komodo ships it.") for i in range(2)]
    card = InsightCard(
        title="Deploys go through Forgejo",
        content="Komodo deploys what Forgejo has.",
        source_hashes=sources,
        insight_type="pattern",
        confidence=0.7,
    )
    [card_hash] = await store_insights([card], storage, graph=GraphStorage(storage.db_path))

    graph = GraphStorage(storage.db_path)
    # Give both source edges attributes of their own; relinking must leave them
    # alone rather than resetting each to the card's confidence.
    with sqlite3.connect(storage.db_path) as conn:
        for i, src in enumerate(sources):
            conn.execute(
                """UPDATE memory_graph
                      SET similarity = ?, connection_types = ?, metadata = ?, created_at = ?
                    WHERE source_hash = ? AND target_hash = ? AND relationship_type = 'derived_from'""",
                (0.9, json.dumps(["semantic"]), json.dumps({"origin": "manual"}),
                 1_000_000_000.0 + i, src, card_hash),
            )
        conn.commit()
    before = [await graph.get_association(src, card_hash) for src in sources]

    await store_insights([card], storage, graph=graph)  # existing card → relink path

    for i, src in enumerate(sources):
        after = await graph.get_association(src, card_hash)
        assert after["similarity"] == pytest.approx(0.9), f"relink reset edge {i}'s similarity"
        assert after["connection_types"] == ["semantic"]
        assert after["metadata"] == {"origin": "manual"}
        assert after["created_at"] == pytest.approx(before[i]["created_at"])


@pytest.mark.asyncio
async def test_forward_link_survives_a_reverse_edge(temp_db_path):
    """A symmetric edge pointing the other way must not stand in for the card's
    source link. get_association() matches either direction, so a lookup that
    only asks "is there any edge here" reads the reverse edge as a hit and
    never writes the forward derived_from link — which is one-way, so the
    link stays missing."""
    storage = await _storage(temp_db_path)
    sources = [await _store(storage, f"Sync note {i}: nightly job pulls the git digest.") for i in range(2)]
    card = InsightCard(
        title="Nightly sync runs",
        content="A nightly job pulls the digest.",
        source_hashes=sources,
        insight_type="pattern",
        confidence=0.7,
    )
    [card_hash] = await store_insights([card], storage, graph=GraphStorage(storage.db_path))

    graph = GraphStorage(storage.db_path)
    # 'related' is symmetric, so this stores both card→src and src→card.
    await graph.store_association(
        source_hash=card_hash,
        target_hash=sources[0],
        similarity=0.5,
        connection_types=["related"],
        relationship_type="related",
    )
    # Clear the other source's edges so only the reverse-related pair remains,
    # then drop the forward derived_from link: a reverse edge is now the only
    # thing a direction-blind lookup would find.
    with sqlite3.connect(storage.db_path) as conn:
        conn.execute("DELETE FROM memory_graph WHERE source_hash = ? OR target_hash = ?",
                     (sources[1], sources[1]))
        conn.execute(
            "DELETE FROM memory_graph WHERE source_hash = ? AND target_hash = ? AND relationship_type = 'derived_from'",
            (sources[0], card_hash),
        )
        conn.commit()
        remaining = sorted(r[0] for r in conn.execute("SELECT relationship_type FROM memory_graph").fetchall())
    # the symmetric write leaves both directions of the pair
    assert remaining == ["related", "related"], f"setup expected only the reverse pair, got {remaining}"

    await store_insights([card], storage, graph=graph)  # existing card → relink path

    with sqlite3.connect(storage.db_path) as conn:
        restored = conn.execute(
            "SELECT relationship_type FROM memory_graph WHERE source_hash = ? AND target_hash = ?",
            (sources[0], card_hash),
        ).fetchall()
    assert ("derived_from",) in restored, \
        f"reverse edge suppressed the source link; rows now: {restored}"


@pytest.mark.asyncio
async def test_source_link_replaces_an_edge_of_another_type(temp_db_path):
    """A row already occupying source→card must not block the link: the graph's
    primary key is (source_hash, target_hash), so any other relationship between
    the same pair is a different edge than the derived_from link, and leaving it
    in place loses the link permanently."""
    storage = await _storage(temp_db_path)
    sources = [await _store(storage, f"Hook note {i}: the PostToolUse hook tags memories.") for i in range(1)]
    card = InsightCard(
        title="Hooks tag memories",
        content="A PostToolUse hook adds the tag.",
        source_hashes=sources,
        insight_type="pattern",
        confidence=0.7,
    )
    [card_hash] = await store_insights([card], storage, graph=GraphStorage(storage.db_path))

    graph = GraphStorage(storage.db_path)
    # 'supports' is asymmetric, so this occupies source→card with a non-link edge.
    await graph.store_association(
        source_hash=sources[0],
        target_hash=card_hash,
        similarity=0.9,
        connection_types=["supports"],
        relationship_type="supports",
    )

    await store_insights([card], storage, graph=graph)  # existing card → relink path

    with sqlite3.connect(storage.db_path) as conn:
        types = [r[0] for r in conn.execute(
            "SELECT relationship_type FROM memory_graph WHERE source_hash = ? AND target_hash = ?",
            (sources[0], card_hash),
        ).fetchall()]
    assert types == ["derived_from"], f"the link was not written over the existing row: {types}"


@pytest.mark.asyncio
async def test_has_edge_is_exact_when_forward_and_reverse_coexist(temp_db_path):
    """get_association() returns either direction's row with no ordering, so a
    reverse edge can shadow the forward link and make a direction/type check
    on its result rewrite an edge that already exists. has_edge() asks for the
    exact directed, typed edge, so it must report the forward derived_from link
    present even when a reverse edge between the same two memories also exists,
    and must not report a forward link where only a reverse edge exists.

    The forward edge is written with a distinct similarity (0.42) so a relink
    that rewrote it (maintenance writes the card's 0.7 confidence) would change
    the stored value and fail the assertion, not pass silently.
    """
    storage = await _storage(temp_db_path)
    src = await _store(storage, "Deploy note: push Forgejo, then Komodo ships it.")
    card = InsightCard(
        title="Deploys go through Forgejo",
        content="Komodo deploys what Forgejo has.",
        source_hashes=[src],
        insight_type="pattern",
        confidence=0.7,
    )
    [card_hash] = await store_insights([card], storage, graph=GraphStorage(storage.db_path))
    graph = GraphStorage(storage.db_path)

    # read the forward row directly (get_association can return the reverse edge,
    # so select the forward, typed row explicitly) and give it a distinct value
    with sqlite3.connect(storage.db_path) as conn:
        conn.execute(
            "UPDATE memory_graph SET similarity = 0.42 "
            "WHERE source_hash = ? AND target_hash = ? AND relationship_type = 'derived_from'",
            (src, card_hash),
        )
        conn.commit()

    # a reverse edge in the opposite direction (card -> src). get_association()
    # matches either direction, so without an exact check this reverse row could
    # shadow the forward link and make a relink rewrite the existing edge.
    # An asymmetric type (supports) is used so it occupies only card->src and does
    # not collide on the (source_hash, target_hash) primary key with the forward
    # src->card derived_from row.
    await graph.store_association(
        source_hash=card_hash,
        target_hash=src,
        similarity=0.5,
        connection_types=["supports"],
        relationship_type="supports",
    )

    # the exact forward link is present → must not be rewritten
    assert await graph.has_edge(src, card_hash, "derived_from") is True
    # the reverse edge is card->src supports, not a forward derived_from
    assert await graph.has_edge(card_hash, src, "derived_from") is False

    # relinking must leave the existing forward link alone, not reset its value.
    # read the forward row directly to avoid get_association returning the reverse edge.
    def _forward_similarity():
        with sqlite3.connect(storage.db_path) as conn:
            return conn.execute(
                "SELECT similarity FROM memory_graph "
                "WHERE source_hash = ? AND target_hash = ? AND relationship_type = 'derived_from'",
                (src, card_hash),
            ).fetchone()[0]

    before = _forward_similarity()
    await store_insights([card], storage, graph=graph)  # existing card → relink path
    after = _forward_similarity()
    assert after == pytest.approx(before), (
        f"relink reset the existing forward link (before={before}, after={after})"
    )
    assert after == pytest.approx(0.42), "the distinct forward value was overwritten"


@pytest.mark.asyncio
async def test_cloudflare_storage_supports_consolidation_delete():
    """The consolidator applies forgetting through storage.delete_memory(), which
    CloudflareStorage did not have, so forgetting raised AttributeError there.
    The base class now provides it on top of delete()."""
    storage = CloudflareStorage.__new__(CloudflareStorage)
    storage.delete = AsyncMock(return_value=(True, "deleted"))

    assert await storage.delete_memory("abc123") is True
    storage.delete.assert_awaited_once_with("abc123")
