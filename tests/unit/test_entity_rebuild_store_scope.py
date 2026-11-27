"""The rebuild plan keeps shares_entity pairs inside one store when asked (#2123).

Backend-free: the plan is fed Memory objects and a store map directly, and its
output is written to a stand-in graph that refuses a cross-store pair the way
MilvusGraphStorage.store_association does since #2119. The Milvus Lite cases are
in tests/storage/test_milvus_rebuild_store_scope.py.
"""

import pytest

import mcp_memory_service.config as cfg
from mcp_memory_service.models.memory import Memory
from mcp_memory_service.reasoning.entity_rebuild import EntityRebuildPlan
from mcp_memory_service.utils.hashing import generate_content_hash


def _key(name):
    return name


@pytest.fixture(autouse=True)
def _entity_config(monkeypatch):
    monkeypatch.setattr(cfg, "MCP_ENTITY_TAG_INCLUDE", "")
    monkeypatch.setattr(cfg, "MCP_ENTITY_TAG_EXCLUDE", "")
    monkeypatch.setattr(cfg, "MCP_ENTITY_CUSTOM_TERMS", "")


def _memory(index):
    content = f"plain note number {index}"
    return Memory(
        content=content,
        content_hash=generate_content_hash(content),
        tags=["shared"],
        created_at=1_700_000_000.0 + index,
    )


class _OneStoreGraph:
    """Accepts a pair only when both ends are known and in the same store."""

    def __init__(self, store_of):
        self.store_of = store_of
        self.pairs = []
        self.links = []

    async def delete_edges_by_relationship_types(self, kinds):
        return 0

    async def store_entity_link(self, memory_hash, name, kind):
        self.links.append(memory_hash)
        return True

    async def store_association(self, source_hash, target_hash, **edge):
        source = self.store_of.get(source_hash)
        if source is None or source != self.store_of.get(target_hash):
            return False
        self.pairs.append(tuple(sorted((source_hash, target_hash))))
        return True


def _report():
    return {"edges_deleted": 0, "edges_created": {"has_entity": 0, "shares_entity": 0}}


async def test_cross_store_pairs_are_dropped_and_counted_before_apply():
    a, b, c = (_memory(i) for i in range(3))
    store_of = {a.content_hash: "docs", b.content_hash: "docs", c.content_hash: "notes"}
    plan = EntityRebuildPlan([], entity_key=_key, store_of=store_of)
    await plan.build([a, b, c])

    preview = plan.describe()
    assert preview["shares_entity_pairs_after"] == 1
    assert preview["shares_entity_pairs_skipped_cross_store"] == 2
    assert preview["shares_entity_pairs_skipped_unknown_store"] == 0
    assert preview["edges_after"] == {"has_entity": 3, "shares_entity": 2}

    graph, report = _OneStoreGraph(store_of), _report()
    await plan.apply(graph, report)
    assert graph.pairs == [tuple(sorted((a.content_hash, b.content_hash)))]
    assert report["edges_created"] == preview["edges_after"]


async def test_an_endpoint_without_a_store_is_skipped_separately():
    a, b, c = (_memory(i) for i in range(3))
    store_of = {a.content_hash: "docs", b.content_hash: "docs"}
    plan = EntityRebuildPlan([], entity_key=_key, store_of=store_of)
    await plan.build([a, b, c])

    preview = plan.describe()
    assert preview["shares_entity_pairs_skipped_unknown_store"] == 2
    assert preview["shares_entity_pairs_skipped_cross_store"] == 0
    # apply() is never handed a pair the plan could not attribute.
    await plan.apply(_OneStoreGraph(store_of), _report())


async def test_without_a_store_map_every_pair_is_kept():
    memories = [_memory(i) for i in range(3)]
    plan = EntityRebuildPlan([], entity_key=_key)
    await plan.build(list(memories))
    unscoped = plan.describe()
    assert unscoped["shares_entity_pairs_after"] == 3
    assert unscoped["shares_entity_pairs_skipped_cross_store"] == 0
    assert unscoped["shares_entity_pairs_skipped_unknown_store"] == 0

    same_store = EntityRebuildPlan(
        [], entity_key=_key,
        store_of={m.content_hash: "default" for m in memories},
    )
    await same_store.build(list(memories))
    assert same_store.associations == plan.associations
