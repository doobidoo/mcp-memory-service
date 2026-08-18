# Copyright 2024 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-to-end checks for the Milvus parity methods against real Milvus Lite.

The mocked tests in test_milvus_parity_methods.py cover aggregation logic but
cannot catch an invalid Milvus filter expression — the server parses those, not
the client. These run the queries for real.
"""

from __future__ import annotations

import asyncio
import time
import uuid

import pytest

pytest.importorskip("pymilvus", reason="pymilvus required for Milvus integration tests")

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.milvus import MilvusMemoryStorage
from mcp_memory_service.utils.hashing import generate_content_hash


@pytest.fixture()
def milvus_uri(tmp_path):
    return str(tmp_path / "milvus.db")


@pytest.fixture()
async def storage(milvus_uri):
    instance = MilvusMemoryStorage(
        uri=milvus_uri,
        collection_name=f"mcp_memory_{uuid.uuid4().hex[:12]}",
    )
    await instance.initialize()
    yield instance
    await instance.close()


async def _settle(read, done, timeout: float = 5.0, interval: float = 0.05):
    """Poll ``read()`` until ``done(value)`` holds, then return that value.

    Milvus Lite makes freshly inserted rows visible asynchronously (growing
    segments), so a read issued immediately after a write can legitimately miss
    it. Other tests in this directory absorb that with fixed sleeps; polling
    keeps these deterministic without padding the runtime. On timeout the last
    value is returned so the caller's assertion reports the real mismatch.
    """
    deadline = time.monotonic() + timeout
    while True:
        result = await read()
        if done(result) or time.monotonic() >= deadline:
            return result
        await asyncio.sleep(interval)


def _mem(content: str, tags=None, memory_type="note") -> Memory:
    return Memory(
        content=content,
        content_hash=generate_content_hash(content),
        tags=tags if tags is not None else ["t"],
        memory_type=memory_type,
    )


async def _insert_tagless_row(storage, content_hash: str) -> None:
    """Insert a row with a structurally empty tags field.

    Memory.__post_init__ rewrites an empty tag list to ["untagged"], so a row
    with genuinely empty tags can only come from data written before that
    default or from a path that bypasses the model. Those legacy rows are
    exactly what /api/manage/untagged is for, so the test writes one directly.
    """
    now = time.time()
    await storage._call_client(
        "insert",
        collection_name=storage.collection_name,
        data=[{
            "id": content_hash,
            "content": "legacy row",
            "content_lower": "legacy row",
            "tags": "",
            "memory_type": "note",
            "metadata": "{}",
            "created_at": now,
            "updated_at": now,
            "created_at_iso": "",
            "updated_at_iso": "",
            "vector": storage._generate_embedding("legacy row"),
        }],
    )


class TestUntaggedFilterIsValidMilvusSyntax:
    """Milvus parses filter expressions server-side, so only a real query proves
    the predicate is both valid syntax and semantically right."""

    async def test_counts_only_untagged(self, storage):
        await storage.store(_mem("tagged one", tags=["alpha"]))
        await storage.store(_mem("tagged two", tags=["beta"]))
        await _insert_tagless_row(storage, "l" * 64)
        await _insert_tagless_row(storage, "m" * 64)

        assert await _settle(storage.count_untagged_memories, lambda v: v == 2) == 2

    async def test_delete_removes_only_untagged(self, storage):
        await storage.store(_mem("tagged", tags=["alpha"]))
        await _insert_tagless_row(storage, "l" * 64)
        await _settle(storage.count_untagged_memories, lambda v: v == 1)

        count, _ = await storage.delete_untagged_memories()

        assert count == 1
        assert await storage.count_untagged_memories() == 0
        remaining = await storage.get_all_memories()
        assert [m.content for m in remaining] == ["tagged"]

    async def test_zero_when_all_tagged(self, storage):
        await storage.store(_mem("tagged", tags=["alpha"]))
        assert await storage.count_untagged_memories() == 0

    async def test_untagged_sentinel_tag_is_not_counted(self, storage):
        """A memory tagged "untagged" by the model default still has tags."""
        await storage.store(_mem("no tags supplied", tags=[]))
        assert await storage.count_untagged_memories() == 0


class TestTypeCounts:

    async def test_groups_real_rows(self, storage):
        await storage.store(_mem("a", tags=["t"], memory_type="note"))
        await storage.store(_mem("b", tags=["t"], memory_type="note"))
        await storage.store(_mem("c", tags=["t"], memory_type="decision"))

        assert await storage.get_type_counts() == {"note": 2, "decision": 1}

    async def test_empty_store(self, storage):
        assert await storage.get_type_counts() == {}


class TestGraphAnalyticsAgainstRealGraphCollection:

    async def test_no_graph_collection_is_not_an_error(self, storage):
        """A store that has never written an edge must degrade, not raise."""
        assert await storage.get_relationship_type_distribution() == {}
        result = await storage.get_graph_visualization_data()
        assert result["nodes"] == []
        assert result["meta"]["total_nodes"] == 0

    async def test_distribution_and_visualization(self, storage, milvus_uri):
        a = _mem("alpha body", tags=["t"])
        b = _mem("bravo body", tags=["t"])
        c = _mem("charlie body", tags=["t"])
        for m in (a, b, c):
            await storage.store(m)

        # Use the storage's own cached graph handle rather than a second
        # MilvusClient: Milvus Lite shares one connection per URI, so closing
        # an independently-created client tears down the storage's too.
        graph = await storage._get_graph_storage()
        await graph.store_association(
            a.content_hash, b.content_hash, 0.9, ["semantic"],
            relationship_type="causes",
        )
        await graph.store_association(
            a.content_hash, c.content_hash, 0.8, ["semantic"],
            relationship_type="related",
        )

        distribution = await storage.get_relationship_type_distribution()
        assert distribution["causes"] == 1
        # 'related' is symmetric, so store_association writes both directions.
        assert distribution["related"] == 2

        viz = await storage.get_graph_visualization_data()

        # Nodes are memories that appear as an edge *source*, matching the
        # SQLite query's INNER JOIN on source_hash. 'causes' is directed, so b
        # is only ever a target and is not rendered; 'related' is symmetric, so
        # c gets a reverse edge and is.
        assert {n["id"] for n in viz["nodes"]} == {a.content_hash, c.content_hash}

        node_a = next(n for n in viz["nodes"] if n["id"] == a.content_hash)
        assert node_a["connections"] == 2  # distinct targets: b and c
        assert viz["meta"]["total_nodes"] == 2
        assert set(viz["meta"]) == {
            "total_nodes", "total_edges", "min_connections", "limit",
        }

        # The a->b edge is dropped because b is not a rendered node.
        assert all(
            e["source"] in {a.content_hash, c.content_hash}
            and e["target"] in {a.content_hash, c.content_hash}
            for e in viz["edges"]
        )


class TestVersionedUpdate:

    async def test_round_trip(self, storage):
        original = _mem("original text", tags=["keep"], memory_type="note")
        await storage.store(original)

        ok, message, new_hash = await storage.update_memory_versioned(
            original.content_hash, "revised text", reason="typo",
        )
        assert ok, message

        # Both versions exist; the old one points at the new one.
        # Both reads need settling: the new row was just inserted, and the old
        # one was just upserted (a Milvus upsert is delete+insert, so it also
        # passes through a transiently-invisible window).
        def visible(value):
            return value is not None

        old = await _settle(lambda: storage.get_by_hash(original.content_hash), visible)
        new = await _settle(lambda: storage.get_by_hash(new_hash), visible)
        assert old is not None, "superseded original never became visible"
        assert new is not None, "new version never became visible"
        assert old.metadata.get("superseded_by") == new_hash
        assert old.metadata.get("evolution_reason") == "typo"
        assert new.content == "revised text"
        assert new.tags == ["keep"]


class TestStatsShape:

    async def test_reports_backend_and_local_size(self, storage):
        await storage.store(_mem("something", tags=["t"]))

        stats = await storage.get_stats()

        assert stats["storage_backend"] == "milvus"
        assert stats["total_memories"] == 1
        # Milvus Lite is a local file, so the dashboard's size widget works.
        assert stats["database_size_mb"] >= 0
        assert stats["database_size_bytes"] > 0
