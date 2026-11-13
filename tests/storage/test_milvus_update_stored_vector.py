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

"""Milvus updates keep the embedding call out of the read-upsert window (#1966).

Every Milvus update is a read followed by a whole-row ``upsert``, and two
clients on one collection share no lock. An update that keeps the content
keeps its vector too, so ``update_memory_metadata`` and
``update_memories_batch`` read the stored vector with the row and write it
back, as ``mark_superseded_batch`` already did, instead of embedding the
content between the read and the upsert. ``update_memory_versioned`` stores
new content, so it embeds that content before it reads the old row.

Mocked-client style of test_milvus_store_scope.py: these run on CI runners
without pymilvus or a model.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.milvus import MilvusMemoryStorage
from mcp_memory_service.storage.shared import _string_to_tags
from mcp_memory_service.utils.hashing import generate_content_hash

_HASH = "a" * 64
_OTHER = "c" * 64
_STORED = [0.5, 0.5, 0.5, 0.5]
_FRESH = [0.1, 0.2, 0.3, 0.4]


def _make_storage(has_store: bool) -> MilvusMemoryStorage:
    """A MilvusMemoryStorage with __init__ skipped and the client mocked."""
    storage = MilvusMemoryStorage.__new__(MilvusMemoryStorage)
    storage.uri = "./milvus.db"
    storage.collection_name = "unit_test_collection"
    storage.embedding_dimension = 4
    storage.embedding_model_name = "test-model"
    storage.embedding_model = MagicMock()
    storage.embedding_model.encode = MagicMock(
        side_effect=lambda texts, **_: [list(_FRESH) for _ in texts]
    )
    storage._initialized = True
    storage.client = MagicMock()
    storage._has_content_lower = True
    storage._has_bm25 = False
    storage._has_access_collection = False
    storage._has_store = has_store
    storage.semantic_dedup_enabled = False
    storage._generate_embedding = MagicMock(return_value=list(_FRESH))
    storage._detect_conflicts = AsyncMock(return_value=[])
    storage._delete_graph_edges = AsyncMock()
    return storage


def _row(
    content_hash: str = _HASH,
    store: str | None = "team",
    vector: list[float] | None = _STORED,
    content: str = "body",
    **extra: Any,
) -> dict[str, Any]:
    now = time.time()
    row: dict[str, Any] = {
        "id": content_hash,
        "content": content,
        "content_lower": content.lower(),
        "tags": "t",
        "memory_type": "note",
        "metadata": "{}",
        "created_at": now,
        "updated_at": now,
        "created_at_iso": "2026-11-12T00:00:00Z",
        "updated_at_iso": "2026-11-12T00:00:00Z",
    }
    if store is not None:
        row["store"] = store
    if vector is not None:
        row["vector"] = list(vector)
    row.update(extra)
    return row


class _Collection:
    """An in-memory collection behind ``_call_client``.

    ``get`` returns only the requested ``output_fields`` (plus ``id``), as
    Milvus does, so a test can tell whether the vector was asked for.
    """

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows: dict[str, dict[str, Any]] = {r["id"]: dict(r) for r in rows}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def client(self) -> AsyncMock:
        return AsyncMock(side_effect=self._call)

    async def _call(self, method: str, **kwargs: Any) -> Any:
        self.calls.append((method, kwargs))
        if method == "get":
            fields = kwargs.get("output_fields") or []
            out = []
            for content_hash in kwargs.get("ids") or []:
                row = self.rows.get(content_hash)
                if row is not None:
                    picked = {f: row[f] for f in fields if f in row}
                    out.append({"id": row["id"], **picked})
            return out
        if method in ("insert", "upsert"):
            for entity in kwargs["data"]:
                self.rows[entity["id"]] = dict(entity)
            return {"insert_count": len(kwargs["data"])}
        if method == "delete":
            if "ids" in kwargs:
                gone = [h for h in kwargs["ids"] if self.rows.pop(h, None) is not None]
                return {"delete_count": len(gone)}
            match = re.fullmatch(
                r'id == "(?P<id>[^"]+)" and store == "(?P<store>[^"]+)"',
                kwargs["filter"],
            )
            assert match, kwargs["filter"]
            row = self.rows.get(match["id"])
            if row is not None and row.get("store", "default") == match["store"]:
                del self.rows[match["id"]]
                return {"delete_count": 1}
            return {"delete_count": 0}
        return None

    def of(self, method: str) -> list[dict[str, Any]]:
        return [kwargs for m, kwargs in self.calls if m == method]


# -- update_memory_metadata ---------------------------------------------------


@pytest.mark.asyncio
async def test_update_metadata_writes_the_stored_vector_back_without_embedding():
    storage = _make_storage(has_store=True)
    collection = _Collection([_row()])
    storage._call_client = collection.client()

    ok, msg = await storage.update_memory_metadata(_HASH, {"tags": ["x"]})

    assert ok, msg
    (get,) = collection.of("get")
    assert "vector" in get["output_fields"]
    (upsert,) = collection.of("upsert")
    assert upsert["data"][0]["vector"] == _STORED
    assert upsert["data"][0]["store"] == "team"
    storage._generate_embedding.assert_not_called()


@pytest.mark.asyncio
async def test_update_metadata_on_a_legacy_collection_reads_the_vector_separately():
    # get_by_hash does not carry the vector, so it is read with a second get:
    # still no embedding call between the read and the upsert.
    storage = _make_storage(has_store=False)
    collection = _Collection([_row(store=None)])
    storage._call_client = collection.client()

    ok, msg = await storage.update_memory_metadata(_HASH, {"tags": ["x"]})

    assert ok, msg
    gets = collection.of("get")
    assert len(gets) == 2
    assert gets[1]["output_fields"] == ["vector", "metadata"]
    (upsert,) = collection.of("upsert")
    assert upsert["data"][0]["vector"] == _STORED
    storage._generate_embedding.assert_not_called()


@pytest.mark.asyncio
async def test_update_metadata_on_a_legacy_collection_stops_at_a_tombstone():
    # The second get reads the metadata with the vector: a row soft-deleted
    # between the two reads is not found, and nothing is written back.
    storage = _make_storage(has_store=False)
    collection = _Collection([_row(store=None)])
    inner = collection.client()

    async def tombstone_after_first_read(method: str, **kwargs: Any) -> Any:
        result = await inner(method, **kwargs)
        if method == "get":
            collection.rows[_HASH]["metadata"] = json.dumps(
                {"deleted_at": "2026-11-12T21:00:00Z"}
            )
        return result

    storage._call_client = AsyncMock(side_effect=tombstone_after_first_read)

    ok, msg = await storage.update_memory_metadata(_HASH, {"tags": ["x"]})

    assert not ok
    assert "not found" in msg
    assert len(collection.of("get")) == 2
    assert collection.of("upsert") == []
    storage._generate_embedding.assert_not_called()


@pytest.mark.asyncio
async def test_update_metadata_embeds_again_only_when_the_row_has_no_vector():
    storage = _make_storage(has_store=True)
    collection = _Collection([_row(vector=None)])
    storage._call_client = collection.client()

    ok, msg = await storage.update_memory_metadata(_HASH, {"tags": ["x"]})

    assert ok, msg
    storage._generate_embedding.assert_called_once_with("body")
    (upsert,) = collection.of("upsert")
    assert upsert["data"][0]["vector"] == _FRESH


# -- update_memories_batch ----------------------------------------------------


def _memory(content_hash: str, content: str = "body") -> Memory:
    return Memory(
        content=content, content_hash=content_hash, tags=["u"], memory_type="note"
    )


@pytest.mark.asyncio
async def test_update_memories_batch_writes_the_stored_vectors_back_without_encoding():
    storage = _make_storage(has_store=True)
    other = [0.9, 0.9, 0.9, 0.9]
    collection = _Collection(
        [_row(), _row(content_hash=_OTHER, store="ops", vector=other)]
    )
    storage._call_client = collection.client()

    results = await storage.update_memories_batch([_memory(_HASH), _memory(_OTHER)])

    assert results == [True, True]
    (get,) = collection.of("get")
    assert "vector" in get["output_fields"]
    (upsert,) = collection.of("upsert")
    assert [e["vector"] for e in upsert["data"]] == [_STORED, other]
    assert [e["store"] for e in upsert["data"]] == ["team", "ops"]
    storage.embedding_model.encode.assert_not_called()


@pytest.mark.asyncio
async def test_update_memories_batch_encodes_only_the_rows_without_a_vector():
    storage = _make_storage(has_store=True)
    collection = _Collection(
        [_row(), _row(content_hash=_OTHER, vector=None, content="no vector here")]
    )
    storage._call_client = collection.client()

    results = await storage.update_memories_batch([_memory(_HASH), _memory(_OTHER)])

    assert results == [True, True]
    storage.embedding_model.encode.assert_called_once()
    assert storage.embedding_model.encode.call_args.args[0] == ["no vector here"]
    (upsert,) = collection.of("upsert")
    assert [e["vector"] for e in upsert["data"]] == [_STORED, _FRESH]


# -- update_memory_versioned --------------------------------------------------


@pytest.mark.asyncio
async def test_versioned_update_embeds_the_new_content_before_reading_the_old_row():
    storage = _make_storage(has_store=True)
    collection = _Collection([_row()])
    storage._call_client = collection.client()
    order: list[str] = []
    storage._generate_embedding = MagicMock(
        side_effect=lambda text: (order.append(f"embed:{text}"), list(_FRESH))[1]
    )
    original = collection.client()

    async def tracing(method: str, **kwargs: Any) -> Any:
        order.append(method)
        return await original(method, **kwargs)

    storage._call_client = AsyncMock(side_effect=tracing)

    ok, msg, new_hash = await storage.update_memory_versioned(_HASH, "body v2")

    assert ok, msg
    assert new_hash == generate_content_hash("body v2")
    # One embedding call, for the new content, and before the first read;
    # the link on the old row reuses its stored vector.
    assert order[0] == "embed:body v2"
    assert [o for o in order if o.startswith("embed:")] == ["embed:body v2"]
    assert collection.rows[new_hash]["vector"] == _FRESH
    assert collection.rows[_HASH]["vector"] == _STORED
    assert "superseded_by" in collection.rows[_HASH]["metadata"]


@pytest.mark.asyncio
async def test_versioned_update_sees_a_row_moved_by_another_client_while_it_embeds():
    """Two clients, one collection. While A embeds its new content, B deletes
    the row from "team" and stores the same content in "ops". A's read comes
    after its embedding call, so the new version lands in "ops", next to the
    row it supersedes, and B's row keeps B's tags."""
    a = _make_storage(has_store=True)
    b = _make_storage(has_store=True)
    collection = _Collection([_row()])
    a._call_client = collection.client()
    b._call_client = collection.client()

    async def b_moves_the_row() -> None:
        ok, msg = await b.delete(_HASH, store="team")
        assert ok, msg
        ok, msg = await b.store(
            Memory(
                content="body", content_hash=_HASH, tags=["from-b"], memory_type="note"
            ),
            store="ops",
        )
        assert ok, msg

    def embed_while_b_writes(text: str) -> list[float]:
        # B is another process: its writes land while A's embedding call runs.
        worker = threading.Thread(target=lambda: asyncio.run(b_moves_the_row()))
        worker.start()
        worker.join()
        return list(_FRESH)

    a._generate_embedding = MagicMock(side_effect=embed_while_b_writes)

    ok, msg, new_hash = await a.update_memory_versioned(_HASH, "body v2")

    assert ok, msg
    a._generate_embedding.assert_called_once_with("body v2")
    assert collection.rows[new_hash]["store"] == "ops"
    old = collection.rows[_HASH]
    assert old["store"] == "ops"
    assert _string_to_tags(old["tags"]) == ["from-b"]
    assert f'"superseded_by": "{new_hash}"' in old["metadata"]
