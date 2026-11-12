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

"""Milvus keeps the store a memory was written to and confines the
hash-addressed calls to it (#1879).

Before this, the collection had no ``store`` field: ``store()`` dropped the
value, and ``get_by_hash`` / ``update_memory_metadata`` / ``delete`` served a
memory from any store. A collection created before the field existed has
nothing to confine on, so a named store is refused there instead of being
served unscoped.

Mocked-client style of test_milvus_parity_methods.py, so these run on CI
runners without pymilvus. test_milvus_store_scope_e2e.py runs the same
contract against Milvus Lite.
"""

from __future__ import annotations

import logging
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from mcp_memory_service.models.memory import Memory
from mcp_memory_service.storage.milvus import MilvusMemoryStorage

_HASH = "a" * 64


def _make_storage(has_store: bool) -> MilvusMemoryStorage:
    """A MilvusMemoryStorage with __init__ skipped and the client mocked."""
    storage = MilvusMemoryStorage.__new__(MilvusMemoryStorage)
    storage.uri = "./milvus.db"
    storage.collection_name = "unit_test_collection"
    storage.embedding_dimension = 4
    storage.embedding_model_name = "test-model"
    storage.embedding_model = MagicMock()
    storage._initialized = True
    storage.client = MagicMock()
    storage._has_content_lower = True
    storage._has_bm25 = False
    storage._has_access_collection = False
    storage._has_store = has_store
    storage.semantic_dedup_enabled = False
    storage._generate_embedding = MagicMock(return_value=[0.1, 0.2, 0.3, 0.4])
    storage._detect_conflicts = AsyncMock(return_value=[])
    storage._delete_graph_edges = AsyncMock()
    return storage


def _row(store: str | None = "team", content_hash: str = _HASH, **extra) -> dict[str, Any]:
    now = time.time()
    row: dict[str, Any] = {
        "id": content_hash,
        "content": "body",
        "tags": "t",
        "memory_type": "note",
        "metadata": "{}",
        "created_at": now,
        "updated_at": now,
        "created_at_iso": "2026-11-11T00:00:00Z",
        "updated_at_iso": "2026-11-11T00:00:00Z",
    }
    if store is not None:
        row["store"] = store
    row.update(extra)
    return row


def _client(rows: list[dict[str, Any]]) -> AsyncMock:
    """``_call_client`` that answers ``get`` with ``rows`` and records writes."""

    async def call(method: str, **kwargs):
        if method == "get":
            wanted = set(kwargs.get("ids") or [])
            return [r for r in rows if r["id"] in wanted]
        return None

    return AsyncMock(side_effect=call)


def _calls(mock: AsyncMock, method: str) -> list[dict[str, Any]]:
    return [c.kwargs for c in mock.call_args_list if c.args and c.args[0] == method]


def _memory(content: str = "new body") -> Memory:
    return Memory(content=content, content_hash="b" * 64, tags=["t"], memory_type="note")


# -- Writes keep the store ----------------------------------------------------


@pytest.mark.asyncio
async def test_store_writes_the_store_field():
    storage = _make_storage(has_store=True)
    storage._call_client = _client([])

    ok, msg = await storage.store(_memory(), store="team")

    assert ok, msg
    (insert,) = _calls(storage._call_client, "insert")
    assert insert["data"][0]["store"] == "team"


@pytest.mark.asyncio
async def test_store_batch_writes_the_store_field():
    storage = _make_storage(has_store=True)
    storage._call_client = _client([])

    results = await storage.store_batch([_memory("one"), _memory("two")], store="team")

    assert [ok for ok, _ in results] == [True, True]
    (insert,) = _calls(storage._call_client, "insert")
    assert [e["store"] for e in insert["data"]] == ["team", "team"]


@pytest.mark.asyncio
async def test_update_writes_back_the_rows_own_store():
    # upsert replaces the whole row: without the store the row moves to "default".
    storage = _make_storage(has_store=True)
    storage._call_client = _client([_row(store="team")])

    ok, msg = await storage.update_memory_metadata(_HASH, {"tags": ["x"]})

    assert ok, msg
    (upsert,) = _calls(storage._call_client, "upsert")
    assert upsert["data"][0]["store"] == "team"
    get = _calls(storage._call_client, "get")[0]
    assert "store" in get["output_fields"]


@pytest.mark.asyncio
async def test_update_memories_batch_writes_back_each_rows_store():
    storage = _make_storage(has_store=True)
    other = "c" * 64
    storage._call_client = _client([_row(store="team"), _row(store="ops", content_hash=other)])
    storage.embedding_model.encode = MagicMock(return_value=[[0.1] * 4, [0.2] * 4])

    results = await storage.update_memories_batch([
        Memory(content="body", content_hash=_HASH, tags=["x"], memory_type="note"),
        Memory(content="body", content_hash=other, tags=["y"], memory_type="note"),
    ])

    assert results == [True, True]
    (upsert,) = _calls(storage._call_client, "upsert")
    assert {e["id"]: e["store"] for e in upsert["data"]} == {_HASH: "team", other: "ops"}


@pytest.mark.asyncio
async def test_mark_superseded_batch_writes_back_the_rows_store():
    storage = _make_storage(has_store=True)
    storage._call_client = _client([_row(store="team", vector=[0.1, 0.2, 0.3, 0.4])])

    marked = await storage.mark_superseded_batch([("w" * 64, _HASH)])

    assert marked == 1
    (upsert,) = _calls(storage._call_client, "upsert")
    assert upsert["data"][0]["store"] == "team"


_TOMBSTONE = '{"deleted_at": 1762905600.0}'


@pytest.mark.asyncio
async def test_update_memories_batch_skips_a_soft_deleted_row():
    # A tombstone is "not found", as for get_by_hash: not written back, not counted.
    storage = _make_storage(has_store=True)
    gone = "c" * 64
    storage._call_client = _client([
        _row(store="team"),
        _row(store="team", content_hash=gone, metadata=_TOMBSTONE),
    ])
    storage.embedding_model.encode = MagicMock(return_value=[[0.1] * 4])

    results = await storage.update_memories_batch([
        Memory(content="body", content_hash=_HASH, tags=["x"], memory_type="note"),
        Memory(content="body", content_hash=gone, tags=["y"], memory_type="note"),
    ])

    assert results == [True, False]
    (upsert,) = _calls(storage._call_client, "upsert")
    assert [e["id"] for e in upsert["data"]] == [_HASH]


@pytest.mark.asyncio
async def test_mark_superseded_batch_skips_a_soft_deleted_loser():
    storage = _make_storage(has_store=True)
    storage._call_client = _client([
        _row(store="team", vector=[0.1, 0.2, 0.3, 0.4], metadata=_TOMBSTONE),
    ])

    marked = await storage.mark_superseded_batch([("w" * 64, _HASH)])

    assert marked == 0
    assert _calls(storage._call_client, "upsert") == []


@pytest.mark.asyncio
async def test_versioned_update_keeps_the_new_version_in_the_old_store():
    storage = _make_storage(has_store=True)
    storage._call_client = _client([_row(store="team")])

    ok, msg, new_hash = await storage.update_memory_versioned(_HASH, "body, revised")

    assert ok, msg
    (insert,) = _calls(storage._call_client, "insert")
    assert insert["data"][0]["id"] == new_hash
    assert insert["data"][0]["store"] == "team"


# -- Hash-addressed calls are confined ----------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("store", [None, "all", "team"])
async def test_get_by_hash_serves_unscoped_and_own_store(store):
    storage = _make_storage(has_store=True)
    storage._call_client = _client([_row(store="team")])

    assert (await storage.get_by_hash(_HASH, store=store)) is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("store", ["ops", "default"])
async def test_get_by_hash_hides_a_memory_in_another_store(store):
    storage = _make_storage(has_store=True)
    storage._call_client = _client([_row(store="team")])

    assert (await storage.get_by_hash(_HASH, store=store)) is None


@pytest.mark.asyncio
async def test_update_in_another_store_is_refused_without_a_write():
    storage = _make_storage(has_store=True)
    storage._call_client = _client([_row(store="team")])

    ok, msg = await storage.update_memory_metadata(_HASH, {"tags": ["x"]}, store="ops")

    assert ok is False
    assert "not found" in msg
    assert _calls(storage._call_client, "upsert") == []


@pytest.mark.asyncio
async def test_update_in_own_store_keeps_it():
    storage = _make_storage(has_store=True)
    storage._call_client = _client([_row(store="team")])

    ok, msg = await storage.update_memory_metadata(_HASH, {"tags": ["x"]}, store="team")

    assert ok, msg
    (upsert,) = _calls(storage._call_client, "upsert")
    assert upsert["data"][0]["store"] == "team"


@pytest.mark.asyncio
async def test_delete_in_another_store_is_refused_without_a_delete():
    storage = _make_storage(has_store=True)
    storage._call_client = _client([_row(store="team")])

    ok, msg = await storage.delete(_HASH, store="ops")

    assert ok is False
    assert "not found" in msg
    assert _calls(storage._call_client, "delete") == []


@pytest.mark.asyncio
async def test_delete_in_own_store_names_the_store_in_the_delete():
    storage = _make_storage(has_store=True)
    storage._call_client = _client([_row(store="team")])

    ok, msg = await storage.delete(_HASH, store="team")

    assert ok, msg
    (delete,) = _calls(storage._call_client, "delete")
    assert delete["filter"] == f'id == "{_HASH}" and store == "team"'


@pytest.mark.asyncio
async def test_unscoped_delete_is_by_id():
    storage = _make_storage(has_store=True)
    storage._call_client = _client([_row(store="team")])

    ok, msg = await storage.delete(_HASH)

    assert ok, msg
    (delete,) = _calls(storage._call_client, "delete")
    assert delete == {"collection_name": "unit_test_collection", "ids": [_HASH]}


def _client_deleting(rows: list[dict[str, Any]], delete_result: Any) -> AsyncMock:
    """Like ``_client``, but ``delete`` answers ``delete_result``."""

    async def call(method: str, **kwargs):
        if method == "get":
            wanted = set(kwargs.get("ids") or [])
            return [r for r in rows if r["id"] in wanted]
        if method == "delete":
            return delete_result
        return None

    return AsyncMock(side_effect=call)


@pytest.mark.asyncio
async def test_delete_that_removed_nothing_is_not_a_success():
    """Between the read and the delete another client moved the row to a
    different store: the confined filter matches nothing, and the access
    record and graph edges of the row that is there now must stay."""
    storage = _make_storage(has_store=True)
    storage._has_access_collection = True
    storage._access_collection = "unit_test_collection_access"
    storage._call_client = _client_deleting([_row(store="team")], {"delete_count": 0, "cost": 0})

    ok, msg = await storage.delete(_HASH, store="team")

    assert ok is False
    assert "not found in store 'team'" in msg
    assert len(_calls(storage._call_client, "delete")) == 1  # only the row delete, no access delete
    storage._delete_graph_edges.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("delete_result", [
    {"delete_count": 1, "cost": 0},  # pymilvus >= 2.3.2
    [_HASH],                        # primary keys, older servers
    None,                           # no count reported: taken as deleted, as before
])
async def test_delete_that_removed_the_row_cleans_up_after_it(delete_result):
    storage = _make_storage(has_store=True)
    storage._has_access_collection = True
    storage._access_collection = "unit_test_collection_access"
    storage._call_client = _client_deleting([_row(store="team")], delete_result)

    ok, msg = await storage.delete(_HASH, store="team")

    assert ok, msg
    assert len(_calls(storage._call_client, "delete")) == 2  # the row, then its access record
    storage._delete_graph_edges.assert_awaited_once_with([_HASH])


@pytest.mark.asyncio
async def test_unscoped_delete_of_a_row_gone_meanwhile_is_not_a_success():
    storage = _make_storage(has_store=True)
    storage._call_client = _client_deleting([_row(store="team")], {"delete_count": 0, "cost": 0})

    ok, msg = await storage.delete(_HASH)

    assert ok is False
    assert msg == f"Memory with hash {_HASH} not found"
    storage._delete_graph_edges.assert_not_awaited()


# -- Legacy collection: fail closed on a named store -------------------------


@pytest.mark.asyncio
async def test_legacy_collection_refuses_a_named_store_without_reading(caplog):
    storage = _make_storage(has_store=False)
    storage._call_client = _client([_row(store=None)])

    with caplog.at_level(logging.WARNING, logger="mcp_memory_service.storage.milvus"):
        assert (await storage.get_by_hash(_HASH, store="team")) is None
        ok_update, _ = await storage.update_memory_metadata(_HASH, {"tags": ["x"]}, store="team")
        ok_delete, _ = await storage.delete(_HASH, store="team")

    assert ok_update is False
    assert ok_delete is False
    assert storage._call_client.call_args_list == []
    refusals = [r for r in caplog.records if "has no 'store' field" in r.getMessage()]
    assert len(refusals) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("store", [None, "all", "default"])
async def test_legacy_collection_serves_unscoped_calls_as_before(store):
    storage = _make_storage(has_store=False)
    storage._call_client = _client([_row(store=None)])

    assert (await storage.get_by_hash(_HASH, store=store)) is not None
    ok, msg = await storage.update_memory_metadata(_HASH, {"tags": ["x"]}, store=store)
    assert ok, msg
    (upsert,) = _calls(storage._call_client, "upsert")
    assert "store" not in upsert["data"][0]
    ok, msg = await storage.delete(_HASH, store=store)
    assert ok, msg
    (delete,) = _calls(storage._call_client, "delete")
    assert delete["ids"] == [_HASH]


# -- Schema --------------------------------------------------------------------


def _describe(field_names: list[str]) -> dict[str, Any]:
    fields = [{"name": n} for n in field_names]
    fields.append({"name": "vector", "params": {"dim": 4}})
    return {"fields": fields}


@pytest.mark.parametrize("names, expected", [
    (["id", "content", "content_lower", "store", "sparse_vector"], True),
    (["id", "content", "content_lower", "sparse_vector"], False),
])
def test_existing_collection_is_read_for_the_store_field(names, expected):
    storage = _make_storage(has_store=not expected)
    storage.client.describe_collection = MagicMock(return_value=_describe(names))

    storage._validate_existing_collection()

    assert storage._has_store is expected


def test_new_collection_gets_a_store_field_defaulting_to_default(monkeypatch):
    import mcp_memory_service.storage.milvus as milvus_module

    datatype = MagicMock()
    monkeypatch.setattr(milvus_module, "DataType", datatype, raising=False)
    monkeypatch.setattr(milvus_module, "_BM25_IMPORTS_AVAILABLE", False)
    storage = _make_storage(has_store=False)
    storage.client.has_collection = MagicMock(return_value=False)
    schema = MagicMock()
    storage.client.create_schema = MagicMock(return_value=schema)

    storage._ensure_collection()

    fields = {c.kwargs["field_name"]: c.kwargs for c in schema.add_field.call_args_list}
    assert fields["store"]["datatype"] is datatype.VARCHAR
    assert fields["store"]["default_value"] == "default"
    assert storage._has_store is True
