"""Explicit, write-scoped entity maintenance, separate from maintain's scan cap."""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from mcp import types

from ...compat import _sanitize_log_value
from ...reasoning.entity_rebuild import EntityRebuildPlan
from ...storage.graph import GraphStorage
from ...storage.hybrid import HybridMemoryStorage
from ...storage.milvus import MilvusMemoryStorage
from ...storage.milvus_graph import MilvusGraphStorage, _entity_key
from ...storage.read_policy import strict_reads
from ...storage.sqlite_vec import SqliteVecMemoryStorage

logger = logging.getLogger(__name__)
_rebuild_lock = asyncio.Lock()


@asynccontextmanager
async def _open_graph(
    primary: Any, dry_run: bool
) -> AsyncIterator[GraphStorage | MilvusGraphStorage]:
    """Open the graph belonging to this server, without mutating a preview."""
    if isinstance(primary, SqliteVecMemoryStorage):
        graph = GraphStorage(primary.db_path)
    elif isinstance(primary, MilvusMemoryStorage):
        graph = MilvusGraphStorage(
            uri=primary.uri, token=primary.token,
            collection_name=primary.collection_name,
        )
    else:
        raise RuntimeError(
            "Entity rebuild requires graph storage (sqlite_vec, hybrid, or milvus)"
        )
    try:
        if isinstance(graph, MilvusGraphStorage):
            await graph.initialize(create_if_missing=not dry_run)
        yield graph
    finally:
        await graph.close()


async def _confines_store(graph: MilvusGraphStorage) -> bool:
    """Whether the Milvus graph refuses a shares_entity pair across two stores.

    True when the association collection carries ``store``, and for a preview
    of a missing collection, which the real run creates with the field. An
    unreadable schema refuses every write, so the rebuild stops before it
    deletes anything rather than after.
    """
    if graph._has_store is None:
        if await graph._call_client(
            "has_collection", collection_name=graph.collection_name
        ):
            raise RuntimeError(
                "Milvus association collection schema could not be read, so "
                "every graph write would be refused (#2119); no edges deleted"
            )
        return True
    return graph._has_store


async def _rebuild(server: Any, dry_run: bool, report: dict) -> None:
    storage = await server._ensure_storage_initialized()
    primary = storage.primary if isinstance(storage, HybridMemoryStorage) else storage
    async with _open_graph(primary, dry_run) as graph:
        # None means every named store; limit=None drains Milvus's full iterator.
        # On Milvus the same scan carries each row's store, so the list that is
        # counted is the list the pairs are attributed from (#2123).
        with strict_reads():
            if isinstance(primary, MilvusMemoryStorage):
                rows = await primary.get_all_memories_with_store()
            else:
                rows = [
                    (m, None)
                    for m in await primary.get_all_memories(limit=None, store=None)
                ]
            expected = await primary.count_all_memories(store=None)
        if len(rows) != expected:
            raise RuntimeError(
                "Memory scan was incomplete or the store changed; no edges deleted"
            )
        memories = [m for m, _ in rows]
        store_of = None
        if isinstance(graph, MilvusGraphStorage) and await _confines_store(graph):
            store_of = {m.content_hash: store for m, store in rows}
        snapshot = await graph.get_edge_snapshot()
        key = (lambda name: name) if isinstance(graph, GraphStorage) else _entity_key
        plan = EntityRebuildPlan(snapshot, entity_key=key, store_of=store_of)
        await plan.build(memories)
        report.update(plan.describe())
        if not dry_run:
            await plan.apply(graph, report)
        report["success"] = True


async def handle_rebuild_entities(
    server: Any, arguments: dict
) -> list[types.TextContent]:
    """Preview by default; only a literal false permits destructive replacement."""
    dry_run = arguments.get("dry_run", True)
    started = time.monotonic()
    report = {
        "action": "rebuild_entities", "dry_run": dry_run, "success": False,
        "edges_deleted": 0, "edges_created": {"has_entity": 0, "shares_entity": 0},
        "errors": [],
    }
    try:
        if not isinstance(dry_run, bool):
            raise ValueError("dry_run must be a boolean")
        if _rebuild_lock.locked():
            raise RuntimeError("An entity rebuild is already running in this process")
        async with _rebuild_lock:
            await _rebuild(server, dry_run, report)
    except Exception as exc:
        report["success"] = False
        report["errors"].append(str(exc))
        logger.error("Entity rebuild failed: %s", _sanitize_log_value(exc))
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return [types.TextContent(type="text", text=json.dumps(report, indent=2))]
