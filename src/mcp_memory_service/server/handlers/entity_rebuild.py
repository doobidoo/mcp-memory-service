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


async def _rebuild(server: Any, dry_run: bool, report: dict) -> None:
    storage = await server._ensure_storage_initialized()
    primary = storage.primary if isinstance(storage, HybridMemoryStorage) else storage
    async with _open_graph(primary, dry_run) as graph:
        # None means every named store; limit=None drains Milvus's full iterator.
        with strict_reads():
            memories = await primary.get_all_memories(limit=None, store=None)
            expected = await primary.count_all_memories(store=None)
        if len(memories) != expected:
            raise RuntimeError(
                "Memory scan was incomplete or the store changed; no edges deleted"
            )
        # Milvus listings include tombstones stored inside metadata, unlike
        # SQLite's deleted_at SQL filter. Verify the raw count before filtering.
        if isinstance(graph, MilvusGraphStorage):
            memories = [
                m for m in memories if m.metadata.get("deleted_at") is None
            ]
        snapshot = await graph.get_edge_snapshot()
        key = (lambda name: name) if isinstance(graph, GraphStorage) else _entity_key
        plan = EntityRebuildPlan(snapshot, entity_key=key)
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
