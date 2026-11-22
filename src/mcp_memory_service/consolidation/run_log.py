"""One memory per consolidation run, so the history syncs like any memory (#2097)."""

import os
import socket
from datetime import datetime

from ..models.memory import Memory
from ..storage.base import MemoryStorage
from ..utils.hashing import generate_content_hash
from .base import ConsolidationReport

RUN_LOG_TYPE = "consolidation-run"

_MAX_ERRORS = 10
_MAX_ERROR_CHARS = 500


def build_run_log(
    report: ConsolidationReport, run_id: str, trigger: str, success: bool
) -> Memory:
    """The log memory for one run, filed under ``MCP_AGENT_ID`` or else the hostname."""
    agent_id = os.environ.get("MCP_AGENT_ID")
    hostname = socket.gethostname()
    node = agent_id or hostname
    ended_at = datetime.now()
    duration = (ended_at - report.start_time).total_seconds()
    status = "" if success else ", failed"
    content = (
        f"Consolidation {report.time_horizon} on {node}: "
        f"{report.memories_processed} processed, "
        f"{report.associations_discovered} associations, "
        f"{report.clusters_created} clusters, "
        f"{report.memories_compressed} compressed, "
        f"{report.memories_archived} archived, "
        f"{duration:.0f} s{status} (run {run_id})"
    )
    return Memory(
        content=content,
        content_hash=generate_content_hash(content),
        tags=[RUN_LOG_TYPE, f"horizon:{report.time_horizon}", f"node:{node}"],
        memory_type=RUN_LOG_TYPE,
        metadata={
            "run_id": run_id,
            "node": node,
            "horizon": report.time_horizon,
            "trigger": trigger,
            "agent_id": agent_id,
            "hostname": hostname,
            "started_at": report.start_time.timestamp(),
            "ended_at": ended_at.timestamp(),
            "duration_seconds": duration,
            "success": success,
            "memories_processed": report.memories_processed,
            "associations_discovered": report.associations_discovered,
            "clusters_created": report.clusters_created,
            "memories_compressed": report.memories_compressed,
            "memories_archived": report.memories_archived,
            "errors": [
                error[:_MAX_ERROR_CHARS] for error in report.errors[:_MAX_ERRORS]
            ],
        },
    )


async def store_run_log(storage: MemoryStorage, run_log: Memory, keep: int) -> None:
    """Store *run_log*, then delete this node's runs beyond the newest *keep*."""
    stored, message = await storage.store(run_log, skip_semantic_dedup=True)
    if not stored:
        raise RuntimeError(message)
    node_tags = [f"node:{run_log.metadata['node']}"]
    overflow = (
        await storage.count_all_memories(memory_type=RUN_LOG_TYPE, tags=node_tags)
        - keep
    )
    if overflow <= 0:
        return
    stale = await storage.get_all_memories(
        limit=overflow, offset=keep, memory_type=RUN_LOG_TYPE, tags=node_tags
    )
    for run in stale:
        await storage.delete(run.content_hash)
