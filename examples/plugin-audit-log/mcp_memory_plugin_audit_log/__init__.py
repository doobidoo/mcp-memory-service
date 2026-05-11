"""Example plugin: Audit Log.

Demonstrates all 4 lifecycle hooks by logging events to a JSON Lines file.
Install alongside mcp-memory-service and hooks activate automatically.

Usage:
    pip install -e examples/plugin-audit-log/
    # Restart mcp-memory-service — plugin loads via entry_points discovery
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Audit log file path (configurable via env var)
AUDIT_LOG_PATH = Path(os.getenv(
    "MCP_PLUGIN_AUDIT_LOG_PATH",
    "/tmp/mcp-memory-audit.jsonl"
))


def register(ctx: Any) -> None:
    """Entry point called by PluginRegistry at startup."""
    logger.info("audit-log plugin: registered (log=%s)", AUDIT_LOG_PATH)
    ctx.on("on_store", on_store)
    ctx.on("on_delete", on_delete)
    ctx.on("on_retrieve", on_retrieve)
    ctx.on("on_consolidate", on_consolidate)


def _write_event_sync(event_type: str, data: dict) -> None:
    """Synchronous file write — called via asyncio.to_thread."""
    event = {
        "timestamp": time.time(),
        "event": event_type,
        **data,
    }
    try:
        with AUDIT_LOG_PATH.open("a") as f:
            f.write(json.dumps(event) + "\n")
    except OSError as e:
        logger.warning("audit-log: failed to write event: %s", e)


async def _write_event(event_type: str, data: dict) -> None:
    """Offload blocking file I/O to a thread."""
    await asyncio.to_thread(_write_event_sync, event_type, data)


async def on_store(memory_dict: dict) -> None:
    """Log every memory store event."""
    await _write_event("store", {
        "hash": memory_dict.get("content_hash", "unknown"),
        "memory_type": memory_dict.get("memory_type", ""),
        "tags": memory_dict.get("tags", []),
        "content_length": len(memory_dict.get("content", "")),
    })


async def on_delete(content_hash: str) -> None:
    """Log every memory deletion."""
    await _write_event("delete", {"hash": content_hash})


async def on_retrieve(query: str, results: list[dict]) -> list[dict]:
    """Log retrieval queries and result count. Returns results unchanged."""
    await _write_event("retrieve", {
        "query": query[:100],
        "result_count": len(results),
    })
    return results  # Pass through unmodified


async def on_consolidate(report: dict) -> None:
    """Log consolidation events."""
    await _write_event("consolidate", {
        "memories_processed": report.get("memories_processed", 0),
        "time_horizon": report.get("time_horizon", ""),
    })
