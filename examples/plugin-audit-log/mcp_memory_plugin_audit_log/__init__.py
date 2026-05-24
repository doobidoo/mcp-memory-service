"""Example plugin: Audit Log.

Demonstrates all 4 lifecycle hooks by logging events to a JSON Lines file.
Install alongside mcp-memory-service and hooks activate automatically.

The default log format is privacy-safe: it records counts, lengths and keyed
identifier hashes instead of raw queries, tags or memory contents. Set
``MCP_PLUGIN_AUDIT_LOG_PRIVACY_MODE=raw`` only for local debugging.

Usage:
    pip install -e examples/plugin-audit-log/
    # Restart mcp-memory-service — plugin loads via entry_points discovery
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
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

# Safe is the default because audit logs are often shared during debugging.
# Raw mode preserves the original example fields for local-only inspection.
PRIVACY_MODE = os.getenv("MCP_PLUGIN_AUDIT_LOG_PRIVACY_MODE", "safe").lower()
HMAC_KEY = os.getenv("MCP_PLUGIN_AUDIT_LOG_HMAC_KEY", "")


def register(ctx: Any) -> None:
    """Entry point called by PluginRegistry at startup."""
    global PRIVACY_MODE
    if PRIVACY_MODE not in ("safe", "raw"):
        logger.warning("audit-log: unknown privacy mode %r; using safe", PRIVACY_MODE)
        PRIVACY_MODE = "safe"

    logger.info(
        "audit-log plugin: registered (log=%s, privacy_mode=%s)",
        AUDIT_LOG_PATH,
        PRIVACY_MODE,
    )
    ctx.on("on_store", on_store)
    ctx.on("on_delete", on_delete)
    ctx.on("on_retrieve", on_retrieve)
    ctx.on("on_consolidate", on_consolidate)


def _privacy_mode() -> str:
    """Return the active privacy mode."""
    return PRIVACY_MODE


def _hmac_hash(value: Any) -> str | None:
    """Return a stable HMAC-SHA256 hash, or None when no key is configured."""
    if not HMAC_KEY or value in (None, ""):
        return None
    return "hmac-sha256:" + hmac.new(
        HMAC_KEY.encode("utf-8"),
        str(value).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _hash_metadata() -> dict[str, str | bool]:
    """Build common hash metadata for safe-mode events."""
    metadata: dict[str, str | bool] = {
        "privacy_mode": "safe",
        "raw_query_included": False,
        "raw_content_included": False,
        "raw_tags_included": False,
        "hash_algorithm": "hmac-sha256" if HMAC_KEY else "none",
    }
    if not HMAC_KEY:
        metadata["identifier_hashes_omitted_reason"] = "MCP_PLUGIN_AUDIT_LOG_HMAC_KEY not set"
    return metadata


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
    content_hash = memory_dict.get("content_hash", "unknown")
    if _privacy_mode() == "raw":
        await _write_event("store", {
            "privacy_mode": "raw",
            "hash": content_hash,
            "memory_type": memory_dict.get("memory_type", ""),
            "tags": memory_dict.get("tags", []),
            "content_length": len(memory_dict.get("content") or ""),
        })
        return

    event = {
        **_hash_metadata(),
        "memory_hash_hmac": _hmac_hash(content_hash),
        "memory_type": memory_dict.get("memory_type", ""),
        "tag_count": len(memory_dict.get("tags", []) or []),
        "content_length": len(memory_dict.get("content") or ""),
    }
    # Drop None values so missing keys are explicit via metadata, not null ids.
    await _write_event("store", {k: v for k, v in event.items() if v is not None})


async def on_delete(content_hash: str) -> None:
    """Log every memory deletion."""
    if _privacy_mode() == "raw":
        await _write_event("delete", {"privacy_mode": "raw", "hash": content_hash})
        return

    event = {
        **_hash_metadata(),
        "memory_hash_hmac": _hmac_hash(content_hash),
    }
    await _write_event("delete", {k: v for k, v in event.items() if v is not None})


async def on_retrieve(query: str, results: list[dict]) -> list[dict]:
    """Log retrieval queries and result count. Returns results unchanged."""
    if _privacy_mode() == "raw":
        await _write_event("retrieve", {
            "privacy_mode": "raw",
            "query": query[:100],
            "result_count": len(results),
        })
        return results

    event = {
        **_hash_metadata(),
        "query_hash_hmac": _hmac_hash(query),
        "query_length": len(query or ""),
        "result_count": len(results),
    }
    await _write_event("retrieve", {k: v for k, v in event.items() if v is not None})
    return results  # Pass through unmodified


async def on_consolidate(report: dict) -> None:
    """Log consolidation events."""
    await _write_event("consolidate", {
        "privacy_mode": _privacy_mode(),
        "memories_processed": report.get("memories_processed", 0),
        "time_horizon": report.get("time_horizon", ""),
    })
