"""Plugin: Smart Tagger.

Auto-tags memories on store based on content patterns.
Boosts mistake-note results on retrieve when context matches.

Usage:
    pip install -e examples/plugin-smart-tagger/
    # Restart mcp-memory-service — plugin loads via entry_points discovery

Configuration (env vars):
    MCP_PLUGIN_SMART_TAGGER_ENABLED=true (default: true)
    MCP_PLUGIN_SMART_TAGGER_BOOST=0.15 (default: 0.15 — score boost for matching mistake-notes)
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

ENABLED = os.getenv("MCP_PLUGIN_SMART_TAGGER_ENABLED", "true").lower() == "true"
MISTAKE_BOOST = float(os.getenv("MCP_PLUGIN_SMART_TAGGER_BOOST", "0.15"))

# Auto-tag rules: pattern → tag to add
TAG_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(decided|decision|chose|picked)\b", re.I), "decision"),
    (re.compile(r"\b(bug|fix|error|crash|broke)\b", re.I), "bug"),
    (re.compile(r"\b(convention|pattern|always|never|rule)\b", re.I), "convention"),
    (re.compile(r"\b(learned|lesson|mistake|wrong)\b", re.I), "learning"),
    (re.compile(r"\b(postgresql|postgres|redis|sqlite|mysql)\b", re.I), "database"),
    (re.compile(r"\b(docker|k8s|kubernetes|helm|deploy)\b", re.I), "infra"),
    (re.compile(r"\b(react|vue|angular|frontend|css)\b", re.I), "frontend"),
    (re.compile(r"\b(fastapi|spring|express|backend|api)\b", re.I), "backend"),
]


def register(ctx: Any) -> None:
    """Entry point called by PluginRegistry at startup."""
    if not ENABLED:
        logger.info("smart-tagger plugin: disabled via env var")
        return
    logger.info("smart-tagger plugin: registered (boost=%.2f)", MISTAKE_BOOST)
    ctx.on("on_store", on_store)
    ctx.on("on_retrieve", on_retrieve)


async def on_store(memory_dict: dict) -> None:
    """Auto-tag memories based on content patterns."""
    content = memory_dict.get("content", "")
    existing_tags = set(memory_dict.get("tags", []))

    for pattern, tag in TAG_RULES:
        if tag not in existing_tags and pattern.search(content):
            existing_tags.add(tag)

    memory_dict["tags"] = list(existing_tags)


async def on_retrieve(query: str, results: list[dict]) -> list[dict]:
    """Boost mistake-notes that match the query context."""
    if not results:
        return results

    for result in results:
        tags = result.get("tags", [])
        if "mistake-note" in tags or "error-replay" in tags:
            current_score = result.get("similarity", result.get("score", 0.5))
            result["similarity"] = min(1.0, current_score + MISTAKE_BOOST)

    # Re-sort by score after boosting
    results.sort(key=lambda r: r.get("similarity", r.get("score", 0)), reverse=True)
    return results
