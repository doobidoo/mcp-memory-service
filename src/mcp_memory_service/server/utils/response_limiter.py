"""
Response size limiter for MCP Memory Service.

Prevents context window overflow by truncating responses at memory boundaries.
This module ensures that large memory retrieval operations don't crash Claude
or other LLM clients by exceeding their context window limits.

Example:
    >>> from mcp_memory_service.server.utils.response_limiter import (
    ...     truncate_memories,
    ...     format_truncated_response,
    ... )
    >>> memories = [{"content": "test", "content_hash": "abc123"}]
    >>> truncated, meta = truncate_memories(memories, max_chars=1000)
    >>> response = format_truncated_response(truncated, meta)
"""

import os
import sys
from typing import Any, Dict, List, Tuple

# Default max response size (can be overridden by environment variable)
# 0 = unlimited (for backward compatibility)
DEFAULT_MAX_CHARS = 0
raw_value = os.environ.get("MCP_MAX_RESPONSE_CHARS")
if raw_value:
    try:
        value = int(raw_value)
        if value >= 0:
            DEFAULT_MAX_CHARS = value
        else:
            print(
                f"WARNING: Invalid MCP_MAX_RESPONSE_CHARS value: {value}. "
                "Must be non-negative. Using default 0 (unlimited).",
                file=sys.stderr,
            )
    except ValueError:
        print(
            f"WARNING: Invalid MCP_MAX_RESPONSE_CHARS value: '{raw_value}'. "
            "Using default 0 (unlimited).",
            file=sys.stderr,
        )

# Estimated overhead per memory for metadata formatting.
# Accounts for: "=== Memory N ===", "Timestamp:", "Content:", "Hash:",
# "Relevance Score:", "Tags:", separators, and newlines.
MEMORY_OVERHEAD_CHARS = 200


def truncate_memories(
    memories: List[Dict[str, Any]],
    max_chars: int = 0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Truncate a list of memories to fit within max_chars limit.

    Truncates at memory boundaries (never mid-content) to preserve integrity.
    Always includes at least one memory if the input list is non-empty.

    Args:
        memories: List of memory dicts with "content", "content_hash", etc.
        max_chars: Maximum total characters. 0 = unlimited.

    Returns:
        Tuple containing:
            - truncated_memories: List of memories that fit within limit
            - metadata: Dict with truncation details including:
                - total_results: Original count
                - shown_results: Count after truncation
                - total_chars: Original estimated character count (content + overhead)
                - shown_chars: Estimated character count after truncation
                - truncated: Boolean indicating if truncation occurred
                - omitted_count: Number of memories omitted

    Example:
        >>> memories = [{"content": "a" * 1000}, {"content": "b" * 1000}]
        >>> truncated, meta = truncate_memories(memories, max_chars=1500)
        >>> len(truncated)
        1
        >>> meta["truncated"]
        True
    """
    empty_meta = {
        "total_results": 0,
        "shown_results": 0,
        "total_chars": 0,
        "shown_chars": 0,
        "truncated": False,
        "omitted_count": 0,
    }

    if not memories:
        return [], empty_meta

    # Calculate estimated size for each memory, including overhead
    memory_sizes = [(len(m.get("content") or "") + MEMORY_OVERHEAD_CHARS) for m in memories]
    total_estimated_chars = sum(memory_sizes)
    total_results = len(memories)

    # If no limit or under limit, return all
    if max_chars <= 0 or total_estimated_chars <= max_chars:
        return memories, {
            "total_results": total_results,
            "shown_results": total_results,
            "total_chars": total_estimated_chars,
            "shown_chars": total_estimated_chars,
            "truncated": False,
            "omitted_count": 0,
        }

    # Truncate at memory boundaries
    truncated: List[Dict[str, Any]] = []
    current_chars = 0

    for i, memory in enumerate(memories):
        mem_size = memory_sizes[i]

        # Allow at least one memory to be returned, even if it exceeds the limit
        if truncated and current_chars + mem_size > max_chars:
            break

        truncated.append(memory)
        current_chars += mem_size

    shown_results = len(truncated)

    return truncated, {
        "total_results": total_results,
        "shown_results": shown_results,
        "total_chars": total_estimated_chars,
        "shown_chars": current_chars,
        "truncated": True,
        "omitted_count": total_results - shown_results,
    }


def format_truncated_response(
    memories: List[Dict[str, Any]],
    meta: Dict[str, Any],
) -> str:
    """
    Format memories into a response string with truncation warning if needed.

    Args:
        memories: List of memory dicts to format.
        meta: Metadata dict from truncate_memories().

    Returns:
        Formatted string response with optional truncation header.

    Example:
        >>> memories = [{"content": "test", "content_hash": "abc123"}]
        >>> meta = {"truncated": False, "total_results": 1, "shown_results": 1}
        >>> response = format_truncated_response(memories, meta)
        >>> "=== Memory 1 ===" in response
        True
    """
    parts: List[str] = []

    # Add truncation warning header if truncated
    if meta.get("truncated"):
        warning = (
            f"[!] RESPONSE TRUNCATED: Showing {meta['shown_results']} of "
            f"{meta['total_results']} results "
            f"({meta['shown_chars']:,} of {meta['total_chars']:,} chars).\n"
            f"{meta['omitted_count']} result(s) omitted to prevent context overflow.\n"
            f"Use specific queries or hash-based retrieval for full content.\n"
        )
        parts.append(warning)
        parts.append("")

    # Format each memory
    for i, memory in enumerate(memories, 1):
        memory_lines = [f"=== Memory {i} ==="]

        # Add timestamp if available
        created_at = memory.get("created_at")
        if created_at:
            memory_lines.append(f"Timestamp: {created_at}")

        # Add content
        content = memory.get("content", "")
        memory_lines.append(f"Content: {content}")

        # Add hash
        content_hash = memory.get("content_hash", "")
        memory_lines.append(f"Hash: {content_hash}")

        # Add relevance score if available
        score = memory.get("relevance_score") or memory.get("similarity_score")
        if score is not None:
            memory_lines.append(f"Relevance Score: {score:.2f}")

        # Add tags if available
        tags = memory.get("tags", [])
        if tags:
            if isinstance(tags, list):
                memory_lines.append(f"Tags: {', '.join(tags)}")
            else:
                memory_lines.append(f"Tags: {tags}")

        memory_lines.append("---")
        parts.append("\n".join(memory_lines))

    return "\n".join(parts)


def apply_response_limit(
    memories: List[Dict[str, Any]],
    max_chars: int = 0,
    header: str = "",
) -> str:
    """
    Convenience function: truncate and format in one call.

    Args:
        memories: List of memory dicts.
        max_chars: Maximum response size. 0 = use env default or unlimited.
        header: Optional header to prepend to the response.

    Returns:
        Formatted response string (truncated if needed).

    Example:
        >>> memories = [{"content": "test", "content_hash": "abc123"}]
        >>> response = apply_response_limit(memories, max_chars=10000)
        >>> isinstance(response, str)
        True
    """
    # Use environment default if no limit specified
    effective_max = max_chars if max_chars > 0 else DEFAULT_MAX_CHARS

    truncated, meta = truncate_memories(memories, effective_max)
    return header + format_truncated_response(truncated, meta)


def safe_retrieve_response(
    memories: List[Dict[str, Any]],
    max_chars: int = 40000,
) -> str:
    """
    Safe wrapper with sensible default limit.

    Use this as a drop-in replacement for formatting retrieve results.
    Default 40KB limit is safe for most context windows.

    Args:
        memories: List of memory dicts.
        max_chars: Maximum response size (default: 40000).

    Returns:
        Formatted response string (truncated if needed).

    Example:
        >>> memories = [{"content": "important info", "content_hash": "hash1"}]
        >>> response = safe_retrieve_response(memories)
        >>> "important info" in response
        True
    """
    return apply_response_limit(memories, max_chars)
