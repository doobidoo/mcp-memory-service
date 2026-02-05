"""TOON format encoding with security hardening.

Converts memory search results to TOON format with DoS protection:
- 10MB content limit per memory (prevents memory exhaustion)
- 500-level metadata depth limit (prevents stack overflow)
- 5-second timeout on encoding (prevents CPU DoS)
- Header sanitization (prevents header injection)
"""

import re
import signal
from contextlib import contextmanager
from typing import Any

from toon_format import encode


class TimeoutError(Exception):
    """Raised when encoding exceeds timeout limit."""

    pass


@contextmanager
def timeout_context(seconds: int):
    """Context manager for timeout protection using signal.alarm.

    Args:
        seconds: Maximum execution time in seconds

    Raises:
        TimeoutError: If execution exceeds timeout
    """

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation exceeded {seconds} second timeout")

    # Set the signal handler and alarm
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Restore previous handler and cancel alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def validate_memory_for_encoding(memory: dict[str, Any]) -> None:
    """Validate memory meets security requirements for encoding.

    Security limits:
    - Content size: 10MB maximum (prevents memory exhaustion)
    - Metadata depth: 1000 levels maximum (prevents stack overflow)

    Args:
        memory: Memory dictionary to validate

    Raises:
        ValueError: If memory exceeds security limits
    """
    # Validate content size
    content = memory.get("content", "")
    content_size = len(content.encode("utf-8"))
    max_content_size = 10 * 1024 * 1024  # 10MB

    if content_size > max_content_size:
        raise ValueError(f"Memory content size {content_size} bytes exceeds maximum {max_content_size} bytes (10MB limit)")

    # Validate metadata nesting depth
    # Use 500 as limit to stay well within Python's ~1000 recursion limit
    metadata = memory.get("metadata")
    if metadata is not None:
        max_depth = 500
        depth = _get_nested_depth(metadata, max_depth=max_depth)
        if depth > max_depth:
            raise ValueError(f"Metadata nesting depth {depth} exceeds maximum {max_depth} levels")


def _get_nested_depth(obj: Any, current_depth: int = 0, max_depth: int = 500) -> int:
    """Calculate maximum nesting depth with early termination to prevent stack overflow.

    Args:
        obj: Object to measure depth
        current_depth: Current recursion depth
        max_depth: Maximum allowed depth (default 1000)

    Returns:
        Maximum nesting depth (returns early if max_depth exceeded)
    """
    if not isinstance(obj, (dict, list)):
        return current_depth

    # Early termination BEFORE recursing to prevent stack overflow
    if current_depth >= max_depth:
        return current_depth + 1  # Signal overflow

    if isinstance(obj, dict):
        if not obj:
            return current_depth
        # Use iterative approach to find max depth
        max_child_depth = current_depth
        for value in obj.values():
            child_depth = _get_nested_depth(value, current_depth + 1, max_depth)
            if child_depth > max_child_depth:
                max_child_depth = child_depth
            # Early exit if we've already exceeded max_depth
            if max_child_depth > max_depth:
                return max_child_depth
        return max_child_depth

    # isinstance(obj, list)
    if not obj:
        return current_depth
    max_child_depth = current_depth
    for item in obj:
        child_depth = _get_nested_depth(item, current_depth + 1, max_depth)
        if child_depth > max_child_depth:
            max_child_depth = child_depth
        # Early exit if we've already exceeded max_depth
        if max_child_depth > max_depth:
            return max_child_depth
    return max_child_depth


def sanitize_header_value(value: str) -> str:
    """Sanitize string for safe use in headers and logs.

    Removes characters that could cause header injection attacks.
    Allows: alphanumeric, hyphens, spaces
    Truncates to 200 characters maximum.

    Args:
        value: String to sanitize

    Returns:
        Sanitized string safe for headers
    """
    # Strip non-alphanumeric characters except hyphens and spaces
    sanitized = re.sub(r"[^a-zA-Z0-9\s\-]", "", value)

    # Truncate to 200 characters
    max_length = 200
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]

    return sanitized


def _format_pagination_header(pagination: dict[str, Any]) -> str:
    """Format pagination metadata as a header line.

    Creates a comment-style header with pagination information:
    # page=2 total=250 page_size=10 has_more=true total_pages=25

    Args:
        pagination: Dictionary with pagination metadata

    Returns:
        Formatted pagination header string
    """
    # Extract pagination fields with defaults
    page = pagination.get("page", 1)
    total = pagination.get("total", 0)
    page_size = pagination.get("page_size", 10)
    has_more = pagination.get("has_more", False)
    total_pages = pagination.get("total_pages", 0)

    # Format as space-separated key=value pairs
    return f"# page={page} total={total} page_size={page_size} has_more={str(has_more).lower()} total_pages={total_pages}"


def format_search_results_as_toon(results: list[dict[str, Any]], pagination: dict[str, Any] | None = None) -> tuple[str, str]:
    """Convert memory search results to TOON format with security hardening.

    Extracts all memory fields and encodes to TOON with:
    - Content validation (10MB limit)
    - Metadata depth validation (1000 levels)
    - Encoding timeout (5 seconds)
    - Error sanitization for headers
    - Optional pagination metadata header

    Args:
        results: List of memory dictionaries from search
        pagination: Optional pagination metadata (total, page, page_size, has_more, total_pages)

    Returns:
        Tuple of (toon_string, media_type)
        - toon_string: TOON-encoded memory results with optional pagination header
        - media_type: Always "text/plain"

    Raises:
        ValueError: If memory validation fails
        TimeoutError: If encoding exceeds 5 seconds
    """
    # Handle empty results
    if not results:
        # Include pagination info even for empty results if provided
        if pagination:
            header = _format_pagination_header(pagination)
            return (f"{header}\nNo memories found matching your query.", "text/plain")
        return ("No memories found matching your query.", "text/plain")

    try:
        # Validate each memory before encoding
        for memory in results:
            validate_memory_for_encoding(memory)

        # Extract all memory fields for TOON encoding
        memories_for_encoding = []
        for memory in results:
            memory_data = {
                "content": memory.get("content", ""),
                "tags": memory.get("tags", []),
                "metadata": memory.get("metadata", {}),
                "created_at": memory.get("created_at", ""),
                "updated_at": memory.get("updated_at", ""),
                "content_hash": memory.get("content_hash", ""),
            }

            # Add similarity score if present (from semantic search)
            if "similarity_score" in memory:
                memory_data["similarity_score"] = memory["similarity_score"]

            memories_for_encoding.append(memory_data)

        # Encode with 5-second timeout protection
        with timeout_context(5):
            toon_output = encode(memories_for_encoding)

        # Prepend pagination header if provided
        if pagination:
            header = _format_pagination_header(pagination)
            toon_output = f"{header}\n{toon_output}"

        return (toon_output, "text/plain")

    except TimeoutError as e:
        # Sanitize timeout error for headers
        error_msg = sanitize_header_value(str(e))
        return (
            f"TOON encoding timeout: {error_msg}. Try reducing result size with pagination.",
            "text/plain",
        )

    except ValueError as e:
        # Sanitize validation error for headers
        error_msg = sanitize_header_value(str(e))
        return (
            f"Memory validation failed: {error_msg}",
            "text/plain",
        )

    except Exception as e:
        # Sanitize generic errors for headers
        error_msg = sanitize_header_value(str(e))
        return (
            f"TOON encoding error: {error_msg}. Please report this issue.",
            "text/plain",
        )
