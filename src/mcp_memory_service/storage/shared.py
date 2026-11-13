"""Shared utility functions for storage backends.

These helpers were originally defined in milvus.py and are used across multiple
backends (Milvus, Cloudflare, Hybrid). Centralising them here eliminates
duplication and ensures consistent behaviour.
"""

import json
import logging
import threading
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

_versioned_conflict_exclusion: ContextVar[Optional[str]] = ContextVar(
    "versioned_conflict_exclusion", default=None
)


@contextmanager
def _exclude_versioned_parent(content_hash: str) -> Iterator[None]:
    token = _versioned_conflict_exclusion.set(content_hash)
    try:
        yield
    finally:
        _versioned_conflict_exclusion.reset(token)


# A vector computed before the row was read, for ``store()`` to write instead
# of embedding the content again. ``update_memory_versioned`` embeds the new
# content ahead of its read so that no embedding call sits between the read
# and the write (#1966).
_precomputed_store_vector: ContextVar[Optional[List[float]]] = ContextVar(
    "precomputed_store_vector", default=None
)


@contextmanager
def _store_with_vector(embedding: List[float]) -> Iterator[None]:
    token = _precomputed_store_vector.set(embedding)
    try:
        yield
    finally:
        _precomputed_store_vector.reset(token)

# ---------------------------------------------------------------------------
# Bounded LRU embedding cache
# ---------------------------------------------------------------------------

_EMBEDDING_CACHE_MAX = 1024
_EMBEDDING_CACHE: "OrderedDict[str, List[float]]" = OrderedDict()
_EMBEDDING_CACHE_LOCK = threading.Lock()


def _embedding_cache_get(key: str) -> Optional[List[float]]:
    with _EMBEDDING_CACHE_LOCK:
        value = _EMBEDDING_CACHE.get(key)
        if value is None:
            return None
        _EMBEDDING_CACHE.move_to_end(key)
        return value


def _embedding_cache_put(key: str, value: List[float]) -> None:
    with _EMBEDDING_CACHE_LOCK:
        if key in _EMBEDDING_CACHE:
            _EMBEDDING_CACHE.move_to_end(key)
            _EMBEDDING_CACHE[key] = value
            return
        _EMBEDDING_CACHE[key] = value
        while len(_EMBEDDING_CACHE) > _EMBEDDING_CACHE_MAX:
            _EMBEDDING_CACHE.popitem(last=False)


def _embedding_cache_size() -> int:
    with _EMBEDDING_CACHE_LOCK:
        return len(_EMBEDDING_CACHE)


def _embedding_cache_clear() -> int:
    """Clear all entries from the shared embedding cache, returning count cleared."""
    with _EMBEDDING_CACHE_LOCK:
        cleared_count = len(_EMBEDDING_CACHE)
        _EMBEDDING_CACHE.clear()
        return cleared_count


# ---------------------------------------------------------------------------
# String / sanitisation helpers
# ---------------------------------------------------------------------------



def _escape_like(value: str) -> str:
    """Escape Milvus ``like`` wildcards so a user-supplied value matches literally.

    Milvus treats ``%`` and ``_`` as wildcards in ``like`` patterns, and ``\\%``
    and ``\\_`` as the literal characters. Escaping them, instead of dropping
    them, keeps values such as ``upload_id:abc`` matchable: dropping ``_``
    searched for ``uploadid:abc``, which never matched (#1684).

    Backslashes are left alone on purpose. Inside a ``%...%`` pattern, Milvus
    Lite 3.2.2 keeps a backslash before any other character as a literal, while
    newer servers collapse ``\\\\`` to one (milvus-io/milvus#43864), so doubling
    it would break the existing match for tags such as ``foo\\`` (#1107). The
    result still goes through ``escape_expr_value`` before it is placed inside
    a ``"..."`` literal, which handles the expression parser's own unescaping.
    """
    return value.replace("%", "\\%").replace("_", "\\_")


def _tags_to_string(tags: Optional[List[str]]) -> str:
    """Encode a tag list as a comma-delimited string with leading/trailing commas.

    Leading/trailing commas let us match an exact tag with
    ``tags like "%,<tag>,%"`` rather than a substring match.
    """
    if not tags:
        return ""
    clean = [t.strip() for t in tags if isinstance(t, str) and t.strip()]
    if not clean:
        return ""
    return "," + ",".join(clean) + ","


def _string_to_tags(raw: Optional[str]) -> List[str]:
    """Decode the comma-delimited tag string back into a list."""
    if not raw:
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]


def _safe_json_loads(text: str, context: str = "") -> Dict[str, Any]:
    """Parse JSON with defensive fallbacks; always return a dict."""
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.error("JSON decode error in %s: %s", context, exc)
        return {}
    if not isinstance(parsed, dict):
        logger.warning("Non-dict JSON in %s: %s", context, type(parsed).__name__)
        return {}
    return parsed


# Keys that Memory.record_access() writes. The access write-back after a read
# merges only these into the stored metadata JSON (#1464).
ACCESS_METADATA_KEYS = ("access_count", "last_accessed_at", "access_queries")


def _access_metadata_merge_sql(column: str, metadata: Dict[str, Any]) -> tuple:
    """Build a SQL expression that writes only the access keys into ``column``.

    Replacing the whole metadata JSON with an in-memory snapshot drops any key
    another writer committed after the snapshot was read (#1464). The returned
    expression ``json_set``s the access keys present in ``metadata`` into the
    stored JSON instead. Works on SQLite and Cloudflare D1.

    - A JSON object: the access keys are set, every other key is kept.
    - NULL, empty, or valid JSON that is not an object: starts from ``{}``,
      the same fallback as ``_safe_json_loads``.
    - Text that SQLite's strict ``json_valid()`` rejects is left unchanged.
      Python's ``json.dumps`` writes ``NaN`` and ``Infinity``, which
      ``json.loads`` reads back but SQLite does not, so resetting such a row
      to ``{}`` would erase metadata the reader accepts. Callers that can run
      Python on the row merge it with ``_merge_access_metadata_text``.

    Args:
        column: Name of the metadata JSON column (a trusted identifier).
        metadata: The memory's metadata after ``record_access()``.

    Returns:
        ``(sql_expression, params)``; ``params`` binds the expression's
        placeholders in order.
    """
    # No ELSE: text SQLite cannot parse gives NULL here, and COALESCE below
    # keeps the stored value.
    base = (
        f"CASE WHEN {column} IS NULL OR {column} = '' THEN '{{}}' "
        f"WHEN json_valid({column}) THEN "
        f"CASE WHEN json_type({column}) = 'object' THEN {column} ELSE '{{}}' END END"
    )
    keys = [key for key in ACCESS_METADATA_KEYS if key in metadata]
    if not keys:
        return f"COALESCE({base}, {column})", []
    paths = ", ".join(f"'$.{key}', json(?)" for key in keys)
    return (
        f"COALESCE(json_set({base}, {paths}), {column})",
        [json.dumps(metadata[key]) for key in keys],
    )


def _merge_access_metadata_text(stored: Optional[str], metadata: Dict[str, Any]) -> str:
    """Merge the access keys of ``metadata`` into stored metadata text in Python.

    Used for rows whose metadata SQLite's JSON functions reject but Python's
    reader may accept (``NaN``, ``Infinity``). The stored text is parsed with
    ``_safe_json_loads``, so it gets the same ``{}`` fallback as on read.
    """
    merged = _safe_json_loads(stored, "access_metadata_merge")
    for key in ACCESS_METADATA_KEYS:
        if key in metadata:
            merged[key] = metadata[key]
    return json.dumps(merged)
