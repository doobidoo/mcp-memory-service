"""Shared utility functions for storage backends.

These helpers were originally defined in milvus.py and are used across multiple
backends (Milvus, Cloudflare, Hybrid). Centralising them here eliminates
duplication and ensures consistent behaviour.
"""

import json
import logging
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

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


def _sanitize_log_value(value: object) -> str:
    """Sanitize a user-provided value for safe inclusion in log messages."""
    return str(value).replace("\n", "\\n").replace("\r", "\\r").replace("\x1b", "\\x1b")


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
