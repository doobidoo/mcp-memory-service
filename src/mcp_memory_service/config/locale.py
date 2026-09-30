"""Unified locale configuration for all subsystems."""
import os
from functools import lru_cache


@lru_cache(maxsize=None)
def _parse_locales(raw: str) -> tuple[str, ...]:
    return tuple(loc.strip() for loc in raw.split(",") if loc.strip())


def get_active_locales() -> list[str]:
    """Get active locales from MCP_LOCALE (fallback HARVEST_LOCALE, default 'en').

    The env vars are read on every call, so a locale set after another module
    already asked (NLI does at import) is still honored. Only the parsing is cached.
    """
    raw = os.environ.get("MCP_LOCALE") or os.environ.get("HARVEST_LOCALE", "en")
    return list(_parse_locales(raw))


# Kept so existing callers/tests that reset the cache keep working.
get_active_locales.cache_clear = _parse_locales.cache_clear  # type: ignore[attr-defined]
