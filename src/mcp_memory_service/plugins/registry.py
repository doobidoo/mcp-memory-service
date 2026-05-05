"""Plugin registry — discovery via entry_points and hook firing."""

from __future__ import annotations

import logging
from importlib.metadata import entry_points
from typing import Any

from .context import HookName, PluginContext

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "mcp_memory_service.plugins"


class PluginRegistry:
    """Discovers and manages plugins."""

    def __init__(self, ctx: PluginContext) -> None:
        self.ctx = ctx
        self._loaded: list[str] = []

    def discover_and_register(self) -> None:
        """Load all plugins from entry_points and call their register()."""
        eps = entry_points()
        # Python 3.12+ returns SelectableGroups, 3.9-3.11 returns dict
        if hasattr(eps, "select"):
            plugin_eps = eps.select(group=ENTRY_POINT_GROUP)
        else:
            plugin_eps = eps.get(ENTRY_POINT_GROUP, [])

        for ep in plugin_eps:
            try:
                register_fn = ep.load()
                register_fn(self.ctx)
                self._loaded.append(ep.name)
                logger.info("Plugin '%s' registered successfully", ep.name)
            except Exception:
                logger.exception("Failed to load plugin '%s'", ep.name)

    @property
    def loaded_plugins(self) -> list[str]:
        """Names of successfully loaded plugins."""
        return list(self._loaded)

    async def fire(self, hook: HookName, *args: Any, **kwargs: Any) -> Any:
        """Fire a hook, calling all registered handlers in order.

        For on_retrieve: handlers can modify and return the results list.
        For other hooks: handlers are called for side effects only.
        """
        handlers = self.ctx._hooks.get(hook, [])
        if not handlers:
            return args[0] if args else None

        result = args[0] if args else None

        for fn in handlers:
            try:
                if hook == "on_retrieve":
                    # on_retrieve can rerank/filter results
                    result = await fn(*args, **kwargs) or result
                else:
                    await fn(*args, **kwargs)
            except Exception:
                logger.exception(
                    "Error in plugin hook '%s' handler %s", hook, fn.__qualname__
                )

        return result
