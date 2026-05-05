"""Plugin system for mcp-memory-service.

Plugins register via Python entry_points:

    [project.entry-points."mcp_memory_service.plugins"]
    my_plugin = "my_package:register"

Each plugin's `register(ctx: PluginContext)` is called once at startup,
after storage initialization completes.
"""

from .context import PluginContext
from .registry import PluginRegistry

__all__ = ["PluginContext", "PluginRegistry"]
