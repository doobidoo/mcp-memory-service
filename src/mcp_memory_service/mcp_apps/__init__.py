"""MCP Apps extension for mcp-memory-service.

Provides interactive HTML UIs for MCP Apps-capable clients
(VS Code Insiders + chat.mcp.apps.enabled).

Enable via: MEMORY_MCP_APPS=true environment variable

See: https://github.com/github/github-mcp-server/discussions/2048
"""

from .renderer import (
    mcp_apps_enabled,
    render_memory_search_results,
    render_memory_list,
    render_health_dashboard,
)

__all__ = [
    "mcp_apps_enabled",
    "render_memory_search_results",
    "render_memory_list",
    "render_health_dashboard",
]
