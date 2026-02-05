#!/usr/bin/env python3
"""Simple test of FastMCP server structure for memory service."""

import sys

# Add src to path
sys.path.insert(0, "src")

from mcp.server.fastmcp import FastMCP

# Create a simple FastMCP server for testing
mcp = FastMCP("Test Memory Service")


@mcp.tool()
def test_store_memory(content: str, tags: list[str] = None) -> dict:
    """Test memory storage function."""
    return {"success": True, "message": f"Stored: {content}", "tags": tags or []}


@mcp.tool()
def test_health() -> dict:
    """Test health check."""
    return {"status": "healthy", "version": "4.0.0-alpha.1"}


if __name__ == "__main__":
    print("FastMCP Memory Service Test")
    print("Server configured with basic tools")
    print("Available tools:")
    print("- test_store_memory")
    print("- test_health")
    print("\nTo run server: mcp.run(transport='streamable-http')")
