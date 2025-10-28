"""
Services package for MCP Memory Service.

This package contains shared business logic services that provide
consistent behavior across different interfaces (API, MCP tools).
"""

from .memory_service import MemoryService, MemoryResult

__all__ = ["MemoryService", "MemoryResult"]
