# Copyright 2024 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Code Execution API for MCP Memory Service.

This module provides a lightweight, token-efficient interface for direct
Python code execution, replacing verbose MCP tool calls with compact
function calls and results.

Token Efficiency Comparison:
    - Import: ~10 tokens (once per session)
    - search(5 results): ~385 tokens vs ~2,625 (85% reduction)
    - store(): ~15 tokens vs ~150 (90% reduction)
    - health(): ~20 tokens vs ~125 (84% reduction)

Annual Savings (Conservative):
    - 10 users x 5 sessions/day x 365 days x 6,000 tokens = 109.5M tokens/year
    - At $0.15/1M tokens: $16.43/year per 10-user deployment

Performance:
    - First call: ~50ms (includes storage initialization)
    - Subsequent calls: ~5-10ms (connection reused)
    - Memory overhead: <10MB

Usage Example:
    >>> from mcp_memory_service.api import search, store, health
    >>>
    >>> # Search memories (20 tokens)
    >>> results = search("architecture decisions", limit=5)
    >>> for m in results.memories:
    ...     print(f"{m.hash}: {m.preview[:50]}...")
    abc12345: Implemented OAuth 2.1 authentication for...
    def67890: Refactored storage backend to support...
    >>>
    >>> # Store memory (15 tokens)
    >>> hash = store("New memory", tags=["note", "important"])
    >>> print(f"Stored: {hash}")
    Stored: abc12345
    >>>
    >>> # Health check (5 tokens)
    >>> info = health()
    >>> print(f"Backend: {info.backend}, Count: {info.count}")
    Backend: sqlite_vec, Count: 1247

Backward Compatibility:
    This API is designed to work alongside existing MCP tools without
    breaking changes. Users can gradually migrate from tool-based calls
    to code execution as needed.

Implementation:
    - Phase 1 (Current): Core operations (search, store, health)
    - Phase 2: Extended operations (search_by_tag, recall, delete, update)
    - Phase 3: Advanced features (batch operations, streaming)

For More Information:
    - Research: /docs/research/code-execution-interface-implementation.md
    - Documentation: /docs/api/code-execution-interface.md
    - Issue: https://github.com/doobidoo/mcp-memory-service/issues/206
"""

from .types import (
    CompactMemory, CompactSearchResult, CompactHealthInfo,
    CompactConsolidationResult, CompactSchedulerStatus
)
from .operations import (
    search, store, health, consolidate, scheduler_status,
    _consolidate_async, _scheduler_status_async
)
from .client import close, close_async, set_consolidator, set_scheduler

__all__ = [
    # Core operations
    'search',           # Semantic search with compact results
    'store',            # Store new memory
    'health',           # Service health check
    'close',            # Close and cleanup storage resources (sync)
    'close_async',      # Close and cleanup storage resources (async)

    # Consolidation operations
    'consolidate',      # Trigger memory consolidation
    'scheduler_status', # Get consolidation scheduler status

    # Consolidation management (internal use by HTTP server)
    'set_consolidator', # Set global consolidator instance
    'set_scheduler',    # Set global scheduler instance

    # Compact data types
    'CompactMemory',
    'CompactSearchResult',
    'CompactHealthInfo',
    'CompactConsolidationResult',
    'CompactSchedulerStatus',
]

# API version for compatibility tracking
__api_version__ = "1.0.0"

# Module metadata
__doc_url__ = "https://github.com/doobidoo/mcp-memory-service/blob/main/docs/api/code-execution-interface.md"
__issue_url__ = "https://github.com/doobidoo/mcp-memory-service/issues/206"
