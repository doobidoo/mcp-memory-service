# Code Execution Interface API Documentation

## Overview

The Code Execution Interface provides a token-efficient Python API for direct memory operations, achieving 85-95% token reduction compared to MCP tool calls.

**Version:** 1.0.0
**Status:** Phase 1 (Core Operations)
**Issue:** [#206](https://github.com/doobidoo/mcp-memory-service/issues/206)

## Token Efficiency

| Operation | MCP Tools | Code Execution | Reduction |
|-----------|-----------|----------------|-----------|
| search(5 results) | ~2,625 tokens | ~385 tokens | **85%** |
| store() | ~150 tokens | ~15 tokens | **90%** |
| health() | ~125 tokens | ~20 tokens | **84%** |

**Annual Savings (Conservative):**
- 10 users x 5 sessions/day x 365 days x 6,000 tokens = 109.5M tokens/year
- At $0.15/1M tokens: **$16.43/year saved** per 10-user deployment

## Installation

The API is included in mcp-memory-service v8.18.2+. No additional installation required.

```bash
# Ensure you have the latest version
pip install --upgrade mcp-memory-service
```

## Quick Start

```python
from mcp_memory_service.api import search, store, health

# Store a memory (15 tokens)
hash = store("Implemented OAuth 2.1 authentication", tags=["auth", "feature"])
print(f"Stored: {hash}")  # Output: Stored: abc12345

# Search memories (385 tokens for 5 results)
results = search("authentication", limit=5)
print(f"Found {results.total} memories")
for m in results.memories:
    print(f"  {m.hash}: {m.preview[:50]}... (score: {m.score:.2f})")

# Health check (20 tokens)
info = health()
print(f"Backend: {info.backend}, Status: {info.status}, Count: {info.count}")
```

## API Reference

### Core Operations

#### search()

Semantic search with compact results.

```python
def search(
    query: str,
    limit: int = 5,
    tags: Optional[List[str]] = None
) -> CompactSearchResult:
    """
    Search memories using semantic similarity.

    Args:
        query: Search query text (natural language)
        limit: Maximum results to return (default: 5)
        tags: Optional list of tags to filter results

    Returns:
        CompactSearchResult with memories, total count, and query

    Raises:
        ValueError: If query is empty or limit is invalid
        RuntimeError: If storage backend unavailable

    Token Cost: ~25 tokens + ~73 tokens per result

    Example:
        >>> results = search("recent architecture decisions", limit=3)
        >>> for m in results.memories:
        ...     print(f"{m.hash}: {m.preview}")
    """
```

**Performance:**
- First call: ~50ms (includes storage initialization)
- Subsequent calls: ~5-10ms (connection reused)

#### store()

Store a new memory.

```python
def store(
    content: str,
    tags: Optional[Union[str, List[str]]] = None,
    memory_type: str = "note"
) -> str:
    """
    Store a new memory.

    Args:
        content: Memory content text
        tags: Single tag or list of tags (optional)
        memory_type: Memory type classification (default: "note")

    Returns:
        8-character content hash

    Raises:
        ValueError: If content is empty
        RuntimeError: If storage operation fails

    Token Cost: ~15 tokens

    Example:
        >>> hash = store(
        ...     "Fixed authentication bug",
        ...     tags=["bug", "auth"],
        ...     memory_type="fix"
        ... )
        >>> print(f"Stored: {hash}")
        Stored: abc12345
    """
```

**Performance:**
- First call: ~50ms (includes storage initialization)
- Subsequent calls: ~10-20ms (includes embedding generation)

#### health()

Service health and status check.

```python
def health() -> CompactHealthInfo:
    """
    Get service health and status.

    Returns:
        CompactHealthInfo with status, count, and backend

    Token Cost: ~20 tokens

    Example:
        >>> info = health()
        >>> if info.status == 'healthy':
        ...     print(f"{info.count} memories in {info.backend}")
    """
```

**Performance:**
- First call: ~50ms (includes storage initialization)
- Subsequent calls: ~5ms (cached stats)

### Data Types

#### CompactMemory

Minimal memory representation (91% token reduction).

```python
class CompactMemory(NamedTuple):
    hash: str           # 8-character content hash
    preview: str        # First 200 characters
    tags: tuple[str]    # Immutable tags tuple
    created: float      # Unix timestamp
    score: float        # Relevance score (0.0-1.0)
```

**Token Cost:** ~73 tokens (vs ~820 for full Memory object)

#### CompactSearchResult

Search result container.

```python
class CompactSearchResult(NamedTuple):
    memories: tuple[CompactMemory]  # Immutable results
    total: int                       # Total results count
    query: str                       # Original query

    def __repr__(self) -> str:
        return f"SearchResult(found={self.total}, shown={len(self.memories)})"
```

**Token Cost:** ~10 tokens + (73 x num_memories)

#### CompactHealthInfo

Service health information.

```python
class CompactHealthInfo(NamedTuple):
    status: str         # 'healthy' | 'degraded' | 'error'
    count: int          # Total memories
    backend: str        # 'sqlite_vec' | 'cloudflare' | 'hybrid'
```

**Token Cost:** ~20 tokens

## Usage Examples

### Basic Search

```python
from mcp_memory_service.api import search

# Simple search
results = search("authentication", limit=5)
print(f"Found {results.total} memories")

# Search with tag filter
results = search("database", limit=10, tags=["architecture"])
for m in results.memories:
    if m.score > 0.7:  # High relevance only
        print(f"{m.hash}: {m.preview}")
```

### Batch Store

```python
from mcp_memory_service.api import store

# Store multiple memories
changes = [
    "Implemented OAuth 2.1 authentication",
    "Added JWT token validation",
    "Fixed session timeout bug"
]

for change in changes:
    hash_val = store(change, tags=["changelog", "auth"])
    print(f"Stored: {hash_val}")
```

### Health Monitoring

```python
from mcp_memory_service.api import health

info = health()

if info.status != 'healthy':
    print(f"⚠️  Service degraded: {info.status}")
    print(f"Backend: {info.backend}")
    print(f"Memory count: {info.count}")
else:
    print(f"✅ Service healthy ({info.count} memories in {info.backend})")
```

### Error Handling

```python
from mcp_memory_service.api import search, store

try:
    # Store with validation
    if not content.strip():
        raise ValueError("Content cannot be empty")

    hash_val = store(content, tags=["test"])

    # Search with error handling
    results = search("query", limit=5)

    if results.total == 0:
        print("No results found")
    else:
        for m in results.memories:
            print(f"{m.hash}: {m.preview}")

except ValueError as e:
    print(f"Validation error: {e}")
except RuntimeError as e:
    print(f"Storage error: {e}")
```

## Performance Optimization

### Connection Reuse

The API automatically reuses storage connections for optimal performance:

```python
from mcp_memory_service.api import search, store

# First call: ~50ms (initialization)
store("First memory", tags=["test"])

# Subsequent calls: ~10ms (reuses connection)
store("Second memory", tags=["test"])
store("Third memory", tags=["test"])

# Search also reuses connection: ~5ms
results = search("test", limit=5)
```

### Limit Result Count

```python
# For quick checks, use small limits
results = search("query", limit=3)  # ~240 tokens

# For comprehensive results, use larger limits
results = search("query", limit=20)  # ~1,470 tokens
```

## Backward Compatibility

The Code Execution API works alongside existing MCP tools without breaking changes:

- **MCP tools continue working** - No deprecation or removal
- **Gradual migration** - Adopt code execution incrementally
- **Fallback mechanism** - Tools available if code execution fails

## Migration Guide

### From MCP Tools to Code Execution

**Before (MCP Tool):**
```javascript
// Node.js hook using MCP client
const result = await mcpClient.callTool('retrieve_memory', {
    query: 'architecture',
    limit: 5,
    similarity_threshold: 0.7
});
// Result: ~2,625 tokens
```

**After (Code Execution):**
```python
# Python code in hook
from mcp_memory_service.api import search
results = search('architecture', limit=5)
# Result: ~385 tokens (85% reduction)
```

## Troubleshooting

### Storage Initialization Errors

```python
from mcp_memory_service.api import health

info = health()
if info.status == 'error':
    print(f"Storage backend {info.backend} not available")
    # Check configuration:
    # - DATABASE_PATH set correctly
    # - Storage backend initialized
    # - Permissions on database directory
```

### Import Errors

```bash
# Ensure mcp-memory-service is installed
pip list | grep mcp-memory-service

# Verify version (requires 8.18.2+)
python -c "import mcp_memory_service; print(mcp_memory_service.__version__)"
```

### Performance Issues

```python
import time
from mcp_memory_service.api import search

# Measure performance
start = time.perf_counter()
results = search("query", limit=5)
duration_ms = (time.perf_counter() - start) * 1000

if duration_ms > 100:
    print(f"⚠️  Slow search: {duration_ms:.1f}ms (expected: <50ms)")
    # Possible causes:
    # - Cold start (first call after initialization)
    # - Large database requiring optimization
    # - Embedding model not cached
```

## Future Enhancements (Roadmap)

### Phase 2: Extended Operations
- `search_by_tag()` - Tag-based filtering
- `recall()` - Natural language time queries
- `delete()` - Delete by content hash
- `update()` - Update memory metadata

### Phase 3: Advanced Features
- `store_batch()` - Batch store operations
- `search_iter()` - Streaming search results
- Document ingestion API
- Memory consolidation triggers

## Related Documentation

- [Research Document](/docs/research/code-execution-interface-implementation.md)
- [Implementation Summary](/docs/research/code-execution-interface-summary.md)
- [Issue #206](https://github.com/doobidoo/mcp-memory-service/issues/206)
- [CLAUDE.md](/CLAUDE.md) - Project instructions

## Support

For issues, questions, or contributions:
- GitHub Issues: https://github.com/doobidoo/mcp-memory-service/issues
- Documentation: https://github.com/doobidoo/mcp-memory-service/wiki

## License

Copyright 2024 Heinrich Krupp
Licensed under the Apache License, Version 2.0
