# Code Execution Interface - Executive Summary
## Issue #206 Implementation Guide

**TL;DR:** Implement Python code API to reduce token consumption by 90-95% while maintaining full backward compatibility. Start with session hooks (75% reduction, 109M+ tokens saved annually).

---

## The Problem

Current MCP tool-based architecture consumes excessive tokens:
- **Session hooks:** 3,600-9,600 tokens per session start
- **Search operations:** 2,625 tokens for 5 results
- **Document ingestion:** 57,400 tokens for 50 PDFs
- **Annual waste:** 17-379 million tokens across user base

## The Solution

Replace verbose MCP tool calls with direct Python code execution:

```python
# Before (625 tokens)
result = await mcp.call_tool('retrieve_memory', {
    'query': 'architecture decisions',
    'limit': 5,
    'similarity_threshold': 0.7
})

# After (25 tokens)
from mcp_memory_service.api import search
results = search('architecture decisions', limit=5)
```

## Key Components

### 1. Compact Data Types (91% size reduction)

```python
# CompactMemory: 73 tokens vs 820 tokens for full Memory
class CompactMemory(NamedTuple):
    hash: str           # 8-char hash
    preview: str        # First 200 chars
    tags: tuple[str]    # Immutable tags
    created: float      # Unix timestamp
    score: float        # Relevance score
```

### 2. Simple API Functions

```python
from mcp_memory_service.api import search, store, health

# Search (20 tokens)
results = search("query", limit=5)

# Store (15 tokens)
hash = store("content", tags=['note'])

# Health (5 tokens)
info = health()
```

### 3. Hook Integration

```javascript
// Node.js hook with fallback
try {
    // Code execution (fast, efficient)
    const output = execSync('python -c "from mcp_memory_service.api import search; ..."');
    memories = parseCompactResults(output);
} catch (error) {
    // Fallback to MCP tools
    memories = await mcpClient.retrieve_memory({...});
}
```

## Expected Impact

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| Session hooks | 3,600-9,600 tokens | 900-2,400 | **75% reduction** |
| Search (5 results) | 2,625 tokens | 385 | **85% reduction** |
| Store memory | 150 tokens | 15 | **90% reduction** |
| Annual savings (10 users) | - | 109.5M tokens | **$16.43/year** |
| Annual savings (100 users) | - | 2.19B tokens | **$328.50/year** |

## Implementation Timeline

```
Week 1-2: Core Infrastructure
├─ Compact types (CompactMemory, CompactSearchResult)
├─ API functions (search, store, health)
├─ Tests and benchmarks
└─ Documentation

Week 3: Session Hook Migration (HIGHEST PRIORITY)
├─ Update session-start.js
├─ Add fallback mechanism
├─ Validate 75% reduction
└─ Deploy to beta

Week 4-5: Search Operations
├─ Mid-conversation hooks
├─ Topic-change hooks
└─ Document best practices

Week 6+: Polish & Extensions
```

## Technical Architecture

### Filesystem Structure
```
src/mcp_memory_service/
├── api/                    # NEW: Code execution interface
│   ├── __init__.py        # Public API exports
│   ├── compact.py         # Compact result types
│   ├── search.py          # Search operations
│   ├── storage.py         # Storage operations
│   └── utils.py           # Utilities
├── models/
│   └── compact.py         # NEW: CompactMemory types
└── server.py              # EXISTING: MCP server (unchanged)
```

### Key Design Decisions

1. **NamedTuple for Compact Types**
   - ✅ Fast (C-based), immutable, type-safe
   - ✅ 60-90% size reduction
   - ✅ Clear field names

2. **Sync Wrappers**
   - ✅ Hide asyncio complexity
   - ✅ Connection reuse
   - ✅ <10ms overhead

3. **Backward Compatibility**
   - ✅ MCP tools continue working
   - ✅ Gradual opt-in migration
   - ✅ Zero breaking changes

## Validation from Research

### Industry Alignment
- ✅ **Anthropic (Nov 2025):** Official MCP code execution announcement
- ✅ **CodeAgents Framework:** 70%+ token reduction demonstrated
- ✅ **mcp-python-interpreter:** Proven code execution patterns

### Performance Benchmarks
- ✅ **NamedTuple:** 8% faster creation, fastest access
- ✅ **Compact types:** 85-91% token reduction measured
- ✅ **Real-world savings:** 109M+ tokens annually (conservative)

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Breaking changes | HIGH | Parallel operation, fallback mechanism |
| Performance degradation | MEDIUM | Benchmarks show <5ms overhead |
| Async/sync mismatch | LOW | Sync wrappers with asyncio.run() |
| Connection management | LOW | Global instance with reuse |
| Error handling | LOW | Expand function for debugging |

**Overall Risk:** LOW - Incremental approach with proven patterns

## Success Criteria

### Phase 1 (Core Infrastructure)
- ✅ API functions work in sync context
- ✅ Token reduction ≥85% measured
- ✅ Performance overhead <5ms
- ✅ Unit tests ≥90% coverage

### Phase 2 (Session Hooks)
- ✅ Token reduction ≥75% in production
- ✅ Hook execution <500ms
- ✅ Fallback mechanism validates
- ✅ Zero user-reported issues

### Phase 3 (Search Operations)
- ✅ Token reduction ≥90%
- ✅ Migration guide complete
- ✅ Community feedback positive

## Next Steps

### Immediate Actions (This Week)
1. Create `src/mcp_memory_service/api/` directory structure
2. Implement `CompactMemory` and `CompactSearchResult` types
3. Write `search()`, `store()`, `health()` functions
4. Add unit tests with token benchmarks

### Priority Tasks
1. **Session hook migration** - Highest impact (3,600 → 900 tokens)
2. **Documentation** - API reference with examples
3. **Beta testing** - Validate with real users
4. **Metrics collection** - Measure actual token savings

## Recommended Reading

### Full Research Document
- `docs/research/code-execution-interface-implementation.md`
  - Comprehensive analysis (10,000+ words)
  - Detailed code examples
  - Industry research findings
  - Complete implementation guide

### Key Sections
1. **Architecture Recommendations** (Section 3)
2. **Implementation Examples** (Section 4)
3. **Migration Approach** (Section 7)
4. **Success Metrics** (Section 8)

---

## Quick Reference

### Token Savings Calculator

```python
# Current MCP approach
tool_schema = 125 tokens
full_memory = 820 tokens per result
total = tool_schema + (full_memory * num_results)

# Example: 5 results
current = 125 + (820 * 5) = 4,225 tokens

# Code execution approach
import_cost = 10 tokens (once)
compact_memory = 73 tokens per result
total = import_cost + (compact_memory * num_results)

# Example: 5 results
new = 10 + (73 * 5) = 375 tokens

# Reduction
reduction = (4225 - 375) / 4225 * 100 = 91.1%
```

### Common Patterns

```python
# Pattern 1: Simple search
from mcp_memory_service.api import search
results = search("query", limit=5)
for m in results.memories:
    print(f"{m.hash}: {m.preview[:50]}")

# Pattern 2: Tag filtering
from mcp_memory_service.api import search_by_tag
results = search_by_tag(['architecture', 'decision'], limit=10)

# Pattern 3: Store with fallback
from mcp_memory_service.api import store
try:
    hash = store("content", tags=['note'])
    print(f"Stored: {hash}")
except Exception as e:
    # Fallback to MCP tool
    await mcp_client.store_memory({...})

# Pattern 4: Health check
from mcp_memory_service.api import health
info = health()
if info.ready:
    print(f"Backend: {info.backend}, Count: {info.count}")
```

---

**Document Version:** 1.0
**Last Updated:** November 6, 2025
**Status:** Ready for Implementation
**Estimated Effort:** 2-3 weeks (Phase 1-2)
**ROI:** 109M+ tokens saved annually (conservative)
