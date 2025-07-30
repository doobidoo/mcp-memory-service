---
name: Remote Memory Bridge Improvements
about: Track missing tools and API integration improvements for remote memory connections
title: 'Remote Memory Bridge: Missing Tools and API Integration Improvements'
labels: 'enhancement, remote-memory, api-integration, mcp-bridge'
assignees: ''
---

## Issue Summary

Based on the July 30, 2025 remote memory bridge implementation, several improvements are needed for complete feature parity between MCP server and HTTP bridge integration.

## Current Status ✅

**Working Features:**
- Remote memory connection via HTTP-to-MCP bridge ✅
- Basic memory operations (store, retrieve, search, delete, health) ✅
- Tag-based search (100% reliable) ✅
- Claude Desktop integration ✅
- Production-ready bridge with retry logic ✅

## Issues Identified 🔍

### 1. Missing MCP Tools in Bridge

**Current Bridge Tools (5 of 11):**
- ✅ `store_memory` - Store new memories
- ✅ `retrieve_memory` - Semantic search (has issues)
- ✅ `search_by_tag` - Tag-based search (reliable)
- ✅ `delete_memory` - Delete by content hash
- ✅ `check_database_health` - Basic health check

**Missing from Bridge (6+ tools):**
- ❌ `recall_memory` - Time-based natural language retrieval
- ❌ `delete_by_tag` - Bulk deletion by single tag
- ❌ `delete_by_tags` - Bulk deletion by multiple tags  
- ❌ `optimize_db` - Database optimization operations
- ❌ `debug_retrieve` - Similarity analysis for debugging
- ❌ Dashboard variants (if applicable for HTTP API)

### 2. Health Check Missing Information

**Current API Response Missing:**
- Total memory count in database
- Memory storage statistics  
- Recent operation performance metrics
- Database optimization status
- Embedding model performance data

**Expected Enhancement:**
```json
{
  "status": "healthy",
  "storage": {
    "total_memories": 1337,
    "storage_size_mb": 25.4,
    "last_optimization": "2025-07-30T12:00:00Z"
  },
  "performance": {
    "avg_query_time_ms": 1.2,
    "operations_last_hour": 45
  }
}
```

### 3. Technical Issues

#### 🔴 Semantic Search Problem
- **Issue**: Returns 0 results despite stored memories
- **Root Cause**: Embedding indexing delays/issues in SQLite-vec backend
- **Current Workaround**: Use tag-based search (100% reliable)
- **Impact**: Major feature limitation

#### 🟡 URL Construction Edge Cases  
- **Issue**: Potential double forward slash problems in URL building
- **Location**: Bridge endpoint construction logic
- **Impact**: Minor, needs review

#### 🟡 API Endpoint Gaps
- **Issue**: Some MCP operations don't have corresponding HTTP endpoints
- **Missing**: Bulk operations, time-based queries, debug tools
- **Impact**: Feature parity incomplete

## Implementation Plan 📋

### Phase 1: Server API Enhancements
- [ ] Add `/api/recall` endpoint for time-based memory retrieval
- [ ] Add `/api/memories/delete-by-tag` for bulk deletion
- [ ] Add `/api/memories/delete-by-tags` for multi-tag deletion
- [ ] Add `/api/optimize` for database optimization
- [ ] Add `/api/debug/retrieve` for similarity analysis
- [ ] Enhance `/api/health/detailed` with memory count and statistics
- [ ] **Fix semantic search embedding indexing issues** (critical)

### Phase 2: Bridge Updates
- [ ] Add support for all missing MCP tools in `http-mcp-bridge-robust.js`
- [ ] Implement proper error handling for API limitations
- [ ] Review and fix URL construction edge cases
- [ ] Add comprehensive logging for debugging
- [ ] Add input validation for new tool parameters

### Phase 3: Integration Testing
- [ ] Test all MCP tools through HTTP bridge
- [ ] Verify complete feature parity between direct MCP and HTTP bridge
- [ ] Performance testing and optimization
- [ ] Edge case handling verification
- [ ] Semantic search fix validation

### Phase 4: Documentation Updates
- [ ] Update API documentation with complete endpoint list
- [ ] Document known limitations and current workarounds
- [ ] Add troubleshooting guides for common issues
- [ ] Update setup documentation with new features

## Technical Details 🔧

### Files Affected
- `examples/http-mcp-bridge-robust.js` - Bridge implementation
- `src/mcp_memory_service/web/api/` - Server API endpoints
- `src/mcp_memory_service/server.py` - MCP server implementation
- `docs/guides/remote-memory-setup.md` - Documentation

### API Endpoint Analysis
Current server has these MCP tools that need HTTP endpoints:
```python
# From server.py analysis
"recall_memory"           # ❌ Missing HTTP endpoint
"delete_by_tag"          # ❌ Missing HTTP endpoint  
"delete_by_tags"         # ❌ Missing HTTP endpoint
"optimize_db"            # ❌ Missing HTTP endpoint
"debug_retrieve"         # ❌ Missing HTTP endpoint
```

### Bridge Tool Schema Examples
```javascript
// Example missing tool schema for bridge
{
    name: "recall_memory",
    description: "Retrieve memories using natural language time expressions.",
    inputSchema: {
        type: "object",
        properties: {
            query: { type: "string", description: "Time-based query" },
            n_results: { type: "integer", default: 5 }
        },
        required: ["query"]
    }
}
```

## Priority and Labels

**Priority:** High 🔴
**Impact:** Affects completeness and reliability of remote memory connections
**Target Version:** v3.2.0
**Branch:** `feature/remote-memory-bridge`

## Acceptance Criteria

### ✅ Definition of Done
- [ ] All MCP server tools have corresponding HTTP endpoints
- [ ] Bridge supports all available MCP tools (11+ tools)
- [ ] Health endpoint includes comprehensive statistics
- [ ] **Semantic search works reliably** (critical fix)
- [ ] Complete feature parity between MCP and HTTP access
- [ ] All integration tests pass
- [ ] Documentation updated with new features
- [ ] Performance meets expectations (< 2s response times)

### 🧪 Testing Requirements
- [ ] Unit tests for all new endpoints
- [ ] Integration tests for bridge tool coverage
- [ ] Performance benchmarks for semantic search fix
- [ ] End-to-end testing with Claude Desktop
- [ ] Error handling validation

## Related Links

- Remote Memory Setup Guide: `docs/guides/remote-memory-setup.md`
- Bridge Implementation: `examples/http-mcp-bridge-robust.js`
- Original MCP Server: `src/mcp_memory_service/server.py`

---

**Created:** July 30, 2025  
**Branch:** `feature/remote-memory-bridge`  
**Milestone:** v3.2.0 - Complete Remote Memory Integration