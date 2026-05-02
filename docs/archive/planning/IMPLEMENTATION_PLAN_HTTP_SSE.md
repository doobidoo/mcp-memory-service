# HTTP/SSE + SQLite-vec Implementation Plan

**Date**: 2025-07-25  
**Status**: Extracted from previous planning session  
**Context**: Issue #57 - Add HTTP/SSE interface to MCP Memory Service

## Executive Summary

Implement HTTP REST API and Server-Sent Events (SSE) interface for the MCP Memory Service using the existing sqlite-vec backend instead of ChromaDB. This creates a lightweight, edge-ready solution that maintains all existing MCP functionality while adding modern web capabilities.

## Key Architectural Decision

**Combine HTTP/SSE with sqlite-vec backend** instead of ChromaDB for:
- Simplified deployment (single file database)
- Better performance (10x faster operations)
- Edge-ready deployment (Cloudflare Workers, Vercel)
- No external dependencies
- Instant SSE updates via SQLite triggers

## Implementation Phases

### Phase 1: Foundation (Week 1)
- ✅ Create feature branch from sqlite-vec-backend
- ✅ Create PROJECT_STATUS.md tracking file
- [ ] Validate sqlite-vec functionality
- [ ] Add FastAPI dependencies
- [ ] Set up development environment

### Phase 2: HTTP Implementation (Week 2)
- [ ] Create web server structure
- [ ] Implement health check endpoint
- [ ] Add memory CRUD endpoints
- [ ] Add search endpoints
- [ ] OpenAPI documentation

### Phase 3: SSE Implementation (Week 3)
- [ ] Design SSE event architecture
- [ ] Implement SQLite triggers
- [ ] Create /events endpoint
- [ ] Connection management
- [ ] Real-time update testing

### Phase 4: Dashboard (Week 4)
- [ ] Minimal UI design (vanilla JS)
- [ ] Memory visualization
- [ ] SSE connection handling
- [ ] Search interface
- [ ] Responsive design

## Technical Architecture

### Directory Structure
```
src/mcp_memory_service/
├── web/
│   ├── __init__.py
│   ├── app.py          # FastAPI application
│   ├── sse.py          # SSE event handling
│   ├── api/
│   │   ├── __init__.py
│   │   ├── memories.py # Memory CRUD
│   │   ├── search.py   # Search operations
│   │   └── health.py   # Health monitoring
│   └── static/
│       ├── index.html  # Dashboard
│       ├── app.js      # Frontend JS
│       └── style.css   # Styling
├── storage/
│   ├── sqlite_vec.py   # Existing
│   └── sqlite_sse.py   # New: SSE triggers
```

### Server Modes
1. **MCP Mode**: Original stdio protocol (unchanged)
2. **HTTP Mode**: FastAPI server with SSE
3. **Hybrid Mode**: Both protocols simultaneously

### SSE Events
- `memory_stored`: New memory added
- `memory_deleted`: Memory removed
- `search_completed`: Search results ready
- `backup_status`: Backup progress
- `health_update`: System status changes

### API Endpoints
- `GET /api/health` - Health check
- `GET /api/memories` - List memories (paginated)
- `POST /api/memories` - Store new memory
- `GET /api/memories/{id}` - Get specific memory
- `DELETE /api/memories/{id}` - Delete memory
- `POST /api/search` - Semantic search
- `POST /api/search/by-tag` - Tag search
- `POST /api/search/by-time` - Time-based recall
- `GET /events` - SSE endpoint

## Dependencies to Add
```
fastapi>=0.115.0
uvicorn>=0.30.0
python-multipart>=0.0.9
sse-starlette>=2.1.0
aiofiles>=23.2.1
```

## Configuration
```python
# New environment variables
HTTP_ENABLED = 'MCP_HTTP_ENABLED'
HTTP_PORT = 'MCP_HTTP_PORT' (default: 8000)
HTTP_HOST = 'MCP_HTTP_HOST' (default: 0.0.0.0)
CORS_ORIGINS = 'MCP_CORS_ORIGINS'
SSE_HEARTBEAT_INTERVAL = 'MCP_SSE_HEARTBEAT' (default: 30s)
API_KEY = 'MCP_API_KEY' (optional auth)
```

## Performance Targets
- Memory storage: <50ms (vs ChromaDB ~500ms)
- Search response: <100ms for 1M memories
- SSE latency: <10ms from write to event
- Startup time: <1s (vs ChromaDB 5-10s)

## Testing Strategy
- Unit tests for all HTTP endpoints
- Integration tests for SSE connections
- Performance benchmarks vs ChromaDB
- Browser compatibility testing
- Edge deployment validation

## Security Considerations
- Optional API key authentication
- CORS configuration
- Rate limiting
- Input validation
- SSL/TLS documentation

## Migration Path
- Existing MCP users: No changes required
- ChromaDB users: Migration script provided
- New users: SQLite-vec as default for HTTP mode

## Benefits
- **Simplicity**: Single file database, no external services
- **Performance**: Orders of magnitude faster
- **Portability**: Runs anywhere Python runs
- **Reliability**: SQLite's proven track record
- **Modern**: HTTP/SSE/REST for web integration
- **Efficient**: Minimal resource usage
- **Edge-ready**: Deploy to CDN edge locations

## Future Possibilities
- Distributed SQLite with Litestream replication
- Cloudflare Workers deployment with D1
- Offline-first PWA with WASM SQLite
- Federation between multiple instances

## Success Metrics
- HTTP endpoints respond within performance targets
- SSE connections maintain real-time updates <10ms
- Dashboard provides intuitive memory management
- Documentation enables easy deployment
- Migration from ChromaDB is seamless
- Edge deployment works on major platforms

---

This plan represents a significant architectural improvement while maintaining full backward compatibility with existing MCP usage patterns.