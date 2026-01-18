# OAuth Storage Backends

MCP Memory Service v9.0.6+ supports persistent OAuth storage for production multi-worker deployments.

## Overview

Two storage backends are available:
- **Memory**: In-memory storage (default, dev/testing only)
- **SQLite**: Persistent SQLite storage (recommended for production)

## Configuration

Configure via environment variables:

```bash
# Memory backend (default)
export MCP_OAUTH_STORAGE_BACKEND=memory

# SQLite backend (production)
export MCP_OAUTH_STORAGE_BACKEND=sqlite
export MCP_OAUTH_SQLITE_PATH=./data/oauth.db
```

Or in `.env` file:
```
MCP_OAUTH_STORAGE_BACKEND=sqlite
MCP_OAUTH_SQLITE_PATH=./data/oauth.db
```

## Memory Backend

**Use Case**: Development, testing, single-worker deployments

**Characteristics**:
- ✅ Fast (no disk I/O)
- ✅ Simple (no database files)
- ❌ Not persistent (data lost on restart)
- ❌ Not shared across workers

**When to use**:
- Local development
- Testing environments
- Single-worker deployments where persistence isn't required

## SQLite Backend

**Use Case**: Production, multi-worker deployments

**Characteristics**:
- ✅ Persistent (survives server restarts)
- ✅ Multi-worker safe (WAL mode)
- ✅ Multi-instance support (shared file system)
- ✅ Security: Atomic one-time code consumption
- ✅ Performance: <10ms operations
- ✅ Automatic expired token cleanup

**When to use**:
- Production deployments
- Multi-worker configurations (uvicorn --workers N)
- When OAuth state must persist across restarts

**SQLite Features**:
- WAL (Write-Ahead Logging) mode for concurrent access
- Atomic `UPDATE WHERE consumed=0` for replay attack prevention
- Indexes on `expires_at` and `consumed` columns for fast queries
- Foreign key constraints for referential integrity

## Programmatic Usage

```python
from mcp_memory_service.web.oauth.storage import create_oauth_storage

# Memory backend
storage = create_oauth_storage("memory")

# SQLite backend
storage = create_oauth_storage("sqlite", db_path="./data/oauth.db")

# Use storage
await storage.store_client(client)
client = await storage.get_client(client_id)
```

## Migration

No migration needed. The default is memory backend for backward compatibility.

To enable SQLite persistence:
1. Set `MCP_OAUTH_STORAGE_BACKEND=sqlite`
2. Optionally set `MCP_OAUTH_SQLITE_PATH` (defaults to `<base_directory>/oauth.db`)
3. Restart server

The database will be created automatically on first use.

## Performance

Both backends meet <10ms operation targets:

| Operation | Memory | SQLite |
|-----------|--------|--------|
| Store client | ~0.1ms | ~2ms |
| Get client | ~0.1ms | ~2ms |
| Store token | ~0.1ms | ~3ms |
| Get token | ~0.1ms | ~3ms |

*Note: SQLite performance measured with WAL mode on SSD*

## Security

**One-Time Authorization Code Consumption**:
Both backends implement atomic code consumption to prevent replay attacks:

```sql
-- SQLite uses atomic UPDATE WHERE
UPDATE oauth_authorization_codes
SET consumed = 1
WHERE code = ? AND consumed = 0
```

If two requests attempt to use the same code simultaneously, only one succeeds.

## Troubleshooting

**Issue**: SQLite database locked
**Solution**: Ensure WAL mode is enabled (automatic in v9.0.6+)

**Issue**: OAuth state lost after restart
**Solution**: Verify `MCP_OAUTH_STORAGE_BACKEND=sqlite` is set

**Issue**: Slow token operations
**Solution**: Check that SQLite database is on fast storage (SSD)

## Related Issues
- #360: OAuth Persistent Storage Backend
- #361: uvx HTTP Test Failures (fixed as part of v9.0.6)
