# SQLite-vec Backend Guide

## Overview

SQLite-vec is the default storage backend for MCP Memory Service. It provides a lightweight, high-performance vector database solution:

- **Lightweight**: Single file database with no external dependencies
- **Fast**: Optimized vector operations with efficient indexing (~5 ms reads)
- **Portable**: Easy to backup, copy, and share memory databases
- **Reliable**: Built on SQLite's proven reliability and ACID compliance
- **Memory Efficient**: Low memory footprint suited to single-user deployments

## Installation

### Prerequisites

The sqlite-vec backend requires the `sqlite-vec` Python package:

```bash
# Install sqlite-vec
pip install sqlite-vec

# Or with UV (recommended)
uv add sqlite-vec
```

### Verification

You can verify sqlite-vec is available by running:

```python
try:
    import sqlite_vec
    print("✅ sqlite-vec is available")
except ImportError:
    print("❌ sqlite-vec is not installed")
```

## Configuration

### Environment Variables

To use the sqlite-vec backend, set the storage backend environment variable:

```bash
# Primary configuration
export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec

# Optional: Custom database path
export MCP_MEMORY_SQLITE_PATH=/path/to/your/memory.db
```

### Platform-Specific Setup

#### macOS (Bash/Zsh)
```bash
# Add to ~/.bashrc or ~/.zshrc
export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec
export MCP_MEMORY_SQLITE_PATH="$HOME/Library/Application Support/mcp-memory/sqlite_vec.db"
```

#### Windows (PowerShell)
```powershell
# Add to PowerShell profile
$env:MCP_MEMORY_STORAGE_BACKEND = "sqlite_vec"
$env:MCP_MEMORY_SQLITE_PATH = "$env:LOCALAPPDATA\mcp-memory\sqlite_vec.db"
```

#### Windows (Command Prompt)
```cmd
set MCP_MEMORY_STORAGE_BACKEND=sqlite_vec
set MCP_MEMORY_SQLITE_PATH=%LOCALAPPDATA%\mcp-memory\sqlite_vec.db
```

#### Linux
```bash
# Add to ~/.bashrc
export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec
export MCP_MEMORY_SQLITE_PATH="$HOME/.local/share/mcp-memory/sqlite_vec.db"
```

### Claude Desktop Configuration

Update your Claude Desktop MCP configuration:

```json
{
  "mcpServers": {
    "memory": {
      "command": "uv",
      "args": ["--directory", "/path/to/mcp-memory-service", "run", "memory"],
      "env": {
        "MCP_MEMORY_STORAGE_BACKEND": "sqlite_vec"
      }
    }
  }
}
```

## Migrating Legacy ChromaDB Data

ChromaDB was removed as a supported backend in v8.0.0. If you still have data from a ChromaDB install, see the dedicated guide: **[guides/chromadb-migration.md](guides/chromadb-migration.md)**. The script is preserved on the [`chromadb-legacy`](https://github.com/doobidoo/mcp-memory-service/tree/chromadb-legacy) branch.

After migration, set the environment variable and restart Claude Desktop:

```bash
export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec
```

### Migration Verification

After migration, verify your memories are accessible:

```bash
# Test the new backend
python scripts/verify_environment.py

# Check database statistics
python -c "
import asyncio
from src.mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage

async def check_stats():
    storage = SqliteVecMemoryStorage('path/to/your/db')
    await storage.initialize()
    stats = storage.get_stats()
    print(f'Total memories: {stats[\"total_memories\"]}')
    print(f'Database size: {stats[\"database_size_mb\"]} MB')
    storage.close()

asyncio.run(check_stats())
"
```

## Performance Characteristics

### Memory Usage

| Collection Size | SQLite-vec RAM |
|-----------------|----------------|
| 1,000 memories  | ~50 MB         |
| 10,000 memories | ~200 MB        |
| 100,000 memories| ~1 GB          |

### Query Performance

- **Semantic Search**: ~5 ms local reads for typical collections
- **Tag Search**: Fast SQL-indexed lookups
- **Metadata Queries**: Efficient SQL WHERE clauses
- **Startup Time**: Single-file database loads in <1 second

### Storage Characteristics

- **Database File**: Single `.db` file (easy backup/restore)
- **Concurrent Access**: SQLite-level locking (single writer, multiple readers) — enable WAL via `MCP_MEMORY_SQLITE_PRAGMAS=journal_mode=WAL,busy_timeout=15000`

## Advanced Configuration

### Custom Embedding Models

```python
# Initialize with custom model
storage = SqliteVecMemoryStorage(
    db_path="memory.db",
    embedding_model="all-mpnet-base-v2"  # Higher quality, slower
)
```

### Multi-Client Access Configuration

Several MCP clients (Claude Desktop, VS Code, Cursor) can share one SQLite-vec database:

- **Same machine:** every stdio MCP server opens the database directly. WAL mode is on by
  default, so multiple readers and one writer coexist; `busy_timeout` decides how long a
  writer waits instead of failing with "database is locked".
- **Several machines, or many clients:** run one HTTP server (`memory launch`) and point the
  clients at it - REST API, or the MCP-over-HTTP transport. See
  [integration/multi-client.md](integration/multi-client.md).

Earlier releases documented a third mode, "HTTP coordination", in which a stdio server would
detect a running HTTP server and proxy to it through `storage/http_client.py`. That code had not
been able to start since v7.5.0 and was removed in v11.12; there is nothing to configure for it.

#### WAL Mode (Default)

The backend enables WAL mode with these defaults:
- **WAL Mode**: multiple readers + single writer
- **Busy Timeout**: 5 seconds (prevents immediate lock errors)
- **Synchronous**: NORMAL (balanced performance/safety)

#### Custom SQLite Pragmas

You can customize SQLite behavior using environment variables:

```bash
# Recommended configuration (v8.9.0+) - For concurrent HTTP + MCP access
export MCP_MEMORY_SQLITE_PRAGMAS="busy_timeout=15000,cache_size=20000"

# Example configurations for different scenarios:
# High concurrency setup (longer timeout)
export MCP_MEMORY_SQLITE_PRAGMAS="busy_timeout=30000,cache_size=20000,wal_autocheckpoint=1000"

# Performance optimized (use with caution - trades safety for speed)
export MCP_MEMORY_SQLITE_PRAGMAS="synchronous=NORMAL,temp_store=MEMORY,cache_size=50000,busy_timeout=15000"

# Conservative/safe mode (maximum data safety)
export MCP_MEMORY_SQLITE_PRAGMAS="synchronous=FULL,busy_timeout=60000,cache_size=20000"
```

#### Claude Desktop Configuration for Several Clients

```json
{
  "mcpServers": {
    "memory": {
      "command": "uv",
      "args": ["--directory", "/path/to/mcp-memory-service", "run", "memory"],
      "env": {
        "MCP_MEMORY_STORAGE_BACKEND": "sqlite_vec",
        "MCP_MEMORY_SQLITE_PRAGMAS": "journal_mode=WAL,busy_timeout=15000,cache_size=20000"
      }
    }
  }
}
```

Each client gets the same block; they share the database file through WAL. If the HTTP
server and dashboard run on the same machine, the same pragmas keep them from blocking each
other (see the recommended value in `.env.example`).

### Database Optimization

```bash
# Optimize database periodically
python -c "
import asyncio
from src.mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage

async def optimize():
    storage = SqliteVecMemoryStorage('path/to/db')
    await storage.initialize()
    
    # Clean up duplicates
    count, msg = await storage.cleanup_duplicates()
    print(f'Cleaned up {count} duplicates')
    
    # Vacuum database
    storage.conn.execute('VACUUM')
    print('Database vacuumed')
    
    storage.close()

asyncio.run(optimize())
"
```

### Backup and Restore

```bash
# Create backup
python scripts/migrate_storage.py \
  --from sqlite_vec \
  --to sqlite_vec \
  --source-path memory.db \
  --target-path backup.db

# Or simple file copy
cp memory.db memory_backup.db

# Restore from JSON backup
python scripts/migrate_storage.py \
  --restore backup.json \
  --to sqlite_vec \
  --target-path restored_memory.db
```

## Troubleshooting

### Common Issues

#### 1. sqlite-vec Not Found
```
ImportError: No module named 'sqlite_vec'
```
**Solution**: Install sqlite-vec package
```bash
pip install sqlite-vec
# or
uv add sqlite-vec
```

#### 2. Database Lock Errors
```
sqlite3.OperationalError: database is locked
```

**✅ Fixed in v8.9.0** - Proper SQLite pragmas now automatically configured by installer

**For Single Client Issues:**
```bash
# Kill existing processes
pkill -f "mcp-memory-service"
# Restart Claude Desktop
```

**For Multi-Client Setup (Claude Desktop + Claude Code + HTTP Server):**
```bash
# v8.9.0+ Solution: Configure recommended pragma values
export MCP_MEMORY_SQLITE_PRAGMAS="busy_timeout=15000,cache_size=20000"

# Restart all services to apply changes
# Note: Installer automatically sets these for hybrid/sqlite_vec backends

# If issues persist, try longer timeout:
export MCP_MEMORY_SQLITE_PRAGMAS="busy_timeout=30000,cache_size=20000"

# Check for stale lock files (rare)
ls -la /path/to/your/database-wal
ls -la /path/to/your/database-shm

# If stale locks exist (no active processes), remove them
rm /path/to/your/database-wal
rm /path/to/your/database-shm

# 4. Restart all MCP clients
```

**Prevention Tips:**
- Always use WAL mode (enabled by default)
- Configure appropriate busy timeouts for your use case
- Ensure proper shutdown of MCP clients
- Use connection retry logic (built-in)

#### 3. Permission Errors
```
PermissionError: [Errno 13] Permission denied
```
**Solution**: Check database file permissions
```bash
# Fix permissions
chmod 644 /path/to/sqlite_vec.db
chmod 755 /path/to/directory
```

#### 4. Migration Failures
```
Migration failed: No memories found
```
**Solution**: Verify source path and initialize if needed
```bash
# Check source exists
ls -la /path/to/chroma_db
# Use absolute paths in migration
```

### Debug Mode

Enable debug logging for troubleshooting:

```bash
export LOG_LEVEL=DEBUG
export DEBUG_MODE=1
# Run your MCP client
```

### Health Checks

```python
# Check backend health
import asyncio
from src.mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage

async def health_check():
    storage = SqliteVecMemoryStorage('path/to/db')
    await storage.initialize()
    
    stats = storage.get_stats()
    print(f"Backend: {stats['backend']}")
    print(f"Total memories: {stats['total_memories']}")
    print(f"Database size: {stats['database_size_mb']} MB")
    print(f"Embedding model: {stats['embedding_model']}")
    
    storage.close()

asyncio.run(health_check())
```

## Comparison: SQLite-vec vs Cloudflare vs Hybrid

| Feature | SQLite-vec | Cloudflare | Hybrid |
|---------|------------|------------|--------|
| Read latency | ~5 ms (local) | Network-dependent | ~5 ms (local) |
| Setup complexity | Low | Medium (API tokens, D1 + Vectorize) | Medium |
| Multi-device sync | ❌ | ✅ | ✅ (background sync) |
| Offline access | ✅ | ❌ | ✅ |
| External dependencies | None | Cloudflare account | Cloudflare account |
| Recommended for | Single-user / dev | Edge / cloud-only | **Production** |

See [guides/STORAGE_BACKENDS.md](guides/STORAGE_BACKENDS.md) for the full comparison.

## Best Practices

### When to Use SQLite-vec

✅ **Use SQLite-vec when:**
- Single-user or development setup
- Multi-client access on the same machine (Claude Desktop + Claude Code)
- Offline / air-gapped environments
- Portability and backup simplicity are important
- Limited system resources

For production with multi-device sync, use the **Hybrid** backend instead (local SQLite-vec for reads + background Cloudflare sync).

### Multi-Client Tips

1. **Same machine, several MCP clients**
   ```bash
   export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec
   export MCP_MEMORY_SQLITE_PRAGMAS="journal_mode=WAL,busy_timeout=15000,cache_size=20000"
   ```
   Every client opens the same file; WAL lets them coexist, `busy_timeout` keeps a waiting
   writer from failing with "database is locked".

2. **Shared server**
   ```bash
   memory launch                          # one HTTP server owns the database
   curl -k https://localhost:8000/api/health
   ```
   Point remote clients at the REST API or the MCP-over-HTTP transport instead of the file.
   See [integration/multi-client.md](integration/multi-client.md).

### Performance Tips

1. **Regular Optimization**
   ```bash
   # Run monthly
   python scripts/optimize_sqlite_vec.py
   ```

2. **Batch Operations**
   ```python
   # Store memories in batches for better performance
   for batch in chunk_memories(all_memories, 100):
       for memory in batch:
           await storage.store(memory)
   ```

3. **Index Maintenance**
   ```sql
   -- Rebuild indexes periodically
   REINDEX;
   VACUUM;
   ```

## API Reference

The sqlite-vec backend implements the standard `BaseStorage` interface shared by all backends:

```python
# All standard operations work identically
await storage.store(memory)
results = await storage.retrieve(query, n_results=5)
memories = await storage.search_by_tag(["tag1", "tag2"])
success, msg = await storage.delete(content_hash)
success, msg = await storage.update_memory_metadata(hash, updates)
```

See the main API documentation for complete method signatures.

## Contributing

To contribute to sqlite-vec backend development:

1. Run tests: `pytest tests/test_sqlite_vec_storage.py`
2. Check performance: `python tests/performance/test_sqlite_vec_perf.py`
3. Add features following the `MemoryStorage` interface
4. Update this documentation

## Support

For sqlite-vec backend issues:

1. Check [sqlite-vec documentation](https://github.com/asg017/sqlite-vec)
2. Review this guide's troubleshooting section
3. Open an issue on the [MCP Memory Service repository](https://github.com/doobidoo/mcp-memory-service/issues)