# SQLite-vec Backend Guide

## Overview

The MCP Memory Service now supports SQLite-vec as an alternative storage backend. SQLite-vec provides a lightweight, high-performance vector database solution that offers several advantages over ChromaDB:

- **Lightweight**: Single file database with no external dependencies
- **Fast**: Optimized vector operations with efficient indexing
- **Portable**: Easy to backup, copy, and share memory databases
- **Reliable**: Built on SQLite's proven reliability and ACID compliance
- **Memory Efficient**: Lower memory footprint for smaller memory collections

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

## Migration from ChromaDB

### Automatic Migration

Use the provided migration script for easy migration:

```bash
# Simple migration with default paths
python migrate_to_sqlite_vec.py

# Custom migration
python scripts/migrate_storage.py \
  --from chroma \
  --to sqlite_vec \
  --source-path /path/to/chroma_db \
  --target-path /path/to/sqlite_vec.db \
  --backup
```

### Manual Migration Steps

1. **Stop the MCP Memory Service**
   ```bash
   # Stop Claude Desktop or any running instances
   ```

2. **Create a backup** (recommended)
   ```bash
   python scripts/migrate_storage.py \
     --from chroma \
     --to sqlite_vec \
     --source-path ~/.local/share/mcp-memory/chroma_db \
     --target-path ~/.local/share/mcp-memory/sqlite_vec.db \
     --backup \
     --backup-path memory_backup.json
   ```

3. **Set environment variables**
   ```bash
   export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec
   ```

4. **Restart Claude Desktop**

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

| Collection Size | ChromaDB RAM | SQLite-vec RAM | Difference |
|----------------|--------------|----------------|------------|
| 1,000 memories | ~200 MB | ~50 MB | -75% |
| 10,000 memories | ~800 MB | ~200 MB | -75% |
| 100,000 memories | ~4 GB | ~1 GB | -75% |

### Query Performance

- **Semantic Search**: Similar performance to ChromaDB for most use cases
- **Tag Search**: Faster due to SQL indexing
- **Metadata Queries**: Significantly faster with SQL WHERE clauses
- **Startup Time**: 2-3x faster initialization

### Storage Characteristics

- **Database File**: Single `.db` file (easy backup/restore)
- **Disk Usage**: ~30% smaller than ChromaDB for same data
- **Concurrent Access**: SQLite-level locking (single writer, multiple readers)

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

SQLite-vec supports advanced multi-client access through **two complementary approaches**:

1. **Phase 1: WAL Mode** - Direct SQLite access with Write-Ahead Logging
2. **Phase 2: HTTP Coordination** - Automatic HTTP server coordination for seamless multi-client access

#### Phase 1: WAL Mode (Default)

The backend automatically enables WAL mode with these default settings:
- **WAL Mode**: Enables multiple readers + single writer
- **Busy Timeout**: 5 seconds (prevents immediate lock errors)  
- **Synchronous**: NORMAL (balanced performance/safety)

#### Phase 2: HTTP Server Auto-Detection (Advanced)

The system automatically detects the optimal coordination mode:

**Auto-Detection Modes:**
- **`http_client`**: Existing HTTP server detected → Connect as client
- **`http_server`**: No server found, port available → Start HTTP server
- **`direct`**: Port in use by other service → Fall back to WAL mode

**Coordination Flow:**
1. Check if MCP Memory Service HTTP server is running
2. If found → Use HTTP client to connect to existing server
3. If not found and port available → Auto-start HTTP server (optional)
4. If port busy → Fall back to direct SQLite with WAL mode

#### Custom SQLite Pragmas

You can customize SQLite behavior using environment variables:

```bash
# Custom pragma configuration
export MCP_MEMORY_SQLITE_PRAGMAS="busy_timeout=10000,cache_size=20000,mmap_size=268435456"

# Example configurations:
# High concurrency setup
export MCP_MEMORY_SQLITE_PRAGMAS="busy_timeout=30000,wal_autocheckpoint=1000"

# Performance optimized
export MCP_MEMORY_SQLITE_PRAGMAS="synchronous=OFF,temp_store=MEMORY,cache_size=50000"

# Conservative/safe mode
export MCP_MEMORY_SQLITE_PRAGMAS="synchronous=FULL,busy_timeout=60000"
```

#### HTTP Coordination Configuration

Enable automatic HTTP server coordination for optimal multi-client access:

```bash
# Enable HTTP server auto-start
export MCP_HTTP_ENABLED=true

# Configure HTTP server settings (optional)
export MCP_HTTP_PORT=8000
export MCP_HTTP_HOST=localhost

# Combine with SQLite-vec backend
export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec
```

**Coordination Modes Explained:**

1. **Automatic Mode (Recommended)**
   ```bash
   # No configuration needed - auto-detects best mode
   export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec
   ```

2. **Forced HTTP Client Mode**
   ```bash
   # Always connect to existing server (fails if none running)
   export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec
   export MCP_HTTP_ENABLED=false
   # Requires running: python scripts/run_http_server.py
   ```

3. **Direct WAL Mode Only**
   ```bash
   # Disable HTTP coordination entirely
   export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec
   export MCP_HTTP_ENABLED=false
   export MCP_HTTP_ENABLED=false
   ```

#### Multi-Client Claude Desktop Configuration

**Option 1: Automatic Coordination (Recommended)**
```json
{
  "mcpServers": {
    "memory": {
      "command": "uv",
      "args": ["--directory", "/path/to/mcp-memory-service", "run", "memory"],
      "env": {
        "MCP_MEMORY_STORAGE_BACKEND": "sqlite_vec",
        "MCP_HTTP_ENABLED": "true"
      }
    }
  }
}
```

**Option 2: Manual HTTP Server + Client Mode**
```json
{
  "mcpServers": {
    "memory": {
      "command": "uv", 
      "args": ["--directory", "/path/to/mcp-memory-service", "run", "memory"],
      "env": {
        "MCP_MEMORY_STORAGE_BACKEND": "sqlite_vec",
        "MCP_HTTP_ENABLED": "false"
      }
    }
  }
}
```
*Note: Requires manually running `python scripts/run_http_server.py` first*

**Option 3: WAL Mode Only (Simple)**
```json
{
  "mcpServers": {
    "memory": {
      "command": "uv",
      "args": ["--directory", "/path/to/mcp-memory-service", "run", "memory"],
      "env": {
        "MCP_MEMORY_STORAGE_BACKEND": "sqlite_vec",
        "MCP_MEMORY_SQLITE_PRAGMAS": "busy_timeout=10000"
      }
    }
  }
}
```

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
**Solutions**: 

**For Single Client Issues:**
```bash
# Kill existing processes
pkill -f "mcp-memory-service"
# Restart Claude Desktop
```

**For Multi-Client Setup (Claude Desktop + Claude Code):**
```bash
# WAL mode should handle this automatically, but if issues persist:

# 1. Increase busy timeout
export MCP_MEMORY_SQLITE_PRAGMAS="busy_timeout=30000"

# 2. Check for stale lock files
ls -la /path/to/your/database-wal
ls -la /path/to/your/database-shm

# 3. If stale locks exist (no active processes), remove them
rm /path/to/your/database-wal
rm /path/to/your/database-shm

# 4. Restart all MCP clients
```

**Prevention Tips:**
- Always use WAL mode (enabled by default)
- Configure appropriate busy timeouts for your use case
- Ensure proper shutdown of MCP clients
- Use connection retry logic (built-in)

#### 5. HTTP Coordination Issues
```
Failed to initialize HTTP client storage: Connection refused
```
**Solutions:**

**Auto-Detection Problems:**
```bash
# Check if HTTP server auto-start is working
export LOG_LEVEL=DEBUG
export MCP_HTTP_ENABLED=true

# Check coordination mode detection
python -c "
import asyncio
from src.mcp_memory_service.utils.port_detection import detect_server_coordination_mode
print(asyncio.run(detect_server_coordination_mode()))
"
```

**Manual HTTP Server Setup:**
```bash
# Start HTTP server manually in separate terminal
python scripts/run_http_server.py

# Then start MCP clients (they'll auto-detect the running server)
```

**Port Conflicts:**
```bash
# Check what's using the port
netstat -an | grep :8000  # Linux/macOS
netstat -an | findstr :8000  # Windows

# Use different port
export MCP_HTTP_PORT=8001
```

**Fallback to WAL Mode:**
```bash
# Force WAL mode if HTTP coordination fails
export MCP_HTTP_ENABLED=false
export MCP_HTTP_ENABLED=false
```

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

## Comparison: ChromaDB vs SQLite-vec

| Feature | ChromaDB | SQLite-vec | Winner |
|---------|----------|------------|--------|
| Setup Complexity | Medium | Low | SQLite-vec |
| Memory Usage | High | Low | SQLite-vec |
| Query Performance | Excellent | Very Good | ChromaDB |
| Portability | Poor | Excellent | SQLite-vec |
| Backup/Restore | Complex | Simple | SQLite-vec |
| Concurrent Access | Good | Excellent (HTTP + WAL) | SQLite-vec |
| Multi-Client Support | Good | Excellent (HTTP + WAL) | SQLite-vec |
| Ecosystem | Rich | Growing | ChromaDB |
| Reliability | Good | Excellent | SQLite-vec |

## Best Practices

### When to Use SQLite-vec

✅ **Use SQLite-vec when:**
- Memory collections < 100,000 entries
- Multi-client access needed (Claude Desktop + Claude Code + others)
- Seamless setup and coordination required (auto-detection)
- Portability and backup simplicity are important
- Limited system resources
- Simple deployment requirements
- Want both HTTP and direct access capabilities

### When to Use ChromaDB

✅ **Use ChromaDB when:**
- Memory collections > 100,000 entries
- Heavy concurrent usage
- Maximum query performance is critical
- Rich ecosystem features needed
- Distributed setups

### Multi-Client Coordination Tips

1. **Automatic Mode (Recommended)**
   ```bash
   # Let the system choose the best coordination method
   export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec
   export MCP_HTTP_ENABLED=true
   ```

2. **Monitoring Coordination Mode**
   ```bash
   # Check which mode is being used
   export LOG_LEVEL=INFO
   # Look for "Detected coordination mode: ..." in logs
   ```

3. **HTTP Server Management**
   ```bash
   # Manual server control
   python scripts/run_http_server.py  # Start manually
   
   # Check server health
   curl http://localhost:8000/health
   ```

4. **Fallback Strategy**
   ```bash
   # If HTTP coordination fails, system falls back to WAL mode
   # No manual intervention needed - fully automatic
   ```

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

The sqlite-vec backend implements the same `MemoryStorage` interface as ChromaDB:

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
3. Open an issue on the [MCP Memory Service repository](https://github.com/user/mcp-memory-service/issues)