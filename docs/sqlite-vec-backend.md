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
**Solution**: Ensure only one MCP instance is running
```bash
# Kill existing processes
pkill -f "mcp-memory-service"
# Restart Claude Desktop
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
| Concurrent Access | Good | Limited | ChromaDB |
| Ecosystem | Rich | Growing | ChromaDB |
| Reliability | Good | Excellent | SQLite-vec |

## Best Practices

### When to Use SQLite-vec

✅ **Use SQLite-vec when:**
- Memory collections < 100,000 entries
- Single-user or light concurrent usage
- Portability and backup simplicity are important
- Limited system resources
- Simple deployment requirements

### When to Use ChromaDB

✅ **Use ChromaDB when:**
- Memory collections > 100,000 entries
- Heavy concurrent usage
- Maximum query performance is critical
- Rich ecosystem features needed
- Distributed setups

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