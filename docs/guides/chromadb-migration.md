# ChromaDB Migration Guide

> **ChromaDB backend was removed in v8.0.0**. This guide helps you migrate to modern storage backends.

## Quick Migration Path

### Option 1: Hybrid Backend (Recommended)

Best choice for most users - combines fast local storage with cloud synchronization.

```bash
# 1. Convert ChromaDB to SQLite-vec (on the chromadb-legacy-final tag)
git checkout chromadb-legacy-final
python scripts/migration/migrate_chroma_to_sqlite.py \
    --chroma-path /path/to/chroma_db --sqlite-path ~/chroma_export.db

# 2. Switch back to main and configure Hybrid backend
git checkout main
export MCP_MEMORY_STORAGE_BACKEND=hybrid
export MCP_MEMORY_SQLITE_PATH=~/chroma_export.db

# 3. Configure Cloudflare credentials
export CLOUDFLARE_API_TOKEN="your-token"
export CLOUDFLARE_ACCOUNT_ID="your-account"
export CLOUDFLARE_D1_DATABASE_ID="your-d1-id"
export CLOUDFLARE_VECTORIZE_INDEX="mcp-memory-index"

# 4. Export from SQLite-vec to JSON, then import to Cloudflare
python scripts/migration/migrate_to_cloudflare.py export \
    --source sqlite_vec --source-path ~/chroma_export.db --output ~/chromadb_backup.json
python scripts/migration/migrate_to_cloudflare.py import --input ~/chromadb_backup.json

# 5. Verify configuration
python scripts/validation/validate_configuration_complete.py
```

### Option 2: SQLite-vec (Local Only)

For single-device use without cloud synchronization.

```bash
# 1. Convert ChromaDB to SQLite-vec (on the chromadb-legacy-final tag)
git checkout chromadb-legacy-final
python scripts/migration/migrate_chroma_to_sqlite.py \
    --chroma-path /path/to/chroma_db --sqlite-path ~/chroma_export.db

# 2. Switch back to main and configure SQLite-vec backend
git checkout main
export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec
export MCP_MEMORY_SQLITE_PATH=~/chroma_export.db
```

### Option 3: Cloudflare (Cloud Only)

For pure cloud storage without local database.

```bash
# 1. Convert ChromaDB to SQLite-vec (on the chromadb-legacy-final tag)
git checkout chromadb-legacy-final
python scripts/migration/migrate_chroma_to_sqlite.py \
    --chroma-path /path/to/chroma_db --sqlite-path ~/chroma_export.db

# 2. Switch back to main and configure Cloudflare backend
git checkout main
export MCP_MEMORY_STORAGE_BACKEND=cloudflare

# 3. Configure Cloudflare credentials
export CLOUDFLARE_API_TOKEN="your-token"
export CLOUDFLARE_ACCOUNT_ID="your-account"
export CLOUDFLARE_D1_DATABASE_ID="your-d1-id"
export CLOUDFLARE_VECTORIZE_INDEX="mcp-memory-index"

# 4. Export from SQLite-vec and import into Cloudflare
python scripts/migration/migrate_to_cloudflare.py export \
    --source sqlite_vec --source-path ~/chroma_export.db --output ~/chromadb_backup.json
python scripts/migration/migrate_to_cloudflare.py import --input ~/chromadb_backup.json
```

## Backend Comparison

| Feature | Hybrid ⭐ | SQLite-vec | Cloudflare | ChromaDB (Removed) |
|---------|----------|------------|------------|-------------------|
| **Performance** | 5ms (local) | 5ms | Network | 15ms |
| **Multi-device** | ✅ Yes | ❌ No | ✅ Yes | ❌ No |
| **Offline support** | ✅ Yes | ✅ Yes | ❌ No | ✅ Yes |
| **Cloud backup** | ✅ Auto | ❌ No | ✅ Native | ❌ No |
| **Dependencies** | Light | Minimal | None | Heavy (~2GB) |
| **Setup complexity** | Medium | Easy | Medium | Easy |
| **Status** | **Recommended** | Supported | Supported | **REMOVED** |

## Migration Script Details

### Using the Legacy Migration Script

The ChromaDB migration script is preserved in the `chromadb-legacy-final` tag:

```bash
# From chromadb-legacy-final tag
git checkout chromadb-legacy-final
python scripts/migration/migrate_chroma_to_sqlite.py [OPTIONS]

Options:
  --chroma-path PATH  Path to ChromaDB data directory
  --sqlite-path PATH  Path for SQLite-vec database
  --batch-size NUM    Batch size for migration (default: 50)
  --verbose           Enable verbose logging
```

### Manual Migration Steps

If you prefer manual control:

1. **Export from ChromaDB**:
   ```bash
   git checkout chromadb-legacy-final
   python -c "
   from mcp_memory_service.storage.chroma import ChromaMemoryStorage
   import json
   storage = ChromaMemoryStorage(path='./chroma_db')
   memories = storage.get_all_memories()
   with open('export.json', 'w') as f:
       json.dump([m.to_dict() for m in memories], f)
   "
   ```

2. **Import to new backend**:
   ```bash
   git checkout main
   python -c "
   from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
   import json
   storage = SqliteVecMemoryStorage(db_path='./memory.db')
   await storage.initialize()
   with open('export.json') as f:
       memories = json.load(f)
   for mem in memories:
       await storage.store(Memory.from_dict(mem))
   "
   ```

## Data Validation

After migration, verify your data:

```bash
# Check memory count
python -c "
from mcp_memory_service.storage.factory import create_storage_instance
storage = await create_storage_instance('./memory.db')
count = len(await storage.get_all_memories())
print(f'Migrated {count} memories')
"

# Validate SQLite-vec database integrity
python scripts/validation/validate_migration.py ./memory.db
```

## Troubleshooting

### Issue: Migration script not found

**Solution**: The migration script is only available on the `chromadb-legacy-final` tag:
```bash
git checkout chromadb-legacy-final
python scripts/migration/migrate_chroma_to_sqlite.py
```

### Issue: Import errors for ChromaMemoryStorage

**Solution**: You must be on the `chromadb-legacy-final` tag to access ChromaDB code:
```bash
git checkout chromadb-legacy-final  # ChromaDB code available
git checkout main                   # ChromaDB removed (v8.0.0+)
```

### Issue: "ChromaDB not installed" error

**Solution**: Install chromadb on the legacy tag:
```bash
git checkout chromadb-legacy-final
pip install chromadb>=0.5.0 sentence-transformers>=2.2.2
```

### Issue: Large ChromaDB database migration is slow

**Solution**: Use batch mode for faster migration:
```bash
python scripts/migration/migrate_chroma_to_sqlite.py --batch-size 100
```

## Rollback Plan

If you need to rollback to ChromaDB (not recommended):

1. **Stay on v7.x releases** - Do not upgrade to v8.0.0
2. **Use the `chromadb-legacy-final` tag** for reference
3. **Restore ChromaDB database files** from your filesystem backup directory directly.

## Post-Migration Checklist

- [ ] Backup completed successfully
- [ ] Migration script ran without errors
- [ ] Memory count matches between old and new backend
- [ ] Sample queries return expected results
- [ ] Configuration updated (`MCP_MEMORY_STORAGE_BACKEND`)
- [ ] Legacy ChromaDB data directory backed up
- [ ] Validation script passes
- [ ] Application tests pass
- [ ] Claude Desktop/Code integration works

## Support

- **Migration issues**: See [Issue #148](https://github.com/doobidoo/mcp-memory-service/issues/148)
- **Backend setup**: See [STORAGE_BACKENDS.md](./STORAGE_BACKENDS.md)

## Why Was ChromaDB Removed?

- **Performance**: 3x slower than SQLite-vec (15ms vs 5ms)
- **Dependencies**: Required ~2GB PyTorch download
- **Complexity**: 2,841 lines of code removed
- **Better alternatives**: Hybrid backend provides better performance with cloud sync
- **Maintenance**: Reduced long-term maintenance burden

The removal improves the project's maintainability while offering better performance through modern alternatives.
