# ChromaDB Migration Guide

> **ChromaDB backend was removed in v8.0.0**. This guide helps you migrate to modern storage backends.

## Step 1: Convert the Old Database with v7.6.0

Current releases cannot read a ChromaDB directory, and the conversion script was
retired with the rest of the ChromaDB code. v7.6.0 — the last release that
shipped the backend — is tagged `chromadb-legacy-final` and is not on PyPI, so
install it from the tag, run the migration there, then upgrade to the current
release. Python 3.12 is the tested interpreter for this step.

```bash
git clone https://github.com/doobidoo/mcp-memory-service.git mcp-memory-service-legacy
cd mcp-memory-service-legacy
git checkout chromadb-legacy-final

# Python 3.12 is the tested interpreter; the chromadb extra installs the backend
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[chromadb]"

# Convert ChromaDB to a SQLite-vec database
python scripts/migration/migrate_chroma_to_sqlite.py \
    --chroma-path /path/to/chroma_db --sqlite-path ~/chroma_export.db
```

Script options:

| Option | Description |
|--------|-------------|
| `--chroma-path PATH` | Path to the ChromaDB data directory |
| `--sqlite-path PATH` | Path for the new SQLite-vec database |
| `--batch-size NUM` | Batch size for the migration (default: 50) |
| `--verbose` | Enable verbose logging |

Keep the original ChromaDB directory until you have verified the converted
database. The legacy checkout is only needed for this step; the rest of the
guide runs on the current release.

## Step 2: Choose a Backend

### Option 1: Hybrid Backend (Recommended)

Best choice for most users - combines fast local storage with cloud synchronization.

```bash
# 1. Configure the Hybrid backend
export MCP_MEMORY_STORAGE_BACKEND=hybrid
export MCP_MEMORY_SQLITE_PATH=~/chroma_export.db

# 2. Configure Cloudflare credentials
export CLOUDFLARE_API_TOKEN="your-token"
export CLOUDFLARE_ACCOUNT_ID="your-account"
export CLOUDFLARE_D1_DATABASE_ID="your-d1-id"
export CLOUDFLARE_VECTORIZE_INDEX="mcp-memory-index"

# 3. Upload the converted database to Cloudflare
python scripts/migration/migrate_to_cloudflare.py export \
    --source sqlite_vec --source-path ~/chroma_export.db --output ~/chromadb_backup.json
python scripts/migration/migrate_to_cloudflare.py import --input ~/chromadb_backup.json

# 4. Verify configuration
python scripts/validation/validate_configuration_complete.py
```

The `export` lines above only last for the current shell. The validator in step
4 reads the project `.env` file and the `env` block of the memory server in
`claude_desktop_config.json`, and a service started by Claude Desktop reads the
latter. Put `MCP_MEMORY_STORAGE_BACKEND`, `MCP_MEMORY_SQLITE_PATH` and the four
`CLOUDFLARE_*` values in both places before running it, then restart the service.

### Option 2: SQLite-vec (Local Only)

For single-device use without cloud synchronization.

```bash
export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec
export MCP_MEMORY_SQLITE_PATH=~/chroma_export.db
```

### Option 3: Cloudflare (Cloud Only)

For pure cloud storage without a local database.

```bash
# 1. Configure Cloudflare
export MCP_MEMORY_STORAGE_BACKEND=cloudflare
export CLOUDFLARE_API_TOKEN="your-token"
export CLOUDFLARE_ACCOUNT_ID="your-account"
export CLOUDFLARE_D1_DATABASE_ID="your-d1-id"
export CLOUDFLARE_VECTORIZE_INDEX="mcp-memory-index"

# 2. Export from SQLite-vec and import into Cloudflare
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

## Data Validation

`python scripts/database/db_health_check.py` checks the environment, the
dependencies and the schema/search/embedding behaviour of the backend, but it
runs its store and search tests against temporary databases. Its only check of
the configured database is an embedding invariant, and it reports PASS (as
"skipping check") when that file or its `memories` table is missing. A passing
run therefore does not verify the data you just converted.

Check the converted database directly. Point `MCP_MEMORY_SQLITE_PATH` at
`~/chroma_export.db` and compare the row count with the old ChromaDB count:

```bash
sqlite3 ~/chroma_export.db "SELECT COUNT(*) FROM memories;"
```

The v7.6.0 schema has no `deleted_at` column; the current release adds it the
first time it opens the file. If the current service has already opened the
converted database, run the active count instead:

```bash
sqlite3 ~/chroma_export.db "SELECT COUNT(*) FROM memories WHERE deleted_at IS NULL;"
```

Then start the service against it and run a few sample searches for content you
know was in ChromaDB, checking that tags, memory types and timestamps survived.

## Troubleshooting

### Issue: Migration script not found

**Solution**: The script only exists on the `chromadb-legacy-final` tag. Current
releases do not ship it — rerun Step 1 from a checkout of that tag.

### Issue: Import errors for ChromaMemoryStorage

**Solution**: Run the migration inside the v7.6.0 checkout, in the virtualenv
where you installed it with `pip install -e ".[chromadb]"`. The current release
has no `mcp_memory_service.storage.chroma` module.

### Issue: "ChromaDB not installed" error

**Solution**: Install the legacy tag's ChromaDB extra, which pulls in the
backend and its ML dependencies:

```bash
git checkout chromadb-legacy-final
pip install -e ".[chromadb]"
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
- [ ] Health check passes
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
