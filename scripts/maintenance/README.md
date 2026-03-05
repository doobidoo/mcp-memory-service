# Maintenance Scripts

This directory contains maintenance and diagnostic scripts for the MCP Memory Service database.

## Quick Reference

| Script | Purpose | Performance | Use Case |
|--------|---------|-------------|----------|
| [`check_memory_types.py`](#check_memory_typespy-new) | Display type distribution | <1s | Quick health check, pre/post-consolidation validation |
| [`consolidate_memory_types.py`](#consolidate_memory_typespy-new) | Consolidate fragmented types | ~5s for 1000 updates | Type taxonomy cleanup, reduce fragmentation |
| [`migrate_embeddings.py`](#migrate_embeddingspy-new) | Migrate to a different embedding model | ~20s for 1000 memories | Switching models (e.g., 384-dim to 768-dim) |
| [`regenerate_embeddings.py`](#regenerate_embeddingspy) | Regenerate all embeddings | ~5min for 2600 memories | After cosine migration or embedding corruption |
| [`fast_cleanup_duplicates.sh`](#fast_cleanup_duplicatessh) | Fast duplicate removal | <5s for 100+ duplicates | Bulk duplicate cleanup |
| [`find_all_duplicates.py`](#find_all_duplicatespy) | Detect near-duplicates | <2s for 2000 memories | Duplicate detection and analysis |
| [`find_duplicates.py`](#find_duplicatespy) | API-based duplicate finder | Slow (~90s/duplicate) | Detailed duplicate analysis via API |
| [`repair_sqlite_vec_embeddings.py`](#repair_sqlite_vec_embeddingspy) | Fix embedding corruption | Varies | Repair corrupted embeddings |
| [`repair_zero_embeddings.py`](#repair_zero_embeddingspy) | Fix zero-valued embeddings | Varies | Repair zero embeddings |
| [`cleanup_corrupted_encoding.py`](#cleanup_corrupted_encodingpy) | Fix encoding issues | Varies | Repair encoding corruption |
| [`cleanup_association_memories.py`](#cleanup_association_memoriespy) | Remove association memories (local) | <5s | After graph migration (SQLite backend) |
| [`cleanup_association_memories_hybrid.py`](#cleanup_association_memories_hybridpy-new) | Remove association memories (hybrid) | ~30s | After graph migration (hybrid backend, multi-PC) |
| [`backfill_graph_table.py`](#backfill_graph_tablepy) | Migrate associations to graph | <10s | Graph database migration |

## Detailed Documentation

### `check_memory_types.py` 🆕

**Purpose**: Quick diagnostic tool to display memory type distribution in the database.

**When to Use**:
- Before running consolidation to see what needs cleanup
- After consolidation to verify results
- Regular health checks to monitor type fragmentation
- When investigating memory organization issues

**Usage**:
```bash
# Display type distribution (Windows)
python scripts/maintenance/check_memory_types.py

# On macOS/Linux, update the database path in the script first
```

**Output Example**:
```
Memory Type Distribution
============================================================
Total memories: 1,978
Unique types: 128

Memory Type                              Count      %
------------------------------------------------------------
note                                       609  30.8%
session                                     89   4.5%
fix                                         67   3.4%
milestone                                   60   3.0%
reference                                   45   2.3%
...
```

**Performance**: < 1 second for any database size (read-only SQL query)

**Features**:
- Shows top 30 types by frequency
- Displays total memory count and unique type count
- Identifies NULL/empty types as "(empty/NULL)"
- Percentage calculation for easy analysis
- Zero risk (read-only operation)

**Workflow Integration**:
1. Run `check_memory_types.py` to identify fragmentation
2. If types > 150, consider running consolidation
3. Run `consolidate_memory_types.py --dry-run` to preview
4. Execute `consolidate_memory_types.py` to clean up
5. Run `check_memory_types.py` again to verify improvement

### `consolidate_memory_types.py` 🆕

**Purpose**: Consolidates fragmented memory types into a standardized 24-type taxonomy.

**When to Use**:
- Type fragmentation (e.g., `bug-fix`, `bugfix`, `technical-fix` all coexisting)
- Many types with only 1-2 memories
- Inconsistent naming across similar concepts
- After importing memories from external sources
- Monthly maintenance to prevent type proliferation

**Usage**:
```bash
# Preview changes (safe, read-only)
python scripts/maintenance/consolidate_memory_types.py --dry-run

# Execute consolidation
python scripts/maintenance/consolidate_memory_types.py

# Use custom mapping configuration
python scripts/maintenance/consolidate_memory_types.py --config custom_mappings.json
```

**Performance**: ~5 seconds for 1,000 memory updates (Nov 2025 real-world test: 1,049 updates in 5s)

**Safety Features**:
- ✅ Automatic timestamped backup before execution
- ✅ Dry-run mode shows preview without changes
- ✅ Transaction safety (atomic with rollback on error)
- ✅ Database lock detection (prevents concurrent access)
- ✅ HTTP server warning (recommends stopping before execution)
- ✅ Disk space verification (needs 2x database size)
- ✅ Backup verification (size and existence checks)

**Standard 24-Type Taxonomy**:

**Content Types:** `note`, `reference`, `document`, `guide`
**Activity Types:** `session`, `implementation`, `analysis`, `troubleshooting`, `test`
**Artifact Types:** `fix`, `feature`, `release`, `deployment`
**Progress Types:** `milestone`, `status`
**Infrastructure Types:** `configuration`, `infrastructure`, `process`, `security`, `architecture`
**Other Types:** `documentation`, `solution`, `achievement`, `technical`

**Example Consolidations**:
- NULL/empty → `note`
- `bug-fix`, `bugfix`, `technical-fix` → `fix`
- `session-summary`, `session-checkpoint` → `session`
- `project-milestone`, `development-milestone` → `milestone`
- All `technical-*` → base type (remove prefix)
- All `project-*` → base type (remove prefix)

**Typical Results** (from production database, Nov 2025):
```
Before: 342 unique types, 609 NULL/empty, fragmented naming
After:  128 unique types (63% reduction), all valid types
Updated: 1,049 memories (59% of database)
Time: ~5 seconds
```

**Configuration**:

Edit `consolidation_mappings.json` to customize behavior:
```json
{
  "mappings": {
    "old-type-name": "new-type-name",
    "bug-fix": "fix",
    "technical-solution": "solution"
  }
}
```

**Prerequisites**:
```bash
# 1. Stop HTTP server
systemctl --user stop mcp-memory-http.service

# 2. Disconnect MCP clients (Claude Code: /mcp command)

# 3. Verify disk space (need 2x database size)
df -h ~/.local/share/mcp-memory/
```

**Recovery**:
```bash
# If something goes wrong, restore from automatic backup
cp ~/.local/share/mcp-memory/sqlite_vec.db.backup-TIMESTAMP ~/.local/share/mcp-memory/sqlite_vec.db

# Verify restoration
sqlite3 ~/.local/share/mcp-memory/sqlite_vec.db "SELECT COUNT(*), COUNT(DISTINCT memory_type) FROM memories;"
```

**Notes**:
- Creates timestamped backup automatically (e.g., `sqlite_vec.db.backup-20251101-202042`)
- No data loss - only type reassignment
- Safe to run multiple times (idempotent for same mappings)
- Comprehensive reporting shows before/after statistics
- See `consolidation_mappings.json` for full mapping list

**Maintenance Schedule**:
- Run `--dry-run` monthly to check fragmentation
- Execute when unique types exceed 150
- Review custom mappings quarterly

---

### `migrate_embeddings.py` NEW

**Purpose**: Migrates embeddings from one model to another, handling dimension changes. Unlike `regenerate_embeddings.py` (which re-embeds with the current model), this script drops and recreates the vec0 virtual table at the new dimension, re-embeds all active memories via an OpenAI-compatible API, and verifies integrity.

**When to Use**:
- Switching embedding models (e.g., `all-MiniLM-L6-v2` 384-dim to `nomic-embed-text` 768-dim)
- Upgrading to a higher-quality model with different dimensions
- Moving from local ONNX to an external embedding API

**Usage**:
```bash
# Preview (safe, read-only):
python scripts/maintenance/migrate_embeddings.py \
    --url http://localhost:11434/v1/embeddings \
    --model nomic-embed-text \
    --dry-run

# Run migration (Ollama example):
python scripts/maintenance/migrate_embeddings.py \
    --url http://localhost:11434/v1/embeddings \
    --model nomic-embed-text

# OpenAI example:
python scripts/maintenance/migrate_embeddings.py \
    --url https://api.openai.com/v1/embeddings \
    --model text-embedding-3-small \
    --api-key sk-...

# Keep graph edges (skip wipe):
python scripts/maintenance/migrate_embeddings.py \
    --url http://localhost:11434/v1/embeddings \
    --model nomic-embed-text \
    --keep-graph
```

**Performance**: ~20 seconds for 1,000 memories via Ollama (depends on model and hardware)

**Safety Features**:
- Automatic timestamped backup before any changes
- `--dry-run` mode for previewing state
- Service running detection (macOS launchd, Linux systemd)
- Dimension spot-check and KNN search verification after migration
- Count matching validation (memories vs embeddings)

**What It Does**:
1. Validates the target embedding API is reachable and detects output dimension
2. Backs up the database file
3. Reads all active memories
4. Generates new embeddings in batches via the target API
5. Drops and recreates the `memory_embeddings` vec0 table at the new dimension
6. Inserts embeddings with correct rowid mapping
7. Optionally wipes graph edges (recommended since similarity scores change)
8. Updates metadata, rebuilds FTS index, VACUUMs
9. Verifies integrity

**Comparison with `regenerate_embeddings.py`**:

| Script | Handles dimension change | External API | Backup | Dry-run |
|--------|------------------------|--------------|--------|---------|
| `migrate_embeddings.py` | Yes (drop/recreate vec0) | Yes (any OpenAI-compatible) | Yes | Yes |
| `regenerate_embeddings.py` | No (same model) | No (uses configured model) | No | No |

**Prerequisites**:
- Target embedding API must be running (e.g., `ollama serve`, vLLM, OpenAI)
- `requests` Python package (`pip install requests`)
- `sqlite-vec` Python package (auto-detected from mcp-memory-service venv)
- mcp-memory-service should be stopped

---

### `regenerate_embeddings.py`

**Purpose**: Regenerates embeddings for all memories in the database after schema migrations or corruption.

**When to Use**:
- After cosine distance migration
- When embeddings table is dropped but memories are preserved
- After embedding corruption detected

**Usage**:
```bash
/home/hkr/repositories/mcp-memory-service/venv/bin/python scripts/maintenance/regenerate_embeddings.py
```

**Performance**: ~5 minutes for 2600 memories with all-MiniLM-L6-v2 model

**Notes**:
- Uses configured storage backend (hybrid, cloudflare, or sqlite_vec)
- Creates embeddings using sentence-transformers model
- Shows progress every 100 memories
- Safe to run multiple times (idempotent)

---

### `fast_cleanup_duplicates.sh`

**Purpose**: Fast duplicate removal using direct SQL access instead of API calls.

**When to Use**:
- Bulk duplicate cleanup after detecting duplicates
- When API-based deletion is too slow (>1min per duplicate)
- Production cleanup without extended downtime

**Usage**:
```bash
bash scripts/maintenance/fast_cleanup_duplicates.sh
```

**Performance**: <5 seconds for 100+ duplicates

**How It Works**:
1. Stops HTTP server to avoid database locking
2. Uses direct SQL DELETE with timestamp normalization
3. Keeps newest copy of each duplicate group
4. Restarts HTTP server automatically

**Warnings**:
- ⚠️ Requires systemd HTTP server setup (`mcp-memory-http.service`)
- ⚠️ Brief service interruption during cleanup
- ⚠️ Direct database access bypasses Cloudflare sync (background sync handles it later)

---

### `find_all_duplicates.py`

**Purpose**: Fast duplicate detection using content normalization and hash comparison.

**When to Use**:
- Regular duplicate audits
- Before running cleanup operations
- Investigating duplicate memory issues

**Usage**:
```bash
/home/hkr/repositories/mcp-memory-service/venv/bin/python scripts/maintenance/find_all_duplicates.py
```

**Performance**: <2 seconds for 2000 memories

**Detection Method**:
- Normalizes content by removing timestamps (dates, ISO timestamps)
- Groups memories by MD5 hash of normalized content
- Reports duplicate groups with counts

**Output**:
```
Found 23 groups of duplicates
Total memories to delete: 115
Total memories after cleanup: 1601
```

---

### `find_duplicates.py`

**Purpose**: Comprehensive duplicate detection via HTTP API with detailed analysis.

**When to Use**:
- Need detailed duplicate analysis with full metadata
- API-based workflow required
- Integration with external tools

**Usage**:
```bash
/home/hkr/repositories/mcp-memory-service/venv/bin/python scripts/maintenance/find_duplicates.py
```

**Performance**: Slow (~90 seconds per duplicate deletion)

**Features**:
- Loads configuration from Claude hooks config
- Supports self-signed SSL certificates
- Pagination support for large datasets
- Detailed duplicate grouping and reporting

**Notes**:
- 15K script with comprehensive error handling
- Useful for API integration scenarios
- Slower than `find_all_duplicates.py` due to network overhead

---

### `repair_sqlite_vec_embeddings.py`

**Purpose**: Repairs corrupted embeddings in the sqlite-vec virtual table.

**When to Use**:
- Embedding corruption detected
- vec0 extension errors
- Database integrity issues

**Usage**:
```bash
/home/hkr/repositories/mcp-memory-service/venv/bin/python scripts/maintenance/repair_sqlite_vec_embeddings.py
```

**Warnings**:
- ⚠️ Requires vec0 extension to be properly installed
- ⚠️ May drop and recreate embeddings table

---

### `repair_zero_embeddings.py`

**Purpose**: Detects and fixes memories with zero-valued embeddings.

**When to Use**:
- Search results showing 0% similarity scores
- After embedding regeneration failures
- Embedding quality issues

**Usage**:
```bash
/home/hkr/repositories/mcp-memory-service/venv/bin/python scripts/maintenance/repair_zero_embeddings.py
```

---

### `cleanup_corrupted_encoding.py`

**Purpose**: Fixes encoding corruption issues in memory content.

**When to Use**:
- UTF-8 encoding errors
- Display issues with special characters
- After data migration from different encoding

**Usage**:
```bash
/home/hkr/repositories/mcp-memory-service/venv/bin/python scripts/maintenance/cleanup_corrupted_encoding.py
```

---

### `cleanup_association_memories_hybrid.py` 🆕

**Purpose**: Removes association memories from BOTH Cloudflare D1 AND local SQLite. Essential for hybrid backend with multi-PC setups where drift-sync can restore deleted associations.

**When to Use**:
- After graph migration (`backfill_graph_table.py`) when using **hybrid backend**
- When `MCP_GRAPH_STORAGE_MODE=graph_only` is set
- Multi-PC environments where associations were deleted on one PC but restored via Cloudflare sync
- When `cleanup_association_memories.py` (local-only) doesn't prevent restoration

**The Multi-PC Problem**:
```
┌─────────────┐     sync      ┌─────────────┐     sync      ┌─────────────┐
│  Windows PC │ ◄──────────► │  Cloudflare │ ◄──────────► │  Linux PC   │
│ deleted 1441│              │  D1 still   │              │  restored!  │
│ associations│              │  has them   │              │             │
└─────────────┘              └─────────────┘              └─────────────┘
```

**The Solution**: Clean Cloudflare D1 FIRST, then local. Other PCs auto-sync the deletion.

**Usage**:
```bash
# Preview (always start with dry-run)
python scripts/maintenance/cleanup_association_memories_hybrid.py --dry-run

# Execute full cleanup
python scripts/maintenance/cleanup_association_memories_hybrid.py --apply

# Skip Vectorize cleanup (optional - orphaned vectors are harmless)
python scripts/maintenance/cleanup_association_memories_hybrid.py --apply --skip-vectorize

# Only clean Cloudflare (useful from any PC)
python scripts/maintenance/cleanup_association_memories_hybrid.py --apply --cloudflare-only

# Only clean local (if Cloudflare already cleaned)
python scripts/maintenance/cleanup_association_memories_hybrid.py --apply --local-only
```

**Performance**: ~30 seconds for 1,400 associations (Cloudflare API batching)

**Safety Features**:
- ✅ Dry-run mode with detailed preview
- ✅ Automatic local database backup before deletion
- ✅ Confirmation prompt before destructive operations
- ✅ Graph table verification (aborts if graph missing)
- ✅ Cloudflare D1 cleaned first (prevents sync restoration)
- ✅ Robust Vectorize error handling (non-fatal errors)

**Prerequisites**:
1. Graph table must exist: Run `backfill_graph_table.py` first
2. Set `MCP_GRAPH_STORAGE_MODE=graph_only` in environment
3. Cloudflare credentials configured (API token, account ID, D1 database ID)

**Comparison with `cleanup_association_memories.py`**:

| Script | Backend | Cleans | Multi-PC Safe |
|--------|---------|--------|---------------|
| `cleanup_association_memories.py` | SQLite-vec | Local only | ❌ No |
| `cleanup_association_memories_hybrid.py` | Hybrid | Cloudflare + Local | ✅ Yes |

**Typical Results** (from production, Dec 2025):
```
Cloudflare D1:        1,441 memories deleted
Cloudflare Vectorize: 1,441 vectors deleted (or skipped)
Local SQLite:         1,441 memories deleted
Space reclaimed:      ~2.5 MB
```

---

## Best Practices

### Before Running Maintenance Scripts

1. **Backup your database**:
   ```bash
   cp ~/.local/share/mcp-memory/sqlite_vec.db ~/.local/share/mcp-memory/sqlite_vec.db.backup
   ```

2. **Check memory count**:
   ```bash
   sqlite3 ~/.local/share/mcp-memory/sqlite_vec.db "SELECT COUNT(*) FROM memories"
   ```

3. **Stop HTTP server if needed** (for direct database access):
   ```bash
   systemctl --user stop mcp-memory-http.service
   ```

### After Running Maintenance Scripts

1. **Verify results**:
   ```bash
   sqlite3 ~/.local/share/mcp-memory/sqlite_vec.db "SELECT COUNT(*) FROM memories"
   ```

2. **Check for duplicates**:
   ```bash
   /home/hkr/repositories/mcp-memory-service/venv/bin/python scripts/maintenance/find_all_duplicates.py
   ```

3. **Restart HTTP server**:
   ```bash
   systemctl --user start mcp-memory-http.service
   ```

4. **Test search functionality**:
   ```bash
   curl -s "http://127.0.0.1:8000/api/health"
   ```

### Performance Comparison

| Operation | API-based | Direct SQL | Speedup |
|-----------|-----------|------------|---------|
| Delete 1 duplicate | ~90 seconds | ~0.05 seconds | **1800x faster** |
| Delete 100 duplicates | ~2.5 hours | <5 seconds | **1800x faster** |
| Find duplicates | ~30 seconds | <2 seconds | **15x faster** |

**Recommendation**: Use direct SQL scripts (`fast_cleanup_duplicates.sh`, `find_all_duplicates.py`) for production maintenance. API-based scripts are useful for integration and detailed analysis.

## Troubleshooting

### "Database is locked"

**Cause**: HTTP server or MCP server has open connection

**Solution**:
```bash
# Stop HTTP server
systemctl --user stop mcp-memory-http.service

# Disconnect MCP server in Claude Code
# Type: /mcp

# Run maintenance script
bash scripts/maintenance/fast_cleanup_duplicates.sh

# Restart services
systemctl --user start mcp-memory-http.service
```

### "No such module: vec0"

**Cause**: Python sqlite3 module doesn't load vec0 extension automatically

**Solution**: Use scripts that work with the vec0-enabled environment:
- ✅ Use: `fast_cleanup_duplicates.sh` (bash wrapper with Python)
- ✅ Use: `/venv/bin/python` with proper storage backend
- ❌ Avoid: Direct `sqlite3` Python module for virtual table operations

### Slow API Performance

**Cause**: Hybrid backend syncs each operation to Cloudflare

**Solution**: Use direct SQL scripts for bulk operations:
```bash
bash scripts/maintenance/fast_cleanup_duplicates.sh  # NOT Python API scripts
```

## Related Documentation

- [Database Schema](../../docs/database-schema.md) - sqlite-vec table structure
- [Storage Backends](../../CLAUDE.md#storage-backends) - Hybrid, Cloudflare, SQLite-vec
- [Troubleshooting](../../docs/troubleshooting.md) - Common issues and solutions

## Contributing

When adding new maintenance scripts:

1. Add comprehensive docstring explaining purpose and usage
2. Include progress indicators for long-running operations
3. Add error handling and validation
4. Document in this README with performance characteristics
5. Test with both sqlite_vec and hybrid backends
