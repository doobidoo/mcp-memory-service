# SQLite to Qdrant Migration Guide

This guide provides step-by-step instructions for migrating your memory service from SQLite-vec backend to Qdrant backend.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Migration Steps](#migration-steps)
- [Resume After Failure](#resume-after-failure)
- [Validation](#validation)
- [Rollback Strategy](#rollback-strategy)
- [Common Issues](#common-issues)
- [Model Change Migration](#model-change-migration)

---

## Prerequisites

Before starting the migration, ensure you have:

### 1. Backup Your SQLite Database

**CRITICAL**: Always backup your SQLite database before migration. This is your safety net.

```bash
# macOS
cp ~/Library/Application\ Support/mcp-memory/sqlite_vec.db ~/sqlite_vec.db.backup

# Linux
cp ~/.local/share/mcp-memory/sqlite_vec.db ~/sqlite_vec.db.backup

# Windows
copy %LOCALAPPDATA%\mcp-memory\sqlite_vec.db %USERPROFILE%\sqlite_vec.db.backup
```

Verify the backup was created successfully:

```bash
# macOS/Linux
ls -lh ~/sqlite_vec.db.backup

# Windows
dir %USERPROFILE%\sqlite_vec.db.backup
```

### 2. Check Disk Space

Qdrant requires disk space for the new storage. Ensure you have at least 2x your SQLite database size available.

```bash
# Check available disk space
df -h /

# Check SQLite database size (macOS/Linux)
du -h ~/Library/Application\ Support/mcp-memory/sqlite_vec.db

# Windows
dir %LOCALAPPDATA%\mcp-memory\sqlite_vec.db
```

**Minimum Requirements**:
- Free space: 2x SQLite database size
- Recommended: 3x SQLite database size for safety

### 3. Stop Running Services

Ensure no services are currently accessing the databases:

```bash
# Stop HTTP server (if running)
pkill -f "memory.*server"

# Check no processes are using the database
lsof ~/Library/Application\ Support/mcp-memory/sqlite_vec.db  # macOS/Linux
```

### 4. Install Dependencies

Ensure you have the latest version of the migration scripts:

```bash
cd /path/to/mcp-memory-service
git pull
uv sync
```

---

## Migration Steps

### Step 1: Dry Run (Recommended)

**Always run a dry run first** to validate the migration process without writing data:

```bash
python scripts/migration/migrate_sqlite_to_qdrant.py \
  --dry-run \
  --sqlite-path ~/Library/Application\ Support/mcp-memory/sqlite_vec.db \
  --qdrant-path ~/Library/Application\ Support/mcp-memory/qdrant \
  --batch-size 100
```

**What the dry run does**:
- Validates SQLite database is accessible
- Checks memory count and structure
- Simulates migration batches
- Reports estimated time
- **Does NOT write to Qdrant**

**Expected Output**:
```
INFO - SQLite path: /Users/yourname/Library/Application Support/mcp-memory/sqlite_vec.db
INFO - Qdrant path: /Users/yourname/Library/Application Support/mcp-memory/qdrant
INFO - Checkpoint path: migration_checkpoint.json
INFO - Initializing storage backends...
INFO - Found 5000 memories in SQLite
INFO - DRY RUN MODE - No data will be written to Qdrant
INFO - Processing batch 1 (100 memories)...
...
INFO - Migration Complete!
INFO - Total memories: 5000
INFO - Successfully migrated: 5000
INFO - Failed migrations: 0
INFO - Duration: 45.23 seconds
INFO - Rate: 110.5 memories/sec
INFO - DRY RUN - No data was actually written to Qdrant
```

### Step 2: Run Migration

Once the dry run succeeds, run the actual migration:

```bash
python scripts/migration/migrate_sqlite_to_qdrant.py \
  --sqlite-path ~/Library/Application\ Support/mcp-memory/sqlite_vec.db \
  --qdrant-path ~/Library/Application\ Support/mcp-memory/qdrant \
  --batch-size 100 \
  --checkpoint migration_checkpoint.json
```

**Command Options**:
- `--sqlite-path`: Full path to your SQLite database
- `--qdrant-path`: Directory where Qdrant will store data
- `--batch-size`: Number of memories per batch (default: 100, adjust based on memory)
- `--checkpoint`: Checkpoint file path (default: `migration_checkpoint.json`)

**Migration Process**:
1. **Initialize**: Connects to SQLite (read-only) and Qdrant (write)
2. **Count**: Reports total memories to migrate
3. **Batch Processing**: Migrates memories in batches of 100
4. **Validation**: Validates each batch after writing (embeddings, tags, timestamps)
5. **Checkpoint**: Saves progress after every batch
6. **Progress**: Reports every 1000 memories with ETA

**Expected Output**:
```
INFO - Initializing storage backends...
INFO - Fetching all memories from SQLite...
INFO - Found 5000 memories in SQLite
INFO - Processing batch 1 (100 memories)...
INFO - Processing batch 2 (100 memories)...
...
INFO - Progress: 1000/5000 (20.0%) Rate: 115.3 memories/sec, ETA: 35s
...
============================================================
Migration Complete!
============================================================
Total memories: 5000
Successfully migrated: 5000
Failed migrations: 0
Duration: 43.41 seconds
Rate: 115.2 memories/sec
```

### Step 3: Monitor Progress

The migration script provides real-time progress updates:

- **Batch Processing**: Shows current batch number
- **Progress Reports**: Every 1000 memories, displays:
  - Percentage complete
  - Migration rate (memories/sec)
  - Estimated time remaining (ETA)
- **Checkpoint Saves**: Automatic after every batch

**Example Progress**:
```
INFO - Progress: 3000/5000 (60.0%) Rate: 112.8 memories/sec, ETA: 18s
```

---

## Resume After Failure

The migration script supports **checkpoint-based resume** if interrupted.

### When to Resume

Resume the migration if:
- Network interruption occurred
- Process was killed (Ctrl+C, system crash)
- Qdrant connection timed out
- Batch validation failed

### Resume Command

```bash
python scripts/migration/migrate_sqlite_to_qdrant.py \
  --resume \
  --checkpoint migration_checkpoint.json \
  --sqlite-path ~/Library/Application\ Support/mcp-memory/sqlite_vec.db \
  --qdrant-path ~/Library/Application\ Support/mcp-memory/qdrant
```

**What `--resume` does**:
1. Loads checkpoint file (`migration_checkpoint.json`)
2. Reads `migrated_count` and `last_successful_hash`
3. Skips already-migrated memories
4. Continues from last successful batch

### Checkpoint File Location

The checkpoint file is created in the **current working directory** by default:

```bash
# Check checkpoint location
ls -la migration_checkpoint.json

# View checkpoint contents
cat migration_checkpoint.json
```

**Checkpoint Structure**:
```json
{
  "total_memories": 5000,
  "migrated_count": 3000,
  "failed_hashes": [],
  "last_successful_hash": "abc123def456...",
  "started_at": "2025-01-16T10:30:00Z",
  "last_updated_at": "2025-01-16T10:35:23Z"
}
```

### Example Resume Scenario

1. **Migration interrupted at 3000/5000**:
```
INFO - Progress: 3000/5000 (60.0%) Rate: 115.2 memories/sec, ETA: 17s
^C  # User pressed Ctrl+C
```

2. **Resume migration**:
```bash
python scripts/migration/migrate_sqlite_to_qdrant.py --resume
```

3. **Output shows resume**:
```
INFO - Loaded checkpoint: 3000/5000 memories migrated
INFO - Resuming migration from checkpoint: 3000 memories already migrated
INFO - Skipping 3000 already migrated memories
INFO - Processing batch 31 (100 memories)...
```

---

## Validation

After migration completes, validate data integrity.

### Automatic Batch Validation

The migration script automatically validates **each batch** after writing:

**Validation Checks** (per batch):
- **Count**: Batch size matches retrieved count
- **Embeddings**: Cosine similarity >0.99 for sample memories
- **Tags**: All tags preserved (set equality)
- **Timestamps**: Within 1ms tolerance

**Example Validation Output**:
```
INFO - Processing batch 50 (100 memories)...
DEBUG - Batch validated successfully
```

If validation fails:
```
WARNING - Batch validation failed: Embedding mismatch for abc123: similarity=0.95
```

### Manual Post-Migration Validation

Run the validation script to verify entire migration:

```bash
python scripts/migration/validate_migration.py \
  --source ~/Library/Application\ Support/mcp-memory/sqlite_vec.db \
  --target ~/Library/Application\ Support/mcp-memory/qdrant \
  --sample-size 100
```

**Validation Script Checks**:
1. **Count Validation**: Total memory counts match
2. **Embedding Similarity**: Sample-based cosine similarity >0.99
3. **Metadata Preservation**: Tags and metadata identical
4. **Timestamp Preservation**: Timestamps within 1ms

**Expected Output**:
```
======================================================================
MIGRATION VALIDATION REPORT
======================================================================
Count Validation: ✅ PASS
  Source: 5,000 memories
  Target: 5,000 memories

Embedding Similarity: ✅ PASS
  Average Cosine Similarity: 0.9998
  Threshold: >0.99

Metadata Preservation: ✅ PASS

Timestamp Preservation: ✅ PASS
  Tolerance: 1ms

No errors found ✅

======================================================================
OVERALL: ✅ VALIDATION PASSED
======================================================================
```

### Validation Failure Handling

If validation fails:

1. **Review Errors**:
```
Errors Found: 3
----------------------------------------------------------------------
  - abc123: Embedding mismatch (similarity=0.95 < 0.99)
  - def456: Tags mismatch - source: {'important', 'test'}, target: {'test'}
  - ghi789: Timestamp mismatch - diff: 5.234ms
```

2. **Check Failed Hashes**:
```bash
cat migration_checkpoint.json | grep failed_hashes
```

3. **Options**:
   - **Retry migration**: Use `--resume` to retry failed batches
   - **Rollback**: See [Rollback Strategy](#rollback-strategy)
   - **Manual investigation**: Check specific memory hashes

---

## Rollback Strategy

If migration fails or validation errors occur, rollback to SQLite.

### Rollback Steps

#### 1. Stop All Services

```bash
# Stop HTTP server
pkill -f "memory.*server"

# Stop MCP servers (if using Claude Desktop)
# Restart Claude Desktop to disconnect MCP servers
```

#### 2. Delete Qdrant Collection

```bash
# Remove Qdrant storage directory
rm -rf ~/Library/Application\ Support/mcp-memory/qdrant  # macOS
rm -rf ~/.local/share/mcp-memory/qdrant                  # Linux
rmdir /s %LOCALAPPDATA%\mcp-memory\qdrant                # Windows
```

#### 3. Restore SQLite Backup (if corrupted)

```bash
# Restore from backup
cp ~/sqlite_vec.db.backup ~/Library/Application\ Support/mcp-memory/sqlite_vec.db  # macOS
cp ~/sqlite_vec.db.backup ~/.local/share/mcp-memory/sqlite_vec.db                  # Linux
copy %USERPROFILE%\sqlite_vec.db.backup %LOCALAPPDATA%\mcp-memory\sqlite_vec.db    # Windows
```

#### 4. Update Configuration

Switch backend to `sqlite_vec`:

```bash
# Update .env or environment variable
export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec

# Verify configuration
python scripts/validation/diagnose_backend_config.py
```

#### 5. Restart Services

```bash
# Start HTTP server (if using)
uv run memory server --http

# Restart Claude Desktop (if using MCP)
```

### Verification After Rollback

```bash
# Check health endpoint
curl http://127.0.0.1:8000/api/health

# Expected output should show:
# {"status": "healthy", "storage": "sqlite-vec", "memory_count": 5000}
```

---

## Common Issues

### Issue 1: Database Locked Error

**Symptom**:
```
ERROR - database is locked
```

**Cause**: Another process is accessing SQLite database

**Solution**:
```bash
# Find processes using database
lsof ~/Library/Application\ Support/mcp-memory/sqlite_vec.db

# Kill processes
pkill -f "memory.*server"

# Retry migration
python scripts/migration/migrate_sqlite_to_qdrant.py --resume
```

### Issue 2: Qdrant Connection Timeout

**Symptom**:
```
ERROR - Qdrant connection timeout
```

**Cause**: Qdrant initialization taking too long

**Solution**:
```bash
# Check Qdrant path permissions
ls -la ~/Library/Application\ Support/mcp-memory/

# Ensure directory is writable
chmod 755 ~/Library/Application\ Support/mcp-memory/

# Retry with smaller batch size
python scripts/migration/migrate_sqlite_to_qdrant.py --resume --batch-size 50
```

### Issue 3: Memory/Disk Space Exhausted

**Symptom**:
```
ERROR - No space left on device
```

**Cause**: Insufficient disk space

**Solution**:
```bash
# Check disk space
df -h /

# Free up space (if needed)
# Then resume migration
python scripts/migration/migrate_sqlite_to_qdrant.py --resume
```

### Issue 4: Embedding Dimension Mismatch

**Symptom**:
```
ERROR - Embedding dimension mismatch: expected 1536, got 768
```

**Cause**: SQLite contains embeddings from different model

**Solution**:
This requires **model change migration** (see next section).

### Issue 5: Checkpoint File Corrupted

**Symptom**:
```
ERROR - Failed to load checkpoint: Invalid JSON
```

**Cause**: Checkpoint file was corrupted

**Solution**:
```bash
# Remove corrupted checkpoint
rm migration_checkpoint.json

# Restart migration (will start from beginning)
python scripts/migration/migrate_sqlite_to_qdrant.py
```

**Note**: If significant progress was made, this is wasteful. Check if you can manually edit checkpoint file.

### Issue 6: Validation Failures

**Symptom**:
```
WARNING - Batch validation failed: Embedding mismatch for abc123: similarity=0.95
```

**Cause**: Embeddings not identical after migration (rare, but possible due to floating-point precision)

**Solution**:
- If similarity >0.95, this is likely acceptable (floating-point rounding)
- If <0.95, investigate specific memory:
```bash
# Check source embedding
sqlite3 ~/Library/Application\ Support/mcp-memory/sqlite_vec.db \
  "SELECT content_hash, content FROM memories WHERE content_hash='abc123';"
```

---

## Model Change Migration

If you're changing embedding models (e.g., from `text-embedding-3-small` to `text-embedding-3-large`), you need to **re-embed all memories**.

### Why Re-embedding is Required

- Different models produce embeddings with **different dimensions**
- Example: `text-embedding-3-small` = 1536 dimensions, `text-embedding-3-large` = 3072 dimensions
- Qdrant collections are dimension-locked (cannot mix dimensions)

### Model Change Migration Process

#### 1. Backup Current Database

```bash
# Backup SQLite
cp ~/Library/Application\ Support/mcp-memory/sqlite_vec.db ~/sqlite_vec.db.backup

# Backup Qdrant (if already using Qdrant)
tar -czf ~/qdrant_backup.tar.gz ~/Library/Application\ Support/mcp-memory/qdrant/
```

#### 2. Set New Embedding Model

```bash
# Update environment variable
export MCP_EMBEDDING_MODEL=text-embedding-3-large

# Verify configuration
python scripts/validation/diagnose_backend_config.py
```

#### 3. Create New Qdrant Collection

The migration script will create a new collection with the correct dimensions:

```bash
python scripts/migration/migrate_sqlite_to_qdrant.py \
  --sqlite-path ~/Library/Application\ Support/mcp-memory/sqlite_vec.db \
  --qdrant-path ~/Library/Application\ Support/mcp-memory/qdrant_new \
  --batch-size 50
```

**Note**: Use a **new Qdrant path** to avoid conflicts.

#### 4. Re-embedding Process

The migration script will:
1. Read memories from SQLite
2. **Re-generate embeddings** using new model (`MCP_EMBEDDING_MODEL`)
3. Store new embeddings in Qdrant
4. Preserve content, tags, timestamps

**Expected Time Estimates** (approximate, depends on API rate limits):
- **100 memories**: ~30 seconds
- **1,000 memories**: ~5 minutes
- **10,000 memories**: ~50 minutes
- **100,000 memories**: ~8 hours

**Factors Affecting Speed**:
- Embedding API rate limits (OpenAI: ~3000 requests/min for Tier 1)
- Network latency
- GPU availability (for local models)
- Batch size

#### 5. Validate New Embeddings

```bash
python scripts/migration/validate_migration.py \
  --source ~/Library/Application\ Support/mcp-memory/sqlite_vec.db \
  --target ~/Library/Application\ Support/mcp-memory/qdrant_new \
  --sample-size 100
```

**Note**: Embedding validation will **fail** because embeddings are different. Validate **count, tags, timestamps** instead.

#### 6. Switch to New Qdrant Path

Update configuration:

```bash
# Update .env
export MCP_MEMORY_QDRANT_PATH=~/Library/Application\ Support/mcp-memory/qdrant_new

# Restart services
uv run memory server --http
```

### GPU Acceleration for Re-embedding

If using local embedding models, GPU acceleration significantly reduces re-embedding time.

**GPU Support**:
- **macOS**: Metal Performance Shaders (MPS) - automatic
- **Windows**: CUDA (NVIDIA) or DirectML (AMD/Intel)
- **Linux**: CUDA (NVIDIA) or ROCm (AMD)

**Example Re-embedding with Local Model**:
```bash
# Use local model (automatically detects GPU)
export MCP_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

python scripts/migration/migrate_sqlite_to_qdrant.py \
  --sqlite-path ~/Library/Application\ Support/mcp-memory/sqlite_vec.db \
  --qdrant-path ~/Library/Application\ Support/mcp-memory/qdrant_new \
  --batch-size 100
```

**Expected Speedup with GPU**:
- CPU: ~10-20 embeddings/sec
- GPU: ~100-500 embeddings/sec (10-50x faster)

---

## Summary

**Migration Checklist**:
- [ ] Backup SQLite database (`cp` command)
- [ ] Check disk space (`df -h`)
- [ ] Stop running services (`pkill`)
- [ ] Run dry run (`--dry-run`)
- [ ] Run migration (`migrate_sqlite_to_qdrant.py`)
- [ ] Monitor progress (checkpoints every batch)
- [ ] Validate migration (`validate_migration.py`)
- [ ] Update backend configuration (`MCP_MEMORY_STORAGE_BACKEND=qdrant`)
- [ ] Restart services and verify

**Key Points**:
- **Always backup first** - no exceptions
- **Dry run first** - validate before writing
- **Resume capability** - interruptions are safe
- **Automatic validation** - each batch is verified
- **Rollback strategy** - delete Qdrant, restore SQLite

**Support**:
- Migration issues: Check [Common Issues](#common-issues)
- Backend configuration: Run `python scripts/validation/diagnose_backend_config.py`
- Validation failures: Review checkpoint file (`migration_checkpoint.json`)

---

**Last Updated**: 2025-01-16
**Script Version**: migrate_sqlite_to_qdrant.py (v1.0)
