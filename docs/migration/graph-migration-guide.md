# Graph Database Migration Guide

**Version**: 1.0
**Target Release**: v8.51.0
**Estimated Time**: 10-15 minutes
**Difficulty**: Easy

## Overview

This guide walks you through migrating from memory-based association storage to the new graph database architecture. The migration provides:

- ✅ **30x faster queries** (150ms → 5ms for find_connected)
- ✅ **97% storage reduction** (~2-3 MB reclaimed for typical deployments)
- ✅ **Zero downtime** migration process
- ✅ **Rollback support** if needed

## Who Should Migrate?

**Immediate migration recommended if**:
- You have 500+ associations (check with `consolidation/status`)
- Query latency >100ms is noticeable
- Database size is growing rapidly
- You use consolidation system regularly

**Can defer migration if**:
- You have <100 associations
- Consolidation is disabled
- Storage space is not a concern
- You're evaluating the service

## Prerequisites

### Check Your Current State

```bash
# 1. Check how many associations you have
sqlite3 ~/.local/share/mcp-memory-service/memory.db \
  "SELECT COUNT(*) FROM memories WHERE tags LIKE '%association%' AND tags LIKE '%discovered%';"
# Example output: 1449

# 2. Check current database size
ls -lh ~/.local/share/mcp-memory-service/memory.db
# Example output: -rw-r--r-- 1 user user 20M Dec 14 02:15 memory.db

# 3. Check consolidation service status
curl http://127.0.0.1:8000/api/consolidation/status | jq '.jobs_executed'
# If consolidation is enabled, you likely have associations to migrate
```

### Required Software

- MCP Memory Service v8.51.0+
- Python 3.11+
- SQLite 3.35+ (for recursive CTEs)
- Disk space: ~10% of current database size (for migration overhead)

## Migration Process

### Step 1: Upgrade to v8.51.0

```bash
# Backup your database first (IMPORTANT!)
cp ~/.local/share/mcp-memory-service/memory.db \
   ~/.local/share/mcp-memory-service/memory.db.backup-$(date +%Y%m%d)

# Upgrade to v8.51.0
pip install --upgrade mcp-memory-service==8.51.0

# Verify version
python -c "import mcp_memory_service; print(mcp_memory_service.__version__)"
# Expected output: 8.51.0
```

**What happens after upgrade**:
- ✅ Service starts in `dual_write` mode by default (zero breaking changes)
- ✅ New associations written to both memories + graph tables
- ✅ Existing associations continue working as before
- ✅ Graph table created automatically on first use

### Step 2: Stop HTTP Server (Recommended)

```bash
# Linux/macOS with systemd
systemctl --user stop mcp-memory-http.service

# Or check for running processes
lsof | grep memory.db
# If processes are using the database, stop them to avoid locks
```

**Why stop the server?**
- Prevents database locks during migration
- Ensures clean transaction commits
- Avoids race conditions during backfill

### Step 3: Run Backfill Script (Dry Run)

```bash
# Preview what will be migrated (read-only, safe)
python scripts/maintenance/backfill_graph_table.py --dry-run
```

**Expected output**:
```
╔════════════════════════════════════════════════════════╗
║  Graph Table Backfill - Dry Run Mode                  ║
╚════════════════════════════════════════════════════════╝

[✓] Database exists: /Users/user/.local/share/mcp-memory-service/memory.db
[✓] Database is not locked
[!] HTTP server is running (recommend stopping first)
[✓] Sufficient disk space available (85.2 GB free)
[✓] Graph table created successfully

📊 Association Analysis:
   Total memories: 5,309
   Association memories: 1,449 (27.3%)

🔍 Extraction Results:
   Valid associations found: 1,435
   Skipped (missing metadata): 14
   Ready for migration: 1,435 bidirectional edges

📦 Graph Table Status:
   Existing associations: 0
   Duplicates to skip: 0
   New associations to add: 1,435

💾 Estimated Impact:
   Memory table size: ~724.5 KB (1,449 × 500 bytes)
   Graph table size: ~71.8 KB (1,435 × 50 bytes)
   Storage reduction: 90.1% (after cleanup)

[DRY RUN] No changes made. Use --apply to execute migration.
```

**Review checklist**:
- ✅ Association count matches expectations (check consolidation logs)
- ✅ Disk space is sufficient (need ~10% of database size)
- ✅ HTTP server warning (if shown, stop it for clean migration)
- ✅ No critical errors in output

### Step 4: Execute Migration

```bash
# Run actual migration (writes to database)
python scripts/maintenance/backfill_graph_table.py --apply
```

**Expected output**:
```
╔════════════════════════════════════════════════════════╗
║  Graph Table Backfill - LIVE MODE                     ║
╚════════════════════════════════════════════════════════╝

[✓] Pre-flight checks passed
[✓] Graph table initialized

🚀 Migrating Associations:

Processing batch 1/15: ████████████████ 100 associations (6.9%)
Processing batch 2/15: ████████████████ 200 associations (13.9%)
Processing batch 3/15: ████████████████ 300 associations (20.9%)
...
Processing batch 15/15: ████████████████ 1,435 associations (100%)

✅ Migration Complete!

📊 Results:
   Total processed: 1,435
   Successfully migrated: 1,435
   Failed: 0
   Duration: 3.2 seconds

🎯 Next Steps:
   1. Restart HTTP server
   2. Switch to graph_only mode (export MCP_GRAPH_STORAGE_MODE=graph_only)
   3. Run cleanup script to reclaim storage (optional)
```

**What if migration fails?**
- Script uses transactions - partial migrations are rolled back
- Check error message for details (usually database locks or disk space)
- Restore backup if needed: `cp memory.db.backup-* memory.db`
- See [Troubleshooting](#troubleshooting) section below

### Step 5: Restart Services

```bash
# Restart HTTP server
systemctl --user restart mcp-memory-http.service

# Or reconnect MCP server in Claude Code
# Run: /mcp

# Verify services are running
curl http://127.0.0.1:8000/api/health
# Expected: {"status": "healthy", "storage": "hybrid"}
```

### Step 6: Switch to graph_only Mode

```bash
# Update .env file
echo "export MCP_GRAPH_STORAGE_MODE=graph_only" >> ~/.bashrc
# Or add to your .env file

# Apply environment change
source ~/.bashrc

# Restart services to pick up new mode
systemctl --user restart mcp-memory-http.service
```

**Verify mode switch**:
```bash
# Check server logs for confirmation
journalctl --user -u mcp-memory-http.service | grep "Graph Storage Mode"
# Expected: INFO - Graph Storage Mode: graph_only
```

**What changes in graph_only mode?**
- ✅ New associations ONLY written to graph table (not memories)
- ✅ 97% storage reduction for new associations
- ✅ No search pollution from association memories
- ✅ 30x faster graph queries

### Step 7: Cleanup (Optional - Reclaim Storage)

**⚠️ Warning**: This step is OPTIONAL. Only proceed if you want to reclaim storage space. Associations will remain queryable via graph table.

```bash
# Preview what will be deleted (read-only, safe)
python scripts/maintenance/cleanup_association_memories.py --dry-run
```

**Expected output**:
```
╔════════════════════════════════════════════════════════╗
║  Association Memory Cleanup - Dry Run Mode            ║
╚════════════════════════════════════════════════════════╝

[✓] Database exists and is accessible
[✓] Graph table exists with 1,435 associations
[!] HTTP server is running (recommend stopping first)

📊 Analysis:
   Total association memories: 1,449
   Verified in graph table: 138 (safe to delete)
   Orphaned (no graph match): 1,311 (will be preserved)

💾 Estimated Impact:
   Space to reclaim: ~67 KB (138 memories × ~500 bytes)
   Database size before: 20.40 MB
   Estimated size after: 20.33 MB (0.3% reduction)

[DRY RUN] No deletions performed. Use --apply or no flag to execute.
```

**If you want to proceed with cleanup**:
```bash
# Interactive cleanup (prompts for confirmation)
python scripts/maintenance/cleanup_association_memories.py

# Prompt:
# Delete 138 memories? This will reclaim ~67 KB. (y/N)
# Enter 'y' to confirm

# Or automated cleanup (no confirmation)
python scripts/maintenance/cleanup_association_memories.py --force
```

**Expected output**:
```
╔════════════════════════════════════════════════════════╗
║  Association Memory Cleanup - LIVE MODE               ║
╚════════════════════════════════════════════════════════╝

[✓] Pre-flight checks passed

🗑️  Deleting Verified Associations:

Processing batch 1/2: ████████████████ 100 memories (72.5%)
Processing batch 2/2: ████████████████ 138 memories (100%)

✅ Deletion Complete!

📊 Results:
   Deleted: 138 memories
   Failed: 0
   Preserved (orphaned): 1,311 memories

💾 Running VACUUM to reclaim space...

✅ VACUUM Complete!

📊 Storage Impact:
   Database size before: 20.40 MB
   Database size after: 19.55 MB
   Space reclaimed: 864 KB (4.1% reduction)

🎯 Migration Complete!
   Mode: graph_only
   Associations: 1,435 in graph table
   Storage: Optimized
```

### Step 7b: Hybrid Backend Cleanup (Multi-PC) 🆕

**⚠️ IMPORTANT**: If you use the **hybrid backend** with multiple PCs, you MUST use the hybrid cleanup script instead. The standard cleanup script only deletes local associations - Cloudflare drift-sync will restore them!

**The Problem**:
```
┌─────────────┐     sync      ┌─────────────┐     sync      ┌─────────────┐
│  Windows PC │ ◄──────────► │  Cloudflare │ ◄──────────► │  Linux PC   │
│ deleted 1441│              │  D1 still   │              │  restored!  │
│ associations│              │  has them   │              │             │
└─────────────┘              └─────────────┘              └─────────────┘
```

**The Solution**: Use `cleanup_association_memories_hybrid.py` to clean Cloudflare D1 FIRST:

```bash
# Preview (always start with dry-run)
python scripts/maintenance/cleanup_association_memories_hybrid.py --dry-run

# Execute full cleanup (cleans Cloudflare + local)
python scripts/maintenance/cleanup_association_memories_hybrid.py --apply

# Skip Vectorize if it causes errors (orphaned vectors are harmless)
python scripts/maintenance/cleanup_association_memories_hybrid.py --apply --skip-vectorize
```

**How It Works**:
1. Deletes from Cloudflare D1 first (prevents sync restoration)
2. Optionally deletes from Cloudflare Vectorize (orphaned vectors harmless)
3. Deletes from local SQLite
4. Other PCs automatically sync the deletion

**Prerequisites**:
- Cloudflare credentials configured (`CLOUDFLARE_API_TOKEN`, etc.)
- Graph table populated (`backfill_graph_table.py` completed)
- `MCP_GRAPH_STORAGE_MODE=graph_only` set in environment

**Which Script to Use?**

| Backend | Script | Multi-PC |
|---------|--------|----------|
| SQLite-vec only | `cleanup_association_memories.py` | N/A |
| Hybrid (single PC) | Either script works | N/A |
| Hybrid (multi-PC) | `cleanup_association_memories_hybrid.py` | ✅ Required |

See [scripts/maintenance/README.md](../../scripts/maintenance/README.md) for detailed documentation.

## Verification

### Test Graph Queries

```bash
# Test 1: Verify graph table has data
sqlite3 ~/.local/share/mcp-memory-service/memory.db \
  "SELECT COUNT(*) FROM memory_graph;"
# Expected: 1,435 (or your association count × 2 for bidirectional)

# Test 2: Verify indexes exist
sqlite3 ~/.local/share/mcp-memory-service/memory.db \
  "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='memory_graph';"
# Expected:
# idx_graph_source
# idx_graph_target
# idx_graph_bidirectional

# Test 3: Query performance test
time sqlite3 ~/.local/share/mcp-memory-service/memory.db \
  "SELECT target_hash FROM memory_graph WHERE source_hash='YOUR_HASH' LIMIT 10;"
# Expected: <0.01s (10ms)
```

### Test Association Queries

```python
# Python test script (save as test_graph.py)
from mcp_memory_service.storage.graph import GraphStorage
import asyncio
import time

async def test_graph_performance():
    db_path = "~/.local/share/mcp-memory-service/memory.db"
    graph = GraphStorage(db_path)

    # Get a sample hash from your database
    sample_hash = "YOUR_HASH_HERE"

    # Test find_connected
    start = time.time()
    connected = await graph.find_connected(sample_hash, max_hops=2)
    elapsed = time.time() - start

    print(f"find_connected: {len(connected)} results in {elapsed*1000:.2f}ms")
    print(f"✅ Performance: {'PASS' if elapsed < 0.01 else 'FAIL'} (<10ms target)")

asyncio.run(test_graph_performance())
```

**Run the test**:
```bash
python test_graph.py
# Expected output:
# find_connected: 15 results in 4.32ms
# ✅ Performance: PASS (<10ms target)
```

## Rollback Procedure

If you need to revert to the old system:

### Option 1: Switch Back to memories_only Mode

```bash
# Update environment variable
export MCP_GRAPH_STORAGE_MODE=memories_only

# Restart services
systemctl --user restart mcp-memory-http.service
```

**Effect**:
- ✅ Associations written to memories table (old behavior)
- ✅ Graph table ignored but preserved
- ✅ Can re-enable graph mode later

### Option 2: Full Rollback to v8.50.x

```bash
# Stop services
systemctl --user stop mcp-memory-http.service

# Restore database backup
cp ~/.local/share/mcp-memory-service/memory.db.backup-YYYYMMDD \
   ~/.local/share/mcp-memory-service/memory.db

# Downgrade to v8.50.x
pip install mcp-memory-service==8.50.1

# Restart services
systemctl --user start mcp-memory-http.service
```

## Troubleshooting

### Issue: "Database is locked" during migration

**Cause**: HTTP server or other process is using the database

**Solution**:
```bash
# Find processes using the database
lsof | grep memory.db

# Stop HTTP server
systemctl --user stop mcp-memory-http.service

# Or kill specific processes
kill <PID>

# Re-run migration
python scripts/maintenance/backfill_graph_table.py --apply
```

### Issue: "Graph table not created"

**Cause**: Migration 008 not run automatically

**Solution**:
```bash
# Manually create graph table
sqlite3 ~/.local/share/mcp-memory-service/memory.db < \
  src/mcp_memory_service/storage/migrations/008_add_graph_table.sql

# Verify table exists
sqlite3 ~/.local/share/mcp-memory-service/memory.db \
  "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_graph';"
# Expected: memory_graph
```

### Issue: "No associations found" in dry-run

**Cause**: Consolidation system hasn't created associations yet

**Solution**:
```bash
# Check if consolidation is enabled
curl http://127.0.0.1:8000/api/consolidation/status

# If disabled, enable it
export MCP_CONSOLIDATION_ENABLED=true

# Trigger manual consolidation to create associations
curl -X POST http://127.0.0.1:8000/api/consolidation/trigger \
  -H "Content-Type: application/json" \
  -d '{"time_horizon": "weekly", "immediate": true}'

# Wait for completion (check logs)
journalctl --user -u mcp-memory-http.service -f | grep consolidation

# Re-run backfill after associations are created
python scripts/maintenance/backfill_graph_table.py --dry-run
```

### Issue: Cleanup script deletes fewer memories than expected

**Cause**: Conservative verification only deletes memories confirmed in graph table

**Explanation**:
- Script prioritizes data integrity over space savings
- Only deletes memories that have matching entries in graph table
- Orphaned associations (missing metadata during backfill) are preserved
- This is intentional and safe behavior

**Expected behavior**:
- If you have 1,449 association memories but only 138 verified in graph
- Script will delete 138 and preserve 1,311 orphaned associations
- You still get query performance benefits from graph table
- Storage reclamation is smaller but safe

### Issue: Slow queries after migration

**Cause**: Indexes not created or need rebuilding

**Solution**:
```bash
# Rebuild indexes
sqlite3 ~/.local/share/mcp-memory-service/memory.db <<EOF
REINDEX memory_graph;
ANALYZE memory_graph;
EOF

# Verify indexes
sqlite3 ~/.local/share/mcp-memory-service/memory.db \
  "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='memory_graph';"
```

## Best Practices

### Database Maintenance

```bash
# Monthly: VACUUM to reclaim space
sqlite3 ~/.local/share/mcp-memory-service/memory.db "VACUUM;"

# Monthly: ANALYZE for query optimization
sqlite3 ~/.local/share/mcp-memory-service/memory.db "ANALYZE;"

# Weekly: Check database integrity
sqlite3 ~/.local/share/mcp-memory-service/memory.db "PRAGMA integrity_check;"
# Expected: ok
```

### Backup Strategy

```bash
# Before any migration or cleanup
cp ~/.local/share/mcp-memory-service/memory.db \
   ~/.local/share/mcp-memory-service/memory.db.backup-$(date +%Y%m%d-%H%M)

# Automated daily backups (add to crontab)
0 2 * * * cp ~/.local/share/mcp-memory-service/memory.db \
  ~/.local/share/mcp-memory-service/memory.db.backup-$(date +\%Y\%m\%d)

# Keep last 7 days of backups
find ~/.local/share/mcp-memory-service -name "memory.db.backup-*" -mtime +7 -delete
```

### Monitoring

```bash
# Check association count growth
sqlite3 ~/.local/share/mcp-memory-service/memory.db \
  "SELECT COUNT(*)/2 AS associations FROM memory_graph;"
# Divide by 2 because bidirectional edges

# Check database size trend
ls -lh ~/.local/share/mcp-memory-service/memory.db

# Check consolidation activity
curl http://127.0.0.1:8000/api/consolidation/status | jq '.jobs_executed'
```

## FAQ

### Q: Can I use dual_write mode permanently?

**A**: Yes, but not recommended. Dual_write mode:
- ✅ Provides safety during migration
- ✅ Allows rollback to memories_only
- ❌ Uses ~3% more storage (associations in both places)
- ❌ Slower writes (two operations per association)

**Recommendation**: Use dual_write for 1-2 weeks to verify graph queries work correctly, then switch to graph_only.

### Q: What happens to associations created during migration?

**A**: In dual_write mode (default after upgrade):
- ✅ New associations written to both memories + graph tables
- ✅ Backfill script only processes existing associations (skips duplicates)
- ✅ No data loss or duplication

### Q: Can I delete the graph table and go back to memories_only?

**A**: Yes, completely safe:
```bash
# Switch mode
export MCP_GRAPH_STORAGE_MODE=memories_only

# Drop graph table (optional, saves minimal space)
sqlite3 ~/.local/share/mcp-memory-service/memory.db "DROP TABLE IF EXISTS memory_graph;"

# Restart services
systemctl --user restart mcp-memory-http.service
```

### Q: How do I verify migration success?

**A**: Use the verification section above, key indicators:
1. ✅ Graph table has associations (count > 0)
2. ✅ Indexes exist (3 indexes on memory_graph)
3. ✅ Query performance <10ms for 1-hop queries
4. ✅ Consolidation system continues creating new associations
5. ✅ No errors in server logs

### Q: Will this affect my existing memories?

**A**: No impact on regular memories:
- ✅ Only association memories are affected
- ✅ Regular semantic search unchanged
- ✅ Memory retrieval performance unaffected
- ✅ Embeddings and metadata preserved

## Support

If you encounter issues not covered in this guide:

1. **Check logs**: `journalctl --user -u mcp-memory-http.service | tail -100`
2. **Run diagnostics**: `python scripts/validation/diagnose_backend_config.py`
3. **Open an issue**: [GitHub Issues](https://github.com/doobidoo/mcp-memory-service/issues)
4. **Documentation**: [Architecture Guide](../architecture/graph-database-design.md)

---

**Last Updated**: 2025-12-14
**Version**: 1.0 (for MCP Memory Service v8.51.0)
