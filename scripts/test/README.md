# Test Environment Scripts

**CRITICAL:** These scripts protect production data during manual testing.

## ⚠️ NEVER Test Against Production Database!

**Always use these scripts before manual testing to prevent data loss.**

---

## Quick Start

### 1. Backup Production (MANDATORY before testing)

```bash
./scripts/test/backup-before-test.sh
```

**What it does:**
- Finds production database automatically
- Creates timestamped backup
- Shows memory count and database size
- Generates restore script for emergency use

**Output:**
- Backup: `~/Library/Application Support/mcp-memory/backups/manual_backup_YYYYMMDD_HHMMSS.db`
- Restore script: `backups/restore_YYYYMMDD_HHMMSS.sh`

---

### 2. Setup Test Environment

```bash
source scripts/test/setup-test-environment.sh
```

**IMPORTANT:** Use `source` or `.` to apply environment variables to your current shell.

**What it does:**
- Creates isolated test database in `test_data/`
- Configures test environment variables:
  - `MCP_MEMORY_SQLITE_PATH`: Test database path
  - `MCP_HTTP_PORT`: 8001 (different from production)
  - `MCP_API_KEY`: test-key-12345
  - `MCP_ALLOW_ANONYMOUS_ACCESS`: false

**Test Server:**
```bash
# After sourcing setup script:
memory server --http

# Access dashboard:
open http://localhost:8001/

# Authenticate with: test-key-12345
```

---

### 3. Run Your Tests

Test against `http://localhost:8001/` (NOT port 8000)

**Verify isolation:**
```bash
echo $MCP_MEMORY_SQLITE_PATH
# Should show: /path/to/project/test_data/test_TIMESTAMP/test_memories.db

echo $MCP_HTTP_PORT
# Should show: 8001
```

---

### 4. Cleanup After Testing

```bash
./scripts/test/cleanup-test-environment.sh
```

**What it does:**
- Stops test server (port 8001)
- Removes test databases (with confirmation)
- Shows commands to reset environment

**Manual reset:**
```bash
unset MCP_MEMORY_SQLITE_PATH MCP_HTTP_PORT MCP_API_KEY
export MCP_ALLOW_ANONYMOUS_ACCESS=true
memory server --http
```

---

## Emergency: Restore from Backup

### Option 1: Quick Restore Script

```bash
# Run the auto-generated restore script
~/Library/Application\ Support/mcp-memory/backups/restore_YYYYMMDD_HHMMSS.sh
```

### Option 2: Manual Restore

```bash
# 1. Stop server
pkill -f "memory server"

# 2. Find backup
ls -lht ~/Library/Application\ Support/mcp-memory/backups/

# 3. Restore (replace TIMESTAMP)
cp ~/Library/Application\ Support/mcp-memory/backups/manual_backup_TIMESTAMP.db \
   ~/Library/Application\ Support/mcp-memory/sqlite_vec.db

# 4. Remove WAL files
rm -f ~/Library/Application\ Support/mcp-memory/sqlite_vec.db-{shm,wal}

# 5. Restart server
memory server --http
```

---

## File Overview

| Script | Purpose | Usage |
|--------|---------|-------|
| `backup-before-test.sh` | Create production backup | `./scripts/test/backup-before-test.sh` |
| `setup-test-environment.sh` | Configure test environment | `source scripts/test/setup-test-environment.sh` |
| `cleanup-test-environment.sh` | Remove test data | `./scripts/test/cleanup-test-environment.sh` |

---

## Testing Workflow (Complete)

```bash
# STEP 1: Backup production (MANDATORY)
./scripts/test/backup-before-test.sh

# STEP 2: Setup test environment
source scripts/test/setup-test-environment.sh

# STEP 3: Start test server
memory server --http
# Opens on http://localhost:8001

# STEP 4: Run your tests
# - Browser testing at http://localhost:8001
# - API testing with curl localhost:8001/api/*
# - Authentication: test-key-12345

# STEP 5: Stop test server
pkill -f "memory server"

# STEP 6: Cleanup test environment
./scripts/test/cleanup-test-environment.sh

# STEP 7: Reset to production
unset MCP_MEMORY_SQLITE_PATH MCP_HTTP_PORT MCP_API_KEY
export MCP_ALLOW_ANONYMOUS_ACCESS=true
memory server --http
```

---

## Common Mistakes to Avoid

❌ **DON'T:**
- Start server without running setup script
- Test on port 8000 (production)
- Skip backup before testing
- Forget to cleanup after testing
- Use production database path

✅ **DO:**
- Always create backup first
- Always source setup-test-environment.sh
- Always test on port 8001
- Always cleanup after testing
- Verify test isolation before running tests

---

## Verification Checklist

Before running tests, verify:

- [ ] Backup created: `ls ~/Library/Application\ Support/mcp-memory/backups/`
- [ ] Test environment active: `echo $MCP_MEMORY_SQLITE_PATH` (should show test_data/)
- [ ] Test port configured: `echo $MCP_HTTP_PORT` (should show 8001)
- [ ] Server running on test port: `curl http://localhost:8001/api/health`
- [ ] Production server stopped: `curl http://localhost:8000/api/health` (should fail)

---

## Troubleshooting

**Problem:** Can't find production database

**Solution:**
```bash
# Manually specify location
export PROD_DB="/path/to/sqlite_vec.db"
./scripts/test/backup-before-test.sh
```

---

**Problem:** Test server on wrong port

**Solution:**
```bash
# Re-source setup script
source scripts/test/setup-test-environment.sh

# Verify
echo $MCP_HTTP_PORT  # Should be 8001
```

---

**Problem:** Accidentally tested on production

**Solution:**
```bash
# 1. Stop all servers immediately
pkill -f "memory server"

# 2. Find most recent backup
ls -lht ~/Library/Application\ Support/mcp-memory/backups/

# 3. Restore from backup (see Emergency Restore above)

# 4. Document what happened in memory system
```

---

## Questions?

See main documentation:
- `CLAUDE.md` - Development guidelines
- `docs/testing/` - Testing best practices (to be created)
- `test_auth_implementation.md` - Dashboard auth testing guide

---

**Remember:** These scripts exist to protect your data. Use them every time! 🛡️
