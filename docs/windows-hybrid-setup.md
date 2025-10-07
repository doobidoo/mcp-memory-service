# Windows Hybrid Backend Configuration

**Machine**: Windows PC (heinrich.krupp)
**Date**: October 6, 2025
**Version**: v8.3.0
**Storage Backend**: Hybrid (SQLite-vec + Cloudflare)

## Configuration Overview

All three Claude environments are configured with **identical hybrid backend settings**:

### 1. Claude Code (MCP Server)
**Configuration File**: `C:\Users\heinrich.krupp\.claude.json` (lines 2944-2967)

```json
{
  "memory": {
    "type": "stdio",
    "command": "uv",
    "args": [
      "--directory",
      "C:/REPOSITORIES/mcp-memory-service",
      "run",
      "python",
      "-m",
      "mcp_memory_service.server"
    ],
    "env": {
      "MCP_MEMORY_STORAGE_BACKEND": "hybrid",
      "CLOUDFLARE_API_TOKEN": "Y9qwW1rYkwiE63iWYASxnzfTQlIn-mtwCihRTwZa",
      "CLOUDFLARE_ACCOUNT_ID": "be0e35a26715043ef8df90253268c33f",
      "CLOUDFLARE_D1_DATABASE_ID": "f745e9b4-ba8e-4d47-b38f-12af91060d5a",
      "CLOUDFLARE_VECTORIZE_INDEX": "mcp-memory-index",
      "MCP_HYBRID_SYNC_INTERVAL": "300",
      "MCP_HYBRID_BATCH_SIZE": "50",
      "MCP_HYBRID_SYNC_ON_STARTUP": "true",
      "MCP_MEMORY_BACKUPS_PATH": "C:\\Users\\heinrich.krupp\\AppData\\Local\\mcp-memory\\backups",
      "MCP_MEMORY_SQLITE_PATH": "C:\\Users\\heinrich.krupp\\AppData\\Local\\mcp-memory\\backups\\sqlite_vec.db"
    }
  }
}
```

**Status**: ✅ Connected
**Verification**: `claude mcp list` shows memory server connected

### 2. Claude Desktop
**Configuration File**: `C:\Users\heinrich.krupp\AppData\Roaming\Claude\claude_desktop_config.json` (lines 3-20)

```json
{
  "mcpServers": {
    "memory": {
      "command": "uv",
      "args": [
        "--directory", "C:/REPOSITORIES/mcp-memory-service",
        "run", "python", "-m", "mcp_memory_service.server"
      ],
      "env": {
        "MCP_MEMORY_BACKUPS_PATH": "C:\\Users\\heinrich.krupp\\AppData\\Local\\mcp-memory\\backups",
        "MCP_MEMORY_STORAGE_BACKEND": "hybrid",
        "MCP_MEMORY_SQLITE_PATH": "C:\\Users\\heinrich.krupp\\AppData\\Local\\mcp-memory\\backups\\sqlite_vec.db",
        "CLOUDFLARE_API_TOKEN": "Y9qwW1rYkwiE63iWYASxnzfTQlIn-mtwCihRTwZa",
        "CLOUDFLARE_ACCOUNT_ID": "be0e35a26715043ef8df90253268c33f",
        "CLOUDFLARE_D1_DATABASE_ID": "f745e9b4-ba8e-4d47-b38f-12af91060d5a",
        "CLOUDFLARE_VECTORIZE_INDEX": "mcp-memory-index",
        "MCP_HYBRID_SYNC_INTERVAL": "300",
        "MCP_HYBRID_BATCH_SIZE": "50",
        "MCP_HYBRID_SYNC_ON_STARTUP": "true"
      }
    }
  }
}
```

**Status**: ✅ Configured
**Restart Required**: After config changes

### 3. HTTP Dashboard Server
**Configuration File**: `C:\REPOSITORIES\mcp-memory-service\.env`

```env
MCP_MEMORY_STORAGE_BACKEND=hybrid

# Cloudflare Configuration
CLOUDFLARE_API_TOKEN=Y9qwW1rYkwiE63iWYASxnzfTQlIn-mtwCihRTwZa
CLOUDFLARE_ACCOUNT_ID=be0e35a26715043ef8df90253268c33f
CLOUDFLARE_D1_DATABASE_ID=f745e9b4-ba8e-4d47-b38f-12af91060d5a
CLOUDFLARE_VECTORIZE_INDEX=mcp-memory-index

# Hybrid Backend Settings
MCP_HYBRID_SYNC_INTERVAL=300
MCP_HYBRID_BATCH_SIZE=50
MCP_HYBRID_SYNC_ON_STARTUP=true

# Paths
MCP_MEMORY_BACKUPS_PATH=C:\Users\heinrich.krupp\AppData\Local\mcp-memory\backups
MCP_MEMORY_SQLITE_PATH=C:\Users\heinrich.krupp\AppData\Local\mcp-memory\backups\sqlite_vec.db
```

**Status**: ✅ Running on port 8000
**Start Command**: `MCP_MEMORY_STORAGE_BACKEND=hybrid uv run python scripts/server/run_http_server.py`

## Dashboard Access

- **URL**: http://localhost:8000/
- **API Docs**: http://localhost:8000/api/docs
- **Health Check**: http://localhost:8000/api/health

## Storage Locations

### Primary Storage (SQLite-vec)
**Path**: `C:\Users\heinrich.krupp\AppData\Local\mcp-memory\backups\sqlite_vec.db`
**Type**: Local, fast (5ms read/write)
**Purpose**: All user-facing operations

### Secondary Storage (Cloudflare)
**D1 Database**: `f745e9b4-ba8e-4d47-b38f-12af91060d5a`
**Vectorize Index**: `mcp-memory-index`
**Account**: `be0e35a26715043ef8df90253268c33f`
**Purpose**: Cloud persistence, multi-device sync

## Sync Status (October 6, 2025)

### Initial Sync Progress
- **Cloudflare Total**: 1,309 memories
- **Local SQLite**: 1,139 memories (86% complete)
- **Synced**: 81 memories since startup
- **Remaining**: 170 memories
- **Status**: ✅ Actively syncing in background

### Sync Configuration (v8.3.0+)
- **Interval**: 300 seconds (5 minutes)
- **Batch Size**: 50 memories per batch
- **Startup Sync**: Enabled
- **Max Empty Batches**: 20 (configurable via `MCP_HYBRID_MAX_EMPTY_BATCHES`)
- **Min Check Count**: 1000 (configurable via `MCP_HYBRID_MIN_CHECK_COUNT`)

### Performance Notes
- Initial sync processes ~10 memories every 30-40 seconds
- Each memory requires embedding generation (all-MiniLM-L6-v2 model)
- Cloudflare API rate limiting may slow sync
- Background sync continues while server is accessible
- Expected completion time: ~5-10 minutes for 170 remaining memories

## Verification Commands

### Check Memory Counts
```bash
# Local SQLite count
python -c "import sqlite3; conn = sqlite3.connect(r'C:\Users\heinrich.krupp\AppData\Local\mcp-memory\backups\sqlite_vec.db'); cursor = conn.cursor(); cursor.execute('SELECT COUNT(*) FROM memories'); print(f'Local: {cursor.fetchone()[0]}'); conn.close()"

# Cloudflare count
curl -s "https://api.cloudflare.com/client/v4/accounts/be0e35a26715043ef8df90253268c33f/d1/database/f745e9b4-ba8e-4d47-b38f-12af91060d5a/query" -H "Authorization: Bearer Y9qwW1rYkwiE63iWYASxnzfTQlIn-mtwCihRTwZa" -H "Content-Type: application/json" -d '{"sql":"SELECT COUNT(*) as count FROM memories"}' | python -c "import sys, json; data = json.load(sys.stdin); print(f'Cloudflare: {data[\"result\"][0][\"results\"][0][\"count\"]}')"

# Dashboard API count
curl -s "http://127.0.0.1:8000/api/memories?page=1&page_size=1" | python -c "import sys, json; data = json.load(sys.stdin); print(f'Dashboard: {data[\"total\"]}')"
```

### Check HTTP Server Health
```bash
curl -s http://127.0.0.1:8000/api/health | python -m json.tool
```

### Check MCP Connection
```bash
claude mcp list | grep memory
```

## Troubleshooting

### If Dashboard Shows Fewer Memories
1. ✅ Verify HTTP server is running: `curl http://localhost:8000/api/health`
2. ✅ Check sync is active: Look for "Initial sync progress" in server logs
3. ✅ Wait for sync to complete: Background process takes 5-10 minutes
4. ❌ **Don't restart server** - This will reset sync progress

### If Sync Stops
1. Check server logs for errors: `BashOutput <server-id>`
2. Verify Cloudflare credentials are correct
3. Check network connectivity
4. Look for "Completed sync" or early break condition messages

### Port Conflicts
- HTTP Server default port changed from 8888 to 8000 in v8.3.0
- Update any bookmarks or scripts to use port 8000
- Dashboard hooks may need endpoint update

## Update History

### October 6, 2025 - v8.3.0 Upgrade
- Upgraded from v7.4.0 to v8.3.0 (50 commits ahead)
- Reset to remote develop branch
- Reinstalled in editable mode: `uv pip install -e .`
- Restarted HTTP server with hybrid backend
- Verified all three environments use identical configuration

### Configuration Changes
- ✅ All environments use `hybrid` backend
- ✅ Cloudflare credentials consistent across all configs
- ✅ SQLite path standardized to backups directory
- ✅ Sync settings unified (300s interval, 50 batch size)

## Next Steps

1. ⏳ Wait for initial sync to complete (170 memories remaining)
2. 📝 Update memory awareness hooks to use correct endpoint
3. ✅ Verify sync completion with count comparison
4. 📊 Monitor dashboard for full memory availability
