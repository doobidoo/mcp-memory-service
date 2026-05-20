# Storage Backends - Complete Reference

## Backend Comparison

| Backend | Performance | Use Case | Installation |
|---------|-------------|----------|--------------|
| **Hybrid** ⚡ | **Fast (5ms read)** | **🌟 Production (Recommended)** | `install.py --storage-backend hybrid` |
| **Cloudflare** ☁️ | Network dependent | Cloud-only deployment | `install.py --storage-backend cloudflare` |
| **SQLite-Vec** 🪶 | Fast (5ms read) | Development, single-user local | `install.py --storage-backend sqlite_vec` |

## Hybrid Backend (v6.21.0+) - RECOMMENDED

The **Hybrid backend** provides the best of both worlds - **SQLite-vec speed with Cloudflare persistence**.

### Configuration

```bash
# Enable hybrid backend
export MCP_MEMORY_STORAGE_BACKEND=hybrid

# Hybrid-specific configuration
export MCP_HYBRID_SYNC_INTERVAL=300    # Background sync every 5 minutes
export MCP_HYBRID_BATCH_SIZE=50        # Sync 50 operations at a time
export MCP_HYBRID_SYNC_ON_STARTUP=true # Initial sync on startup

# Drift detection configuration (v8.25.0+)
export MCP_HYBRID_SYNC_UPDATES=true              # Enable metadata sync (default: true)
export MCP_HYBRID_DRIFT_CHECK_INTERVAL=3600      # Seconds between drift checks (default: 1 hour)
export MCP_HYBRID_DRIFT_BATCH_SIZE=100           # Memories to check per scan (default: 100)

# Requires Cloudflare credentials (same as cloudflare backend)
export CLOUDFLARE_API_TOKEN="your-token"
export CLOUDFLARE_ACCOUNT_ID="your-account"
export CLOUDFLARE_D1_DATABASE_ID="your-d1-id"
export CLOUDFLARE_VECTORIZE_INDEX="mcp-memory-index"
```

### Key Benefits

- ✅ **5ms read/write performance** (SQLite-vec speed)
- ✅ **Zero user-facing latency** - Cloud sync happens in background
- ✅ **Multi-device synchronization** - Access memories everywhere
- ✅ **Graceful offline operation** - Works without internet, syncs when available
- ✅ **Automatic failover** - Falls back to SQLite-only if Cloudflare unavailable
- ✅ **Drift detection (v8.25.0+)** - Automatic metadata sync prevents data loss

### Architecture

- **Primary Storage**: SQLite-vec (all user operations)
- **Secondary Storage**: Cloudflare (background sync)
- **Background Service**: Async queue with retry logic and health monitoring

## Database Lock Prevention (v8.9.0+)

**CRITICAL**: After adding `MCP_MEMORY_SQLITE_PRAGMAS` to `.env`, you **MUST restart all servers**:

```bash
# Add to .env
MCP_MEMORY_SQLITE_PRAGMAS=busy_timeout=15000,cache_size=20000

# Restart HTTP server
memory restart

# Restart MCP servers
# Use /mcp in Claude Code to reconnect, or restart Claude Desktop

# Verify
# Check logs for: "Custom pragma from env: busy_timeout=15000"
```

**Why restart is required**: SQLite pragmas are **per-connection**, not global. Long-running servers (days/weeks old) won't pick up new `.env` settings automatically.

**Symptoms of missing pragmas:**
- "database is locked" errors despite v8.9.0+ installation
- `PRAGMA busy_timeout` returns `0` instead of `15000`
- Concurrent HTTP + MCP access fails

## Graph Table Architecture (v8.51.0+)

### Graph Table: Local-Only (NOT Synced to Cloudflare)

**Rationale:**
- **Derived data**: Graph associations can be reconstructed from memory metadata
- **Performance**: Local graph queries are 30x faster (5ms vs 150ms)
- **Complexity**: Cloudflare doesn't have native graph capabilities (no recursive CTEs in D1)
- **Restoration**: Can rebuild graph table locally from association metadata in memories

**Implementation:**
- Graph table exists only in SQLite-vec backend
- Associations stored as regular memories with `tags=['association', 'discovered']`
- Memory metadata contains: `{similarity, connection_types, source_hash, target_hash}`
- Cloudflare syncs the **memory objects**, NOT the graph table

### Cluster Summaries: Sync to Cloudflare

**Rationale:**
- **User-facing content**: Summaries contain valuable semantic information
- **Multi-device access**: Users should see cluster summaries on all devices
- **Restoration strategy**: Graph table can be rebuilt from synced association memories

**Recovery Process** (new device or SQLite database deleted):
1. Pull all memories from Cloudflare (including associations and cluster summaries)
2. Extract association memories (tag='association')
3. Rebuild graph table locally from association metadata (~2-3 seconds for 1,435 associations)
4. Graph queries immediately functional (5ms performance restored)

## Installer Enhancements (v6.16.0+)

- **Interactive backend selection** with usage-based recommendations
- **Automatic Cloudflare credential setup** and `.env` file generation
- **Connection testing** during installation to validate configuration
- **Graceful fallbacks** from cloud to local backends if setup fails

## Multi-PC Configuration

See [`.claude/directives/README.md`](README.md) for multi-PC sync guidelines. Key points:

**Must match across all PCs:**
- Cloudflare credentials (API token, account ID, database ID, vectorize index)
- Content length limits (5000 for both Cloudflare and Hybrid)
- Graph storage mode (graph_only)

**Can differ per PC:**
- SQLite pragmas (adjust per machine specs)
- Archive paths (OS-specific)
- Cache directories (local storage)
