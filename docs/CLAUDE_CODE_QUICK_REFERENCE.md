# Claude Code Quick Reference for MCP Memory Service

**One-page cheat sheet for efficient development with Claude Code**

---

## 🎯 Essential Keybindings

| Key | Action | Use Case |
|-----|--------|----------|
| `Shift+Tab` | Auto-accept edits | Fast iteration on suggested changes |
| `Esc` | Cancel operation | Stop unwanted actions |
| `Ctrl+R` | Verbose output | Debug when things go wrong |
| `#` | Create memory | Store important decisions |
| `@` | Add to context | Include files/dirs (`@src/`, `@tests/`) |
| `!` | Bash mode | Quick shell commands |

---

## 🚀 Common Tasks

### Memory Operations

```bash
# Store information
/memory-store "Hybrid backend uses SQLite primary + Cloudflare secondary"

# Retrieve information
/memory-recall "how to configure Cloudflare backend"

# Check service health
/memory-health
```

### Development Workflow

```bash
# 1. Start with context
@src/mcp_memory_service/storage/
@tests/test_storage.py

# 2. Make changes incrementally
# Accept suggestions with Shift+Tab

# 3. Test immediately
pytest tests/test_storage.py -v

# 4. Document decisions
/memory-store "Changed X because Y"
```

### Backend Configuration

```bash
# Check current backend
python scripts/server/check_http_server.py -v

# Validate configuration
python scripts/validation/validate_configuration_complete.py

# Diagnose issues
python scripts/validation/diagnose_backend_config.py
```

### Synchronization

```bash
# Check sync status
python scripts/sync/sync_memory_backends.py --status

# Preview sync (dry run)
python scripts/sync/sync_memory_backends.py --dry-run

# Execute sync
python scripts/sync/sync_memory_backends.py --direction bidirectional
```

---

## 🏗️ Project-Specific Context

### Key Files to Add

| Purpose | Files to Include |
|---------|-----------------|
| **Storage backends** | `@src/mcp_memory_service/storage/` |
| **MCP protocol** | `@src/mcp_memory_service/server.py` |
| **Web interface** | `@src/mcp_memory_service/web/` |
| **Configuration** | `@.env.example`, `@src/mcp_memory_service/config.py` |
| **Tests** | `@tests/test_*.py` |
| **Scripts** | `@scripts/server/`, `@scripts/sync/` |

### Common Debugging Patterns

```bash
# 1. HTTP Server not responding
python scripts/server/check_http_server.py -v
tasklist | findstr python  # Check if running
scripts/server/start_http_server.bat  # Restart

# 2. Wrong backend active
python scripts/validation/diagnose_backend_config.py
# Check: .env file, environment variables, Claude Desktop config

# 3. Missing memories
python scripts/sync/sync_memory_backends.py --status
# Compare: Cloudflare count vs SQLite count

# 4. Service logs
@http_server.log  # Add to context for troubleshooting
```

---

## 📚 Architecture Quick Reference

### Storage Backends

| Backend | Performance | Use Case | Config Variable |
|---------|-------------|----------|-----------------|
| **Hybrid** ⭐ | 5ms read | Production (recommended) | `MCP_MEMORY_STORAGE_BACKEND=hybrid` |
| **SQLite-vec** | 5ms read | Development, single-user | `MCP_MEMORY_STORAGE_BACKEND=sqlite_vec` |
| **Cloudflare** | Network-dependent | Legacy cloud-only | `MCP_MEMORY_STORAGE_BACKEND=cloudflare` |

### Key Directories

```
mcp-memory-service/
├── src/mcp_memory_service/
│   ├── server.py              # MCP protocol implementation
│   ├── storage/
│   │   ├── base.py            # Abstract storage interface
│   │   ├── sqlite_vec.py      # SQLite-vec backend
│   │   ├── cloudflare.py      # Cloudflare backend
│   │   └── hybrid.py          # Hybrid backend (recommended)
│   ├── web/
│   │   ├── app.py             # FastAPI server
│   │   └── static/            # Dashboard UI
│   └── config.py              # Configuration management
├── scripts/
│   ├── server/                # HTTP server management
│   ├── sync/                  # Backend synchronization
│   └── validation/            # Configuration validation
└── tests/                     # Test suite
```

---

## 🔧 Environment Variables

**Essential Configuration** (in `.env` file):

```bash
# Backend Selection
MCP_MEMORY_STORAGE_BACKEND=hybrid  # hybrid|sqlite_vec|cloudflare

# Cloudflare (required for hybrid/cloudflare backends)
CLOUDFLARE_API_TOKEN=your-token
CLOUDFLARE_ACCOUNT_ID=your-account
CLOUDFLARE_D1_DATABASE_ID=your-d1-id
CLOUDFLARE_VECTORIZE_INDEX=mcp-memory-index

# Hybrid-Specific
MCP_HYBRID_SYNC_INTERVAL=300      # 5 minutes
MCP_HYBRID_BATCH_SIZE=50
MCP_HYBRID_SYNC_ON_STARTUP=true

# HTTP Server
MCP_HTTP_ENABLED=true
MCP_HTTPS_ENABLED=true
MCP_API_KEY=your-generated-key
```

---

## 🐛 Troubleshooting Checklist

### HTTP Server Issues
- [ ] Check if server is running: `python scripts/server/check_http_server.py -v`
- [ ] Review logs: `@http_server.log`
- [ ] Restart server: `scripts/server/start_http_server.bat`
- [ ] Verify port 8000 is free: `netstat -ano | findstr :8000`

### Backend Configuration Issues
- [ ] Run diagnostic: `python scripts/validation/diagnose_backend_config.py`
- [ ] Check `.env` file exists and has correct values
- [ ] Verify Cloudflare credentials are valid
- [ ] Confirm environment variables loaded: check server startup logs

### Missing Memories
- [ ] Check sync status: `python scripts/sync/sync_memory_backends.py --status`
- [ ] Compare memory counts: Cloudflare vs SQLite
- [ ] Run manual sync: `python scripts/sync/sync_memory_backends.py --dry-run`
- [ ] Check for duplicates: Look for content hash matches

### Performance Issues
- [ ] Verify backend: Hybrid should show ~5ms read times
- [ ] Check disk space: Litestream requires adequate space
- [ ] Monitor background sync: Check `http_server.log` for sync logs
- [ ] Review embedding model cache: Should be loaded once

---

## 💡 Pro Tips

### Efficient Context Management

```bash
# Start specific, expand as needed
@src/mcp_memory_service/storage/hybrid.py  # Specific file
@src/mcp_memory_service/storage/           # Whole module if needed

# Remove context when done
# Use Esc to cancel unnecessary context additions
```

### Multi-Step Tasks

```bash
# Always use TodoWrite for complex tasks
# Claude will create and manage task list automatically

# Example: "Implement new backend"
# 1. Research existing backends
# 2. Create new backend class
# 3. Implement abstract methods
# 4. Add configuration
# 5. Write tests
# 6. Update documentation
```

### Testing Strategy

```bash
# Test incrementally
pytest tests/test_storage.py::TestHybridBackend -v

# Run full suite before committing
pytest tests/ -v

# Check coverage
pytest tests/ --cov=src/mcp_memory_service --cov-report=term
```

### Git Workflow with Claude Code

```bash
# Let Claude help with commits
git status  # Claude reviews changes
git diff    # Claude explains changes

# Use semantic commits
git commit -m "feat: add new backend support"
git commit -m "fix: resolve sync timing issue"
git commit -m "docs: update configuration guide"
```

---

## 📖 Additional Resources

- **Full Documentation**: `@CLAUDE.md` (project-specific guide)
- **Global Best Practices**: `~/.claude/CLAUDE.md` (cross-project)
- **Wiki**: https://github.com/doobidoo/mcp-memory-service/wiki
- **Troubleshooting**: See Wiki for comprehensive troubleshooting guide

---

**Last Updated**: 2025-10-08
