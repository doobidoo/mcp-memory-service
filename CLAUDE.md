# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Table of Contents
- [Quick Reference](#quick-reference) - Essential commands and endpoints
- [Overview](#overview) - Project description and purpose
- [Key Commands](#key-commands) - Development, testing, and build commands
- [Architecture](#architecture) - Core components and design patterns
- [CLI Migration Guide](#cli-migration-guide) - Migrating from legacy commands
- [Environment Variables](#environment-variables) - Configuration options
- [Memory Storage Commands](#memory-storage-commands) - Claude Code integration
- [Development Tips](#development-tips) - Best practices and workflows
- [Deployment & Debugging](#deployment--debugging) - Production deployment
- [Git Configuration](#git-configuration) - Automated workflows and cleanup
- [Production Deployment](#production-deployment) - Cloudflare backend setup
- [API Reference](#api-reference) - REST endpoints and MCP operations
- [Performance Tuning](#performance-tuning) - Optimization strategies
- [Monitoring & Observability](#monitoring--observability) - Health checks and logging

## Memory Context Reference (Optional)

This project supports enhanced context through MCP Memory Service integration. If you have a local MCP Memory Service instance running, you can store and retrieve project context to reduce token usage in Claude Code sessions.

### Setup Memory Context (Optional)
1. **Deploy MCP Memory Service**: Follow deployment instructions for your environment
2. **Store Reference Memories**: Use the `distributable-reference` tag for shareable context
3. **Create Local CLAUDE_MEMORY.md**: Add your memory hashes (git-ignored)

### Memory Categories for Storage
- **Project Structure**: Server architecture, file locations, component relationships
- **Key Commands**: Installation, testing, debugging, deployment commands
- **Environment Variables**: Configuration options and platform-specific settings
- **Recent Changes**: Version history, resolved issues, breaking changes
- **Testing Practices**: Framework preferences, test patterns, validation steps

### Local Memory Configuration
Create `CLAUDE_MEMORY.md` (git-ignored) with your memory service details:
```markdown
# Local Memory Hashes
- Memory Service: https://your-instance:port
- Auth: Bearer your-api-key
- Retrieve Command: curl -k -s -X POST https://your-instance:port/mcp -H "Authorization: Bearer your-key" -d '{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "search_by_tag", "arguments": {"tags": ["claude-code-reference"]}}}'
```

### Memory Management Guidelines
- **Review Schedule**: Quarterly review of reference memories for accuracy
- **Distribution**: Use `distributable-reference` tag for team sharing
- **Export Tool**: `./scripts/export_distributable_memories.sh` for network distribution
- **Documentation**: Keep memory hashes in local files, not in version control

## Overview

MCP Memory Service is a Model Context Protocol server that provides semantic memory and persistent storage capabilities for Claude Desktop using ChromaDB and sentence transformers. The project enables long-term memory storage with semantic search across conversations.

## Key Commands

### Development
- **Install dependencies**: `python install.py` (platform-aware installation)
- **Run server**: `uv run memory server`
- **Run tests**: `pytest tests/`
- **Run specific test**: `pytest tests/unit/test_memory_models.py::TestMemoryModel::test_memory_creation`
- **Check environment**: `python scripts/verify_environment.py`
- **Debug with MCP Inspector**: `npx @modelcontextprotocol/inspector uv --directory /path/to/repo run memory server`
- **Check documentation links**: `python scripts/check_documentation_links.py` (validates all internal markdown links)
- **Test Docker functionality**: `python scripts/test_docker_functionality.py` (comprehensive Docker container verification)
- **Find and remove duplicates**: `python scripts/find_duplicates.py --execute` (removes duplicate memories from database)
- **Clean corrupted encoding**: `python scripts/cleanup_corrupted_encoding.py --execute` (removes memories with corrupted emoji encoding)
- **Setup git merge drivers**: `./scripts/setup-git-merge-drivers.sh` (one-time setup for new contributors)
- **Store memory**: `claude /memory-store "content"` - Store information via Claude Code commands (requires command installation)

### Claude Code Memory Awareness (v6.2.3)
- **Install hooks system**: `cd claude-hooks && ./install.sh` (one-command installation)
- **Test hooks system**: `cd claude-hooks && npm test` (10 integration tests, 100% pass rate)
- **Verify project detection**: Test project context detection across multiple languages
- **Memory injection testing**: Validate automatic memory context injection during Claude Code sessions
- **Session consolidation**: Verify session-end hooks capture and store conversation outcomes

### Build & Package
- **Build package**: `python -m build`
- **Install locally**: `pip install -e .`

## Architecture

### Core Components

1. **Server Layer** (`src/mcp_memory_service/server.py`)
   - Implements MCP protocol with async request handlers
   - Global model and embedding caches for performance
   - Handles all memory operations (store, retrieve, search, delete)

2. **Storage Abstraction** (`src/mcp_memory_service/storage/`)
   - `base.py`: Abstract interface for storage backends
   - `chroma.py`: ChromaDB implementation (local/multi-client)
   - `chroma_enhanced.py`: Extended ChromaDB features (time parsing, advanced search)
   - `sqlite_vec.py`: SQLite-Vec implementation (fast, single-client)
   - `cloudflare.py`: Cloudflare backend (Vectorize + D1, production-ready)
   - `http_client.py`: HTTP client abstraction for remote backends

3. **Models** (`src/mcp_memory_service/models/memory.py`)
   - `Memory`: Core dataclass for memory entries
   - `MemoryMetadata`: Metadata structure
   - All models use Python dataclasses with type hints

4. **Configuration** (`src/mcp_memory_service/config.py`)
   - Environment-based configuration
   - Platform-specific optimizations
   - Hardware acceleration detection

5. **Web Interface** (`src/mcp_memory_service/web/`)
   - `app.py`: FastAPI web application with dashboard
   - `api/`: REST API endpoints (memories, search, health)
   - `sse.py`: Server-sent events for real-time updates
   - Web UI at `https://localhost:8443/` for memory management

6. **Claude Code Hooks System** (`claude-hooks/`)
   - `core/session-start.js`: Automatic memory injection hook
   - `core/session-end.js`: Session consolidation and outcome storage
   - `utilities/project-detector.js`: Multi-language project context detection
   - `utilities/memory-scorer.js`: Advanced relevance scoring algorithms
   - `utilities/context-formatter.js`: Memory presentation and formatting

### Key Design Patterns

- **Async/Await**: All I/O operations are async
- **Type Safety**: Comprehensive type hints (Python 3.10+)
- **Error Handling**: Specific exception types with clear messages
- **Caching**: Global caches for models and embeddings to improve performance
- **Platform Detection**: Automatic hardware optimization (CUDA, MPS, DirectML, ROCm)
- **Hook-Based Architecture**: Event-driven memory awareness with session lifecycle management
- **Multi-Factor Scoring**: Advanced algorithms for memory relevance and context selection

### MCP Protocol Operations

Memory operations implemented:
- `store_memory`: Store new memories with tags and metadata
- `retrieve_memory`: Basic retrieval by query
- `recall_memory`: Natural language time-based retrieval
- `search_by_tag`: Tag-based search
- `delete_memory`: Delete specific memories
- `delete_by_tag/tags`: Bulk deletion by tags
- `optimize_db`: Database optimization
- `check_database_health`: Health monitoring
- `debug_retrieve`: Similarity analysis for debugging

### Testing

Tests are organized by type:
- `tests/unit/`: Unit tests for individual components
- `tests/integration/`: Integration tests for full workflows
- `tests/performance/`: Performance benchmarks

Run tests with coverage: `pytest --cov=src/mcp_memory_service tests/`

## CLI Migration Guide

### Migrating from Legacy Commands (v6.3.0+)

The MCP Memory Service CLI has been consolidated to provide a consistent interface. Here's how to migrate from legacy patterns:

#### ✅ Recommended (Current)
```bash
# Primary server command
uv run memory server

# With options
uv run memory server --debug --chroma-path ./custom-path

# Other commands
uv run memory status
uv run memory ingest-directory ./docs
```

#### ⚠️ Deprecated (Still Works with Warnings)
```bash
# Legacy server command (shows deprecation warning)
uv run memory-server --debug

# This will continue to work but shows:
# "DeprecationWarning: The 'memory-server' command is deprecated. 
#  Please use 'memory server' instead."
```

#### ❌ No Longer Supported
```bash
# These patterns have been removed/updated:
uv run memory                    # Now: uv run memory server
python scripts/run_memory_server.py  # Now: uv run memory server
```

#### Migration Checklist
1. **Update scripts**: Replace `memory-server` with `memory server`
2. **Update documentation**: Use `uv run memory server` as the standard pattern
3. **Check CI/CD**: Update deployment scripts to use new commands
4. **Remove legacy patterns**: Phase out deprecated commands before they're removed

#### Backward Compatibility
- The `memory-server` command still works but shows deprecation warnings
- All existing arguments (`--debug`, `--chroma-path`) are supported
- No breaking changes to core functionality
- Migration period allows gradual transition

### Environment Variables

#### Core Configuration
- `MCP_MEMORY_STORAGE_BACKEND`: Storage backend (`sqlite_vec`, `chroma`, `cloudflare`) (default: `sqlite_vec`)
- `MCP_MEMORY_CHROMA_PATH`: ChromaDB storage location (default: `~/.mcp_memory_chroma`)
- `MCP_MEMORY_BACKUPS_PATH`: Backup location (default: `~/.mcp_memory_backups`)
- `LOG_LEVEL`: Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) (default: `INFO`)

#### Web Interface & HTTP
- `MCP_HTTP_ENABLED`: Enable HTTP server (default: `true`)
- `MCP_HTTP_PORT`: HTTP server port (default: `8443`)
- `MCP_HTTPS_ENABLED`: Enable HTTPS with self-signed cert (default: `true`)
- `MCP_API_KEY`: API key for HTTP authentication (optional, generates if unset)
- `MCP_HTTP_AUTO_START`: Auto-start HTTP server with MCP server (default: `true`)

#### Client & Discovery
- `MCP_MEMORY_INCLUDE_HOSTNAME`: Enable automatic machine identification (default: `false`)
  - When enabled, adds client hostname as `source:hostname` tag to stored memories
  - Clients can specify hostname via `client_hostname` parameter or `X-Client-Hostname` header
  - Fallback to server hostname if client doesn't provide one
- `MCP_MDNS_ENABLED`: Enable mDNS service discovery (default: `true`)
- `MCP_MDNS_SERVICE_NAME`: mDNS service name (default: `"MCP Memory Service"`)

#### Cloudflare Backend
- `CLOUDFLARE_API_TOKEN`: Cloudflare API token with Workers and D1 permissions
- `CLOUDFLARE_ACCOUNT_ID`: Cloudflare account ID
- `CLOUDFLARE_VECTORIZE_INDEX`: Vectorize index name (default: `mcp-memory-index`)
- `CLOUDFLARE_D1_DATABASE`: D1 database name (default: `mcp-memory-metadata`)
- `CLOUDFLARE_R2_BUCKET`: Optional R2 bucket for large content storage

#### Memory Consolidation
- `MCP_CONSOLIDATION_ENABLED`: Enable automatic memory consolidation (default: `false`)
- `MCP_CONSOLIDATION_INTERVAL`: Consolidation interval in hours (default: `24`)
- `MCP_CONSOLIDATION_MIN_MEMORIES`: Minimum memories to trigger consolidation (default: `100`)

#### Platform-Specific
- `PYTORCH_ENABLE_MPS_FALLBACK`: Enable MPS fallback on macOS (default: `1`)
- `MCP_MEMORY_USE_ONNX`: Use ONNX runtime for embeddings (default: `false`)
- `HF_HUB_OFFLINE`: Prevent HuggingFace model downloads (default: `1`)
- `TRANSFORMERS_OFFLINE`: Prevent transformer model downloads (default: `1`)

#### API Key Configuration

The `MCP_API_KEY` environment variable enables HTTP API authentication:

```bash
# Generate a secure API key
export MCP_API_KEY="$(openssl rand -base64 32)"

# Or set manually
export MCP_API_KEY="your-secure-api-key-here"
```

When set, all HTTP API requests require the `Authorization: Bearer <api_key>` header. This is essential for production deployments and multi-client setups.

### Platform Support

The codebase includes platform-specific optimizations:
- **macOS**: MPS acceleration for Apple Silicon, CPU fallback for Intel
- **Windows**: CUDA, DirectML, or CPU
- **Linux**: CUDA, ROCm, or CPU

Hardware detection is automatic via `utils/system_detection.py`.

### Memory Storage Commands

#### Claude Code Commands (Recommended)
Install Claude Code commands first: `python scripts/claude_commands_utils.py`

**Basic Usage:**
```bash
claude /memory-store "content to store"
claude /memory-recall "what did we decide about databases?"
claude /memory-search --tags "architecture,database"
claude /memory-health  # Check service status
claude /memory-context  # Store current session context
```

**Advanced Usage:**
```bash
claude /memory-store --tags "decision,architecture" "Database backend choice"
claude /memory-store --type "note" --project "my-app" "Important reminder"
```

**Features:**
- Automatic project context detection and smart tagging
- Git repository information and recent commits capture
- Client hostname tracking for multi-machine usage
- Integrated with MCP Memory Service at `https://narrowbox.local:8443`
- No setup required after command installation

#### Direct API Usage (Alternative)
For direct API access without Claude Code commands:
```bash
curl -k -s -X POST https://narrowbox.local:8443/api/memories \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer mcp-test-key" \
  -d '{"content": "your content", "tags": ["tag1", "tag2"]}'
```

#### Command Installation
To install Claude Code commands:
1. Run: `python scripts/claude_commands_utils.py`
2. Commands will be available as `claude /memory-store`, etc.
3. Test installation: `claude /memory-health`

**Troubleshooting:**
- If commands timeout: Use direct API calls as fallback
- If installation fails: Ensure Claude Code CLI is installed
- For path issues: Commands auto-detect project location

### Development Tips

1. When modifying storage backends, ensure compatibility with the abstract base class
2. Memory operations should handle duplicates gracefully (content hashing)
3. Time parsing supports natural language ("yesterday", "last week")
4. Use the debug_retrieve operation to analyze similarity scores
5. The server maintains global state for models - be careful with concurrent modifications
6. All new features should include corresponding tests
7. Use semantic commit messages for version management
8. Use `claude /memory-store` to capture important decisions and context during development

### Deployment & Debugging

#### Remote Server Deployment Process
When deploying changes to production servers (e.g., 10.0.1.30:8443):

1. **Pre-deployment**: Test changes locally first
2. **Deploy**: SSH to server, `git pull`, restart service
3. **Verify**: Check `/api/health` for new version number
4. **Test**: Verify web frontend loads at https://server:8443/

#### Web Frontend Debugging
Common issues and solutions for internal server errors:

**Python Syntax Errors in HTML Templates**:
- **Problem**: F-strings or `.format()` with CSS cause syntax errors
- **Symptom**: `SyntaxError: invalid decimal literal` or `KeyError` with CSS fragments
- **Solution**: Use string concatenation instead: `"text" + variable + "more text"`
- **Example**: Replace `f"CSS {color: blue;}"` with `"CSS " + color_var + ": blue;"`

**Cache-Related Issues**:
- **Problem**: Code changes not taking effect despite restart
- **Symptoms**: Old behavior persists, web frontend shows errors
- **Solution**: Clear all Python caches:
  ```bash
  # Clear Python bytecode cache
  find . -name "*.pyc" -delete
  find . -name "__pycache__" -type d -exec rm -rf {} +
  
  # Clear UV cache (can be several GB)
  uv cache clean
  
  # Kill all running server processes
  pkill -f "run_server.py"
  pkill -f "python -m src.mcp_memory_service.server"
  
  # Restart with fresh environment
  ```

**Environment Reset Procedure**:
When persistent issues occur:
1. Stop all services: `sudo systemctl stop mcp-memory-service`
2. Clear caches (see above)
3. Reset UV environment: `rm -rf .venv && uv sync`
4. Restart service: `sudo systemctl start mcp-memory-service`
5. Monitor logs: `journalctl -u mcp-memory-service -f`

#### Version Mismatch Issues
- **Problem**: Health endpoints show old version after deployment
- **Check**: Verify `__version__` import in affected files
- **Files to verify**: `server.py`, `web/app.py`, `web/api/health.py`
- **Solution**: Ensure all files import `from . import __version__` or `from ... import __version__`

#### Backend Method Compatibility
- **Problem**: Web API calls methods not available in storage backend
- **Example**: API calls `search_by_tags` but backend only has `search_by_tag`
- **Solution**: Implement missing methods with backward compatibility
- **Check**: Verify API endpoints match storage backend method names

### Git Configuration

#### Automated uv.lock Conflict Resolution

The repository includes automated resolution for `uv.lock` conflicts:

1. **For new contributors**: Run `./scripts/setup-git-merge-drivers.sh` once after cloning
2. **How it works**: 
   - Git automatically resolves `uv.lock` conflicts using the incoming version
   - Then runs `uv sync` to regenerate the lock file based on your `pyproject.toml`
   - Ensures consistent dependency resolution across all environments

Files involved:
- `.gitattributes`: Defines merge strategy for `uv.lock`
- `scripts/uv-lock-merge.sh`: Custom merge driver script
- `scripts/setup-git-merge-drivers.sh`: One-time setup for contributors

#### Feature Branch Lifecycle and Cleanup

Follow this complete workflow for feature development to maintain repository cleanliness:

**1. Feature Development**
```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Develop and commit changes
git add .
git commit -m "feat: implement your feature"
```

**2. Integration Process**
```bash
# Update to latest main before integration
git fetch origin
git rebase origin/main

# Resolve any conflicts during rebase
# Test thoroughly after rebase
```

**3. Pull Request Workflow**
```bash
# Push feature branch
git push -u origin feature/your-feature-name

# Create PR via GitHub or gh CLI
gh pr create --title "feat: your feature" --body "Description"

# Address code review feedback
# Push additional commits if needed
```

**4. Post-Merge Cleanup (Essential)**
```bash
# After PR is merged to main, clean up immediately:

# 1. Switch back to main
git checkout main
git pull origin main

# 2. Delete local feature branch
git branch -d feature/your-feature-name

# 3. Delete remote feature branch
git push origin --delete feature/your-feature-name

# 4. Verify cleanup complete
git branch -a  # Should not show deleted feature branch
```

**5. Release Management**
```bash
# For version releases, follow semantic versioning:

# Update version files
# pyproject.toml: version = "x.y.z" 
# src/mcp_memory_service/__init__.py: __version__ = "x.y.z"

# Update CHANGELOG.md with release notes
# Commit version changes
git add pyproject.toml src/mcp_memory_service/__init__.py CHANGELOG.md
git commit -m "chore: bump version to vx.y.z"

# Create annotated release tag
git tag -a vx.y.z -m "Release vx.y.z: Brief description"

# Push commits and tags
git push origin main
git push origin vx.y.z

# Create GitHub release (optional)
gh release create vx.y.z --title "vx.y.z" --notes "Release notes"
```

**Branch Cleanup Best Practices:**
- Always delete feature branches after successful merge
- Never leave stale feature branches in the repository
- Use `git branch -a` regularly to check for cleanup opportunities
- Feature branches should be short-lived (days to weeks, not months)
- Clean up both local and remote branches to avoid confusion

This workflow ensures:
- Clean git history without stale branches
- Clear feature development lifecycle
- Proper release management with semantic versioning
- Team coordination through standardized practices

### Claude Code Commands Troubleshooting

#### Installation Issues
1. **Command not found**: Ensure Claude Code CLI is installed
   ```bash
   # Check Claude Code installation
   claude --version
   
   # Install commands if Claude CLI is available
   python scripts/claude_commands_utils.py
   ```

2. **Commands directory not accessible**:
   ```bash
   # Check permissions on ~/.claude/commands
   ls -la ~/.claude/commands
   
   # Create directory if missing
   mkdir -p ~/.claude/commands
   ```

3. **Installation script fails**:
   ```bash
   # Run with verbose output for debugging
   python scripts/claude_commands_utils.py --verbose
   
   # Check for existing backups
   ls ~/.claude/commands/backup_*
   ```

#### Command Execution Issues
1. **Commands timeout or hang**:
   - Use direct API calls as fallback:
     ```bash
     curl -k -s -X POST https://narrowbox.local:8443/api/memories \
       -H "Content-Type: application/json" \
       -H "Authorization: Bearer mcp-test-key" \
       -d '{"content": "your content", "tags": ["tag1"]}'
     ```

2. **Path detection errors** (e.g., "Path /home/user/Repositories not found"):
   - Commands auto-detect repository location
   - Ensure you're in a valid git repository
   - Check that MCP Memory Service is running at `https://narrowbox.local:8443`

3. **Authentication failures**:
   - Verify MCP Memory Service is accessible: `curl -k https://narrowbox.local:8443/api/health`
   - Check API key configuration in service
   - Test with `claude /memory-health` command

#### Service Connectivity Issues
1. **Service not responding**:
   ```bash
   # Check service status
   systemctl status mcp-memory-service  # Linux
   ps aux | grep memory-service         # Process check
   
   # Test direct connectivity
   curl -k https://narrowbox.local:8443/api/health
   ```

2. **SSL/TLS errors**:
   - Commands use `-k` flag to accept self-signed certificates
   - Ensure service is running on HTTPS port 8443
   - Check firewall settings for port access

3. **Wrong endpoint**:
   - Default endpoint: `https://narrowbox.local:8443`
   - Update commands if using different hostname/port
   - Check MDNS resolution: `avahi-resolve-host-name narrowbox.local`

#### Quick Fixes
- **Reset command installation**: `python scripts/claude_commands_utils.py --uninstall && python scripts/claude_commands_utils.py`
- **Test basic functionality**: `claude /memory-health`
- **Use direct API**: Always available as fallback method
- **Check service logs**: `journalctl -u mcp-memory-service -f`

## Production Deployment

### Cloudflare Backend Setup

The Cloudflare backend provides enterprise-grade scalability with global edge distribution:

#### Prerequisites
1. **Cloudflare Account**: Workers, D1, and Vectorize enabled
2. **API Token**: Create with permissions for Workers, D1, and Vectorize
3. **Resources**: Pre-create D1 database and Vectorize index

#### Configuration
```bash
# Required environment variables
export CLOUDFLARE_API_TOKEN="your-api-token"
export CLOUDFLARE_ACCOUNT_ID="your-account-id"
export CLOUDFLARE_VECTORIZE_INDEX="mcp-memory-index"
export CLOUDFLARE_D1_DATABASE="mcp-memory-metadata"

# Optional: Large content storage
export CLOUDFLARE_R2_BUCKET="mcp-memory-content"

# Enable Cloudflare backend
export MCP_MEMORY_STORAGE_BACKEND="cloudflare"
```

#### Resource Creation
```bash
# Create required Cloudflare resources
python scripts/setup_cloudflare_resources.py

# Or manually:
# wrangler d1 create mcp-memory-metadata
# wrangler vectorize create mcp-memory-index --dimensions=384
```

#### Deployment Checklist
- [ ] API token configured with correct permissions
- [ ] D1 database created and accessible
- [ ] Vectorize index created with 384 dimensions
- [ ] R2 bucket created (optional, for large content)
- [ ] Service deployed with environment variables
- [ ] Health endpoint accessible: `/api/health`
- [ ] Performance baseline established

### Multi-Client Production Setup

For production environments with multiple clients:

```bash
# Enable production features
export MCP_HTTP_ENABLED=true
export MCP_HTTPS_ENABLED=true
export MCP_API_KEY="$(openssl rand -base64 32)"
export MCP_MDNS_ENABLED=true
export MCP_CONSOLIDATION_ENABLED=true

# Start production server
uv run memory server
```

#### Load Balancing
- Use nginx/HAProxy for SSL termination and load balancing
- Configure health checks against `/api/health`
- Enable sticky sessions for WebSocket connections
- Monitor memory usage and response times

#### Security Best Practices
- Always use HTTPS in production (`MCP_HTTPS_ENABLED=true`)
- Generate strong API keys (`openssl rand -base64 32`)
- Use Cloudflare Access for additional authentication layers
- Enable request rate limiting
- Monitor API usage and implement quotas

## API Reference

### REST API Endpoints

#### Memory Operations
```http
POST /api/memories
Content-Type: application/json
Authorization: Bearer <api_key>

{
  "content": "Memory content",
  "tags": ["tag1", "tag2"],
  "metadata": {
    "type": "note",
    "project": "my-project"
  }
}
```

```http
GET /api/memories/search?query=text&limit=10&tags=tag1,tag2
Authorization: Bearer <api_key>
```

```http
DELETE /api/memories/<hash>
Authorization: Bearer <api_key>
```

#### Health & Status
```http
GET /api/health
# Response: {"status": "healthy", "version": "6.2.3", "backend": "cloudflare"}

GET /api/status
Authorization: Bearer <api_key>
# Response: {"memories_count": 1234, "tags_count": 56, "disk_usage": "1.2GB"}
```

### Server-Sent Events (SSE)

Real-time memory updates:
```javascript
const eventSource = new EventSource('/api/events', {
  headers: { 'Authorization': 'Bearer ' + apiKey }
});

eventSource.addEventListener('memory_stored', (event) => {
  const memory = JSON.parse(event.data);
  console.log('New memory:', memory);
});

eventSource.addEventListener('memory_deleted', (event) => {
  const { hash } = JSON.parse(event.data);
  console.log('Deleted memory:', hash);
});
```

### MCP Protocol Operations

Standard MCP operations for desktop clients:
- `store_memory(content, tags, metadata)`: Store new memory
- `retrieve_memory(query, limit)`: Basic semantic search
- `recall_memory(query, time_filter)`: Time-aware retrieval  
- `search_by_tag(tags)`: Tag-based filtering
- `delete_memory(hash)`: Remove specific memory
- `check_database_health()`: Health diagnostics
- `optimize_db()`: Database maintenance

## Performance Tuning

### Backend Performance Characteristics

| Backend | Read Latency | Write Latency | Memory Usage | Scalability | Best For |
|---------|--------------|---------------|--------------|-------------|----------|
| SQLite-Vec | ~5ms | ~10ms | Low (50MB) | Single-client | Development, personal use |
| ChromaDB | ~15ms | ~25ms | Medium (200MB) | Multi-client | Team collaboration |
| Cloudflare | ~50ms* | ~100ms* | Minimal | Global | Production, enterprise |

*Network dependent, includes global edge caching benefits

### Optimization Strategies

#### Memory Caching
```bash
# Enable model caching (default: true)
export MCP_MEMORY_CACHE_MODELS=true

# Increase embedding cache size
export MCP_MEMORY_EMBEDDING_CACHE_SIZE=1000

# Enable query result caching
export MCP_MEMORY_QUERY_CACHE_TTL=3600
```

#### Database Optimization
```bash
# Run periodic optimization
claude /memory-optimize

# Or via script
python scripts/cleanup_memories.py --optimize

# SQLite-Vec specific tuning
export SQLITE_VEC_BATCH_SIZE=100
export SQLITE_VEC_CACHE_SIZE=2000
```

#### Hardware Acceleration
```bash
# GPU acceleration (NVIDIA)
export CUDA_VISIBLE_DEVICES=0

# Apple Silicon (MPS)
export PYTORCH_ENABLE_MPS_FALLBACK=1

# CPU optimization
export OMP_NUM_THREADS=4
export TORCH_NUM_THREADS=4
```

### Load Testing

Performance benchmarking:
```bash
# Run comprehensive performance tests
pytest tests/performance/ -v

# Memory operation benchmarks
python scripts/run_complete_test.py --benchmark

# Concurrent client testing
python tests/integration/test_concurrent_clients.py
```

## Monitoring & Observability

### Health Monitoring

#### Health Check Endpoints
```bash
# Basic health check
curl https://localhost:8443/api/health

# Detailed system status  
curl -H "Authorization: Bearer $MCP_API_KEY" \
     https://localhost:8443/api/status

# Database health diagnostics
claude /memory-health
```

#### System Metrics
Monitor these key metrics:
- **Response Time**: API endpoint latency (target: <100ms)
- **Memory Usage**: Process RAM consumption
- **Database Size**: Storage growth rate
- **Error Rate**: Failed operations percentage
- **Active Connections**: Concurrent client count

#### Logging Configuration
```bash
# Enable detailed logging
export LOG_LEVEL=DEBUG

# Log to file
export MCP_LOG_FILE="/var/log/mcp-memory.log"

# Structured logging for monitoring systems
export MCP_LOG_FORMAT=json
```

### Production Monitoring

#### Systemd Service Monitoring
```bash
# Check service status
systemctl status mcp-memory-service

# View real-time logs
journalctl -u mcp-memory-service -f

# Check for errors
journalctl -u mcp-memory-service --since "1 hour ago" | grep ERROR
```

#### Automated Health Checks
```bash
# Setup cron-based health monitoring
# Add to crontab:
*/5 * * * * curl -f https://localhost:8443/api/health || systemctl restart mcp-memory-service
```

#### Performance Alerts
Set up monitoring alerts for:
- Response time > 500ms
- Memory usage > 1GB
- Disk usage > 80%
- Error rate > 1%
- Service downtime

### Backup & Recovery

#### Automated Backups
```bash
# Setup backup cron job
python scripts/setup_backup_cron.sh

# Manual backup
python scripts/backup_memories.py --compress

# Restore from backup
python scripts/restore_memories.py backup_2024-01-01.tar.gz
```

#### Database Synchronization
```bash
# Export memories for distribution
python scripts/export_distributable_memories.sh

# Sync between environments
python scripts/sync/export_memories.py --target production
python scripts/sync/import_memories.py --source production
```

### Common Issues

1. **MPS Fallback**: On macOS, if MPS fails, set `PYTORCH_ENABLE_MPS_FALLBACK=1`
2. **ONNX Runtime**: For compatibility issues, use `MCP_MEMORY_USE_ONNX=true`
3. **ChromaDB Persistence**: Ensure write permissions for storage paths
4. **Memory Usage**: Model loading is deferred until first use to reduce startup time
5. **uv.lock Conflicts**: Should resolve automatically; if not, ensure git merge drivers are set up

### GitHub Integration

6. **GitHub Discussion Content**: Can import GitHub discussions, issues, and PR content via GraphQL API into memory for project context
7. **Content Classification**: Marketing content and strategic decisions should be stored in MEMORY (not committed to git) using `claude /memory-store` commands

## Storage Backend Comparison

| Backend | Performance | Scalability | Features | Use Case |
|---------|------------|-------------|----------|----------|
| **SQLite-Vec** | Fast | Medium | Full-text, vector search | Single-client, development |
| **ChromaDB** | Fast | Medium | Advanced metadata | Multi-client, research |
| **Cloudflare** | Very Fast | High | Global distribution, analytics | Production, enterprise |

### Backend Selection Guidelines
- **Development**: SQLite-Vec (default, fastest setup)
- **Multi-client**: ChromaDB (best local sharing)
- **Production**: Cloudflare (scalable, distributed)

## Quick Reference

### Essential Commands
```bash
# Setup
python install.py
uv run memory server

# Memory Operations
claude /memory-store "content"
claude /memory-recall "query"
claude /memory-health

# Testing
pytest tests/

# Development
uv run memory server
```

### Key Endpoints
- **Health**: `https://localhost:8443/api/health`
- **Web UI**: `https://localhost:8443/`
- **API**: `https://localhost:8443/api/memories`