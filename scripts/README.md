# MCP Memory Service Scripts

This directory contains organized utility scripts for maintaining, managing, and operating the MCP Memory Service. Scripts are categorized by function for easy navigation and maintenance.

## 📁 Directory Structure

```
scripts/
├── backup/          # Backup and restore operations
├── database/        # Database analysis and health monitoring
├── development/     # Development tools and debugging utilities
├── installation/    # Setup and installation scripts
├── linux/           # Linux service management shortcuts (v7.5.1+)
├── maintenance/     # Database cleanup and repair operations
├── migration/       # Data migration and schema updates
├── server/          # Server runtime and operational scripts
├── service/         # Service management and deployment
├── sync/            # Backend synchronization utilities
├── testing/         # Test scripts and validation
├── utils/           # General utility scripts and wrappers
├── validation/      # Configuration and system validation
├── run/             # Runtime execution scripts
├── archive/         # Deprecated scripts (kept for reference)
└── README.md        # This file
```

## 🚀 Quick Reference

### Essential Daily Operations
```bash
# Service Management
./service/memory_service_manager.sh status           # Check service status
./service/memory_service_manager.sh start-cloudflare # Start with Cloudflare backend

# Backend Synchronization
./sync/claude_sync_commands.py status               # Check sync status
./sync/claude_sync_commands.py backup               # Backup Cloudflare → SQLite
./sync/claude_sync_commands.py sync                 # Bidirectional sync
./sync/sync_now.py --verbose                        # Manual on-demand hybrid sync (v7.5.1+)

# Configuration Validation
./validation/validate_configuration_complete.py     # Comprehensive config validation
./validation/diagnose_backend_config.py             # Cloudflare backend diagnostics
./validation/verify_environment.py                  # Check environment setup

# Database Health
./database/simple_timestamp_check.py                # Quick health check
./database/db_health_check.py                       # Comprehensive health analysis
```

## 📂 Detailed Directory Guide

### 🔄 **sync/** - Backend Synchronization
Essential for managing dual-backend setups and data synchronization.

#### Cloudflare Hybrid Sync (Cloud Backend)
| Script | Purpose | Quick Usage |
|--------|---------|-------------|
| `sync_memory_backends.py` | Core bidirectional sync engine | `python sync/sync_memory_backends.py --status` |
| `claude_sync_commands.py` | User-friendly sync wrapper | `python sync/claude_sync_commands.py backup` |
| `sync_now.py` | Manual on-demand hybrid sync (v7.5.1+) | `python sync/sync_now.py --verbose` |
| `export_memories.py` | Export memories to JSON | `python sync/export_memories.py` |
| `import_memories.py` | Import memories from JSON | `python sync/import_memories.py data.json` |

**Key Features:**
- ✅ Bidirectional Cloudflare ↔ SQLite synchronization
- ✅ Intelligent deduplication using content hashing
- ✅ Dry-run mode for safe testing
- ✅ Comprehensive status reporting

#### Litestream Sync (Local Network HTTP API)
Located in `sync/litestream/` - Git-like staging workflow for syncing to central SQLite-vec HTTP API server.

| Script | Purpose | Quick Usage |
|--------|---------|-------------|
| `memory_sync.sh` | Main sync orchestrator (stash → pull → apply → push) | `./sync/litestream/memory_sync.sh sync` |
| `push_to_remote.sh` | Push staged changes to remote API | `./sync/litestream/push_to_remote.sh` |
| `pull_remote_changes.sh` | Pull latest from remote | `./sync/litestream/pull_remote_changes.sh` |
| `stash_local_changes.sh` | Stash local changes to staging | `./sync/litestream/stash_local_changes.sh` |
| `apply_local_changes.sh` | Apply staged changes locally | `./sync/litestream/apply_local_changes.sh` |
| `setup_local_litestream.sh` | Initialize local Litestream setup | `./sync/litestream/setup_local_litestream.sh` |
| `setup_remote_litestream.sh` | Setup remote API server | `./sync/litestream/setup_remote_litestream.sh` |

**Key Features:**
- ✅ Git-like staging database workflow
- ✅ Conflict detection and resolution
- ✅ Multi-device local network synchronization
- ✅ Sync to central HTTP API (e.g., `https://narrowbox.local:8443/api/memories`)
- ✅ macOS launchd service for automatic replication

**Note:** Litestream sync (local network) is separate from Cloudflare hybrid sync (cloud backend)

### 🛠️ **service/** - Service Management
Linux service management for production deployments.

| Script | Purpose | Quick Usage |
|--------|---------|-------------|
| `memory_service_manager.sh` | Complete service lifecycle management | `./service/memory_service_manager.sh start-cloudflare` |
| `service_control.sh` | Basic service control operations | `./service/service_control.sh restart` |
| `service_utils.py` | Service utility functions | Used by other service scripts |
| `deploy_dual_services.sh` | Deploy dual-backend architecture | `./service/deploy_dual_services.sh` |
| `update_service.sh` | Update running service | `./service/update_service.sh` |

**Key Features:**
- ✅ Dual-backend configuration management
- ✅ Environment file handling (.env, .env.sqlite)
- ✅ Service health monitoring
- ✅ Integrated sync operations

### ✅ **validation/** - Configuration & System Validation
Ensure proper setup and configuration.

| Script | Purpose | Quick Usage |
|--------|---------|-------------|
| `validate_configuration_complete.py` | Comprehensive config validation | `python validation/validate_configuration_complete.py` |
| `diagnose_backend_config.py` | Cloudflare backend diagnostics | `python validation/diagnose_backend_config.py` |
| `validate_memories.py` | Memory data validation | `python validation/validate_memories.py` |
| `validate_migration.py` | Migration validation | `python validation/validate_migration.py` |
| `verify_environment.py` | Environment setup checker | `python validation/verify_environment.py` |
| `verify_pytorch_windows.py` | PyTorch Windows validation | `python validation/verify_pytorch_windows.py` |
| `verify_torch.py` | PyTorch installation check | `python validation/verify_torch.py` |
| `check_documentation_links.py` | Documentation link validator | `python validation/check_documentation_links.py` |

**Key Features:**
- ✅ Claude Code global configuration validation
- ✅ Cloudflare credentials verification
- ✅ Environment conflict detection
- ✅ Comprehensive error reporting with solutions

### 🗄️ **database/** - Database Analysis & Health
Monitor and analyze database health and performance.

| Script | Purpose | Quick Usage |
|--------|---------|-------------|
| `simple_timestamp_check.py` | Quick timestamp health check | `python database/simple_timestamp_check.py` |
| `db_health_check.py` | Comprehensive health analysis | `python database/db_health_check.py` |
| `analyze_sqlite_vec_db.py` | Detailed SQLite-vec analysis | `python database/analyze_sqlite_vec_db.py` |
| `check_sqlite_vec_status.py` | SQLite-vec status checker | `python database/check_sqlite_vec_status.py` |

**Exit Codes (for CI/CD):**
- `0` - Excellent/Good health
- `1` - Warning status
- `2` - Critical issues
- `3` - Analysis failed

### 🧹 **maintenance/** - Database Cleanup & Repair
Scripts for maintaining database integrity and performance.

| Script | Purpose | Quick Usage |
|--------|---------|-------------|
| `migrate_embeddings.py` | **NEW:** Migrate to a different embedding model (handles dimension changes) | `python maintenance/migrate_embeddings.py --url ... --model ... --dry-run` |
| `improve_memory_ontology.py` | **NEW (v9.3.0+):** Re-classify memory types and standardize tags | `python maintenance/improve_memory_ontology.py --dry-run` |
| `update_graph_relationship_types.py` | **NEW (v9.3.0+):** Infer relationship types using RelationshipInferenceEngine | `python maintenance/update_graph_relationship_types.py --dry-run` |
| `cleanup_memories.py` | General memory cleanup, enhanced with HTTP/HTTPS auto-detection | `python maintenance/cleanup_memories.py` |
| `find_duplicates.py` | Find and remove duplicate memories | `python maintenance/find_duplicates.py --execute` |
| `cleanup_corrupted_encoding.py` | Fix corrupted emoji encoding | `python maintenance/cleanup_corrupted_encoding.py --execute` |
| `repair_memories.py` | Repair corrupted memory entries | `python maintenance/repair_memories.py` |
| `repair_sqlite_vec_embeddings.py` | Fix embedding inconsistencies | `python maintenance/repair_sqlite_vec_embeddings.py` |
| `repair_zero_embeddings.py` | Fix zero/null embeddings | `python maintenance/repair_zero_embeddings.py` |

**New in v9.3.0+:**
- `improve_memory_ontology.py` - Batch re-classify memory types using high-confidence patterns, convert legacy tags to namespace format (topic:, proj:, q:)
- `update_graph_relationship_types.py` - Use RelationshipInferenceEngine to infer relationship types (causes, fixes, contradicts, supports, follows, related) for existing associations

**Safety Features:**
- ✅ Dry-run mode available for all scripts
- ✅ Comprehensive backup recommendations
- ✅ Detailed reporting of changes

### 💾 **backup/** - Backup & Restore Operations
Data protection and recovery operations.

| Script | Purpose | Quick Usage |
|--------|---------|-------------|
| `backup_memories.py` | Create memory backups | `python backup/backup_memories.py` |
| `restore_memories.py` | Restore from backups | `python backup/restore_memories.py backup.json` |
| `backup_sqlite_vec.sh` | SQLite-vec database backup | `./backup/backup_sqlite_vec.sh` |
| `export_distributable_memories.sh` | Create distributable exports | `./backup/export_distributable_memories.sh` |

### 🔄 **migration/** - Data Migration & Schema Updates
Handle database migrations and data transformations.

| Script | Purpose | Quick Usage |
|--------|---------|-------------|
| `migrate_to_cloudflare.py` | Migrate to Cloudflare backend | `python migration/migrate_to_cloudflare.py` |
| `migrate_chroma_to_sqlite.py` | ChromaDB → SQLite migration | `python migration/migrate_chroma_to_sqlite.py` |
| `migrate_sqlite_vec_embeddings.py` | Update embedding format | `python migration/migrate_sqlite_vec_embeddings.py` |
| `migrate_timestamps.py` | Fix timestamp issues | `python migration/migrate_timestamps.py` |
| `cleanup_mcp_timestamps.py` | Clean timestamp proliferation | `python migration/cleanup_mcp_timestamps.py` |
| `verify_mcp_timestamps.py` | Verify timestamp consistency | `python migration/verify_mcp_timestamps.py` |

### 🏠 **installation/** - Setup & Installation
Platform-specific installation and setup scripts.

| Script | Purpose | Quick Usage |
|--------|---------|-------------|
| `install.py` | Platform-aware installer with backend selection | `python installation/install.py --storage-backend hybrid` |
| `install_linux_service.py` | Linux service installation | `python installation/install_linux_service.py` |
| `install_macos_service.py` | macOS service setup | `python installation/install_macos_service.py` |
| `install_windows_service.py` | Windows service installation | `python installation/install_windows_service.py` |
| `install_uv.py` | UV package manager installation | `python installation/install_uv.py` |
| `setup_cloudflare_resources.py` | Cloudflare resource setup | `python installation/setup_cloudflare_resources.py` |
| `setup_claude_mcp.sh` | Claude MCP configuration | `./installation/setup_claude_mcp.sh` |
| `setup_backup_cron.sh` | Automated backup scheduling | `./installation/setup_backup_cron.sh` |

### 🐧 **linux/** - Linux Service Management Shortcuts
Quick service management wrappers for Linux systemd deployments (v7.5.1+).

| Script | Purpose | Quick Usage |
|--------|---------|-------------|
| `service_status.sh` | Check systemd service status | `./linux/service_status.sh` |
| `start_service.sh` | Start mcp-memory service | `./linux/start_service.sh` |
| `stop_service.sh` | Stop mcp-memory service | `./linux/stop_service.sh` |
| `view_logs.sh` | View service logs | `./linux/view_logs.sh` |
| `uninstall_service.sh` | Remove systemd service | `./linux/uninstall_service.sh` |

**Key Features:**
- ✅ Simple wrappers for systemd service management
- ✅ User-level service control (~/.config/systemd/user/)
- ✅ Quick status and log viewing
- ✅ Clean uninstall capabilities

### 🖥️ **server/** - Server Runtime & Operations
Scripts for running and managing the memory server.

| Script | Purpose | Quick Usage |
|--------|---------|-------------|
| `run_memory_server.py` | Start memory server | `python server/run_memory_server.py` |
| `run_http_server.py` | Start HTTP API server | `python server/run_http_server.py` |
| `check_server_health.py` | Health check endpoint | `python server/check_server_health.py` |
| `memory_offline.py` | Offline memory operations | `python server/memory_offline.py` |
| `preload_models.py` | Pre-load ML models | `python server/preload_models.py` |

### 🧪 **testing/** - Test Scripts & Validation
Comprehensive testing and validation scripts.

| Script | Purpose | Quick Usage |
|--------|---------|-------------|
| `run_complete_test.py` | Complete system test | `python testing/run_complete_test.py` |
| `test_memory_api.py` | API functionality tests | `python testing/test_memory_api.py` |
| `test_cloudflare_backend.py` | Cloudflare backend tests | `python testing/test_cloudflare_backend.py` |
| `test_sqlite_vec_embeddings.py` | Embedding system tests | `python testing/test_sqlite_vec_embeddings.py` |
| `simple_test.py` | Basic functionality test | `python testing/simple_test.py` |

### 🔧 **utils/** - General Utilities
Helper scripts and utility functions.

| Script | Purpose | Quick Usage |
|--------|---------|-------------|
| `claude_commands_utils.py` | Claude command utilities | Used by Claude Code hooks |
| `query_memories.py` | Direct memory querying | `python utils/query_memories.py "search term"` |
| `memory_wrapper_uv.py` | UV package manager wrapper | Used by other scripts |
| `generate_personalized_claude_md.sh` | Generate custom CLAUDE.md | `./utils/generate_personalized_claude_md.sh` |

### 🛠️ **development/** - Development Tools
Tools for developers and debugging.

| Script | Purpose | Quick Usage |
|--------|---------|-------------|
| `setup-git-merge-drivers.sh` | Configure git merge drivers | `./development/setup-git-merge-drivers.sh` |
| `fix_mdns.sh` | Fix mDNS issues | `./development/fix_mdns.sh` |
| `uv-lock-merge.sh` | Handle UV lock file merges | `./development/uv-lock-merge.sh` |
| `find_orphaned_files.py` | Find orphaned database files | `python development/find_orphaned_files.py` |

## 🎯 Common Use Cases

### Initial Setup
```bash
# 1. Validate environment
python validation/verify_environment.py

# 2. Install appropriate service
python installation/install_linux_service.py

# 3. Validate configuration
python validation/validate_config.py

# 4. Start service
./service/memory_service_manager.sh start-cloudflare
```

### Daily Operations
```bash
# Check overall health
./service/memory_service_manager.sh status
python database/simple_timestamp_check.py

# Sync backends
python sync/claude_sync_commands.py sync

# Backup
python sync/claude_sync_commands.py backup
```

### Troubleshooting
```bash
# Validate configuration
python validation/validate_config.py

# Check database health
python database/db_health_check.py

# Fix common issues
python maintenance/find_duplicates.py --execute
python maintenance/cleanup_corrupted_encoding.py --execute
```

### Migration & Upgrades
```bash
# Before migration - backup
python backup/backup_memories.py

# Migrate to new backend
python migration/migrate_to_cloudflare.py

# Verify migration
python validation/validate_memories.py
```

## 🚨 Safety Guidelines

### Before Running Maintenance Scripts
1. **Always backup first**: `python backup/backup_memories.py`
2. **Use dry-run mode**: Most scripts support `--dry-run` or similar
3. **Test with small datasets** when possible
4. **Check database health**: `python database/simple_timestamp_check.py`

### Script Execution Order
1. **Validation** scripts first (check environment)
2. **Backup** before any data modifications
3. **Maintenance** operations (cleanup, repair)
4. **Verification** after changes
5. **Service restart** if needed

## 🔗 Integration with Documentation

This scripts directory integrates with:
- **CLAUDE.md**: Essential commands for Claude Code users
- **AGENTS.md**: Agent development and release process
- **Wiki**: Detailed setup and troubleshooting guides
- **GitHub Actions**: CI/CD pipeline integration

## 📝 Adding New Scripts

When adding new scripts:
1. **Choose appropriate category** based on primary function
2. **Follow naming conventions**: `snake_case.py` or `kebab-case.sh`
3. **Include proper documentation** in script headers
4. **Add safety mechanisms** for data-modifying operations
5. **Update this README** with script description
6. **Test with multiple backends** (SQLite-vec, Cloudflare)

## 🆘 Getting Help

- **Configuration issues**: Run `python validation/validate_config.py`
- **Database problems**: Run `python database/db_health_check.py`
- **Documentation links**: Run `python validation/check_documentation_links.py`
- **General health**: Run `./service/memory_service_manager.sh status`

For complex issues, check the [project wiki](https://github.com/doobidoo/mcp-memory-service/wiki) or create an issue with the output from relevant diagnostic scripts.