# Migration Guide

> **ChromaDB was removed in v8.0.0.** This page covers moving between the
> backends current releases support. Legacy ChromaDB data needs one extra step on
> the v7.6.0 release; see
> [ChromaDB Migration](chromadb-migration.md).

## Current Backends

| From | To | How |
|------|----|-----|
| SQLite-vec | Cloudflare | `scripts/migration/migrate_to_cloudflare.py` |
| SQLite-vec | Hybrid | Configure `MCP_MEMORY_STORAGE_BACKEND=hybrid` — startup syncs Cloudflare from the local database |
| Cloudflare | Hybrid | Configure `MCP_MEMORY_STORAGE_BACKEND=hybrid` — startup picks the populated side |
| Cloudflare | SQLite-vec | No dedicated script; run Hybrid until the local file is populated, or export through the REST API |

Choosing a backend, credentials and configuration are covered in
[Storage Backends](STORAGE_BACKENDS.md). The commands below are the operational
short version.

### SQLite-vec → Cloudflare

```bash
python scripts/migration/migrate_to_cloudflare.py migrate \
  --source sqlite_vec --source-path /path/to/sqlite_vec.db
```

Use the `export` and `import` subcommands instead of `migrate` if you want an
intermediate JSON backup.

### SQLite-vec → Hybrid / Cloudflare → Hybrid

```bash
export MCP_MEMORY_STORAGE_BACKEND=hybrid
```

Restart the service. Hybrid initialization reconciles the local SQLite-vec store
with the Cloudflare mirror.

### Cloudflare → SQLite-vec

There is no dedicated migration script. Either run in `hybrid` mode until the
local database is populated, or export memories through the REST API
(`GET /api/memories`) and import them with a small custom script.

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `MCP_MEMORY_STORAGE_BACKEND` | `sqlite_vec`, `cloudflare`, `hybrid` or `milvus` | `sqlite_vec` |
| `MCP_MEMORY_SQLITE_PATH` | SQLite-vec database file | `~/.local/share/mcp-memory/sqlite_vec.db` |
| `MCP_HTTP_ENABLED` | Enable HTTP/SSE interface | `false` |
| `MCP_HTTP_PORT` | HTTP server port | `8000` |

Add the values to `.env` and to the `env` block of the memory server in
`claude_desktop_config.json`; a service started by Claude Desktop reads the
latter, not your shell.

## Backup Before Migrating

```bash
# Copy the database and its WAL/SHM files
./scripts/backup/backup_sqlite_vec.sh

# Or take a consistent SQLite backup
sqlite3 memory.db ".backup memory_backup.db"
```

Stop the service before replacing a database file, then restart it with
`memory launch`.

## Verify the Result

```bash
# Check the configured database, schema and search
python scripts/database/db_health_check.py

# Check the environment
python scripts/validation/verify_environment.py
```

## Next Steps

1. ✅ Test memory operations (store, retrieve, search)
2. ✅ Confirm tags and timestamps survived the migration
3. ✅ Update any scripts or tools that reference storage paths
4. ✅ Keep the old database until the new one has been verified
5. ✅ Remove the old data once you are satisfied

## Support

If you encounter issues, check the migration output, verify the environment
variables, and open an issue with the exact commands and error output:
[GitHub Issues](https://github.com/doobidoo/mcp-memory-service/issues).
