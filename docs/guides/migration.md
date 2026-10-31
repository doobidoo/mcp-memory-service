# Migration Guide

> **ChromaDB was removed in v8.0.0.** This page covers moving between the
> backends current releases support. Legacy ChromaDB data needs one extra step on
> the v7.6.0 release; see
> [ChromaDB Migration](chromadb-migration.md).

## Current Backends

| From | To | How |
|------|----|-----|
| SQLite-vec | Cloudflare | `scripts/migration/migrate_to_cloudflare.py` |
| SQLite-vec | Hybrid | Upload with `scripts/migration/migrate_to_cloudflare.py migrate`, then configure `MCP_MEMORY_STORAGE_BACKEND=hybrid` — startup does not upload existing local memories |
| Cloudflare | Hybrid | Configure `MCP_MEMORY_STORAGE_BACKEND=hybrid` — startup pulls missing Cloudflare memories into the local database |
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

### SQLite-vec → Hybrid

Switching to `hybrid` does not upload a database that already exists locally.
Initial sync only pulls Cloudflare memories that are missing from the local
store, and the background sync service only pushes operations queued after
startup. Upload first:

```bash
memory stop
# Quit every client that uses the database too, then:

python scripts/migration/migrate_to_cloudflare.py migrate \
  --source sqlite_vec --source-path /path/to/sqlite_vec.db

export MCP_MEMORY_STORAGE_BACKEND=hybrid
```

Restart the service once both steps are done. Without the upload, existing
memories stay local-only and are not backed up to Cloudflare.

### Cloudflare → Hybrid

```bash
export MCP_MEMORY_STORAGE_BACKEND=hybrid
```

Restart the service. Hybrid initialization reconciles the local SQLite-vec store
with Cloudflare by pulling remote memories that are missing locally.

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

Take the backup with SQLite itself. `.backup` produces one consistent file even
while the database is open:

```bash
sqlite3 ~/.local/share/mcp-memory/sqlite_vec.db ".backup memory_backup.db"
```

`scripts/backup/backup_sqlite_vec.sh` copies the main database, WAL and SHM in
separate `cp` calls without a SQLite lock, so a write or checkpoint between
copies can leave the files disagreeing. Stop the HTTP server and every client
that opens the database before using it.

To restore, the server is not the only writer: stdio MCP servers started by
Claude Desktop or an editor keep the database open too. Stop all of them, move
the old database and its `-wal`/`-shm` sidecars aside, then put the backup in
place:

```bash
memory stop
# Quit every client that uses the database, then:
mv ~/.local/share/mcp-memory/sqlite_vec.db ~/.local/share/mcp-memory/sqlite_vec.db.before-restore
rm -f ~/.local/share/mcp-memory/sqlite_vec.db-wal ~/.local/share/mcp-memory/sqlite_vec.db-shm
cp memory_backup.db ~/.local/share/mcp-memory/sqlite_vec.db
memory launch
```

Removing the sidecars matters: a WAL left by the old database can be replayed
over the restored file, losing or corrupting the restored rows.

## Verify the Result

`scripts/database/db_health_check.py` exercises store, search and embedding
checks against temporary databases, not the one you migrated. Its only check of
the configured database is an embedding invariant, and it reports PASS while
"skipping check" if the file or its `memories` table is missing, so a green run
does not prove the migration worked.

Check the converted database directly, then exercise it through the service:

```bash
# Active row count in the database this guide migrates
sqlite3 /path/to/sqlite_vec.db "SELECT COUNT(*) FROM memories WHERE deleted_at IS NULL;"

# Run schema/search checks (temporary databases) and the configured-DB embedding check
python scripts/database/db_health_check.py

# Check the environment
python scripts/validation/verify_environment.py
```

Then start the service and run sample searches for content you know existed
before the migration; confirm tags, memory types and timestamps survived.

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
