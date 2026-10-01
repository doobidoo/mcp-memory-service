# Import from mem0

The mem0 converter turns a mem0 export into the JSON shape consumed by
`MemoryImporter`; it does not write to storage itself.

```bash
python -m mcp_memory_service.sync.converters.mem0 \
  ~/mem0-export.json \
  ~/mem0-import.json

python scripts/sync/import_memories.py ~/mem0-import.json
```

Supported inputs are the official mem0 OSS Qdrant export
(`kind: mem0_oss_qdrant_export`, `records[]`), platform `get_all()` responses
(`results[]`), structured exports that contain a `memories[]` array, and bare
arrays of memory records. Each record needs a non-empty `memory` or `content`
field. When `created_at` is absent, the converter uses `updated_at` first, then
the export's `exported_at`; if neither is available it uses the conversion time.
The timestamp source is recorded in `metadata.mem0_timestamp_source`.

The import command uses the SQLite path from the service configuration. Pass
`--db-path /path/to/sqlite_vec.db` only when intentionally overriding that path.

## Field mapping

| mem0 field | MCP Memory Service field |
|---|---|
| `memory` / `content` | `content` |
| `created_at` / `updated_at` | Unix timestamps in `created_at` / `updated_at` |
| `user_id` | `user:<id>` tag and `metadata.mem0_user_id` |
| `agent_id` | `metadata.agent_id`, `agent:<id>` tag, and `metadata.mem0_agent_id` |
| `run_id` | `sys:mem0-run:<id>` tag and `metadata.mem0_run_id` |
| `metadata` | merged into target `metadata` |
| `source_payload` | preserved as `metadata.mem0_source_payload` |
| `id`, `app_id`, `actor_id`, `role`, `hash`, `categories` | preserved as `mem0_*` metadata |

The converter computes `content_hash` with the target service's normalization
instead of reusing mem0's hash. This makes the existing `MemoryImporter`
deduplicate the same content across repeated imports and migrations. If one
export contains the same normalized content under different mem0 identities,
the converter emits one memory with all tags and identity metadata retained.
