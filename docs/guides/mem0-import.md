# Import from mem0

The mem0 converter turns a mem0 export into the JSON shape consumed by
`MemoryImporter`; it does not write to storage itself.

```bash
python -m mcp_memory_service.sync.converters.mem0 \
  ~/mem0-export.json \
  ~/mem0-import.json

python scripts/sync/import_memories.py \
  --db-path ~/.local/share/mcp-memory/sqlite_vec.db \
  ~/mem0-import.json
```

Supported inputs are the official mem0 OSS Qdrant export
(`kind: mem0_oss_qdrant_export`, `records[]`), platform `get_all()` responses
(`results[]`), structured exports that contain a `memories[]` array, and bare
arrays of memory records. Each record needs a non-empty `memory` or `content`
field. If `created_at` is absent, the export's `exported_at` value is used and
recorded in metadata.

## Field mapping

| mem0 field | MCP Memory Service field |
|---|---|
| `memory` / `content` | `content` |
| `created_at` / `updated_at` | Unix timestamps in `created_at` / `updated_at` |
| `user_id` | `user:<id>` tag and `metadata.mem0_user_id` |
| `agent_id` | `metadata.agent_id`, `agent:<id>` tag, and `metadata.mem0_agent_id` |
| `run_id` | `run:<id>` tag and `metadata.mem0_run_id` |
| `metadata` | merged into target `metadata` |
| `id`, `app_id`, `actor_id`, `role`, `hash`, `categories` | preserved as `mem0_*` metadata |

The converter computes `content_hash` with the target service's normalization
instead of reusing mem0's hash. This makes the existing `MemoryImporter`
deduplicate the same content across repeated imports and migrations.
