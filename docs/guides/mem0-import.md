# Import from mem0

The mem0 converter turns a mem0 export into the JSON shape consumed by
`MemoryImporter`; it does not write to storage itself.

```bash
python -m mcp_memory_service.sync.converters.mem0 \
  ~/mem0-export.json \
  ~/mem0-import.json && \
python scripts/sync/import_memories.py ~/mem0-import.json
```

Supported inputs are the official mem0 OSS Qdrant export
(`kind: mem0_oss_qdrant_export`, `records[]`), platform `get_all()` responses
(`results[]`), structured exports that contain a `memories[]` array, and bare
arrays of memory records. Each record needs a non-empty `memory` or `content`
field. When `created_at` is absent, the converter uses `updated_at` first, then
the export's `exported_at`; if neither is available it uses the conversion time.
The timestamp source is recorded in `metadata.mem0_timestamp_source`. Malformed
records are skipped individually, and timestamp fallbacks or skips are recorded
in `export_metadata.conversion_warnings` so one bad record cannot discard the
valid records in a batch.

The import command writes directly into a local SQLite-vec file. It uses the
configured SQLite path (`MCP_MEMORY_SQLITE_PATH`) on both the `sqlite_vec` and
`hybrid` backends, so run it with the same environment as the service, or pass
`--db-path /path/to/sqlite_vec.db` explicitly. Two limits apply:

- On `hybrid`, the imported rows bypass the service's sync queue, so they exist
  only in the local SQLite file until a sync pushes them. Trigger one on the
  running service with `POST /api/sync/force` (write access required). Its
  response reports `success: false` when Cloudflare is unreachable, but it does
  not report individual write failures, so a successful response does not prove
  that every imported memory reached Cloudflare. Check the imported
  `content_hash` values on the Cloudflare side before relying on the remote copy.
- The `cloudflare` backend has no local SQLite file, so this command cannot
  import into it.

The converter exits with status 1 when the export contains records but none of
them could be converted, so a script that chains both commands with `&&` stops
before importing an empty file.

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

The converter computes `content_hash` from the normalized content plus the
mapped mem0 identity (`user_id`, `agent_id`, `run_id`, and source tags) instead
of reusing mem0's hash. The same text under different identities therefore
imports as distinct memories, while repeated imports of the same identity still
deduplicate through the existing `MemoryImporter`.
