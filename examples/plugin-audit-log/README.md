# Audit Log Plugin — Example

Demonstrates all 4 lifecycle hooks by writing events to a JSON Lines file.

By default the example writes a **privacy-safe audit log**: raw memory content,
raw tags and raw retrieval queries are omitted. Configure an HMAC key when you
want stable correlation IDs across events.

## Install

```bash
pip install -e examples/plugin-audit-log/
# or
uv pip install -e examples/plugin-audit-log/
```

Restart mcp-memory-service — the plugin loads automatically via `entry_points` discovery.

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `MCP_PLUGIN_AUDIT_LOG_PATH` | `/tmp/mcp-memory-audit.jsonl` | Path to the audit log file |
| `MCP_PLUGIN_AUDIT_LOG_PRIVACY_MODE` | `safe` | `safe` omits raw queries/tags/content; `raw` preserves the original debug fields for local-only inspection. **Warning:** raw mode can write retrieval queries, tags and derived memory identifiers to disk; use it only for short-lived local debugging when you control the log destination. |
| `MCP_PLUGIN_AUDIT_LOG_HMAC_KEY` | unset | Optional secret used to emit stable `hmac-sha256:*` query/memory identifiers in safe mode |

## Events

Each line in the audit log is a JSON object.

Safe mode with `MCP_PLUGIN_AUDIT_LOG_HMAC_KEY`:

```json
{"timestamp": 1715000000.0, "event": "store", "privacy_mode": "safe", "memory_hash_hmac": "hmac-sha256:abc123...", "memory_type": "note", "tag_count": 1, "content_length": 150, "raw_query_included": false, "raw_content_included": false, "raw_tags_included": false, "hash_algorithm": "hmac-sha256"}
{"timestamp": 1715000001.0, "event": "retrieve", "privacy_mode": "safe", "query_hash_hmac": "hmac-sha256:def456...", "query_length": 13, "result_count": 5, "raw_query_included": false, "raw_content_included": false, "raw_tags_included": false, "hash_algorithm": "hmac-sha256"}
{"timestamp": 1715000002.0, "event": "delete", "privacy_mode": "safe", "memory_hash_hmac": "hmac-sha256:789abc...", "raw_query_included": false, "raw_content_included": false, "raw_tags_included": false, "hash_algorithm": "hmac-sha256"}
{"timestamp": 1715000003.0, "event": "consolidate", "privacy_mode": "safe", "memories_processed": 42, "time_horizon": "7d"}
```

Safe mode without `MCP_PLUGIN_AUDIT_LOG_HMAC_KEY` still logs counts and lengths,
but omits stable query/memory identifiers instead of falling back to guessable
plain hashes:

```json
{"timestamp": 1715000001.0, "event": "retrieve", "privacy_mode": "safe", "query_length": 13, "result_count": 5, "hash_algorithm": "none", "identifier_hashes_omitted_reason": "MCP_PLUGIN_AUDIT_LOG_HMAC_KEY not set"}
```

Raw mode is available when you need the original local debugging behavior and
control the log destination:

```bash
MCP_PLUGIN_AUDIT_LOG_PRIVACY_MODE=raw
```

```json
{"timestamp": 1715000000.0, "event": "store", "privacy_mode": "raw", "hash": "abc123...", "memory_type": "note", "tags": ["project"], "content_length": 150}
{"timestamp": 1715000001.0, "event": "retrieve", "privacy_mode": "raw", "query": "how to deploy", "result_count": 5}
{"timestamp": 1715000002.0, "event": "delete", "privacy_mode": "raw", "hash": "def456..."}
```

## Hooks Used

| Hook | Purpose |
|------|---------|
| `on_store` | Log memory type, content length, tag count and optional HMAC of the memory hash |
| `on_delete` | Log optional HMAC of the deleted memory hash |
| `on_retrieve` | Log query length, result count and optional HMAC of the query (returns results unchanged) |
| `on_consolidate` | Log consolidation stats |

## Writing Your Own Plugin

1. Create a Python package with an `entry_points` declaration:

```toml
[project.entry-points."mcp_memory_service.plugins"]
my_plugin = "my_package:register"
```

2. Implement `register(ctx)` that subscribes to hooks:

```python
def register(ctx):
    ctx.on("on_store", my_store_handler)
    ctx.on("on_retrieve", my_retrieve_handler)
```

3. Install alongside mcp-memory-service and restart.
