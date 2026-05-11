# Audit Log Plugin — Example

Demonstrates all 4 lifecycle hooks by writing events to a JSON Lines file.

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

## Events

Each line in the audit log is a JSON object:

```json
{"timestamp": 1715000000.0, "event": "store", "hash": "abc123...", "memory_type": "note", "tags": ["project"], "content_length": 150}
{"timestamp": 1715000001.0, "event": "retrieve", "query": "how to deploy", "result_count": 5}
{"timestamp": 1715000002.0, "event": "delete", "hash": "def456..."}
{"timestamp": 1715000003.0, "event": "consolidate", "memories_processed": 42, "time_horizon": "7d"}
```

## Hooks Used

| Hook | Purpose |
|------|---------|
| `on_store` | Log hash, type, tags, content length |
| `on_delete` | Log deleted hash |
| `on_retrieve` | Log query and result count (returns results unchanged) |
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
