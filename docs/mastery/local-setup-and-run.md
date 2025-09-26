# MCP Memory Service — Local Setup and Run

Follow these steps to run the service locally, switch storage backends, and validate functionality.

## 1) Install Dependencies

Using uv (recommended):

```
uv sync
```

Using pip:

```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

If using SQLite-vec backend (recommended):

```
uv add sqlite-vec sentence-transformers torch
# or
pip install sqlite-vec sentence-transformers torch
```

## 2) Choose Storage Backend

SQLite-vec (default):

```
export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec
# optional custom DB path
export MCP_MEMORY_SQLITE_PATH="$HOME/.local/share/mcp-memory/sqlite_vec.db"
```

ChromaDB (deprecated):

```
export MCP_MEMORY_STORAGE_BACKEND=chroma
export MCP_MEMORY_CHROMA_PATH="$HOME/.local/share/mcp-memory/chroma_db"
```

Cloudflare:

```
export MCP_MEMORY_STORAGE_BACKEND=cloudflare
export CLOUDFLARE_API_TOKEN=...
export CLOUDFLARE_ACCOUNT_ID=...
export CLOUDFLARE_VECTORIZE_INDEX=...
export CLOUDFLARE_D1_DATABASE_ID=...
```

## 3) Run the Server

Stdio MCP server (integrates with Claude Desktop):

```
uv run memory server
```

FastMCP HTTP server (for Claude Code / remote):

```
uv run mcp-memory-server
```

Configure Claude Desktop example (~/.claude/config.json):

```
{
  "mcpServers": {
    "memory": {
      "command": "uv",
      "args": ["--directory", "/path/to/mcp-memory-service", "run", "memory", "server"],
      "env": { "MCP_MEMORY_STORAGE_BACKEND": "sqlite_vec" }
    }
  }
}
```

## 4) Verify Health and Basic Ops

CLI status:

```
uv run memory status
```

MCP tool flow (via client):
- store_memory → retrieve_memory → search_by_tag → delete_memory

## 5) Run Tests

```
pytest -q
# or
uv run pytest -q
```

See also: `docs/mastery/testing-guide.md` and `docs/sqlite-vec-backend.md`.

