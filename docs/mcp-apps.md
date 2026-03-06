# MCP Apps Support (Insiders Mode)

> **Status:** Experimental — tracks [github/github-mcp-server Discussion #2048](https://github.com/github/github-mcp-server/discussions/2048)

MCP Apps is an extension to the Model Context Protocol that allows servers to deliver
interactive HTML interfaces directly within the conversation. Instead of returning
plain text for the LLM to relay, tools render forms and dashboards inside a
sandboxed iframe in the chat UI.

## Activation

```bash
# Enable MCP Apps HTML rendering
export MEMORY_MCP_APPS=true
```

With a GitHub MCP Server remote endpoint, additionally append `/insiders` to the URL
or set `X-MCP-Insiders: true` in the request header.

## Client Requirements

Currently tested and working with:

- **VS Code Insiders** — enable via `chat.mcp.apps.enabled` setting

Claude Desktop does not yet support MCP Apps (as of March 2026).

## Tools with MCP Apps UI

| Tool | UI | Description |
|------|----|-------------|
| `retrieve_memory` | Memory card grid | Similarity scores, tags, copy-hash buttons |
| `list_memories` | Paginated browser | Pagination, tag badges, type indicators |
| `check_database_health` | Stats dashboard | Memory count, backend status, health indicator |

## How It Works

When `MEMORY_MCP_APPS=true`:

1. The tool executes normally and collects results
2. Instead of returning JSON text, it calls the renderer in `mcp_apps/renderer.py`
3. The renderer returns an `EmbeddedResource` with `mimeType=text/html`
4. MCP Apps-capable clients render the HTML in a sandboxed iframe
5. Clients without MCP Apps support fall back to standard text output

## Security

- All HTML is self-contained — no external CDN dependencies
- Compatible with strict Content Security Policy sandbox
- No data is sent outside the iframe
- Hash copy functionality uses `navigator.clipboard` (requires secure context)

## Protocol Note

The MCP Apps protocol is in Insiders Mode as of March 2026. The response format
uses `EmbeddedResource(mimeType="text/html")` based on the reference implementation
in the GitHub MCP Server. This may evolve as the spec stabilizes.

Relevant files:
- `src/mcp_memory_service/mcp_apps/renderer.py` — HTML generators
- `src/mcp_memory_service/mcp_apps/__init__.py` — public API
- `src/mcp_memory_service/mcp_server.py` — tool integration points
