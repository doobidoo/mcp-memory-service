- **`mcp-memory-server` served the full MCP tool surface unauthenticated on any bind (GHSA-26rx-6fvr-qjqg, NotAFlightRisk).**
  The FastMCP entry point builds its server with no auth and calls `mcp.run("streamable-http")`
  without the bind check the SSE and Streamable HTTP transports got for GHSA-2hh8-qjxc-43x3.
  With `MCP_HTTP_HOST=0.0.0.0`, the setting the docs use for network access, any caller
  could read, store and delete memories, and setting `MCP_API_KEY` or OAuth changed nothing
  because this entry point never consults either. It now refuses to start on anything but a
  loopback address, whatever auth is configured. For network clients run
  `memory server --streamable-http`, which enforces `MCP_API_KEY` or OAuth; note that it
  binds `MCP_SSE_HOST` (or `--sse-host`), not `MCP_HTTP_HOST`.
