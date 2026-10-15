- **A read-only OAuth token could still write memories over SSE, Streamable HTTP and `prompts/get` (GHSA-9x7v-2vjj-xghg, gigio1023).**
  The `write` scope check from GHSA-2r68-g678-7qr3 lived in the FastAPI JSON-RPC shim only.
  SSE and Streamable HTTP authenticated the token, dropped its scope and dispatched straight
  into `MemoryServer.call_tool`, so a `read` token could call `memory_store`, `memory_update`
  and `memory_delete`; the `learning_session` prompt stored a memory without going through
  `call_tool` at all. The transports now record the granted scope on each request, and
  `call_tool` and `learning_session` refuse writes without `write`, per message. The
  `readOnlyHint` classification moved onto `MemoryServer` so the shim and the transports share
  it. The `mcp` SDK floor rises from 1.8.0 to 1.9.2, the first release that hands the
  per-message request to handlers. Only affects servers with `MCP_OAUTH_ENABLED=true`; stdio
  and API-key callers are unchanged.
