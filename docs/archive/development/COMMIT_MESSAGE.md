fix: Support flexible MCP memory server naming conventions

The hook installer was hardcoded to check for a memory server named
exactly 'memory', but Claude Code allows users to configure MCP servers
with any name they choose. This caused false "standalone" detection even
when a memory MCP server was properly configured and connected.

Changes:
- Check multiple common memory server names (memory-service, memory,
  mcp-memory-service, extended-memory)
- Fallback to 'claude mcp list' grep detection for any memory-related
  server
- Support HTTP MCP server format (URL field instead of Command field)
- Update validation to accept http type and URL format
- Maintain backward compatibility with original 'memory' name

Fixes installation failures for users who configured their memory MCP
servers with descriptive names like 'memory-service' (common for HTTP
servers) or 'extended-memory' (older installations).

Testing:
- Verified with HTTP MCP server named 'memory-service'
- Confirmed backward compatibility with 'memory' name
- Tested fallback detection mechanism
- All test cases documented in TESTING_NOTES.md

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
