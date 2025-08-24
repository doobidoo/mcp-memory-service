# Memory Context Loading Prompt

Use this prompt at the start of Claude Code sessions on machines in your local network:

---

## Prompt for Claude Code

```
Load MCP Memory Service context for this project. Before we begin working, please retrieve and incorporate all stored knowledge about this codebase from my local memory service:

**Memory Service Endpoint**: https://your-server-ip:8443/mcp
**Authorization**: Bearer your-api-key

Execute this command to load context:
```bash
curl -k -s -X POST https://your-server-ip:8443/mcp \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "retrieve_memory", "arguments": {"query": "claude-code-reference distributable-reference", "limit": 20}}}' \
  | jq -r '.result.content[0].text'
```

This memory contains:
- Complete project structure and architecture
- All key development, testing, and deployment commands
- Environment variables and configuration patterns
- Recent changes including v5.0.2 ONNX implementation details
- Issue management approaches and current project status
- Testing practices and platform-specific optimizations
- Remote service deployment and health monitoring

After loading this context, you'll have comprehensive knowledge of the MCP Memory Service project equivalent to extensive codebase exploration, which will significantly reduce token usage and improve response accuracy.

Please confirm successful context loading and summarize the key project information you've retrieved.
```

---

## Alternative Short Prompt

For quick context loading:

```
Load project context from memory service: curl -k -s -X POST https://your-server-ip:8443/mcp -H "Content-Type: application/json" -H "Authorization: Bearer your-api-key" -d '{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "retrieve_memory", "arguments": {"query": "claude-code-reference", "limit": 20}}}' | jq -r '.result.content[0].text'

Incorporate this MCP Memory Service project knowledge before proceeding.
```

---

## Network Distribution

1. **Copy this prompt file** to other machines in your network
2. **Update IP address** if memory service moves
3. **Test connectivity** with: `curl -k -s https://your-server-ip:8443/api/health`
4. **Use at session start** for instant project context

This eliminates repetitive codebase discovery across all your development machines.