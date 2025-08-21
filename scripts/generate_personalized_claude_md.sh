#!/bin/bash

# Generate personalized CLAUDE.md with memory context for local network distribution
# Usage: ./generate_personalized_claude_md.sh [target_machine_ip] [output_file]

TARGET_IP="${1:-10.0.1.30}"
OUTPUT_FILE="${2:-CLAUDE_PERSONALIZED.md}"
MCP_ENDPOINT="https://${TARGET_IP}:8443/mcp"
API_KEY="test-key-123"

echo "Generating personalized CLAUDE.md for network distribution..."
echo "Target: $TARGET_IP"
echo "Output: $OUTPUT_FILE"

# Create the personalized CLAUDE.md with embedded memory context
cat > "$OUTPUT_FILE" << 'EOF'
# CLAUDE.md - Personalized with Memory Context

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
**This version includes pre-loaded memory context from your local MCP Memory Service.**

## Memory Context Integration

Your local memory service contains essential project context. Here's the prompt to retrieve it:

```
Load MCP Memory Service context for this project. Retrieve all memories tagged with 'claude-code-reference' and 'distributable-reference' from the following endpoint and incorporate the knowledge into your understanding of this codebase:

Memory Service: https://TARGET_IP:8443/mcp
Authorization: Bearer test-key-123

Use this command to fetch context:
curl -k -s -X POST https://TARGET_IP:8443/mcp \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test-key-123" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "retrieve_memory", "arguments": {"query": "claude-code-reference distributable-reference", "limit": 20}}}' \
  | jq -r '.result.content[0].text'

This will provide you with:
- Project structure and architecture details
- Key commands for development, testing, and deployment  
- Environment variables and configuration options
- Recent changes including v5.0.2 ONNX implementation
- Issue management patterns and current status
- Testing practices and platform-specific optimizations
- Remote service deployment information

After loading this context, you'll have comprehensive knowledge of the MCP Memory Service project without needing to discover the codebase structure through file reading.
```

## Quick Memory Retrieval Commands

If memory context fails to load automatically, use these commands:

### Get All Project Context
```bash
curl -k -s -X POST https://TARGET_IP:8443/mcp \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test-key-123" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "retrieve_memory", "arguments": {"query": "claude-code-reference", "limit": 20}}}' \
  | jq -r '.result.content[0].text'
```

### Check Memory Service Health
```bash
curl -k -s -X POST https://TARGET_IP:8443/mcp \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test-key-123" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "check_database_health", "arguments": {}}}' \
  | jq -r '.result.content[0].text'
```

## Memory Categories Available
- **Project Structure**: Server architecture, file locations, component relationships
- **Key Commands**: Installation, testing, debugging, deployment commands  
- **Environment Variables**: Configuration options and platform-specific settings
- **Recent Changes**: Version history, resolved issues, breaking changes
- **Testing Practices**: Framework preferences, test patterns, validation steps
- **Current Status**: Active issues, recent work, development context

EOF

# Replace TARGET_IP placeholder with actual IP
sed -i "s/TARGET_IP/$TARGET_IP/g" "$OUTPUT_FILE"

# Append the original CLAUDE.md content (without the memory section)
echo "" >> "$OUTPUT_FILE"
echo "## Original Project Documentation" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Extract content from original CLAUDE.md starting after memory section
awk '/^## Overview/{print; getline; while(getline > 0) print}' CLAUDE.md >> "$OUTPUT_FILE"

echo "✅ Personalized CLAUDE.md generated: $OUTPUT_FILE"
echo ""
echo "Distribution instructions:"
echo "1. Copy $OUTPUT_FILE to target machines as CLAUDE.md"
echo "2. Ensure target machines can access https://$TARGET_IP:8443"
echo "3. Claude Code will automatically use memory context on those machines"
echo ""
echo "Network test command:"
echo "curl -k -s https://$TARGET_IP:8443/api/health"