#!/bin/bash
# Test the robust bridge connection
echo "Testing robust bridge connection to memory.local..."

# Set environment variables
export MCP_MEMORY_HTTP_ENDPOINT="https://memory.local/api"
export MCP_MEMORY_API_KEY="mcp-0b1ccbde2197a08dcb12d41af4044be6"
export NODE_TLS_REJECT_UNAUTHORIZED=0

# Test initialization
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | timeout 30 node ./examples/http-mcp-bridge-robust.js

echo "Robust bridge test complete!"