#!/bin/bash
# Test the robust bridge with tools support
echo "Testing tools support in robust bridge..."

# Set environment variables
export MCP_MEMORY_HTTP_ENDPOINT="https://memory.local/api"
export MCP_MEMORY_API_KEY="mcp-0b1ccbde2197a08dcb12d41af4044be6"
export NODE_TLS_REJECT_UNAUTHORIZED=0

# Start bridge in background
node ./examples/http-mcp-bridge-robust.js > bridge_output.txt 2>&1 &
BRIDGE_PID=$!

# Give it time to start
sleep 2

# Test initialize
echo "Testing initialize..."
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | node ./examples/http-mcp-bridge-robust.js > test_init.txt &
sleep 3

# Test list tools
echo "Testing list tools..."
echo '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' | node ./examples/http-mcp-bridge-robust.js > test_list.txt &
sleep 3

# Display results
echo "=== Initialize Result ==="
cat test_init.txt 2>/dev/null || echo "No output"

echo "=== List Tools Result ==="
cat test_list.txt 2>/dev/null || echo "No output"

# Clean up
kill $BRIDGE_PID 2>/dev/null || true
rm -f test_init.txt test_list.txt bridge_output.txt

echo "Test complete!"