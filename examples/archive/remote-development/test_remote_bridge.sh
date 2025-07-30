#!/bin/bash

# Test remote memory bridge connection
echo "Testing connection to memory.local..."

# Define environment variables
export MCP_MEMORY_HTTP_ENDPOINT="https://memory.local/api"
export MCP_MEMORY_API_KEY="mcp-0b1ccbde2197a08dcb12d41af4044be6"
export NODE_TLS_REJECT_UNAUTHORIZED=0

# Start the bridge in the background
node ./examples/http-mcp-bridge.js > bridge_output.txt 2> bridge_error.txt &
BRIDGE_PID=$!

# Give it a moment to start up
sleep 2

# Send an initialize request
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | node ./examples/http-mcp-bridge.js > test_output.txt

# Check the database health
echo '{"jsonrpc":"2.0","id":2,"method":"check_database_health","params":{}}' | node ./examples/http-mcp-bridge.js >> test_output.txt

# Display the results
echo "Test Results:"
cat test_output.txt

# Clean up
kill $BRIDGE_PID
echo "Test complete!"