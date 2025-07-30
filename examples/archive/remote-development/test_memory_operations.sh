#!/bin/bash

# Test memory operations through the bridge
echo "Testing memory operations with memory.local..."

# Define environment variables
export MCP_MEMORY_HTTP_ENDPOINT="https://memory.local/api"
export MCP_MEMORY_API_KEY="mcp-0b1ccbde2197a08dcb12d41af4044be6"
export NODE_TLS_REJECT_UNAUTHORIZED=0

# Start the bridge
echo "Starting MCP bridge..."
node ./examples/http-mcp-bridge.js > bridge_output.txt 2> bridge_error.txt &
BRIDGE_PID=$!

# Give it a moment to start up
sleep 2

# Send an initialize request
echo "Initializing..."
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | node ./examples/http-mcp-bridge.js > test_output.txt

# Store a test memory
echo "Storing a test memory..."
echo '{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "store_memory",
  "params": {
    "content": "This is a test memory created on '$(date)'",
    "metadata": {
      "tags": ["test", "remote-bridge"],
      "type": "note"
    }
  }
}' | node ./examples/http-mcp-bridge.js >> test_output.txt

# Retrieve memories
echo "Retrieving memories..."
echo '{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "retrieve_memory",
  "params": {
    "query": "test memory",
    "n_results": 5
  }
}' | node ./examples/http-mcp-bridge.js >> test_output.txt

# Search by tag
echo "Searching by tag..."
echo '{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "search_by_tag",
  "params": {
    "tags": ["remote-bridge"]
  }
}' | node ./examples/http-mcp-bridge.js >> test_output.txt

# Display the results
echo "Test Results:"
cat test_output.txt

# Clean up
kill $BRIDGE_PID
echo "Test complete!"