#!/bin/bash

# Test memory operations with simple single-line requests
echo "Testing memory operations with memory.local..."

# Define environment variables
export MCP_MEMORY_HTTP_ENDPOINT="https://memory.local/api"
export MCP_MEMORY_API_KEY="mcp-0b1ccbde2197a08dcb12d41af4044be6"
export NODE_TLS_REJECT_UNAUTHORIZED=0

# Create a temp file with our requests
cat > requests.txt << EOF
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}
{"jsonrpc":"2.0","id":2,"method":"check_database_health","params":{}}
{"jsonrpc":"2.0","id":3,"method":"store_memory","params":{"content":"Test memory from remote bridge","metadata":{"tags":["test","remote"],"type":"note"}}}
{"jsonrpc":"2.0","id":4,"method":"retrieve_memory","params":{"query":"test memory","n_results":5}}
EOF

# Process each request individually
echo "Testing initialize..."
head -n 1 requests.txt | node ./examples/http-mcp-bridge.js > test_output.txt

echo "Testing health check..."
sed -n '2p' requests.txt | node ./examples/http-mcp-bridge.js >> test_output.txt

echo "Testing store memory..."
sed -n '3p' requests.txt | node ./examples/http-mcp-bridge.js >> test_output.txt

echo "Testing retrieve memory..."
sed -n '4p' requests.txt | node ./examples/http-mcp-bridge.js >> test_output.txt

# Display the results
echo "Test Results:"
cat test_output.txt

# Clean up
rm requests.txt
echo "Test complete!"