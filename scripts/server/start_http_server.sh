#!/usr/bin/env bash
# Start the MCP Memory Service HTTP server in the background on Unix/macOS

set -e

echo "Starting MCP Memory Service HTTP server..."

# Check if server is already running
if python scripts/server/check_http_server.py -q; then
    echo "✅ HTTP server is already running!"
    python scripts/server/check_http_server.py -v
    exit 0
fi

# Start the server in the background
nohup uv run python scripts/server/run_http_server.py > /tmp/mcp-http-server.log 2>&1 &
SERVER_PID=$!

echo "Server started with PID: $SERVER_PID"
echo "Logs available at: /tmp/mcp-http-server.log"

# Wait a moment for server to start
sleep 3

# Check if it started successfully
if python scripts/server/check_http_server.py -v; then
    echo ""
    echo "✅ HTTP server started successfully!"
    echo "PID: $SERVER_PID"
else
    echo ""
    echo "⚠️ Server may still be starting... Check logs at /tmp/mcp-http-server.log"
fi
