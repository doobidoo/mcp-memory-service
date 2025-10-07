#!/usr/bin/env bash
# Start the MCP Memory Service HTTP server in the background on Unix/macOS

set -e

echo "Starting MCP Memory Service HTTP server..."

# Check if server is already running
if uv run python scripts/server/check_http_server.py -q; then
    echo "✅ HTTP server is already running!"
    uv run python scripts/server/check_http_server.py -v
    exit 0
fi

# Start the server in the background
nohup uv run python scripts/server/run_http_server.py > /tmp/mcp-http-server.log 2>&1 &
SERVER_PID=$!

echo "Server started with PID: $SERVER_PID"
echo "Logs available at: /tmp/mcp-http-server.log"

# Wait up to 5 seconds for the server to start
for i in {1..5}; do
    if uv run python scripts/server/check_http_server.py -q; then
        break
    fi
    sleep 1
done

# Check if it started successfully
if uv run python scripts/server/check_http_server.py -v; then
    echo ""
    echo "✅ HTTP server started successfully!"
    echo "PID: $SERVER_PID"
else
    echo ""
    echo "⚠️ Server may still be starting... Check logs at /tmp/mcp-http-server.log"
fi
