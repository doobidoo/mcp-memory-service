#!/bin/bash
# Unified Docker entrypoint script for MCP Memory Service
# Supports both MCP protocol mode and HTTP server mode

set -e

echo "[INFO] Starting MCP Memory Service in Docker container"

# Function to handle signals
handle_signal() {
    echo "[INFO] Received signal, shutting down..."
    if [ -n "$SERVER_PID" ]; then
        kill -TERM $SERVER_PID 2>/dev/null || true
    fi
    exit 0
}

# Set up signal handlers
trap handle_signal SIGTERM SIGINT

# Determine mode based on environment variable
MODE="${MCP_MODE:-mcp}"
echo "[INFO] Running in $MODE mode"

if [ "$MODE" = "http" ] || [ "$MODE" = "api" ]; then
    # HTTP Server Mode
    echo "[INFO] Starting HTTP server with FastAPI/Uvicorn"
    
    # Ensure we have the HTTP server file
    if [ ! -f "/app/run_server.py" ]; then
        echo "[ERROR] run_server.py not found. Please ensure it's copied in the Dockerfile"
        exit 1
    fi
    
    # Start the HTTP server
    exec python /app/run_server.py "$@"
    
elif [ "$MODE" = "streamable-http" ]; then
    # Streamable HTTP Transport Mode (for Claude.ai remote MCP)
    echo "[INFO] Starting MCP server with Streamable HTTP transport"

    HOST="${MCP_SSE_HOST:-0.0.0.0}"
    PORT="${MCP_SSE_PORT:-8765}"
    echo "[INFO] Listening on ${HOST}:${PORT}"

    exec python -m mcp_memory_service.cli.main server --streamable-http --sse-host "$HOST" --sse-port "$PORT" "$@"

elif [ "$MODE" = "mcp" ]; then
    # MCP Protocol Mode (stdin/stdout)
    echo "[INFO] Starting MCP protocol server (stdin/stdout communication)"

    # Function to keep stdin alive
    keep_stdin_alive() {
        while true; do
            # Send newline to stdin every 30 seconds to keep the pipe open
            echo "" 2>/dev/null || break
            sleep 30
        done
    }

    # Start the keep-alive process in the background
    keep_stdin_alive &
    KEEPALIVE_PID=$!

    # Run the MCP server
    python -u -m mcp_memory_service.server "$@" &
    SERVER_PID=$!

    # Wait for the server process
    wait $SERVER_PID
    SERVER_EXIT_CODE=$?

    # Clean up the keep-alive process
    kill $KEEPALIVE_PID 2>/dev/null || true

    exit $SERVER_EXIT_CODE

else
    echo "[ERROR] Unknown mode: $MODE. Use 'mcp', 'http', or 'streamable-http'"
    exit 1
fi