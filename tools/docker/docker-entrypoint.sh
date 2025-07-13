#!/bin/bash
# Docker entrypoint script for MCP Memory Service

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

# Function to keep stdin alive
keep_stdin_alive() {
    while true; do
        # Send newline to stdin every 30 seconds to keep the pipe open
        echo "" 2>/dev/null || break
        sleep 30
    done
}

# Check if running in standalone mode
if [ "${MCP_STANDALONE_MODE}" = "1" ]; then
    echo "[INFO] Running in standalone mode"
    exec /usr/local/bin/docker-entrypoint-persistent.sh "$@"
fi

# Check if UV_ACTIVE is set
if [ "${UV_ACTIVE}" = "1" ]; then
    echo "[INFO] Running with UV wrapper"
    # Start the keep-alive process in the background
    keep_stdin_alive &
    KEEPALIVE_PID=$!
    
    # Run the server
    python -u uv_wrapper.py "$@" &
    SERVER_PID=$!
    
    # Wait for the server process
    wait $SERVER_PID
    SERVER_EXIT_CODE=$?
    
    # Clean up the keep-alive process
    kill $KEEPALIVE_PID 2>/dev/null || true
    
    exit $SERVER_EXIT_CODE
else
    echo "[INFO] Running directly with Python"
    # Start the keep-alive process in the background
    keep_stdin_alive &
    KEEPALIVE_PID=$!
    
    # Run the server
    python -u -m mcp_memory_service.server "$@" &
    SERVER_PID=$!
    
    # Wait for the server process
    wait $SERVER_PID
    SERVER_EXIT_CODE=$?
    
    # Clean up the keep-alive process
    kill $KEEPALIVE_PID 2>/dev/null || true
    
    exit $SERVER_EXIT_CODE
fi