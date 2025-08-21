#!/bin/bash
# Docker entrypoint script for MCP Memory Service - Persistent mode
# This script keeps the container running even when there's no active MCP client

set -e

echo "[INFO] Starting MCP Memory Service in Docker container (persistent mode)"

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

# Create named pipes for stdio communication
FIFO_DIR="/tmp/mcp-memory-fifo"
mkdir -p "$FIFO_DIR"
STDIN_FIFO="$FIFO_DIR/stdin"
STDOUT_FIFO="$FIFO_DIR/stdout"

# Remove old pipes if they exist
rm -f "$STDIN_FIFO" "$STDOUT_FIFO"

# Create new named pipes
mkfifo "$STDIN_FIFO"
mkfifo "$STDOUT_FIFO"

echo "[INFO] Created named pipes for stdio communication"

# Start the server in the background with the named pipes
if [ "${UV_ACTIVE}" = "1" ]; then
    echo "[INFO] Running with UV wrapper (persistent mode)"
    python -u uv_wrapper.py < "$STDIN_FIFO" > "$STDOUT_FIFO" 2>&1 &
else
    echo "[INFO] Running directly with Python (persistent mode)"
    python -u -m mcp_memory_service.server < "$STDIN_FIFO" > "$STDOUT_FIFO" 2>&1 &
fi

SERVER_PID=$!
echo "[INFO] Server started with PID: $SERVER_PID"

# Keep the stdin pipe open to prevent the server from exiting
exec 3> "$STDIN_FIFO"

# Monitor the server process
while true; do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "[ERROR] Server process exited unexpectedly"
        exit 1
    fi
    
    # Send a keep-alive message every 30 seconds
    echo "" >&3
    
    sleep 30
done