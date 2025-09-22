#!/bin/bash
# Memory Service Manager for Claude Code on Linux
# Manages dual backend setup with Cloudflare primary and SQLite-vec backup

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Service configuration
CLOUDFLARE_ENV="$PROJECT_DIR/.env"
SQLITE_ENV="$PROJECT_DIR/.env.sqlite"

# Create SQLite-vec environment file if it doesn't exist
if [ ! -f "$SQLITE_ENV" ]; then
    cat > "$SQLITE_ENV" << EOF
# SQLite-vec Configuration for MCP Memory Service (Backup)
MCP_MEMORY_STORAGE_BACKEND=sqlite_vec
MCP_MEMORY_SQLITE_PATH="$HOME/.local/share/mcp-memory/primary_sqlite_vec.db"
EOF
    echo "Created SQLite-vec environment configuration: $SQLITE_ENV"
fi

usage() {
    echo "Memory Service Manager for Claude Code"
    echo ""
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  start-cloudflare    Start memory server with Cloudflare backend"
    echo "  start-sqlite        Start memory server with SQLite-vec backend"
    echo "  status             Show current backend and sync status"
    echo "  sync-backup        Backup Cloudflare → SQLite-vec"
    echo "  sync-restore       Restore SQLite-vec → Cloudflare"
    echo "  sync-both          Bidirectional sync"
    echo "  stop               Stop any running memory server"
    echo ""
}

start_memory_service() {
    local backend="$1"
    local env_file="$2"

    echo "Starting memory service with $backend backend..."

    # Stop any existing service
    pkill -f "memory server" 2>/dev/null || true
    sleep 2

    # Start new service
    cd "$PROJECT_DIR"
    if [ -f "$env_file" ]; then
        echo "Loading environment from: $env_file"
        set -a
        source "$env_file"
        set +a
    fi

    echo "Starting: uv run memory server"
    nohup uv run memory server > /tmp/memory-service-$backend.log 2>&1 &

    # Wait a moment and check if it started
    sleep 3
    if pgrep -f "memory server" > /dev/null; then
        echo "Memory service started successfully with $backend backend"
        echo "Logs: /tmp/memory-service-$backend.log"

        # Save active backend to state file for reliable detection
        echo "$backend" > /tmp/memory-service-backend.state
    else
        echo "Failed to start memory service"
        echo "Check logs: /tmp/memory-service-$backend.log"
        return 1
    fi
}

show_status() {
    echo "=== Memory Service Status ==="

    # Check if service is running
    if pgrep -f "memory server" > /dev/null; then
        echo "Service: Running"

        # Check which backend is active using state file
        if [ -f "/tmp/memory-service-backend.state" ]; then
            local active_backend=$(cat /tmp/memory-service-backend.state)
            echo "Active Backend: $active_backend (from state file)"
        else
            echo "Active Backend: Unknown (no state file found)"
        fi
    else
        echo "Service: Not running"
        # Clean up state file if service is not running
        [ -f "/tmp/memory-service-backend.state" ] && rm -f /tmp/memory-service-backend.state
    fi

    echo ""
    echo "=== Sync Status ==="
    cd "$PROJECT_DIR"
    uv run python scripts/claude_sync_commands.py status
}

sync_memories() {
    local direction="$1"
    echo "Syncing memories: $direction"
    cd "$PROJECT_DIR"
    uv run python scripts/claude_sync_commands.py "$direction"
}

stop_service() {
    echo "Stopping memory service..."
    pkill -f "memory server" 2>/dev/null || true
    sleep 2
    if ! pgrep -f "memory server" > /dev/null; then
        echo "Memory service stopped"
        # Clean up state file when service is stopped
        [ -f "/tmp/memory-service-backend.state" ] && rm -f /tmp/memory-service-backend.state
    else
        echo "Failed to stop memory service"
        return 1
    fi
}

# Main command handling
case "$1" in
    start-cloudflare)
        start_memory_service "cloudflare" "$CLOUDFLARE_ENV"
        ;;
    start-sqlite)
        start_memory_service "sqlite" "$SQLITE_ENV"
        ;;
    status)
        show_status
        ;;
    sync-backup)
        sync_memories "backup"
        ;;
    sync-restore)
        sync_memories "restore"
        ;;
    sync-both)
        sync_memories "sync"
        ;;
    stop)
        stop_service
        ;;
    *)
        usage
        exit 1
        ;;
esac
