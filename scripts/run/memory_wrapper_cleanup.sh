#!/bin/bash
# MCP Memory Service Wrapper with Orphan Cleanup
# 
# Cleans up orphaned MCP memory processes before starting the server.
# Orphaned processes cause SQLite "database locked" errors.
#
# Usage in MCP config:
# {
#   "memory": {
#     "command": "/path/to/mcp-memory-service/scripts/run/memory_wrapper_cleanup.sh",
#     "args": [],
#     "env": { ... }
#   }
# }

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

log() {
    echo "[mcp-memory-wrapper] $1" >&2
}

# Find and kill orphaned MCP memory processes
cleanup_orphans() {
    local count=0
    
    for pid in $(pgrep -f "mcp-memory-service" 2>/dev/null || true); do
        # Skip our own process tree
        if [ "$pid" = "$$" ]; then
            continue
        fi
        
        # Get parent PID
        local ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
        
        # ppid=1 means orphaned (parent is init/launchd)
        if [ "$ppid" = "1" ]; then
            log "Killing orphaned process: $pid"
            kill -9 "$pid" 2>/dev/null || true
            ((count++)) || true
        fi
    done
    
    if [ "$count" -gt 0 ]; then
        log "Cleaned up $count orphaned process(es)"
    else
        log "No orphaned processes found"
    fi
}

# Find uv executable
find_uv() {
    if command -v uv &>/dev/null; then
        echo "uv"
    elif [ -x "$HOME/.local/bin/uv" ]; then
        echo "$HOME/.local/bin/uv"
    elif [ -x "$HOME/.cargo/bin/uv" ]; then
        echo "$HOME/.cargo/bin/uv"
    else
        log "ERROR: uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
}

main() {
    log "Starting ($(uname -s) $(uname -r))"
    
    # Step 1: Cleanup orphans
    cleanup_orphans
    
    # Step 2: Start server
    cd "$PROJECT_DIR"
    
    UV=$(find_uv)
    log "Starting server with: $UV run memory"
    
    # exec replaces this shell - clean signal handling, no subprocess
    exec "$UV" run memory "$@"
}

main "$@"
