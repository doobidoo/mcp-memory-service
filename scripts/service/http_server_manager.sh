#!/bin/bash
# MCP Memory HTTP Server Manager
# Smart auto-start and restart management for macOS/Linux

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Configuration
HTTP_SERVER_SCRIPT="$PROJECT_DIR/scripts/server/run_http_server.py"
PID_FILE="/tmp/mcp-memory-http-server.pid"
START_TIME_FILE="/tmp/mcp-memory-http-server-start.time"
LOG_FILE="/tmp/mcp-memory-http-server.log"
ENV_FILE="$PROJECT_DIR/.env"
HTTP_ENDPOINT="http://127.0.0.1:8000"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function: Get package version from _version.py
get_package_version() {
    local version_file="$PROJECT_DIR/src/mcp_memory_service/_version.py"
    if [ -f "$version_file" ]; then
        grep -E '^__version__\s*=\s*["\047]' "$version_file" | sed 's/^__version__[^"'\'']*["'\'']\([^"'\'']*\)["'\''].*/\1/'
    else
        echo "unknown"
    fi
}

# Function: Get running server version from health endpoint
get_server_version() {
    # Try HTTP first
    local version=$(curl -s --max-time 2 "$HTTP_ENDPOINT/api/health" 2>/dev/null | \
        python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('version', 'unknown'))" 2>/dev/null)

    if [ "$version" != "unknown" ] && [ -n "$version" ]; then
        echo "$version"
        return
    fi

    # Fallback to HTTPS
    curl -k -s --max-time 2 "https://127.0.0.1:8000/api/health" 2>/dev/null | \
        python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('version', 'unknown'))" 2>/dev/null || echo "unknown"
}

# Function: Check if server process is running
is_process_running() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            return 0  # Running
        fi
    fi
    return 1  # Not running
}

# Function: Check HTTP/HTTPS health
check_http_health() {
    # Try HTTP first
    if curl -s --max-time 2 "$HTTP_ENDPOINT/api/health" > /dev/null 2>&1; then
        return 0
    fi
    # Fallback to HTTPS (with self-signed cert support)
    if curl -k -s --max-time 2 "https://127.0.0.1:8000/api/health" > /dev/null 2>&1; then
        return 0
    fi
    return 1
}

# Function: Get .env modification time
get_env_mtime() {
    if [ -f "$ENV_FILE" ]; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            stat -f %m "$ENV_FILE"
        else
            stat -c %Y "$ENV_FILE"
        fi
    else
        echo "0"
    fi
}

# Function: Get server start time
get_server_start_time() {
    if [ -f "$START_TIME_FILE" ]; then
        cat "$START_TIME_FILE"
    else
        echo "0"
    fi
}

# Function: Find process listening on port 8000
find_port_process() {
    # Try lsof first (most reliable)
    if command -v lsof &>/dev/null; then
        lsof -ti:8000 2>/dev/null || true
    # Fallback to ss (modern Linux systems)
    elif command -v ss &>/dev/null; then
        ss -tlnp 2>/dev/null | grep ":8000 " | sed -n 's/.*pid=\([0-9]*\).*/\1/p'
    # Fallback to netstat (older systems)
    elif command -v netstat &>/dev/null; then
        netstat -tlnp 2>/dev/null | grep ":8000 " | awk '{print $7}' | cut -d'/' -f1
    else
        # No tools available, try grepping process list as last resort
        ps aux | grep -E "run_http_server\.py|uvicorn.*:8000" | grep -v grep | awk '{print $2}'
    fi
}

# Function: Stop server
stop_server() {
    local force=${1:-false}

    # Kill process from PID file if it exists
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            if [ "$force" = true ]; then
                echo "Force stopping server (PID: $pid)..."
                kill -9 "$pid" 2>/dev/null || true
            else
                echo "Stopping server (PID: $pid)..."
                kill "$pid" 2>/dev/null || true
            fi

            # Wait for process to die
            local count=0
            while kill -0 "$pid" 2>/dev/null && [ $count -lt 10 ]; do
                sleep 0.5
                count=$((count + 1))
            done

            # Force kill if still alive
            if kill -0 "$pid" 2>/dev/null; then
                echo "Force killing stubborn process..."
                kill -9 "$pid" 2>/dev/null || true
                sleep 1
            fi
        fi
        rm -f "$PID_FILE"
        rm -f "$START_TIME_FILE"
    fi

    # Also kill any orphaned processes listening on port 8000
    local port_pids=$(find_port_process)
    if [ -n "$port_pids" ]; then
        echo "Found orphaned process(es) on port 8000, killing..."
        for pid in $port_pids; do
            echo "  Killing PID: $pid"
            kill -9 "$pid" 2>/dev/null || true
        done
        sleep 1
    fi
}

# Function: Start server
start_server() {
    local mode=${1:-background}  # background or foreground

    cd "$PROJECT_DIR"

    # Use venv Python directly (avoids uv/.venv vs venv mismatch with Python 3.14)
    local VENV_PYTHON="$PROJECT_DIR/venv/bin/python"
    if [ ! -f "$VENV_PYTHON" ]; then
        echo "Warning: venv not found at $VENV_PYTHON, falling back to uv run"
        VENV_PYTHON="uv run python"
    fi

    if [ "$mode" = "foreground" ]; then
        echo "Starting HTTP server in foreground..."
        $VENV_PYTHON "$HTTP_SERVER_SCRIPT"
    else
        echo "Starting HTTP server in background..."
        nohup $VENV_PYTHON "$HTTP_SERVER_SCRIPT" > "$LOG_FILE" 2>&1 &
        local pid=$!
        echo "$pid" > "$PID_FILE"
        date +%s > "$START_TIME_FILE"

        # Wait for server to initialize (hybrid storage takes ~4 seconds)
        echo "Waiting for server to initialize..."
        local count=0
        local max_wait=10
        while [ $count -lt $max_wait ]; do
            sleep 1
            count=$((count + 1))
            if check_http_health; then
                echo -e "${GREEN}✓${NC} HTTP server started successfully (PID: $pid, ready after ${count}s)"
                echo "  Logs: $LOG_FILE"
                return 0
            fi
        done

        # Failed to start within timeout
        echo -e "${RED}✗${NC} HTTP server failed to start within ${max_wait}s"
        echo "  Check logs: $LOG_FILE"
        cat "$LOG_FILE" | tail -20
        return 1
    fi
}

# Function: Determine if restart is needed
needs_restart() {
    local reason=""

    # Check 1: Process not running
    if ! is_process_running; then
        if check_http_health; then
            # Server responding but PID file missing (orphaned)
            reason="pid_file_missing"
        else
            # Server not running at all
            reason="not_running"
        fi
        echo "$reason"
        return 0
    fi

    # Check 2: HTTP health check fails
    if ! check_http_health; then
        reason="health_check_failed"
        echo "$reason"
        return 0
    fi

    # Check 3: Version mismatch
    local server_version=$(get_server_version)
    local package_version=$(get_package_version)
    if [ "$server_version" != "$package_version" ] && [ "$server_version" != "unknown" ] && [ "$package_version" != "unknown" ]; then
        reason="version_mismatch:$server_version→$package_version"
        echo "$reason"
        return 0
    fi

    # Check 4: Config file changed after server start
    local env_mtime=$(get_env_mtime)
    local server_start=$(get_server_start_time)
    if [ "$env_mtime" -gt "$server_start" ] && [ "$server_start" != "0" ]; then
        reason="config_changed"
        echo "$reason"
        return 0
    fi

    # All checks passed - no restart needed
    echo "healthy"
    return 1
}

# Command: status
cmd_status() {
    echo "=== HTTP Server Status ==="
    echo ""

    if is_process_running; then
        local pid=$(cat "$PID_FILE")
        echo -e "Process: ${GREEN}Running${NC} (PID: $pid)"
    else
        echo -e "Process: ${RED}Not running${NC}"
    fi

    if check_http_health; then
        echo -e "Health: ${GREEN}Healthy${NC}"
        local server_version=$(get_server_version)
        echo "Server Version: $server_version"
    else
        echo -e "Health: ${RED}Unhealthy${NC}"
    fi

    local package_version=$(get_package_version)
    echo "Package Version: $package_version"

    if [ -f "$START_TIME_FILE" ]; then
        local start_time=$(cat "$START_TIME_FILE")
        local current_time=$(date +%s)
        local uptime=$((current_time - start_time))
        echo "Uptime: ${uptime}s"
    fi

    echo ""
    set +e  # Temporarily disable exit-on-error for needs_restart
    restart_reason=$(needs_restart)
    restart_exit_code=$?
    set -e  # Re-enable exit-on-error
    if [ $restart_exit_code -eq 0 ]; then
        echo -e "Status: ${YELLOW}Needs restart${NC} ($restart_reason)"
    else
        echo -e "Status: ${GREEN}Healthy${NC}"
    fi
}

# Command: start
cmd_start() {
    local mode=${1:-background}

    if is_process_running && check_http_health; then
        echo -e "${GREEN}✓${NC} HTTP server is already running"
        return 0
    fi

    if is_process_running; then
        echo "Stale process detected, cleaning up..."
        stop_server
    fi

    start_server "$mode"
}

# Command: stop
cmd_stop() {
    local had_process=false

    # Check both PID file and port
    if is_process_running; then
        had_process=true
    fi

    local port_pids=$(find_port_process)
    if [ -n "$port_pids" ]; then
        had_process=true
    fi

    if [ "$had_process" = false ]; then
        echo "HTTP server is not running"
        return 0
    fi

    stop_server false
    echo -e "${GREEN}✓${NC} HTTP server stopped"
}

# Command: restart
cmd_restart() {
    echo "Restarting HTTP server..."
    cmd_stop
    sleep 1
    cmd_start background
}

# Command: auto-start (smart startup for shell integration)
cmd_auto_start() {
    local silent=${1:-false}

    set +e  # Temporarily disable exit-on-error for needs_restart
    restart_reason=$(needs_restart)
    restart_exit_code=$?
    set -e  # Re-enable exit-on-error
    if [ $restart_exit_code -eq 0 ]; then
        if [ "$silent" != "true" ]; then
            echo "HTTP server needs restart: $restart_reason"
        fi

        stop_server true
        start_server background
    else
        # Server is healthy, do nothing
        if [ "$silent" != "true" ]; then
            echo -e "${GREEN}✓${NC} HTTP server is healthy"
        fi
    fi
}

# Command: logs
cmd_logs() {
    local follow=${1:-false}

    if [ "$follow" = "true" ] || [ "$follow" = "-f" ]; then
        tail -f "$LOG_FILE"
    else
        tail -50 "$LOG_FILE"
    fi
}

# Main command dispatcher
case "${1:-status}" in
    status)
        cmd_status
        ;;
    start)
        cmd_start "${2:-background}"
        ;;
    stop)
        cmd_stop
        ;;
    restart)
        cmd_restart
        ;;
    auto-start)
        cmd_auto_start "${2:-false}"
        ;;
    logs)
        cmd_logs "${2:-false}"
        ;;
    *)
        echo "Usage: $0 {status|start|stop|restart|auto-start|logs}"
        echo ""
        echo "Commands:"
        echo "  status      - Show server status and health"
        echo "  start       - Start server (use 'start foreground' for interactive)"
        echo "  stop        - Stop server"
        echo "  restart     - Restart server"
        echo "  auto-start  - Smart startup (checks health, restarts if needed)"
        echo "  logs        - Show recent logs (use 'logs -f' to follow)"
        echo ""
        echo "Shell Integration:"
        echo "  Add to ~/.zshrc or ~/.bash_profile:"
        echo "  source $0 auto-start true"
        exit 1
        ;;
esac
