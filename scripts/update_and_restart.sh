#!/bin/bash
# MCP Memory Service - One-Command Update & Restart
# Copyright 2026 Heinrich Krupp
# Licensed under Apache License 2.0
#
# Usage: ./scripts/update_and_restart.sh [--no-restart] [--force]
#
# This script provides a streamlined workflow to:
# 1. Pull latest changes from git
# 2. Install updated dependencies (editable mode)
# 3. Restart HTTP dashboard server
# 4. Verify version and health
#
# Target time: <2 minutes (typical: 60-90 seconds)

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
HTTP_MANAGER="$SCRIPT_DIR/service/http_server_manager.sh"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Options
NO_RESTART=false
FORCE=false
START_TIME=$(date +%s)

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-restart)
            NO_RESTART=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --no-restart    Update code but don't restart server"
            echo "  --force         Force update even with uncommitted changes"
            echo "  -h, --help      Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h for help"
            exit 1
            ;;
    esac
done

# Helper functions
log_info() {
    echo -e "${CYAN}ℹ${NC}  $1"
}

log_success() {
    echo -e "${GREEN}✓${NC}  $1"
}

log_warning() {
    echo -e "${YELLOW}⚠${NC}  $1"
}

log_error() {
    echo -e "${RED}✗${NC}  $1"
}

log_step() {
    echo -e "\n${BLUE}▶${NC}  $1"
}

elapsed_time() {
    local current=$(date +%s)
    echo $((current - START_TIME))
}

# Banner
echo ""
echo -e "${CYAN}╔════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  MCP Memory Service - Update & Restart     ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════╝${NC}"
echo ""

cd "$PROJECT_DIR"

# Step 1: Check for uncommitted changes
log_step "Checking repository status..."

if ! git diff-index --quiet HEAD --; then
    if [ "$FORCE" = true ]; then
        log_info "Auto-stashing local changes (--force flag)..."
        git stash push -m "Auto-stash before update $(date '+%Y-%m-%d %H:%M:%S')"
        log_success "Changes stashed"
    else
        log_warning "You have uncommitted changes:"
        git status --short | head -10
        echo ""
        read -p "Stash changes and continue? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "Stashing local changes..."
            git stash push -m "Auto-stash before update $(date '+%Y-%m-%d %H:%M:%S')"
            log_success "Changes stashed"
        else
            log_error "Update cancelled. Use --force to override."
            exit 1
        fi
    fi
fi

# Step 2: Get current version
log_step "Recording current version..."

CURRENT_VERSION=$(grep -E '^__version__\s*=\s*["\047]' src/mcp_memory_service/_version.py | sed 's/^__version__[^"'\'']*["'\'']\([^"'\'']*\)["'\''].*/\1/' || echo "unknown")
log_info "Current version: ${CURRENT_VERSION}"

# Step 3: Pull latest changes
log_step "Pulling latest changes from git..."

BEFORE_COMMIT=$(git rev-parse HEAD)
git pull --rebase

AFTER_COMMIT=$(git rev-parse HEAD)
if [ "$BEFORE_COMMIT" = "$AFTER_COMMIT" ]; then
    log_info "Already up-to-date (no new commits)"
else
    COMMIT_COUNT=$(git rev-list --count ${BEFORE_COMMIT}..${AFTER_COMMIT})
    log_success "Pulled ${COMMIT_COUNT} new commit(s)"

    # Show brief summary of changes
    echo ""
    git log --oneline --graph ${BEFORE_COMMIT}..${AFTER_COMMIT} | head -10
fi

# Step 4: Get new version
NEW_VERSION=$(grep -E '^__version__\s*=\s*["\047]' src/mcp_memory_service/_version.py | sed 's/^__version__[^"'\'']*["'\'']\([^"'\'']*\)["'\''].*/\1/' || echo "unknown")

if [ "$CURRENT_VERSION" != "$NEW_VERSION" ]; then
    log_info "Version change: ${CURRENT_VERSION} → ${NEW_VERSION}"
fi

# Step 5: Install dependencies (editable mode)
log_step "Installing dependencies (editable mode)..."

# Determine Python executable and version
VENV_DIR="$PROJECT_DIR/venv"
VENV_PIP="$VENV_DIR/bin/pip"
VENV_PYTHON="$VENV_DIR/bin/python"

# Check if venv exists and get its Python version
if [ -f "$VENV_PYTHON" ]; then
    VENV_PY_VERSION=$("$VENV_PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "unknown")
    log_info "Existing venv Python version: ${VENV_PY_VERSION}"
else
    VENV_PY_VERSION="none"
fi

# Find best Python (prefer 3.12 or 3.13, avoid 3.14+)
find_compatible_python() {
    for py in python3.12 python3.13 python3.11 python3.10; do
        if command -v "$py" &> /dev/null; then
            local version=$("$py" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
            local minor=$(echo "$version" | cut -d. -f2)
            if [ "$minor" -lt 14 ] 2>/dev/null; then
                echo "$py"
                return 0
            fi
        fi
    done
    # Fallback to python3 if nothing else found
    echo "python3"
}

# Check system Python version
SYS_PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "3.12")
SYS_PY_MINOR=$(echo "$SYS_PY_VERSION" | cut -d. -f2)

# Recreate venv if Python 3.14+ detected or venv missing/broken
NEEDS_VENV_RECREATE=false
if [ "$VENV_PY_VERSION" = "none" ]; then
    log_warning "No venv found, creating..."
    NEEDS_VENV_RECREATE=true
elif [ "$SYS_PY_MINOR" -ge 14 ] 2>/dev/null && [ ! -f "$VENV_DIR/.python312_compat" ]; then
    # System Python is 3.14+, check if venv was created with compatible Python
    VENV_MINOR=$(echo "$VENV_PY_VERSION" | cut -d. -f2)
    if [ "$VENV_MINOR" -ge 14 ] 2>/dev/null; then
        log_warning "Venv uses Python ${VENV_PY_VERSION} (incompatible with some packages)"
        log_info "Recreating venv with compatible Python..."
        NEEDS_VENV_RECREATE=true
    fi
fi

if [ "$NEEDS_VENV_RECREATE" = true ]; then
    COMPAT_PYTHON=$(find_compatible_python)
    COMPAT_VERSION=$("$COMPAT_PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "unknown")

    log_info "Using ${COMPAT_PYTHON} (Python ${COMPAT_VERSION}) for venv..."

    # Backup old venv if exists
    if [ -d "$VENV_DIR" ]; then
        rm -rf "$VENV_DIR.bak" 2>/dev/null || true
        mv "$VENV_DIR" "$VENV_DIR.bak"
    fi

    # Create new venv with compatible Python
    "$COMPAT_PYTHON" -m venv "$VENV_DIR"
    "$VENV_PIP" install --upgrade pip --quiet

    # Mark as compatible
    touch "$VENV_DIR/.python312_compat"

    log_success "Venv created with Python ${COMPAT_VERSION}"
fi

# Always use venv pip for installation (avoids system Python 3.14 issues)
log_info "Installing with venv pip (this may take 1-2 minutes)..."

# Detect platform and hardware using Python-based detection (consistent with install.py)
DETECTION_SCRIPT="$SCRIPT_DIR/utils/detect_platform.py"

if [ ! -f "$DETECTION_SCRIPT" ]; then
    log_warning "Platform detection script not found, using basic detection"
    # Fallback to simple detection
    PLATFORM=$(uname -s)
    if [ "$PLATFORM" = "Darwin" ]; then
        log_info "macOS detected - using CPU-only PyTorch (no CUDA downloads)"
        EXTRA_INDEX="--extra-index-url https://download.pytorch.org/whl/cpu"
    else
        if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
            log_info "NVIDIA GPU detected - using default PyTorch with CUDA"
            EXTRA_INDEX=""
        else
            log_info "No NVIDIA GPU detected - using CPU-only PyTorch"
            EXTRA_INDEX="--extra-index-url https://download.pytorch.org/whl/cpu"
        fi
    fi
    NEEDS_DIRECTML="False"
else
    # Use Python-based detection for comprehensive hardware support
    # Note: stderr not redirected to allow debug warnings from detect_platform.py
    PLATFORM_JSON=$("$VENV_PYTHON" "$DETECTION_SCRIPT")

    if [ $? -eq 0 ]; then
        # Parse JSON fields directly (Bash 3.2 compatible - no mapfile/readarray)
        # Note: Direct parsing is simpler and works on all bash versions
        ACCELERATOR=$(echo "$PLATFORM_JSON" | "$VENV_PYTHON" -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('accelerator', 'cpu'))
except (json.JSONDecodeError, TypeError, ValueError):
    print('cpu')
" 2>/dev/null)

        PYTORCH_INDEX=$(echo "$PLATFORM_JSON" | "$VENV_PYTHON" -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('pytorch_index_url', ''))
except (json.JSONDecodeError, TypeError, ValueError):
    print('')
" 2>/dev/null)

        NEEDS_DIRECTML=$(echo "$PLATFORM_JSON" | "$VENV_PYTHON" -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('needs_directml', False))
except (json.JSONDecodeError, TypeError, ValueError):
    print('False')
" 2>/dev/null)

        CUDA_VER=$(echo "$PLATFORM_JSON" | "$VENV_PYTHON" -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('cuda_version', 'Unknown'))
except (json.JSONDecodeError, TypeError, ValueError):
    print('Unknown')
" 2>/dev/null)

        ROCM_VER=$(echo "$PLATFORM_JSON" | "$VENV_PYTHON" -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('rocm_version', 'Unknown'))
except (json.JSONDecodeError, TypeError, ValueError):
    print('Unknown')
" 2>/dev/null)

        DIRECTML_VER=$(echo "$PLATFORM_JSON" | "$VENV_PYTHON" -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('directml_version', 'Unknown'))
except (json.JSONDecodeError, TypeError, ValueError):
    print('Unknown')
" 2>/dev/null)

        # Validate accelerator value
        if [ -z "$ACCELERATOR" ]; then
            ACCELERATOR="cpu"
        fi

        # Log detected accelerator
        case "$ACCELERATOR" in
            cuda)
                log_info "CUDA detected (${CUDA_VER}) - using optimized PyTorch"
                ;;
            rocm)
                log_info "ROCm detected (${ROCM_VER}) - using optimized PyTorch"
                ;;
            mps)
                log_info "Apple Silicon MPS detected - using MPS-optimized PyTorch"
                ;;
            directml)
                log_info "DirectML detected (${DIRECTML_VER}) - using CPU PyTorch + DirectML package"
                ;;
            *)
                log_info "No GPU acceleration detected - using CPU-only PyTorch"
                ;;
        esac

        # Set extra index URL
        if [ -n "$PYTORCH_INDEX" ]; then
            EXTRA_INDEX="--extra-index-url $PYTORCH_INDEX"
        else
            EXTRA_INDEX=""
        fi
    else
        log_warning "Platform detection failed, using CPU-only PyTorch"
        EXTRA_INDEX="--extra-index-url https://download.pytorch.org/whl/cpu"
        NEEDS_DIRECTML="False"
    fi
fi

"$VENV_PIP" install -e . $EXTRA_INDEX 2>&1 | while IFS= read -r line; do
    # Show progress for key actions only
    if echo "$line" | grep -qE "Processing|Installing|Collecting|Successfully|ERROR|WARNING"; then
        echo "  $line"
    fi
done

# Fallback if filtered output fails
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log_warning "Install had issues, retrying with full output..."
    "$VENV_PIP" install -e . $EXTRA_INDEX
fi

log_success "Dependencies installed"

# Install DirectML if needed (Windows GPU acceleration)
if [ "$NEEDS_DIRECTML" = "True" ]; then
    log_info "Installing torch-directml for DirectML support..."
    INSTALL_OUTPUT=$("$VENV_PIP" install torch-directml>=0.2.0 --quiet 2>&1)
    if [ $? -eq 0 ]; then
        log_success "torch-directml installed"
    else
        log_warning "Failed to install torch-directml, GPU acceleration may not be available"
        log_info "Installation output:"
        echo "$INSTALL_OUTPUT"
    fi
fi

# Verify installation
INSTALLED_VERSION=$("$VENV_PIP" show mcp-memory-service 2>/dev/null | grep "^Version:" | awk '{print $2}' || echo "unknown")
if [ "$INSTALLED_VERSION" != "$NEW_VERSION" ]; then
    log_warning "Installation version mismatch! Expected: ${NEW_VERSION}, Got: ${INSTALLED_VERSION}"
    log_warning "Retrying installation..."
    "$VENV_PIP" install -e . --force-reinstall --quiet
    INSTALLED_VERSION=$("$VENV_PIP" show mcp-memory-service 2>/dev/null | grep "^Version:" | awk '{print $2}' || echo "unknown")
fi

log_info "Installed version: ${INSTALLED_VERSION}"

# Step 6: Restart server (if requested)
if [ "$NO_RESTART" = true ]; then
    log_warning "Skipping server restart (--no-restart flag)"
else
    log_step "Restarting HTTP dashboard server..."

    if [ ! -x "$HTTP_MANAGER" ]; then
        log_error "HTTP manager not found or not executable: $HTTP_MANAGER"
        log_info "Starting server manually..."

        # Fallback: manual start using venv python
        pkill -f "run_http_server.py" 2>/dev/null || true
        sleep 2
        nohup "$VENV_PYTHON" scripts/server/run_http_server.py > /tmp/mcp-memory-update.log 2>&1 &
        sleep 8
    else
        # Use http_server_manager for smart restart
        "$HTTP_MANAGER" restart
    fi

    # Step 7: Health check
    log_step "Verifying server health..."

    # Try both HTTP and HTTPS (server may use either)
    HEALTH_URL_HTTP="http://127.0.0.1:8000/api/health"
    HEALTH_URL_HTTPS="https://127.0.0.1:8000/api/health"
    MAX_WAIT=15
    WAIT_COUNT=0

    while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
        # Try HTTPS first (most common), then HTTP
        if curl -sk --max-time 2 "$HEALTH_URL_HTTPS" > /dev/null 2>&1; then
            # Get health data
            HEALTH_DATA=$(curl -sk --max-time 2 "$HEALTH_URL_HTTPS")
            HEALTH_URL="$HEALTH_URL_HTTPS"
        elif curl -s --max-time 2 "$HEALTH_URL_HTTP" > /dev/null 2>&1; then
            # Get health data
            HEALTH_DATA=$(curl -s --max-time 2 "$HEALTH_URL_HTTP")
            HEALTH_URL="$HEALTH_URL_HTTP"
        else
            HEALTH_DATA=""
        fi

        if [ -n "$HEALTH_DATA" ]; then
            SERVER_VERSION=$(echo "$HEALTH_DATA" | "$VENV_PYTHON" -c "import sys, json; data=json.load(sys.stdin); print(data.get('version', 'unknown'))" 2>/dev/null || echo "unknown")

            if [ "$SERVER_VERSION" = "$NEW_VERSION" ]; then
                log_success "Server healthy and running version ${SERVER_VERSION}"
                break
            else
                log_warning "Server running old version: ${SERVER_VERSION} (expected: ${NEW_VERSION})"
                log_info "Waiting for server to reload... (${WAIT_COUNT}s)"
            fi
        fi

        sleep 1
        WAIT_COUNT=$((WAIT_COUNT + 1))
    done

    if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
        log_error "Server health check timeout after ${MAX_WAIT}s"
        log_info "Check logs: tail -50 /tmp/mcp-memory-*.log"
        exit 1
    fi
fi

# Step 8: Summary
TOTAL_TIME=$(elapsed_time)

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║          Update Complete! 🎉               ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════╝${NC}"
echo ""
log_success "Version: ${CURRENT_VERSION} → ${NEW_VERSION}"
log_success "Total time: ${TOTAL_TIME}s"
echo ""

# Show correct protocol based on health check result
if [ "$NO_RESTART" = false ] && [ -n "${HEALTH_URL:-}" ]; then
    PROTOCOL=$(echo "$HEALTH_URL" | grep -o "^https\?")
    log_info "Dashboard: ${PROTOCOL}://localhost:8000"
    log_info "API Docs:  ${PROTOCOL}://localhost:8000/api/docs"
else
    log_info "Dashboard: https://localhost:8000 (or http://localhost:8000)"
    log_info "API Docs:  https://localhost:8000/api/docs"
fi
echo ""

if [ "$CURRENT_VERSION" != "$NEW_VERSION" ]; then
    log_info "New version deployed. Check CHANGELOG.md for details:"
    echo ""
    grep -A 20 "## \[${NEW_VERSION}\]" CHANGELOG.md 2>/dev/null || log_warning "CHANGELOG not updated for ${NEW_VERSION}"
fi

exit 0
