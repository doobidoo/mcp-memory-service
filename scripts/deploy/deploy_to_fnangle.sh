#!/bin/bash
#
# deploy_to_fnangle.sh - Automated deployment to Windows WSL2 via SSH
#
# Deploys MCP Memory Service to fnangle (Windows machine with WSL2 + RTX 4090)
# via SSH. Handles file sync, GPU setup, and Docker deployment.
#
# Prerequisites:
#   - SSH access to fnangle configured in ~/.ssh/config or known_hosts
#   - WSL2 installed on fnangle
#   - Docker Desktop for Windows running with WSL2 backend
#
# Usage:
#   ./deploy_to_fnangle.sh [--user USERNAME] [--no-gpu]
#

set -euo pipefail

# Cleanup handler for interruptions
cleanup() {
    local exit_code=$?
    if [[ -f /tmp/mcp-memory-deploy.tar.gz ]]; then
        echo -e "\n${YELLOW}Cleaning up temporary files...${NC}"
        rm -f /tmp/mcp-memory-deploy.tar.gz
    fi
    if [[ -f /tmp/docker-compose.override.yml ]]; then
        rm -f /tmp/docker-compose.override.yml
    fi
    exit $exit_code
}
trap cleanup EXIT INT TERM

# Color output
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly RED='\033[0;31m'
readonly NC='\033[0m'

# Configuration
WINDOWS_HOST="fnangle"
WINDOWS_USER="${USER}"  # Default to current user, override with --user
WSL_DISTRO="Ubuntu"     # Default WSL distribution
PROJECT_NAME="mcp-memory-service"
REMOTE_PROJECT_DIR="/mnt/c/Users/\$USER/code/27B/mcp/$PROJECT_NAME"
USE_GPU=true
SKIP_MIGRATION=false

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --user)
            WINDOWS_USER="$2"
            shift 2
            ;;
        --no-gpu)
            USE_GPU=false
            shift
            ;;
        --skip-migration)
            SKIP_MIGRATION=true
            shift
            ;;
        --help)
            grep "^#" "$0" | grep -v "#!/bin/bash" | sed 's/^# //' | sed 's/^#//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Update remote project dir with actual username
REMOTE_PROJECT_DIR="/mnt/c/Users/${WINDOWS_USER}/code/27B/mcp/$PROJECT_NAME"

# Display banner
echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}  Deploy to fnangle (Windows WSL2 + RTX 4090)               ${CYAN}║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Target:${NC} ${WINDOWS_USER}@${WINDOWS_HOST}"
echo -e "${BLUE}GPU:${NC} $([ "$USE_GPU" == "true" ] && echo "Enabled (RTX 4090)" || echo "Disabled")"
echo ""

# Test SSH connection
echo -e "${BLUE}Testing SSH connection to ${WINDOWS_USER}@${WINDOWS_HOST}...${NC}"
echo -e "${CYAN}→ ssh -o ConnectTimeout=5 ${WINDOWS_USER}@${WINDOWS_HOST} 'echo SSH OK'${NC}"

if ! SSH_TEST=$(ssh -o ConnectTimeout=5 "${WINDOWS_USER}@${WINDOWS_HOST}" "echo 'SSH OK'" 2>&1); then
    echo -e "${RED}ERROR: Cannot connect to ${WINDOWS_USER}@${WINDOWS_HOST}${NC}"
    echo -e "${YELLOW}Error output: $SSH_TEST${NC}"
    echo "Check SSH config or add to ~/.ssh/config:"
    cat <<EOF

Host fnangle
    HostName <ip-address>
    User ${WINDOWS_USER}
    IdentityFile ~/.ssh/id_rsa

EOF
    exit 1
fi
echo -e "${GREEN}✓ SSH connection successful: $SSH_TEST${NC}"

# Check WSL is available
echo -e "${BLUE}Checking WSL2 availability...${NC}"
echo -e "${CYAN}→ ssh ${WINDOWS_USER}@${WINDOWS_HOST} 'wsl echo WSL OK'${NC}"

WSL_TEST=$(ssh "${WINDOWS_USER}@${WINDOWS_HOST}" "wsl echo 'WSL OK'" 2>&1)
if [[ $? -ne 0 ]]; then
    echo -e "${RED}ERROR: WSL not accessible on fnangle${NC}"
    echo -e "${YELLOW}Error output: $WSL_TEST${NC}"
    exit 1
fi
echo -e "${GREEN}✓ WSL2 is available: $WSL_TEST${NC}"

# Check Docker
echo -e "${BLUE}Checking Docker...${NC}"
echo -e "${CYAN}→ ssh ${WINDOWS_USER}@${WINDOWS_HOST} 'wsl docker info'${NC}"

DOCKER_TEST=$(ssh "${WINDOWS_USER}@${WINDOWS_HOST}" 'wsl bash -c "docker info 2>&1 | head -5"')
if [[ $? -ne 0 ]]; then
    echo -e "${RED}ERROR: Docker not running or not accessible in WSL${NC}"
    echo "Start Docker Desktop on fnangle"
    echo -e "${YELLOW}Error output: $DOCKER_TEST${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker is running${NC}"
echo -e "${CYAN}  Docker info:${NC}"
echo "$DOCKER_TEST" | sed 's/^/    /'

# Check GPU (if enabled)
if [[ "$USE_GPU" == "true" ]]; then
    echo -e "${BLUE}Checking NVIDIA GPU...${NC}"
    echo -e "${CYAN}→ ssh ${WINDOWS_USER}@${WINDOWS_HOST} 'wsl nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'${NC}"

    GPU_INFO=$(ssh "${WINDOWS_USER}@${WINDOWS_HOST}" "wsl nvidia-smi --query-gpu=name,memory.total --format=csv,noheader" 2>&1)
    GPU_EXIT=$?

    if [[ $GPU_EXIT -eq 0 ]] && [[ -n "$GPU_INFO" ]]; then
        echo -e "${GREEN}✓ GPU detected: $GPU_INFO${NC}"

        # Show full nvidia-smi output for details
        echo -e "${CYAN}  Full GPU info:${NC}"
        ssh "${WINDOWS_USER}@${WINDOWS_HOST}" 'wsl bash -c "nvidia-smi 2>&1 | head -15"' | sed 's/^/    /'
    else
        echo -e "${YELLOW}⚠ GPU not detected or nvidia-smi not available${NC}"
        echo -e "${YELLOW}  Error: $GPU_INFO${NC}"
        echo -e "${YELLOW}  Continuing without GPU acceleration${NC}"
        USE_GPU=false
    fi
fi

echo ""

# Sync project files to fnangle
echo -e "${BLUE}Syncing project files to fnangle...${NC}"

# Create remote directory structure
echo -e "${CYAN}→ Creating remote directory: ${REMOTE_PROJECT_DIR}${NC}"
ssh "${WINDOWS_USER}@${WINDOWS_HOST}" "wsl bash -c 'mkdir -p \"${REMOTE_PROJECT_DIR}\"'"
echo -e "${GREEN}✓ Remote directory created${NC}"

# Rsync project files (excluding .git, node_modules, etc.)
# Use WSL path directly - OpenSSH on Windows doesn't understand C:\ paths
echo -e "${CYAN}→ rsync to ${WINDOWS_USER}@${WINDOWS_HOST}:${REMOTE_PROJECT_DIR}${NC}"
echo -e "${YELLOW}  Excluding: .git, node_modules, __pycache__, .venv, data/${NC}"

rsync -avz --progress \
    --exclude='.git' \
    --exclude='node_modules' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.venv' \
    --exclude='data/' \
    --exclude='.pytest_cache' \
    --exclude='.mypy_cache' \
    --exclude='.ruff_cache' \
    --exclude='.coverage' \
    --exclude='dist' \
    --exclude='build' \
    --exclude='.DS_Store' \
    -e ssh \
    "$PROJECT_ROOT/" \
    "${WINDOWS_USER}@${WINDOWS_HOST}:${REMOTE_PROJECT_DIR}" 2>&1 | tee /tmp/rsync.log || {
        echo -e "${RED}ERROR: rsync failed, trying tar fallback...${NC}"

        # Try direct WSL path via ssh
        echo -e "${CYAN}→ Creating tar archive of project files${NC}"
        tar -czf /tmp/mcp-memory-deploy.tar.gz \
            --exclude='.git' \
            --exclude='node_modules' \
            --exclude='__pycache__' \
            --exclude='*.pyc' \
            --exclude='.venv' \
            --exclude='data/' \
            -C "$PROJECT_ROOT" .

        TAR_SIZE=$(du -h /tmp/mcp-memory-deploy.tar.gz | cut -f1)
        echo -e "${GREEN}✓ Tar archive created: $TAR_SIZE${NC}"

        echo -e "${CYAN}→ Copying tar to fnangle (Windows home)${NC}"
        if ! scp /tmp/mcp-memory-deploy.tar.gz "${WINDOWS_USER}@${WINDOWS_HOST}:code/27B/mcp/mcp-memory-service/"; then
            echo -e "${RED}ERROR: scp failed - cannot copy archive to remote host${NC}"
            rm -f /tmp/mcp-memory-deploy.tar.gz
            exit 1
        fi
        echo -e "${GREEN}✓ Tar copied${NC}"

        echo -e "${CYAN}→ Extracting tar on fnangle to ${REMOTE_PROJECT_DIR}${NC}"
        CMD="tar -xzf ${REMOTE_PROJECT_DIR}/mcp-memory-deploy.tar.gz -C ${REMOTE_PROJECT_DIR} && rm -f ${REMOTE_PROJECT_DIR}/mcp-memory-deploy.tar.gz"
        if ! ssh "${WINDOWS_USER}@${WINDOWS_HOST}" "wsl bash -c \"$CMD\""; then
            echo -e "${RED}ERROR: tar extraction failed on remote host${NC}"
            rm -f /tmp/mcp-memory-deploy.tar.gz
            exit 1
        fi
        echo -e "${GREEN}✓ Files extracted${NC}"

        rm -f /tmp/mcp-memory-deploy.tar.gz
    }

echo -e "${GREEN}✓ Files synced successfully${NC}"

# Verify directory exists and count files
if ssh "${WINDOWS_USER}@${WINDOWS_HOST}" 'wsl bash -c "test -d '"${REMOTE_PROJECT_DIR}"'"'; then
    FILE_COUNT=$(ssh "${WINDOWS_USER}@${WINDOWS_HOST}" 'wsl bash -c "find '"${REMOTE_PROJECT_DIR}"' -type f 2>/dev/null | wc -l"')
    if [[ -n "$FILE_COUNT" ]] && [[ "$FILE_COUNT" -gt 0 ]]; then
        echo -e "${CYAN}  Total files synced: $FILE_COUNT${NC}"
    else
        echo -e "${YELLOW}  Warning: No files found in remote directory${NC}"
    fi
else
    echo -e "${RED}ERROR: Remote directory does not exist after sync${NC}"
    exit 1
fi
echo ""

# Create GPU-enabled docker-compose if needed
if [[ "$USE_GPU" == "true" ]]; then
    echo -e "${BLUE}Configuring GPU support for RTX 4090...${NC}"
    echo -e "${CYAN}→ Creating docker-compose.override.yml with NVIDIA GPU configuration${NC}"

    # Create file locally, then copy to remote (avoids quote escaping nightmare)
    cat > /tmp/docker-compose.override.yml << 'EOF'
version: "3.8"

services:
  mcp-memory:
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - MCP_MEMORY_DEVICE=cuda
      - MCP_MEMORY_EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
EOF

    # Copy to remote (to Windows home so it's accessible)
    if ! scp /tmp/docker-compose.override.yml "${WINDOWS_USER}@${WINDOWS_HOST}:docker-compose.override.yml"; then
        echo -e "${RED}ERROR: Failed to copy GPU config to remote${NC}"
        rm -f /tmp/docker-compose.override.yml
        exit 1
    fi

    if ! ssh "${WINDOWS_USER}@${WINDOWS_HOST}" 'wsl bash -c "mkdir -p '"${REMOTE_PROJECT_DIR}"'/deployment/local-wsl && mv /mnt/c/Users/fish/docker-compose.override.yml '"${REMOTE_PROJECT_DIR}"'/deployment/local-wsl/"'; then
        echo -e "${RED}ERROR: Failed to move GPU config to correct location${NC}"
        rm -f /tmp/docker-compose.override.yml
        exit 1
    fi

    rm -f /tmp/docker-compose.override.yml

    echo -e "${GREEN}✓ GPU configuration created${NC}"
    echo -e "${CYAN}  GPU device: CUDA device 0 (RTX 4090)${NC}"
    echo -e "${CYAN}  Embedding model: all-mpnet-base-v2 (768 dims, GPU-accelerated)${NC}"
fi

echo ""

# Run deployment script on fnangle
echo -e "${BLUE}Running deployment on fnangle (this will show live output)...${NC}"
echo -e "${CYAN}→ ssh -t ${WINDOWS_USER}@${WINDOWS_HOST} 'wsl bash scripts/deploy/deploy_local_wsl.sh'${NC}"
echo ""
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}  Live deployment output from fnangle:${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

MIGRATION_FLAG=""
[[ "$SKIP_MIGRATION" == "true" ]] && MIGRATION_FLAG="--skip-migration"

ssh -t "${WINDOWS_USER}@${WINDOWS_HOST}" 'wsl bash -c "cd '"${REMOTE_PROJECT_DIR}"' && bash scripts/deploy/deploy_local_wsl.sh '"${MIGRATION_FLAG}"'"'

DEPLOY_EXIT=$?

echo ""

if [[ $DEPLOY_EXIT -eq 0 ]]; then
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║${NC}  🎉 Deployment Successful!                                 ${GREEN}║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # Get Tailscale IP if available
    TAILSCALE_IP=$(ssh "${WINDOWS_USER}@${WINDOWS_HOST}" "wsl ip addr show tailscale0 2>/dev/null | grep 'inet ' | awk '{print \$2}' | cut -d/ -f1" || echo "")

    echo -e "${CYAN}Access URLs:${NC}"
    echo "  From fnangle:     http://localhost:8000"
    [[ -n "$TAILSCALE_IP" ]] && echo "  Via Tailscale:    http://$TAILSCALE_IP:8000"
    echo "  SSH tunnel:       ssh -L 8000:localhost:8000 ${WINDOWS_USER}@${WINDOWS_HOST}"
    echo ""

    echo -e "${CYAN}Useful Commands:${NC}"
    echo "  View logs:   ssh ${WINDOWS_USER}@${WINDOWS_HOST} \"wsl bash -c 'cd \\\"${REMOTE_PROJECT_DIR}/deployment/local-wsl\\\" && docker compose logs -f'\""
    echo "  Restart:     ssh ${WINDOWS_USER}@${WINDOWS_HOST} \"wsl bash -c 'cd \\\"${REMOTE_PROJECT_DIR}/deployment/local-wsl\\\" && docker compose restart'\""
    echo "  GPU check:   ssh ${WINDOWS_USER}@${WINDOWS_HOST} \"wsl docker exec mcp-memory nvidia-smi\""
    echo ""

    # Test health endpoint
    echo -e "${BLUE}Testing health endpoint...${NC}"
    if ssh "${WINDOWS_USER}@${WINDOWS_HOST}" "wsl bash -c 'curl -sf http://localhost:8000/api/health | grep -q '\\\"status\\\"'\\\"ok\\\"'"; then
        HEALTH_OUTPUT=$(ssh "${WINDOWS_USER}@${WINDOWS_HOST}" "wsl bash -c 'curl -sf http://localhost:8000/api/health'")
        HEALTH_EXIT=$?
        if [[ $HEALTH_EXIT -eq 0 ]] && [[ -n "$HEALTH_OUTPUT" ]]; then
            echo -e "${GREEN}✓ Service is healthy${NC}"
            echo "$HEALTH_OUTPUT" | jq . 2>/dev/null || echo "$HEALTH_OUTPUT"
        else
            echo -e "${YELLOW}⚠ Health check unavailable (service may still be starting)${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ Health check pending (service may still be starting)${NC}"
    fi

    echo ""

else
    echo -e "${RED}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║${NC}  Deployment Failed                                         ${RED}║${NC}"
    echo -e "${RED}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Check logs with:"
    echo "  ssh ${WINDOWS_USER}@${WINDOWS_HOST} \"wsl bash -c 'cd \\\"${REMOTE_PROJECT_DIR}/deployment/local-wsl\\\" && docker compose logs'\""
    exit 1
fi
