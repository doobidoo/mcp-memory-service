#!/bin/bash
#
# deploy_local_wsl.sh - Deploy MCP Memory Service locally on Windows WSL2
#
# Deploys containerized MCP Memory Service using Docker Desktop on WSL2.
# No cloud infrastructure needed - runs on localhost with Tailscale access.
#
# Prerequisites:
#   - Docker Desktop for Windows with WSL2 backend
#   - Tailscale installed on Windows (optional, for remote access)
#
# Usage:
#   ./deploy_local_wsl.sh [--skip-migration]
#

set -euo pipefail

# Color output
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly RED='\033[0;31m'
readonly NC='\033[0m'

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPLOYMENT_DIR="$PROJECT_ROOT/deployment/local-wsl"

# Parse arguments
WITH_MIGRATION=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-migration)
            WITH_MIGRATION=false
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

# Display banner
echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}  MCP Memory Service - Local WSL2 Deployment                ${CYAN}║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check prerequisites
echo -e "${BLUE}Checking prerequisites...${NC}"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}ERROR: Docker not found${NC}"
    echo "Install Docker Desktop for Windows with WSL2 backend"
    echo "https://docs.docker.com/desktop/install/windows-install/"
    exit 1
fi

# Check Docker is running
if ! docker info &> /dev/null; then
    echo -e "${RED}ERROR: Docker is not running${NC}"
    echo "Start Docker Desktop and try again"
    exit 1
fi

echo -e "${GREEN}✓ Docker Desktop is running${NC}"

# Check if running in WSL
if ! grep -qi microsoft /proc/version 2>/dev/null; then
    echo -e "${YELLOW}WARNING: Not running in WSL2${NC}"
    echo "This script is optimized for WSL2, but will continue anyway"
fi

# Get Tailscale IP if available
TAILSCALE_IP=$(ip addr show tailscale0 2>/dev/null | grep "inet " | awk '{print $2}' | cut -d/ -f1 || echo "")

if [[ -n "$TAILSCALE_IP" ]]; then
    echo -e "${GREEN}✓ Tailscale connected: $TAILSCALE_IP${NC}"
    ACCESS_URL="http://$TAILSCALE_IP:8000"
else
    echo -e "${YELLOW}⚠ Tailscale not detected (install on Windows for remote access)${NC}"
    ACCESS_URL="http://localhost:8000"
fi

echo ""

# Create deployment directory if needed
if [[ ! -d "$DEPLOYMENT_DIR" ]]; then
    echo -e "${BLUE}Creating deployment directory...${NC}"
    mkdir -p "$DEPLOYMENT_DIR"
fi

# Navigate to deployment directory
cd "$DEPLOYMENT_DIR"

# Build and start containers
echo -e "${BLUE}Building Docker images...${NC}"
docker compose build --progress=plain

echo ""
echo -e "${BLUE}Starting containers...${NC}"
docker compose up -d

echo ""
echo -e "${BLUE}Waiting for service to become healthy...${NC}"

# Wait for health check
for i in {1..30}; do
    if curl -s --connect-timeout 2 "http://localhost:8000/api/health" | grep -q '"status":"ok"'; then
        echo -e "${GREEN}✓ Service is healthy${NC}"
        break
    fi

    if [[ $i -eq 30 ]]; then
        echo -e "${RED}ERROR: Service health check timeout${NC}"
        echo "Check logs: docker compose logs mcp-memory"
        exit 1
    fi

    echo -n "."
    sleep 2
done

echo ""

# Optional migration
if [[ "$WITH_MIGRATION" == "true" ]]; then
    echo -e "${BLUE}Running Cloudflare migration...${NC}"

    # Check credentials
    if [[ -z "${CLOUDFLARE_API_TOKEN:-}" ]] || [[ -z "${CLOUDFLARE_ACCOUNT_ID:-}" ]] || [[ -z "${CLOUDFLARE_D1_DATABASE_ID:-}" ]]; then
        echo -e "${YELLOW}Skipping migration: Cloudflare credentials not set${NC}"
        echo "Set CLOUDFLARE_API_TOKEN, CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_D1_DATABASE_ID to migrate"
    else
        MIGRATION_SCRIPT="$PROJECT_ROOT/scripts/migrate/cloudflare_to_oracle.py"
        if [[ -f "$MIGRATION_SCRIPT" ]]; then
            python "$MIGRATION_SCRIPT" --target-url "http://localhost:8000" || \
                echo -e "${YELLOW}Migration failed, but service is running${NC}"
        fi
    fi
fi

# Success!
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║${NC}  🎉 Deployment Complete!                                   ${GREEN}║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Display access information
echo -e "${CYAN}Access URLs:${NC}"
echo "  Local:       http://localhost:8000"
[[ -n "$TAILSCALE_IP" ]] && echo "  Tailscale:   http://$TAILSCALE_IP:8000"
echo ""

# Display client configuration
echo -e "${CYAN}Client Configuration:${NC}"
echo ""
cat <<EOF
{
  "mcpServers": {
    "memory": {
      "command": "uv",
      "args": ["run", "memory", "server"],
      "env": {
        "MCP_MEMORY_STORAGE_BACKEND": "http_client",
        "MCP_HTTP_CLIENT_ENDPOINT": "$ACCESS_URL"
      }
    }
  }
}
EOF
echo ""

# Display useful commands
echo -e "${CYAN}Useful Commands:${NC}"
echo "  View logs:       docker compose logs -f"
echo "  Restart:         docker compose restart"
echo "  Stop:            docker compose down"
echo "  Check health:    curl http://localhost:8000/api/health"
echo ""

echo -e "${GREEN}✓ MCP Memory Service is running!${NC}"
echo ""
