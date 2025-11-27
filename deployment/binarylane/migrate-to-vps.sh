#!/bin/bash
# MCP Memory Service - VPS Migration Script
# Migrates Qdrant data and codebase to BinaryLane VPS
#
# Usage: ./migrate-to-vps.sh [--data-only|--code-only|--all]
#
# Prerequisites:
#   - Tailscale connected (hostname 'mem' reachable)
#   - VPS bootstrapped with bootstrap.sh
#   - Local Docker running with Qdrant data

set -euo pipefail

# Configuration
VPS_HOST="${VPS_HOST:-mem}"
VPS_USER="${VPS_USER:-fish}"
VPS_APP_DIR="/opt/mcp-memory"
LOCAL_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
QDRANT_VOLUME="mcp-memory-service_qdrant-server-data"
BACKUP_DIR="/tmp/mcp-migration-$$"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Parse arguments
ACTION="${1:-all}"

case "$ACTION" in
    --data-only|data)
        MIGRATE_DATA=true
        MIGRATE_CODE=false
        ;;
    --code-only|code)
        MIGRATE_DATA=false
        MIGRATE_CODE=true
        ;;
    --all|all)
        MIGRATE_DATA=true
        MIGRATE_CODE=true
        ;;
    --help|-h)
        echo "Usage: $0 [--data-only|--code-only|--all]"
        echo ""
        echo "Options:"
        echo "  --data-only, data    Only migrate Qdrant data"
        echo "  --code-only, code    Only sync codebase"
        echo "  --all, all           Migrate both (default)"
        echo ""
        echo "Environment variables:"
        echo "  VPS_HOST   Hostname/IP of VPS (default: mem)"
        echo "  VPS_USER   SSH user (default: fish)"
        exit 0
        ;;
    *)
        log_error "Unknown option: $ACTION"
        echo "Use --help for usage"
        exit 1
        ;;
esac

echo "========================================="
echo "MCP Memory Service - VPS Migration"
echo "========================================="
echo ""
echo "VPS Target: ${VPS_USER}@${VPS_HOST}:${VPS_APP_DIR}"
echo "Local Project: ${LOCAL_PROJECT_DIR}"
echo "Migrate Data: ${MIGRATE_DATA}"
echo "Migrate Code: ${MIGRATE_CODE}"
echo ""

# Check prerequisites
log_info "Checking prerequisites..."

# Check Tailscale/VPS connectivity
if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "${VPS_USER}@${VPS_HOST}" "echo 'Connected'" &>/dev/null; then
    log_error "Cannot connect to ${VPS_USER}@${VPS_HOST}"
    echo "Ensure:"
    echo "  1. Tailscale is running: tailscale status"
    echo "  2. VPS is online: ssh ${VPS_USER}@${VPS_HOST}"
    exit 1
fi
log_info "VPS connectivity: OK"

# Check Docker locally (if migrating data)
if [ "$MIGRATE_DATA" = true ]; then
    if ! docker info &>/dev/null; then
        log_error "Docker is not running locally"
        exit 1
    fi
    log_info "Local Docker: OK"

    # Check Qdrant volume exists
    if ! docker volume inspect "$QDRANT_VOLUME" &>/dev/null; then
        log_error "Qdrant volume '$QDRANT_VOLUME' not found"
        echo "Available volumes:"
        docker volume ls --format '  {{.Name}}' | grep -i qdrant || echo "  (none with 'qdrant' in name)"
        exit 1
    fi
    log_info "Qdrant volume found: $QDRANT_VOLUME"
fi

# Create backup directory
mkdir -p "$BACKUP_DIR"
trap "rm -rf $BACKUP_DIR" EXIT

echo ""

# ============================================
# STEP 1: Sync Codebase
# ============================================
if [ "$MIGRATE_CODE" = true ]; then
    log_info "=== Step 1: Syncing Codebase ==="

    # Files/dirs to exclude from sync
    RSYNC_EXCLUDES=(
        --exclude='.git'
        --exclude='.venv'
        --exclude='__pycache__'
        --exclude='*.pyc'
        --exclude='.pytest_cache'
        --exclude='.coverage'
        --exclude='htmlcov'
        --exclude='.eggs'
        --exclude='*.egg-info'
        --exclude='dist'
        --exclude='build'
        --exclude='.env'
        --exclude='.env.local'
        --exclude='*.db'
        --exclude='*.sqlite'
        --exclude='data/'
        --exclude='backups/'
        --exclude='.DS_Store'
        --exclude='*.tar.gz'
        --exclude='.serena'
        --exclude='.spec-workflow'
    )

    log_info "Syncing project files..."
    rsync -avz --progress --delete \
        "${RSYNC_EXCLUDES[@]}" \
        "${LOCAL_PROJECT_DIR}/" \
        "${VPS_USER}@${VPS_HOST}:${VPS_APP_DIR}/src/"

    log_info "Codebase sync complete"
    echo ""
fi

# ============================================
# STEP 2: Migrate Qdrant Data
# ============================================
if [ "$MIGRATE_DATA" = true ]; then
    log_info "=== Step 2: Migrating Qdrant Data ==="

    # Get memory count before migration
    log_info "Checking local Qdrant memory count..."
    LOCAL_COUNT=$(docker exec mcp-memory-service-qdrant-server-1 \
        curl -s http://localhost:6333/collections/memories 2>/dev/null | \
        grep -o '"points_count":[0-9]*' | grep -o '[0-9]*' || echo "unknown")
    log_info "Local memories: ${LOCAL_COUNT}"

    # Export Qdrant data
    log_info "Exporting Qdrant data from Docker volume..."
    BACKUP_FILE="${BACKUP_DIR}/qdrant-data.tar.gz"

    docker run --rm \
        -v "${QDRANT_VOLUME}:/source:ro" \
        -v "${BACKUP_DIR}:/backup" \
        alpine tar -czvf /backup/qdrant-data.tar.gz -C /source .

    BACKUP_SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
    log_info "Backup created: ${BACKUP_SIZE}"

    # Transfer to VPS
    log_info "Transferring backup to VPS..."
    scp -C "$BACKUP_FILE" "${VPS_USER}@${VPS_HOST}:${VPS_APP_DIR}/"

    # Stop existing containers on VPS (if running)
    log_info "Stopping existing containers on VPS..."
    ssh "${VPS_USER}@${VPS_HOST}" "cd ${VPS_APP_DIR} && docker compose -f docker-compose.qdrant.yml down 2>/dev/null || true"

    # Ensure Qdrant data directory exists
    ssh "${VPS_USER}@${VPS_HOST}" "mkdir -p ${VPS_APP_DIR}/data/qdrant"

    # Extract data on VPS
    log_info "Extracting Qdrant data on VPS..."
    ssh "${VPS_USER}@${VPS_HOST}" "cd ${VPS_APP_DIR} && tar -xzvf qdrant-data.tar.gz -C data/qdrant/"

    # Clean up remote backup file
    ssh "${VPS_USER}@${VPS_HOST}" "rm -f ${VPS_APP_DIR}/qdrant-data.tar.gz"

    log_info "Qdrant data migration complete"
    echo ""
fi

# ============================================
# STEP 3: Deploy on VPS
# ============================================
log_info "=== Step 3: Deploying Services ==="

# Ensure docker-compose file exists on VPS
if ! ssh "${VPS_USER}@${VPS_HOST}" "test -f ${VPS_APP_DIR}/docker-compose.qdrant.yml"; then
    log_info "Copying docker-compose.qdrant.yml to VPS..."
    scp "${LOCAL_PROJECT_DIR}/deployment/binarylane/docker-compose.qdrant.yml" \
        "${VPS_USER}@${VPS_HOST}:${VPS_APP_DIR}/"
fi

# Ensure .env exists
if ! ssh "${VPS_USER}@${VPS_HOST}" "test -f ${VPS_APP_DIR}/.env"; then
    log_warn ".env not found on VPS, copying template..."
    scp "${LOCAL_PROJECT_DIR}/deployment/binarylane/.env.template" \
        "${VPS_USER}@${VPS_HOST}:${VPS_APP_DIR}/.env"
    log_warn "Please review ${VPS_APP_DIR}/.env on VPS before starting"
fi

# Start services
log_info "Starting services on VPS..."
ssh "${VPS_USER}@${VPS_HOST}" "cd ${VPS_APP_DIR} && docker compose -f docker-compose.qdrant.yml up -d"

# Wait for services to be healthy
log_info "Waiting for services to start (30s)..."
sleep 30

# ============================================
# STEP 4: Verify Migration
# ============================================
log_info "=== Step 4: Verifying Migration ==="

# Check Qdrant health
if ssh "${VPS_USER}@${VPS_HOST}" "curl -s http://localhost:6333/health" | grep -q "ok"; then
    log_info "Qdrant health: OK"
else
    log_warn "Qdrant health check failed (may still be starting)"
fi

# Check MCP server health
if ssh "${VPS_USER}@${VPS_HOST}" "curl -s http://localhost:8001/health 2>/dev/null" | grep -q -i "healthy\|ok"; then
    log_info "MCP server health: OK"
else
    log_warn "MCP server health check pending (may still be loading model)"
fi

# Get memory count on VPS
if [ "$MIGRATE_DATA" = true ]; then
    log_info "Checking VPS memory count..."
    sleep 5  # Give Qdrant time to load
    VPS_COUNT=$(ssh "${VPS_USER}@${VPS_HOST}" \
        "curl -s http://localhost:6333/collections/memories 2>/dev/null" | \
        grep -o '"points_count":[0-9]*' | grep -o '[0-9]*' || echo "unknown")

    log_info "VPS memories: ${VPS_COUNT}"

    if [ "$LOCAL_COUNT" != "unknown" ] && [ "$VPS_COUNT" != "unknown" ]; then
        if [ "$LOCAL_COUNT" = "$VPS_COUNT" ]; then
            log_info "Memory count matches: ${LOCAL_COUNT}"
        else
            log_warn "Memory count mismatch! Local: ${LOCAL_COUNT}, VPS: ${VPS_COUNT}"
        fi
    fi
fi

echo ""
echo "========================================="
echo "=== Migration Complete ==="
echo "========================================="
echo ""
echo "Services running on VPS:"
ssh "${VPS_USER}@${VPS_HOST}" "cd ${VPS_APP_DIR} && docker compose -f docker-compose.qdrant.yml ps"
echo ""
echo "To check logs:"
echo "  ssh ${VPS_USER}@${VPS_HOST} 'cd ${VPS_APP_DIR} && docker compose -f docker-compose.qdrant.yml logs -f'"
echo ""
echo "To update Claude config (in ~/.claude.json):"
echo '  Change: "http://localhost:8001/mcp"'
echo '  To:     "http://mem:8001/mcp"'
echo ""
echo "Test with:"
echo "  curl http://${VPS_HOST}:8001/health"
