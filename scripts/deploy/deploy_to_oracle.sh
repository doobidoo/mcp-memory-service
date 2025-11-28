#!/bin/bash
#
# deploy_to_oracle.sh - Master deployment orchestration script
#
# Automates complete deployment of MCP Memory Service to Oracle Cloud Free Tier
# with Tailscale VPN integration, Docker containerization, and optional data migration.
#
# Usage:
#   ./deploy_to_oracle.sh [OPTIONS]
#
# Options:
#   --with-migration       Run Cloudflare to Oracle data migration after deployment
#   --skip-provisioning    Skip Oracle instance provisioning (use existing instance)
#   --skip-tailscale       Skip Tailscale setup (already configured)
#   --dry-run              Simulate deployment without making changes
#   --help                 Show this help message
#
# Environment Variables:
#   OCI_COMPARTMENT_ID         - Oracle compartment OCID (required for provisioning)
#   OCI_SSH_KEY_FILE           - Path to SSH public key (default: ~/.ssh/id_rsa.pub)
#   TAILSCALE_AUTH_KEY         - Tailscale auth key (required for setup)
#   CLOUDFLARE_API_TOKEN       - Cloudflare API token (required for migration)
#   CLOUDFLARE_ACCOUNT_ID      - Cloudflare account ID (required for migration)
#   CLOUDFLARE_D1_DATABASE_ID  - Cloudflare D1 database ID (required for migration)
#

set -euo pipefail

# Script metadata
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPLOYMENT_DIR="$PROJECT_ROOT/deployment/oracle"

# Color output for logging
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*" >&2
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*" >&2
}

log_warn() {
    echo -e "${YELLOW}[WARNING]${NC} $*" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

log_step() {
    echo -e "${CYAN}==>${NC} $*" >&2
}

# Error handler
error_exit() {
    log_error "$1"
    log_error "Deployment failed. Check logs above for details."
    exit 1
}

# Cleanup handler for dry-run mode
cleanup() {
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Dry-run complete. No changes were made."
    fi
}
trap cleanup EXIT

# Parse command line arguments
WITH_MIGRATION=false
SKIP_PROVISIONING=false
SKIP_TAILSCALE=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-migration)
            WITH_MIGRATION=true
            shift
            ;;
        --skip-provisioning)
            SKIP_PROVISIONING=true
            shift
            ;;
        --skip-tailscale)
            SKIP_TAILSCALE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            grep "^#" "$0" | grep -v "#!/bin/bash" | sed 's/^# //' | sed 's/^#//'
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Run with --help for usage information"
            exit 1
            ;;
    esac
done

# Display deployment banner
echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}  Oracle Server Deployment - MCP Memory Service            ${CYAN}║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    log_warn "DRY-RUN MODE: No actual changes will be made"
    echo ""
fi

# Prerequisites check
log_step "Checking prerequisites..."

# Check required scripts exist
REQUIRED_SCRIPTS=(
    "$SCRIPT_DIR/provision_oracle.sh"
    "$SCRIPT_DIR/setup_tailscale.sh"
)

for script in "${REQUIRED_SCRIPTS[@]}"; do
    if [[ ! -x "$script" ]]; then
        error_exit "Required script not found or not executable: $script"
    fi
done

# Check deployment files exist
if [[ ! -f "$DEPLOYMENT_DIR/docker-compose.yml" ]]; then
    error_exit "docker-compose.yml not found in $DEPLOYMENT_DIR"
fi

if [[ ! -f "$DEPLOYMENT_DIR/Dockerfile" ]]; then
    error_exit "Dockerfile not found in $DEPLOYMENT_DIR"
fi

# Check required tools
REQUIRED_TOOLS=("ssh" "rsync" "jq")
for tool in "${REQUIRED_TOOLS[@]}"; do
    if ! command -v "$tool" &> /dev/null; then
        error_exit "$tool is required but not installed. Install it and try again."
    fi
done

log_success "All prerequisites satisfied"
echo ""

# Variables to track deployment state
PUBLIC_IP=""
TAILSCALE_IP=""
INSTANCE_OCID=""
SSH_USER="ubuntu"
SSH_KEY="${OCI_SSH_KEY_FILE:-$HOME/.ssh/id_rsa}"

# Step 1: Provision Oracle Cloud instance
if [[ "$SKIP_PROVISIONING" == "true" ]]; then
    log_step "Skipping Oracle instance provisioning (--skip-provisioning)"

    # User must provide PUBLIC_IP manually
    if [[ -z "${OCI_PUBLIC_IP:-}" ]]; then
        log_error "When using --skip-provisioning, you must set OCI_PUBLIC_IP environment variable"
        error_exit "Example: export OCI_PUBLIC_IP=123.45.67.89"
    fi
    PUBLIC_IP="$OCI_PUBLIC_IP"
    log_info "Using existing instance at $PUBLIC_IP"
else
    log_step "Step 1/5: Provisioning Oracle Cloud instance..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Would run: $SCRIPT_DIR/provision_oracle.sh"
        PUBLIC_IP="1.2.3.4"
        INSTANCE_OCID="ocid1.instance.oc1.ap-melbourne-1.dryrun"
    else
        # Run provision script with real-time output
        log_info "Running provision_oracle.sh (this may take 5-10 minutes)..."

        # Use tee to show output AND capture it
        PROVISION_OUTPUT=$("$SCRIPT_DIR/provision_oracle.sh" 2>&1 | tee /dev/tty) || error_exit "Oracle provisioning failed"

        # Extract JSON from last line of output
        PROVISION_JSON=$(echo "$PROVISION_OUTPUT" | tail -n1)

        # Parse JSON output
        PUBLIC_IP=$(echo "$PROVISION_JSON" | jq -r '.public_ip')
        INSTANCE_OCID=$(echo "$PROVISION_JSON" | jq -r '.instance_ocid')

        if [[ -z "$PUBLIC_IP" || "$PUBLIC_IP" == "null" ]]; then
            error_exit "Failed to get public IP from provisioning script"
        fi

        log_success "Instance provisioned: $PUBLIC_IP (OCID: $INSTANCE_OCID)"
    fi
fi

echo ""

# Step 2: Setup Tailscale VPN on Oracle instance
if [[ "$SKIP_TAILSCALE" == "true" ]]; then
    log_step "Skipping Tailscale setup (--skip-tailscale)"

    # User must provide TAILSCALE_IP manually
    if [[ -z "${ORACLE_TAILSCALE_IP:-}" ]]; then
        log_error "When using --skip-tailscale, you must set ORACLE_TAILSCALE_IP environment variable"
        error_exit "Example: export ORACLE_TAILSCALE_IP=100.64.1.2"
    fi
    TAILSCALE_IP="$ORACLE_TAILSCALE_IP"
    log_info "Using existing Tailscale IP: $TAILSCALE_IP"
else
    log_step "Step 2/5: Setting up Tailscale VPN..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Would run: setup_tailscale.sh via SSH on $PUBLIC_IP"
        TAILSCALE_IP="100.64.0.1"
    else
        # Wait for SSH to become available
        log_info "Waiting for SSH to become available on $PUBLIC_IP..."
        for i in {1..30}; do
            if ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=5 -o BatchMode=yes \
                "$SSH_USER@$PUBLIC_IP" "echo 'SSH ready'" &>/dev/null; then
                log_success "SSH connection established"
                break
            fi

            if [[ $i -eq 30 ]]; then
                error_exit "SSH connection timeout after 150 seconds"
            fi

            if [[ $((i % 5)) -eq 0 ]]; then
                log_info "Still waiting for SSH... ($((i * 5))/150 seconds)"
            fi
            sleep 5
        done

        # Copy Tailscale setup script to remote instance
        log_info "Copying setup_tailscale.sh to remote instance..."
        scp -i "$SSH_KEY" -o StrictHostKeyChecking=no \
            "$SCRIPT_DIR/setup_tailscale.sh" "$SSH_USER@$PUBLIC_IP:/tmp/" || \
            error_exit "Failed to copy setup_tailscale.sh to remote instance"

        # Run Tailscale setup on remote instance
        log_info "Running Tailscale setup on remote instance (installing packages, authenticating)..."
        TAILSCALE_OUTPUT=$(ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no \
            "$SSH_USER@$PUBLIC_IP" \
            "sudo bash /tmp/setup_tailscale.sh" 2>&1 | tee /dev/tty) || \
            error_exit "Tailscale setup failed"

        # Extract JSON from last line
        TAILSCALE_JSON=$(echo "$TAILSCALE_OUTPUT" | tail -n1)
        TAILSCALE_IP=$(echo "$TAILSCALE_JSON" | jq -r '.tailscale_ip')

        if [[ -z "$TAILSCALE_IP" || "$TAILSCALE_IP" == "null" ]]; then
            error_exit "Failed to get Tailscale IP from setup script"
        fi

        log_success "Tailscale configured: $TAILSCALE_IP"
    fi
fi

echo ""

# Step 3: Deploy application files and Docker configuration
log_step "Step 3/5: Deploying application files..."

if [[ "$DRY_RUN" == "true" ]]; then
    log_info "Would rsync deployment files to $PUBLIC_IP:/opt/mcp-memory"
    log_info "Would create .env file with TAILSCALE_IP=$TAILSCALE_IP"
else
    # Create deployment directory on remote
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no \
        "$SSH_USER@$PUBLIC_IP" \
        "sudo mkdir -p /opt/mcp-memory && sudo chown $SSH_USER:$SSH_USER /opt/mcp-memory" || \
        error_exit "Failed to create deployment directory"

    # Rsync deployment files
    log_info "Syncing deployment files (Dockerfile, docker-compose.yml, configs)..."
    rsync -avz --progress --delete -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
        "$DEPLOYMENT_DIR/" "$SSH_USER@$PUBLIC_IP:/opt/mcp-memory/" || \
        error_exit "Failed to sync deployment files"

    # Verify .env file exists and has TAILSCALE_IP
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no \
        "$SSH_USER@$PUBLIC_IP" \
        "grep -q 'TAILSCALE_IP=' /opt/mcp-memory/.env" || \
        log_warn ".env file may not have TAILSCALE_IP set correctly"

    log_success "Application files deployed"
fi

echo ""

# Step 4: Build and start Docker containers
log_step "Step 4/5: Building and starting Docker containers..."

if [[ "$DRY_RUN" == "true" ]]; then
    log_info "Would run: docker compose build"
    log_info "Would run: docker compose up -d"
else
    # Install Docker if not present
    log_info "Ensuring Docker is installed..."
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no \
        "$SSH_USER@$PUBLIC_IP" \
        "command -v docker &> /dev/null || (curl -fsSL https://get.docker.com | sudo sh && sudo usermod -aG docker $SSH_USER)" || \
        error_exit "Failed to install Docker"

    # Build Docker images
    log_info "Building Docker images (this may take 5-10 minutes for ARM64 build)..."
    log_info "You'll see output from: apt packages, Python dependencies, multi-stage build..."
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no \
        "$SSH_USER@$PUBLIC_IP" \
        "cd /opt/mcp-memory && docker compose build --progress=plain" 2>&1 | \
        grep -E "(Step|FROM|RUN|COPY|#|Building|built|naming)" || \
        error_exit "Docker build failed"

    log_success "Docker build complete"

    # Start containers
    log_info "Starting containers (mcp-memory + rclone backup)..."
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no \
        "$SSH_USER@$PUBLIC_IP" \
        "cd /opt/mcp-memory && docker compose up -d" || \
        error_exit "Failed to start containers"

    log_success "Containers started"
fi

echo ""

# Step 5: Verify deployment health
log_step "Step 5/5: Verifying deployment..."

if [[ "$DRY_RUN" == "true" ]]; then
    log_info "Would verify health endpoint at http://$TAILSCALE_IP:8000/api/health"
else
    # Wait for health check to pass
    log_info "Waiting for health check to pass (http://$TAILSCALE_IP:8000/api/health)..."
    log_info "Note: Ensure you're on Tailscale network to access this IP"
    HEALTH_CHECK_PASSED=false

    for i in {1..30}; do
        # Use curl from local machine if on Tailscale network
        HEALTH_RESPONSE=$(curl -s --connect-timeout 5 "http://$TAILSCALE_IP:8000/api/health" 2>&1)

        if echo "$HEALTH_RESPONSE" | grep -q '"status":"ok"'; then
            HEALTH_CHECK_PASSED=true
            log_success "Health check passed: $HEALTH_RESPONSE"
            break
        fi

        if [[ $i -eq 1 ]] || [[ $((i % 5)) -eq 0 ]]; then
            log_info "Attempt $i/30: Still waiting for health check... ($((i * 2))/60 seconds)"
            [[ -n "$HEALTH_RESPONSE" ]] && log_info "Response: $HEALTH_RESPONSE"
        fi
        sleep 2
    done

    if [[ "$HEALTH_CHECK_PASSED" != "true" ]]; then
        log_error "Health check failed after 60 seconds"
        log_info "Check container logs: ssh -i $SSH_KEY $SSH_USER@$PUBLIC_IP 'cd /opt/mcp-memory && docker compose logs'"
        error_exit "Deployment verification failed"
    fi
fi

echo ""

# Optional: Run data migration
if [[ "$WITH_MIGRATION" == "true" ]]; then
    log_step "Running Cloudflare to Oracle data migration..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Would run: python scripts/migrate/cloudflare_to_oracle.py --target-url http://$TAILSCALE_IP:8000"
    else
        # Check migration script exists
        MIGRATION_SCRIPT="$PROJECT_ROOT/scripts/migrate/cloudflare_to_oracle.py"
        if [[ ! -f "$MIGRATION_SCRIPT" ]]; then
            error_exit "Migration script not found: $MIGRATION_SCRIPT"
        fi

        # Verify Cloudflare credentials
        if [[ -z "${CLOUDFLARE_API_TOKEN:-}" ]] || [[ -z "${CLOUDFLARE_ACCOUNT_ID:-}" ]] || [[ -z "${CLOUDFLARE_D1_DATABASE_ID:-}" ]]; then
            error_exit "Cloudflare credentials required for migration. Set CLOUDFLARE_API_TOKEN, CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_D1_DATABASE_ID"
        fi

        # Run migration
        log_info "Starting data migration (this may take several minutes)..."
        if python "$MIGRATION_SCRIPT" --target-url "http://$TAILSCALE_IP:8000"; then
            log_success "Data migration completed successfully"
        else
            log_error "Data migration failed"
            log_warn "Deployment is functional, but migration did not complete"
        fi
    fi
    echo ""
fi

# Deployment complete!
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║${NC}  Deployment Complete!                                      ${GREEN}║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Display client configuration instructions
log_info "Client Configuration:"
echo ""
echo "Add to your Claude Desktop or Claude Code configuration:"
echo ""
echo -e "${CYAN}MCP_MEMORY_STORAGE_BACKEND=http_client${NC}"
echo -e "${CYAN}MCP_HTTP_CLIENT_ENDPOINT=http://$TAILSCALE_IP:8000${NC}"
echo ""
echo "Or use in .claude.json:"
echo ""
cat <<EOF
{
  "mcpServers": {
    "memory": {
      "command": "uv",
      "args": ["run", "memory", "server"],
      "env": {
        "MCP_MEMORY_STORAGE_BACKEND": "http_client",
        "MCP_HTTP_CLIENT_ENDPOINT": "http://$TAILSCALE_IP:8000"
      }
    }
  }
}
EOF
echo ""

# Display useful commands
log_info "Useful Commands:"
echo ""
echo "  Check container status:"
echo "    ssh -i $SSH_KEY $SSH_USER@$PUBLIC_IP 'cd /opt/mcp-memory && docker compose ps'"
echo ""
echo "  View logs:"
echo "    ssh -i $SSH_KEY $SSH_USER@$PUBLIC_IP 'cd /opt/mcp-memory && docker compose logs -f'"
echo ""
echo "  Restart containers:"
echo "    ssh -i $SSH_KEY $SSH_USER@$PUBLIC_IP 'cd /opt/mcp-memory && docker compose restart'"
echo ""
echo "  Test health endpoint (from Tailscale network):"
echo "    curl http://$TAILSCALE_IP:8000/api/health"
echo ""

# Display access information
log_info "Access Information:"
echo ""
echo "  Public IP (SSH only):  $PUBLIC_IP"
echo "  Tailscale IP:          $TAILSCALE_IP"
echo "  HTTP Endpoint:         http://$TAILSCALE_IP:8000"
echo "  Dashboard:             http://$TAILSCALE_IP:8000/"
echo ""

if [[ -n "$INSTANCE_OCID" ]]; then
    echo "  Instance OCID:         $INSTANCE_OCID"
    echo ""
fi

log_success "MCP Memory Service is now running on Oracle Cloud!"
echo ""
