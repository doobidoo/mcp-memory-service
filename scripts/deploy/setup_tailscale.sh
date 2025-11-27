#!/bin/bash
#
# setup_tailscale.sh - Tailscale VPN installation and configuration
#
# Installs Tailscale, authenticates with user's tailnet, and obtains
# Tailscale IP (100.x.x.x) for HTTP server binding.
#
# Usage:
#   ./setup_tailscale.sh [--auth-key KEY] [--env-file PATH]
#
# Environment Variables:
#   TAILSCALE_AUTH_KEY - Ephemeral auth key from Tailscale admin console
#   MCP_ENV_FILE       - Path to .env file (default: /opt/mcp-memory/.env)
#

set -euo pipefail

# Color output for logging
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

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

# Parse command line arguments
AUTH_KEY="${TAILSCALE_AUTH_KEY:-}"
ENV_FILE="${MCP_ENV_FILE:-/opt/mcp-memory/.env}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --auth-key)
            AUTH_KEY="$2"
            shift 2
            ;;
        --env-file)
            ENV_FILE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--auth-key KEY] [--env-file PATH]"
            echo ""
            echo "Installs and configures Tailscale VPN for MCP Memory Service"
            echo ""
            echo "Options:"
            echo "  --auth-key KEY       Tailscale auth key (or set TAILSCALE_AUTH_KEY)"
            echo "  --env-file PATH      Path to .env file (default: /opt/mcp-memory/.env)"
            echo "  --help               Show this help message"
            echo ""
            echo "Generate auth key: https://login.tailscale.com/admin/settings/keys"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Run with --help for usage information"
            exit 1
            ;;
    esac
done

# Check if running as root (required for Tailscale installation)
if [[ $EUID -ne 0 ]]; then
    log_error "This script must be run as root (use sudo)"
    exit 1
fi

# Check if Tailscale is already installed
if command -v tailscale &> /dev/null; then
    TAILSCALE_VERSION=$(tailscale version | head -n1 | awk '{print $1}')
    log_success "Tailscale already installed (version $TAILSCALE_VERSION)"
else
    # Install Tailscale
    log_info "Installing Tailscale..."
    if ! curl -fsSL https://tailscale.com/install.sh | sh; then
        log_error "Tailscale installation failed"
        exit 1
    fi
    log_success "Tailscale installed successfully"
fi

# Check if Tailscale is already authenticated
TAILSCALE_STATUS=$(tailscale status --json 2>/dev/null || echo '{"BackendState":"NeedsLogin"}')
BACKEND_STATE=$(echo "$TAILSCALE_STATUS" | grep -o '"BackendState":"[^"]*"' | cut -d'"' -f4)

if [[ "$BACKEND_STATE" == "Running" ]]; then
    log_success "Tailscale already authenticated and running"
else
    # Authenticate with Tailscale
    log_info "Authenticating with Tailscale..."

    if [[ -n "$AUTH_KEY" ]]; then
        log_info "Using provided auth key for authentication"
        # Never log the actual auth key
        if ! tailscale up --authkey="$AUTH_KEY" --accept-routes --ssh 2>&1 | grep -v "$AUTH_KEY"; then
            log_error "Tailscale authentication failed with auth key"
            log_info "Verify auth key is valid and not expired: https://login.tailscale.com/admin/settings/keys"
            exit 1
        fi
        log_success "Authenticated with auth key"
    else
        log_warn "No auth key provided, starting interactive authentication"
        log_info "You will need to visit a URL to authenticate this device"
        log_info ""

        # Start interactive authentication
        tailscale up --accept-routes --ssh

        log_info ""
        log_success "Interactive authentication complete"
    fi
fi

# Wait for Tailscale IP to be assigned
log_info "Waiting for Tailscale IP assignment (up to 60 seconds)..."
TAILSCALE_IP=""
for i in {1..30}; do
    TAILSCALE_IP=$(tailscale ip -4 2>/dev/null || echo "")
    if [[ -n "$TAILSCALE_IP" ]]; then
        log_success "Tailscale IP obtained: $TAILSCALE_IP"
        break
    fi

    if [[ $((i % 5)) -eq 0 ]]; then
        log_info "Still waiting for IP assignment... ($((i * 2))/60 seconds)"
    fi
    sleep 2
done

if [[ -z "$TAILSCALE_IP" ]]; then
    log_error "Failed to obtain Tailscale IP after 60 seconds"
    log_info "Check Tailscale status: tailscale status"
    log_info "Check Tailscale logs: journalctl -u tailscaled -n 50"
    exit 1
fi

# Validate IP is in Tailscale range (100.x.x.x)
if [[ ! "$TAILSCALE_IP" =~ ^100\. ]]; then
    log_error "Invalid Tailscale IP: $TAILSCALE_IP (expected 100.x.x.x)"
    exit 1
fi

# Verify Tailscale connectivity with self-ping
log_info "Verifying Tailscale connectivity..."
if timeout 10 tailscale ping --c 1 "$TAILSCALE_IP" &> /dev/null; then
    log_success "Tailscale connectivity verified (self-ping successful)"
else
    log_warn "Tailscale self-ping failed, but IP was obtained"
    log_info "This may be normal - proceeding anyway"
fi

# Create .env file directory if it doesn't exist
ENV_DIR=$(dirname "$ENV_FILE")
if [[ ! -d "$ENV_DIR" ]]; then
    log_info "Creating directory: $ENV_DIR"
    mkdir -p "$ENV_DIR"
fi

# Write TAILSCALE_IP to .env file
log_info "Writing Tailscale IP to $ENV_FILE"

# Check if .env file exists and already has TAILSCALE_IP
if [[ -f "$ENV_FILE" ]] && grep -q "^TAILSCALE_IP=" "$ENV_FILE"; then
    # Update existing TAILSCALE_IP
    sed -i "s|^TAILSCALE_IP=.*|TAILSCALE_IP=$TAILSCALE_IP|" "$ENV_FILE"
    log_info "Updated existing TAILSCALE_IP in $ENV_FILE"
else
    # Append TAILSCALE_IP
    echo "TAILSCALE_IP=$TAILSCALE_IP" >> "$ENV_FILE"
    log_info "Added TAILSCALE_IP to $ENV_FILE"
fi

# Secure .env file permissions (readable by owner only)
chmod 600 "$ENV_FILE"
log_info "Set .env file permissions to 600"

# Get Tailscale device name and network info
DEVICE_NAME=$(tailscale status --json 2>/dev/null | grep -o '"HostName":"[^"]*"' | head -n1 | cut -d'"' -f4 || echo "unknown")
TAILNET=$(tailscale status --json 2>/dev/null | grep -o '"MagicDNSSuffix":"[^"]*"' | cut -d'"' -f4 || echo "unknown")

log_success "Tailscale setup complete!"
log_info "Device name: $DEVICE_NAME"
log_info "Tailscale IP: $TAILSCALE_IP"
log_info "Tailnet: $TAILNET"
log_info "MagicDNS name: ${DEVICE_NAME}${TAILNET}"

# Output JSON for orchestration script
cat <<EOF
{
  "tailscale_ip": "$TAILSCALE_IP",
  "device_name": "$DEVICE_NAME",
  "tailnet": "$TAILNET",
  "magic_dns": "${DEVICE_NAME}${TAILNET}",
  "env_file": "$ENV_FILE",
  "status": "configured"
}
EOF

# Display next steps
log_info ""
log_info "Next steps:"
log_info "1. Verify .env file: cat $ENV_FILE"
log_info "2. Test Tailscale connection from another device: tailscale ping $TAILSCALE_IP"
log_info "3. HTTP server will bind to: http://$TAILSCALE_IP:8000"
