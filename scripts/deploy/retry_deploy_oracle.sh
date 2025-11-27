#!/bin/bash
#
# retry_deploy_oracle.sh - Automated retry loop for Oracle deployment
#
# Keeps retrying deployment every 15 minutes until capacity becomes available.
# Uses 2 cores / 12GB configuration for better availability.
#
# Usage:
#   ./retry_deploy_oracle.sh [--interval SECONDS]
#
# Options:
#   --interval SECONDS    Retry interval in seconds (default: 900 = 15 minutes)
#   --4-cores             Try 4 cores / 24GB instead of 2 cores / 12GB
#   --help                Show this help message
#

set -euo pipefail

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Color output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m'

# Default configuration
RETRY_INTERVAL=900  # 15 minutes
USE_4_CORES=false
ATTEMPT=1

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --interval)
            RETRY_INTERVAL="$2"
            shift 2
            ;;
        --4-cores)
            USE_4_CORES=true
            shift
            ;;
        --help)
            grep "^#" "$0" | grep -v "#!/bin/bash" | sed 's/^# //' | sed 's/^#//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage"
            exit 1
            ;;
    esac
done

# Set shape configuration
if [[ "$USE_4_CORES" == "true" ]]; then
    export OCI_SHAPE_OCPUS=4
    export OCI_SHAPE_MEMORY_GB=24
    CONFIG_DESC="4 cores / 24GB (full free tier)"
else
    export OCI_SHAPE_OCPUS=2
    export OCI_SHAPE_MEMORY_GB=12
    CONFIG_DESC="2 cores / 12GB (better availability)"
fi

# Display banner
echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}  Oracle Deployment - Automated Retry Loop                  ${CYAN}║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo "  Shape: VM.Standard.A1.Flex"
echo "  Resources: $CONFIG_DESC"
echo "  Retry interval: ${RETRY_INTERVAL}s ($((RETRY_INTERVAL / 60)) minutes)"
echo ""
echo -e "${YELLOW}This will keep trying until capacity becomes available.${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop.${NC}"
echo ""

# Verify credentials are set
if [[ -z "${OCI_COMPARTMENT_ID:-}" ]]; then
    echo -e "${RED}ERROR: OCI_COMPARTMENT_ID not set${NC}"
    echo "Set it with: export OCI_COMPARTMENT_ID=your-compartment-ocid"
    exit 1
fi

if [[ -z "${TAILSCALE_AUTH_KEY:-}" ]]; then
    echo -e "${YELLOW}WARNING: TAILSCALE_AUTH_KEY not set${NC}"
    echo "Set it with: export TAILSCALE_AUTH_KEY=tskey-auth-..."
    echo ""
fi

# Retry loop
while true; do
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Attempt #$ATTEMPT${NC} - $(date '+%Y-%m-%d %H:%M:%S')"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    # Run deployment
    if bash "$SCRIPT_DIR/deploy_to_oracle.sh"; then
        echo ""
        echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║${NC}  🎉 Deployment Successful!                                 ${GREEN}║${NC}"
        echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        echo -e "${BLUE}Success after $ATTEMPT attempt(s)${NC}"
        echo ""
        exit 0
    fi

    # Deployment failed, prepare to retry
    echo ""
    echo -e "${YELLOW}Deployment failed (attempt #$ATTEMPT)${NC}"
    echo -e "${BLUE}Next retry in ${RETRY_INTERVAL}s ($((RETRY_INTERVAL / 60)) minutes) at $(date -d "+${RETRY_INTERVAL} seconds" '+%H:%M:%S' 2>/dev/null || date -v +${RETRY_INTERVAL}S '+%H:%M:%S' 2>/dev/null || echo "soon")${NC}"
    echo ""
    echo -e "${CYAN}💡 Tip: Oracle capacity fluctuates - best times are early morning or late night${NC}"
    echo ""

    ATTEMPT=$((ATTEMPT + 1))
    sleep "$RETRY_INTERVAL"
done
