#!/bin/bash
#
# provision_oracle.sh - Oracle Cloud Infrastructure provisioning automation
#
# Provisions VM.Standard.A1.Flex (4 ARM cores, 24GB RAM) on Oracle Free Tier
# in ap-melbourne-1 region with Ubuntu 24.04 LTS.
#
# Usage:
#   ./provision_oracle.sh [--compartment-id OCID] [--ssh-key-file PATH]
#
# Environment Variables:
#   OCI_COMPARTMENT_ID      - Oracle compartment OCID (required)
#   OCI_SSH_KEY_FILE        - Path to SSH public key (default: ~/.ssh/id_rsa.pub)
#   OCI_AVAILABILITY_DOMAIN - AD to provision in (optional, auto-selects if not set)
#   OCI_REGION              - Region to provision in (default: ap-melbourne-1)
#   OCI_SHAPE_OCPUS         - Number of OCPUs (default: 4, try 2 for better availability)
#   OCI_SHAPE_MEMORY_GB     - Memory in GB (default: 24, try 12 with 2 OCPUs)
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
COMPARTMENT_ID="${OCI_COMPARTMENT_ID:-}"
SSH_KEY_FILE="${OCI_SSH_KEY_FILE:-$HOME/.ssh/id_rsa.pub}"
AVAILABILITY_DOMAIN="${OCI_AVAILABILITY_DOMAIN:-}"
REGION="${OCI_REGION:-ap-melbourne-1}"
INSTANCE_NAME="oracle-vps-au"
SHAPE="VM.Standard.A1.Flex"
OCPUS="${OCI_SHAPE_OCPUS:-4}"
MEMORY_GB="${OCI_SHAPE_MEMORY_GB:-24}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --compartment-id)
            COMPARTMENT_ID="$2"
            shift 2
            ;;
        --ssh-key-file)
            SSH_KEY_FILE="$2"
            shift 2
            ;;
        --availability-domain)
            AVAILABILITY_DOMAIN="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--compartment-id OCID] [--ssh-key-file PATH]"
            echo ""
            echo "Provisions Oracle Cloud Compute instance for MCP Memory Service"
            echo ""
            echo "Options:"
            echo "  --compartment-id OCID    Oracle compartment OCID (or set OCI_COMPARTMENT_ID)"
            echo "  --ssh-key-file PATH      SSH public key file (default: ~/.ssh/id_rsa.pub)"
            echo "  --availability-domain AD Availability domain (auto-selects if not set)"
            echo "  --help                   Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Run with --help for usage information"
            exit 1
            ;;
    esac
done

# Validate prerequisites
if [[ -z "$COMPARTMENT_ID" ]]; then
    log_error "Compartment ID is required. Set OCI_COMPARTMENT_ID environment variable or use --compartment-id"
    log_info "Find your compartment ID: oci iam compartment list"
    exit 1
fi

if [[ ! -f "$SSH_KEY_FILE" ]]; then
    log_error "SSH key file not found: $SSH_KEY_FILE"
    log_info "Generate one with: ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa"
    exit 1
fi

# Check OCI CLI is installed and configured
if ! command -v oci &> /dev/null; then
    log_error "OCI CLI not found. Install from: https://docs.oracle.com/en-us/iaas/Content/API/SDKDocs/cliinstall.htm"
    exit 1
fi

# Verify OCI authentication
log_info "Verifying OCI authentication..."
if ! oci iam region list --query 'data[0].name' --raw-output &> /dev/null; then
    log_error "OCI authentication failed. Run: oci session authenticate --region $REGION"
    log_info "Or configure API key: https://docs.oracle.com/en-us/iaas/Content/API/Concepts/apisigningkey.htm"
    exit 1
fi
log_success "OCI authentication verified"

# Check if instance already exists (idempotency)
log_info "Checking for existing instance '$INSTANCE_NAME'..."
EXISTING_INSTANCE=$(oci compute instance list \
    --compartment-id "$COMPARTMENT_ID" \
    --display-name "$INSTANCE_NAME" \
    --lifecycle-state RUNNING \
    --query 'data[0].id' \
    --raw-output 2>/dev/null || echo "")

if [[ -n "$EXISTING_INSTANCE" && "$EXISTING_INSTANCE" != "null" ]]; then
    log_success "Instance '$INSTANCE_NAME' already exists (OCID: $EXISTING_INSTANCE)"

    # Get public IP
    PUBLIC_IP=$(oci compute instance list-vnics \
        --instance-id "$EXISTING_INSTANCE" \
        --query 'data[0]."public-ip"' \
        --raw-output)

    # Output JSON for orchestration script
    cat <<EOF
{
  "instance_ocid": "$EXISTING_INSTANCE",
  "public_ip": "$PUBLIC_IP",
  "instance_name": "$INSTANCE_NAME",
  "status": "already_exists"
}
EOF
    exit 0
fi

log_info "Instance does not exist, proceeding with provisioning..."

# Get Ubuntu 24.04 ARM64 image ID
log_info "Finding Ubuntu 24.04 LTS ARM64 image..."
UBUNTU_IMAGE_ID=$(oci compute image list \
    --compartment-id "$COMPARTMENT_ID" \
    --operating-system "Canonical Ubuntu" \
    --operating-system-version "24.04" \
    --shape "$SHAPE" \
    --sort-by TIMECREATED \
    --sort-order DESC \
    --query 'data[0].id' \
    --raw-output 2>/dev/null || echo "")

if [[ -z "$UBUNTU_IMAGE_ID" || "$UBUNTU_IMAGE_ID" == "null" ]]; then
    log_error "Ubuntu 24.04 ARM64 image not found"
    log_info "Listing available images..."
    oci compute image list --compartment-id "$COMPARTMENT_ID" --shape "$SHAPE" --query 'data[].{"Name":"display-name","OS":"operating-system","Version":"operating-system-version"}' --output table
    exit 1
fi
log_success "Found Ubuntu 24.04 image: $UBUNTU_IMAGE_ID"

# Auto-select availability domain if not provided
if [[ -z "$AVAILABILITY_DOMAIN" ]]; then
    log_info "Auto-selecting availability domain..."
    AVAILABILITY_DOMAIN=$(oci iam availability-domain list \
        --compartment-id "$COMPARTMENT_ID" \
        --query 'data[0].name' \
        --raw-output)
    log_info "Selected availability domain: $AVAILABILITY_DOMAIN"
fi

# Get default VCN and subnet (or create if not exists)
log_info "Finding or creating VCN and subnet..."
VCN_ID=$(oci network vcn list \
    --compartment-id "$COMPARTMENT_ID" \
    --lifecycle-state AVAILABLE \
    --query 'data[0].id' \
    --raw-output 2>/dev/null || echo "")

if [[ -z "$VCN_ID" || "$VCN_ID" == "null" ]]; then
    log_warn "No VCN found, creating default VCN..."
    VCN_ID=$(oci network vcn create \
        --compartment-id "$COMPARTMENT_ID" \
        --cidr-block "10.0.0.0/16" \
        --display-name "mcp-memory-vcn" \
        --wait-for-state AVAILABLE \
        --query 'data.id' \
        --raw-output)
    log_success "Created VCN: $VCN_ID"

    # Create internet gateway
    IGW_ID=$(oci network internet-gateway create \
        --compartment-id "$COMPARTMENT_ID" \
        --vcn-id "$VCN_ID" \
        --is-enabled true \
        --display-name "mcp-memory-igw" \
        --wait-for-state AVAILABLE \
        --query 'data.id' \
        --raw-output)
    log_success "Created internet gateway: $IGW_ID"

    # Update default route table
    DEFAULT_ROUTE_TABLE_ID=$(oci network vcn get \
        --vcn-id "$VCN_ID" \
        --query 'data."default-route-table-id"' \
        --raw-output)

    oci network route-table update \
        --rt-id "$DEFAULT_ROUTE_TABLE_ID" \
        --route-rules "[{\"destination\":\"0.0.0.0/0\",\"networkEntityId\":\"$IGW_ID\"}]" \
        --force > /dev/null
    log_success "Updated route table with internet gateway"
fi

# Get or create subnet
SUBNET_ID=$(oci network subnet list \
    --compartment-id "$COMPARTMENT_ID" \
    --vcn-id "$VCN_ID" \
    --lifecycle-state AVAILABLE \
    --query 'data[0].id' \
    --raw-output 2>/dev/null || echo "")

if [[ -z "$SUBNET_ID" || "$SUBNET_ID" == "null" ]]; then
    log_warn "No subnet found, creating default subnet..."
    SUBNET_ID=$(oci network subnet create \
        --compartment-id "$COMPARTMENT_ID" \
        --vcn-id "$VCN_ID" \
        --cidr-block "10.0.1.0/24" \
        --display-name "mcp-memory-subnet" \
        --availability-domain "$AVAILABILITY_DOMAIN" \
        --wait-for-state AVAILABLE \
        --query 'data.id' \
        --raw-output)
    log_success "Created subnet: $SUBNET_ID"
fi

# Get security list and update rules (SSH only)
SECURITY_LIST_ID=$(oci network subnet get \
    --subnet-id "$SUBNET_ID" \
    --query 'data."security-list-ids"[0]' \
    --raw-output)

log_info "Configuring security list to allow SSH only..."
oci network security-list update \
    --security-list-id "$SECURITY_LIST_ID" \
    --ingress-security-rules '[
        {
            "protocol": "6",
            "source": "0.0.0.0/0",
            "tcpOptions": {
                "destinationPortRange": {
                    "min": 22,
                    "max": 22
                }
            },
            "description": "SSH access"
        }
    ]' \
    --egress-security-rules '[
        {
            "protocol": "all",
            "destination": "0.0.0.0/0",
            "description": "Allow all outbound"
        }
    ]' \
    --force > /dev/null
log_success "Security list configured (SSH only)"

# Provision instance with retry logic for capacity issues
log_info "Provisioning Oracle Cloud instance (this may take 2-3 minutes)..."
log_info "Configuration: $SHAPE ($OCPUS cores, ${MEMORY_GB}GB RAM), Ubuntu 24.04 LTS ARM64"

INSTANCE_OCID=""
for attempt in {1..3}; do
    log_info "Attempt $attempt/3..."

    if INSTANCE_OCID=$(timeout 600 oci compute instance launch \
        --compartment-id "$COMPARTMENT_ID" \
        --availability-domain "$AVAILABILITY_DOMAIN" \
        --shape "$SHAPE" \
        --shape-config "{\"ocpus\":$OCPUS,\"memoryInGBs\":$MEMORY_GB}" \
        --image-id "$UBUNTU_IMAGE_ID" \
        --subnet-id "$SUBNET_ID" \
        --assign-public-ip true \
        --ssh-authorized-keys-file "$SSH_KEY_FILE" \
        --display-name "$INSTANCE_NAME" \
        --wait-for-state RUNNING \
        --query 'data.id' \
        --raw-output 2>&1); then
        log_success "Instance provisioned successfully!"
        break
    else
        EXIT_CODE=$?
        if [[ $EXIT_CODE -eq 124 ]]; then
            log_error "Instance provisioning timed out after 10 minutes"
        elif echo "$INSTANCE_OCID" | grep -q "Out of host capacity"; then
            log_warn "Capacity unavailable in $AVAILABILITY_DOMAIN (attempt $attempt/3)"
            if [[ $attempt -lt 3 ]]; then
                log_info "Retrying in 30 seconds..."
                sleep 30
            fi
        else
            log_error "Instance provisioning failed: $INSTANCE_OCID"
            exit 1
        fi

        if [[ $attempt -eq 3 ]]; then
            log_error "Failed to provision instance after 3 attempts"
            log_info "Try again later or use a different availability domain"
            log_info "List ADs: oci iam availability-domain list --compartment-id $COMPARTMENT_ID"
            exit 1
        fi
    fi
done

# Get public IP address
log_info "Retrieving public IP address..."
PUBLIC_IP=$(oci compute instance list-vnics \
    --instance-id "$INSTANCE_OCID" \
    --query 'data[0]."public-ip"' \
    --raw-output)

if [[ -z "$PUBLIC_IP" || "$PUBLIC_IP" == "null" ]]; then
    log_error "Failed to retrieve public IP address"
    exit 1
fi

log_success "Instance provisioned successfully!"
log_info "Instance OCID: $INSTANCE_OCID"
log_info "Public IP: $PUBLIC_IP"
log_info "SSH access: ssh ubuntu@$PUBLIC_IP"

# Wait for SSH to become available
log_info "Waiting for SSH to become available (up to 5 minutes)..."
SSH_READY=false
for i in {1..60}; do
    if timeout 5 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -o BatchMode=yes ubuntu@"$PUBLIC_IP" true 2>/dev/null; then
        SSH_READY=true
        log_success "SSH is ready!"
        break
    fi
    if [[ $((i % 10)) -eq 0 ]]; then
        log_info "Still waiting... ($i/60)"
    fi
    sleep 5
done

if [[ "$SSH_READY" != "true" ]]; then
    log_warn "SSH not ready after 5 minutes, but instance is running"
    log_info "Try connecting manually: ssh ubuntu@$PUBLIC_IP"
fi

# Output JSON for orchestration script
cat <<EOF
{
  "instance_ocid": "$INSTANCE_OCID",
  "public_ip": "$PUBLIC_IP",
  "instance_name": "$INSTANCE_NAME",
  "region": "$REGION",
  "shape": "$SHAPE",
  "status": "provisioned"
}
EOF

log_success "Provisioning complete!"
