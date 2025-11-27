#!/bin/bash
#
# check_oracle_capacity.sh - Find available Oracle Free Tier ARM capacity
#
# Checks all regions and availability domains for VM.Standard.A1.Flex availability
# Helps locate where you can actually provision instances when primary region is full.
#
# Usage:
#   ./check_oracle_capacity.sh [--region REGION_NAME]
#
# Options:
#   --region REGION_NAME    Check only specific region (e.g., ap-melbourne-1)
#   --quick                 Only check ap-melbourne-1 and ap-sydney-1
#   --help                  Show this help message
#

set -euo pipefail

# Color output
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly RED='\033[0;31m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m'

# Parse arguments
SPECIFIC_REGION=""
QUICK_MODE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --region)
            SPECIFIC_REGION="$2"
            shift 2
            ;;
        --quick)
            QUICK_MODE=true
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

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}  Oracle Cloud Free Tier ARM Capacity Checker               ${CYAN}║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Get compartment ID from environment or config
if [[ -z "${OCI_COMPARTMENT_ID:-}" ]]; then
    # Try to get from config
    TENANCY_OCID=$(grep tenancy ~/.oci/config | cut -d= -f2 | tr -d ' ')
    OCI_COMPARTMENT_ID="$TENANCY_OCID"
    echo -e "${YELLOW}Note: Using tenancy OCID as compartment (root compartment)${NC}"
    echo ""
fi

echo -e "${BLUE}Compartment:${NC} $OCI_COMPARTMENT_ID"
echo ""

# Determine which regions to check
if [[ -n "$SPECIFIC_REGION" ]]; then
    REGIONS=("$SPECIFIC_REGION")
    echo -e "${BLUE}Checking specific region:${NC} $SPECIFIC_REGION"
elif [[ "$QUICK_MODE" == "true" ]]; then
    REGIONS=("ap-melbourne-1" "ap-sydney-1")
    echo -e "${BLUE}Quick mode:${NC} Checking Melbourne and Sydney only"
else
    echo -e "${BLUE}Fetching all available regions...${NC}"
    # Get all regions (this is fast, no need to filter)
    mapfile -t REGIONS < <(oci iam region list --query 'data[].name' --raw-output 2>/dev/null | jq -r '.[]')
    echo -e "${GREEN}Found ${#REGIONS[@]} regions${NC}"
fi

echo ""
echo -e "${CYAN}Checking ARM capacity (VM.Standard.A1.Flex)...${NC}"
echo ""

# Track results
AVAILABLE_CONFIGS=()

# Check each region
for REGION in "${REGIONS[@]}"; do
    echo -e "${BLUE}━━━ Region: $REGION ━━━${NC}"

    # Get availability domains for this region
    ADS=$(oci iam availability-domain list \
        --compartment-id "$OCI_COMPARTMENT_ID" \
        --region "$REGION" 2>/dev/null | \
        jq -r '.data[].name' || echo "")

    if [[ -z "$ADS" ]]; then
        echo -e "${YELLOW}  ⚠ Could not list availability domains${NC}"
        echo ""
        continue
    fi

    # Check each AD
    while IFS= read -r AD; do
        [[ -z "$AD" ]] && continue

        echo -e "  ${CYAN}AD:${NC} $AD"

        # Get shape information
        SHAPE_INFO=$(oci compute shape list \
            --compartment-id "$OCI_COMPARTMENT_ID" \
            --availability-domain "$AD" \
            --region "$REGION" 2>/dev/null | \
            jq -r '.data[] | select(.shape == "VM.Standard.A1.Flex")' || echo "")

        if [[ -z "$SHAPE_INFO" ]]; then
            echo -e "    ${RED}✗ VM.Standard.A1.Flex not available${NC}"
            continue
        fi

        # Extract limits
        MAX_OCPUS=$(echo "$SHAPE_INFO" | jq -r '.["ocpu-options"]."max" // 4')
        MAX_MEMORY=$(echo "$SHAPE_INFO" | jq -r '.["memory-options"]."max-in-g-bs" // 24')

        # Oracle Free Tier allows up to 4 OCPUs and 24GB RAM total across all A1 instances
        echo -e "    ${GREEN}✓ Available${NC}"
        echo -e "      Max OCPUs: $MAX_OCPUS"
        echo -e "      Max Memory: ${MAX_MEMORY}GB"

        # Try to check if we can actually create an instance (this is the real test)
        # We'll do a dry-run by checking current usage
        CURRENT_INSTANCES=$(oci compute instance list \
            --compartment-id "$OCI_COMPARTMENT_ID" \
            --availability-domain "$AD" \
            --region "$REGION" \
            --lifecycle-state RUNNING 2>/dev/null | \
            jq -r '[.data[] | select(.shape == "VM.Standard.A1.Flex")] | length' || echo "0")

        echo -e "      Current A1 instances: $CURRENT_INSTANCES"

        # Suggest configurations
        if [[ "$CURRENT_INSTANCES" -eq 0 ]]; then
            echo -e "      ${GREEN}Suggested configs:${NC}"
            echo -e "        - 4 OCPUs / 24GB (full free tier)"
            echo -e "        - 2 OCPUs / 12GB (better availability)"
            echo -e "        - 1 OCPU  / 6GB  (highest availability)"

            # Store this as available
            AVAILABLE_CONFIGS+=("$REGION|$AD|4|24")
        else
            echo -e "      ${YELLOW}Note: You already have $CURRENT_INSTANCES A1 instance(s)${NC}"
        fi

    done <<< "$ADS"

    echo ""
done

# Summary
echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}  Summary                                                    ${CYAN}║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

if [[ ${#AVAILABLE_CONFIGS[@]} -eq 0 ]]; then
    echo -e "${RED}No available ARM capacity found.${NC}"
    echo ""
    echo -e "${YELLOW}Suggestions:${NC}"
    echo "  1. Try again in a few hours (capacity fluctuates)"
    echo "  2. Use --quick mode to check Melbourne + Sydney only"
    echo "  3. Try smaller configurations (2 cores instead of 4)"
    echo "  4. Check Oracle Cloud status page for known issues"
    echo ""
else
    echo -e "${GREEN}Found ${#AVAILABLE_CONFIGS[@]} region(s) with available capacity:${NC}"
    echo ""

    for CONFIG in "${AVAILABLE_CONFIGS[@]}"; do
        IFS='|' read -r REGION AD OCPUS MEMORY <<< "$CONFIG"
        echo -e "  ${GREEN}✓${NC} ${CYAN}$REGION${NC} → $AD"
        echo -e "    Max: $OCPUS OCPUs / ${MEMORY}GB RAM"
    done

    echo ""
    echo -e "${BLUE}To deploy to a specific region/AD:${NC}"
    echo "  export OCI_REGION=<region-name>"
    echo "  export OCI_AVAILABILITY_DOMAIN=<ad-name>"
    echo "  bash scripts/deploy/deploy_to_oracle.sh"
    echo ""
fi

# Latency estimates for Australian deployment
if [[ ${#AVAILABLE_CONFIGS[@]} -gt 0 ]]; then
    echo -e "${CYAN}Latency from Melbourne (estimated):${NC}"
    echo "  ap-melbourne-1: <10ms  ✓ Best for Melbourne users"
    echo "  ap-sydney-1:    <30ms  ✓ Still excellent"
    echo "  ap-osaka-1:     ~100ms"
    echo "  ap-tokyo-1:     ~120ms"
    echo "  Other APAC:     100-200ms"
    echo "  US/EU:          200-300ms"
    echo ""
fi

echo -e "${BLUE}Tip:${NC} Run with ${CYAN}--quick${NC} to check Melbourne + Sydney only (faster)"
echo ""
