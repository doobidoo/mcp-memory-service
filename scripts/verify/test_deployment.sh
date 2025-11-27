#!/bin/bash
#
# test_deployment.sh - Post-deployment verification script
#
# Validates that Oracle server deployment is fully functional by testing:
# - Health endpoint accessibility
# - Memory CRUD operations (create, read, delete)
# - Docker container health
# - Backup configuration
# - Environment configuration
#
# Usage:
#   ./test_deployment.sh [--tailscale-ip IP] [--api-key KEY]
#
# Environment Variables:
#   TAILSCALE_IP  - Tailscale IP of Oracle server (required)
#   MCP_API_KEY   - API key for authentication (optional)
#

set -euo pipefail

# Color output for logging
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color

# Test results tracking
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_TOTAL=0

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

log_error() {
    echo -e "${RED}[✗]${NC} $*"
}

log_test() {
    echo -e "${CYAN}[TEST]${NC} $*"
}

# Test result tracking
pass_test() {
    ((TESTS_PASSED++))
    ((TESTS_TOTAL++))
    log_success "$1"
}

fail_test() {
    ((TESTS_FAILED++))
    ((TESTS_TOTAL++))
    log_error "$1"
}

# Parse command line arguments
TAILSCALE_IP="${TAILSCALE_IP:-}"
MCP_API_KEY="${MCP_API_KEY:-}"
DEPLOYMENT_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tailscale-ip)
            TAILSCALE_IP="$2"
            shift 2
            ;;
        --api-key)
            MCP_API_KEY="$2"
            shift 2
            ;;
        --deployment-dir)
            DEPLOYMENT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--tailscale-ip IP] [--api-key KEY] [--deployment-dir DIR]"
            echo ""
            echo "Post-deployment verification for Oracle server deployment"
            echo ""
            echo "Options:"
            echo "  --tailscale-ip IP      Tailscale IP of Oracle server (or set TAILSCALE_IP)"
            echo "  --api-key KEY          API key for authentication (or set MCP_API_KEY)"
            echo "  --deployment-dir DIR   Path to deployment directory (default: deployment/oracle)"
            echo "  --help                 Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Run with --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required parameters
if [[ -z "$TAILSCALE_IP" ]]; then
    log_error "TAILSCALE_IP is required. Set environment variable or use --tailscale-ip"
    log_info "Example: TAILSCALE_IP=100.64.1.2 $0"
    exit 1
fi

# Set deployment directory
if [[ -z "$DEPLOYMENT_DIR" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
    DEPLOYMENT_DIR="$PROJECT_ROOT/deployment/oracle"
fi

# Display test banner
echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}  Oracle Server Deployment Verification                    ${CYAN}║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
log_info "Testing deployment at: http://$TAILSCALE_IP:8000"
echo ""

# Prepare curl authentication
CURL_AUTH=""
if [[ -n "$MCP_API_KEY" ]]; then
    CURL_AUTH="-H 'Authorization: Bearer $MCP_API_KEY'"
    log_info "Using API key authentication"
fi

# Test data (will be cleaned up)
TEST_MEMORY_CONTENT="Deployment verification test - $(date +%s)"
TEST_MEMORY_HASH=""

# Cleanup function
cleanup() {
    if [[ -n "$TEST_MEMORY_HASH" ]]; then
        log_info "Cleaning up test data..."
        eval curl -s -X DELETE $CURL_AUTH "http://$TAILSCALE_IP:8000/api/memories/$TEST_MEMORY_HASH" > /dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

# ============================================================================
# Test 1: Health Endpoint
# ============================================================================

log_test "Test 1: Health endpoint accessibility"

HEALTH_RESPONSE=$(eval curl -s --connect-timeout 10 $CURL_AUTH "http://$TAILSCALE_IP:8000/api/health" || echo "FAILED")

if [[ "$HEALTH_RESPONSE" == "FAILED" ]]; then
    fail_test "Health endpoint unreachable (connection timeout or network error)"
    log_error "Cannot continue testing if server is unreachable"
    exit 1
fi

if echo "$HEALTH_RESPONSE" | grep -q '"status":"ok"'; then
    pass_test "Health endpoint returns status: ok"

    # Extract and display additional health info
    VERSION=$(echo "$HEALTH_RESPONSE" | grep -o '"version":"[^"]*"' | cut -d'"' -f4 || echo "unknown")
    MEMORY_COUNT=$(echo "$HEALTH_RESPONSE" | grep -o '"memory_count":[0-9]*' | cut -d':' -f2 || echo "unknown")
    QUEUE_SIZE=$(echo "$HEALTH_RESPONSE" | grep -o '"queue_size":[0-9]*' | cut -d':' -f2 || echo "unknown")

    log_info "  Version: $VERSION"
    log_info "  Memory count: $MEMORY_COUNT"
    log_info "  Queue size: $QUEUE_SIZE"
else
    fail_test "Health endpoint returned unexpected response: $HEALTH_RESPONSE"
fi

echo ""

# ============================================================================
# Test 2: Create Memory (POST)
# ============================================================================

log_test "Test 2: Create memory via POST /api/memories"

CREATE_PAYLOAD=$(cat <<EOF
{
  "content": "$TEST_MEMORY_CONTENT",
  "tags": ["test", "deployment", "verification"],
  "metadata": {
    "type": "test",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  }
}
EOF
)

CREATE_RESPONSE=$(eval curl -s -X POST $CURL_AUTH \
    -H "Content-Type: application/json" \
    -d "'$CREATE_PAYLOAD'" \
    "http://$TAILSCALE_IP:8000/api/memories" || echo "FAILED")

if [[ "$CREATE_RESPONSE" == "FAILED" ]]; then
    fail_test "Failed to create memory (connection error)"
else
    # Check if response contains content_hash
    TEST_MEMORY_HASH=$(echo "$CREATE_RESPONSE" | grep -o '"content_hash":"[^"]*"' | cut -d'"' -f4 || echo "")

    if [[ -n "$TEST_MEMORY_HASH" ]]; then
        pass_test "Memory created successfully (hash: ${TEST_MEMORY_HASH:0:16}...)"
    else
        fail_test "Memory creation response missing content_hash: $CREATE_RESPONSE"
    fi
fi

echo ""

# ============================================================================
# Test 3: Search Memory (GET)
# ============================================================================

log_test "Test 3: Search for created memory"

if [[ -z "$TEST_MEMORY_HASH" ]]; then
    fail_test "Cannot test search - memory creation failed"
else
    SEARCH_RESPONSE=$(eval curl -s -X POST $CURL_AUTH \
        -H "Content-Type: application/json" \
        -d "'{\"query\": \"$TEST_MEMORY_CONTENT\", \"limit\": 5}'" \
        "http://$TAILSCALE_IP:8000/api/search" || echo "FAILED")

    if [[ "$SEARCH_RESPONSE" == "FAILED" ]]; then
        fail_test "Search endpoint unreachable"
    elif echo "$SEARCH_RESPONSE" | grep -q "$TEST_MEMORY_HASH"; then
        pass_test "Memory found via semantic search"
    else
        fail_test "Memory not found in search results (may take time to index)"
        log_warn "This might be a timing issue - try again in a few seconds"
    fi
fi

echo ""

# ============================================================================
# Test 4: Retrieve Memory by Hash (GET)
# ============================================================================

log_test "Test 4: Retrieve memory by content hash"

if [[ -z "$TEST_MEMORY_HASH" ]]; then
    fail_test "Cannot test retrieval - memory creation failed"
else
    RETRIEVE_RESPONSE=$(eval curl -s $CURL_AUTH \
        "http://$TAILSCALE_IP:8000/api/memories/$TEST_MEMORY_HASH" || echo "FAILED")

    if [[ "$RETRIEVE_RESPONSE" == "FAILED" ]]; then
        fail_test "Failed to retrieve memory by hash"
    elif echo "$RETRIEVE_RESPONSE" | grep -q "$TEST_MEMORY_CONTENT"; then
        pass_test "Memory retrieved successfully by hash"
    else
        fail_test "Retrieved memory does not match expected content"
    fi
fi

echo ""

# ============================================================================
# Test 5: Delete Memory (DELETE)
# ============================================================================

log_test "Test 5: Delete test memory"

if [[ -z "$TEST_MEMORY_HASH" ]]; then
    fail_test "Cannot test deletion - memory creation failed"
else
    DELETE_RESPONSE=$(eval curl -s -X DELETE $CURL_AUTH \
        "http://$TAILSCALE_IP:8000/api/memories/$TEST_MEMORY_HASH" || echo "FAILED")

    if [[ "$DELETE_RESPONSE" == "FAILED" ]]; then
        fail_test "Failed to delete memory"
    elif echo "$DELETE_RESPONSE" | grep -q '"deleted":true'; then
        pass_test "Memory deleted successfully"
        TEST_MEMORY_HASH=""  # Prevent cleanup from trying again
    else
        fail_test "Delete response unexpected: $DELETE_RESPONSE"
    fi
fi

echo ""

# ============================================================================
# Test 6: Docker Container Health
# ============================================================================

log_test "Test 6: Docker container status"

# Check if we can access docker compose (might be remote)
if [[ -d "$DEPLOYMENT_DIR" ]] && command -v docker &> /dev/null; then
    COMPOSE_PS=$(cd "$DEPLOYMENT_DIR" && docker compose ps --format json 2>/dev/null || echo "FAILED")

    if [[ "$COMPOSE_PS" == "FAILED" ]]; then
        fail_test "Cannot check Docker container status (not running locally or no docker access)"
    else
        # Check for mcp-memory service
        if echo "$COMPOSE_PS" | grep -q '"Service":"mcp-memory"'; then
            if echo "$COMPOSE_PS" | grep -q '"State":"running"'; then
                pass_test "mcp-memory container is running"
            else
                fail_test "mcp-memory container is not running"
            fi
        else
            fail_test "mcp-memory service not found in docker compose"
        fi

        # Check for backup service (optional)
        if echo "$COMPOSE_PS" | grep -q '"Service":"backup"'; then
            if echo "$COMPOSE_PS" | grep -q '"State":"running"'; then
                pass_test "backup container is running"
            else
                log_warn "backup container is not running (may be intentional)"
            fi
        fi
    fi
else
    log_warn "Skipping Docker container check (deployment not local or docker not available)"
fi

echo ""

# ============================================================================
# Test 7: Environment Configuration
# ============================================================================

log_test "Test 7: Environment configuration validation"

if [[ -f "$DEPLOYMENT_DIR/.env" ]]; then
    # Check if TAILSCALE_IP is set in .env
    if grep -q "^TAILSCALE_IP=$TAILSCALE_IP" "$DEPLOYMENT_DIR/.env"; then
        pass_test ".env file has correct TAILSCALE_IP"
    else
        fail_test ".env file missing or incorrect TAILSCALE_IP"
    fi

    # Check if MCP_API_KEY is set (should not be empty)
    if grep -q "^MCP_API_KEY=.\+" "$DEPLOYMENT_DIR/.env"; then
        pass_test ".env file has MCP_API_KEY configured"
    else
        log_warn ".env file missing MCP_API_KEY (may be insecure)"
    fi

    # Check storage backend setting
    if grep -q "^MCP_MEMORY_STORAGE_BACKEND=sqlite_vec" "$DEPLOYMENT_DIR/.env"; then
        pass_test ".env file has correct storage backend (sqlite_vec)"
    else
        fail_test ".env file has incorrect storage backend"
    fi
else
    log_warn "Skipping .env validation (file not found at $DEPLOYMENT_DIR/.env)"
fi

echo ""

# ============================================================================
# Test 8: Backup Configuration (Optional)
# ============================================================================

log_test "Test 8: Backup configuration validation"

if [[ -f "$DEPLOYMENT_DIR/rclone.conf" ]]; then
    # Check if rclone config has R2 remote
    if grep -q "^\[r2\]" "$DEPLOYMENT_DIR/rclone.conf"; then
        pass_test "rclone.conf has R2 remote configured"

        # Check endpoint
        if grep -q "^endpoint = https://" "$DEPLOYMENT_DIR/rclone.conf"; then
            pass_test "rclone.conf has R2 endpoint configured"
        else
            fail_test "rclone.conf missing R2 endpoint"
        fi
    else
        fail_test "rclone.conf missing [r2] remote configuration"
    fi
else
    log_warn "Skipping backup validation (rclone.conf not found - backups may be disabled)"
fi

echo ""

# ============================================================================
# Test Summary
# ============================================================================

echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}  Test Summary                                              ${CYAN}║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

if [[ $TESTS_FAILED -eq 0 ]]; then
    echo -e "${GREEN}All tests passed!${NC} ($TESTS_PASSED/$TESTS_TOTAL)"
    echo ""
    log_success "Oracle server deployment is fully functional"
    echo ""
    log_info "Next steps:"
    echo "  1. Configure MCP clients to use: http://$TAILSCALE_IP:8000"
    echo "  2. Test from multiple devices on your Tailscale network"
    echo "  3. Monitor logs: docker compose logs -f"
    echo "  4. Check backups: docker compose run backup rclone ls r2:mcp-memory-backups"
    echo ""
    exit 0
else
    echo -e "${RED}Some tests failed${NC} ($TESTS_FAILED/$TESTS_TOTAL failed, $TESTS_PASSED/$TESTS_TOTAL passed)"
    echo ""
    log_error "Deployment has issues that need to be addressed"
    echo ""
    log_info "Troubleshooting:"
    echo "  1. Check container logs: docker compose logs"
    echo "  2. Verify .env configuration: cat $DEPLOYMENT_DIR/.env"
    echo "  3. Test connectivity: curl http://$TAILSCALE_IP:8000/api/health"
    echo "  4. Restart containers: docker compose restart"
    echo ""
    exit 1
fi
