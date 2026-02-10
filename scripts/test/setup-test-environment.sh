#!/bin/bash
# Test Environment Setup for MCP Memory Service
#
# CRITICAL: This script MUST be run before any manual testing to protect production data
#
# Usage:
#   source scripts/test/setup-test-environment.sh
#   # OR
#   . scripts/test/setup-test-environment.sh
#
# After sourcing, run: memory server --http
# Test server will be on port 8001 with isolated test database

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧪 MCP Memory Service - Test Environment Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Get project root (assuming script is in scripts/test/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Create test data directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TEST_DATA_DIR="$PROJECT_ROOT/test_data"
TEST_DB_DIR="$TEST_DATA_DIR/test_$TIMESTAMP"

echo "📂 Creating isolated test environment..."
mkdir -p "$TEST_DB_DIR"

# Configure test environment
export MCP_MEMORY_SQLITE_PATH="$TEST_DB_DIR/test_memories.db"
export MCP_HTTP_PORT=8001
export MCP_API_KEY="test-key-12345"
export MCP_ALLOW_ANONYMOUS_ACCESS="false"

# Optional: Reduce logging verbosity for tests
export MCP_LOG_LEVEL="WARNING"

echo ""
echo "✅ Test environment configured:"
echo "   Database:   $MCP_MEMORY_SQLITE_PATH"
echo "   HTTP Port:  $MCP_HTTP_PORT"
echo "   API Key:    $MCP_API_KEY"
echo "   Anonymous:  $MCP_ALLOW_ANONYMOUS_ACCESS"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⚠️  ISOLATED TEST ENVIRONMENT - Safe to test!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🚀 Start test server with:"
echo "   memory server --http"
echo ""
echo "🌐 Access dashboard at:"
echo "   http://localhost:8001/"
echo ""
echo "🧹 Cleanup after testing:"
echo "   scripts/test/cleanup-test-environment.sh"
echo ""
echo "🔄 Return to production:"
echo "   unset MCP_MEMORY_SQLITE_PATH MCP_HTTP_PORT MCP_API_KEY"
echo "   export MCP_ALLOW_ANONYMOUS_ACCESS=true"
echo "   memory server --http"
echo ""
