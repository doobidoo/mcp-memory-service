#!/bin/bash
# Cleanup Test Environment
#
# Removes test databases and resets environment to production
#
# Usage:
#   ./scripts/test/cleanup-test-environment.sh

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧹 MCP Memory Service - Test Environment Cleanup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Get project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
TEST_DATA_DIR="$PROJECT_ROOT/test_data"

# Stop any running test servers
echo "🛑 Stopping test server (if running)..."
pkill -f "memory server.*8001" 2>/dev/null || true
sleep 1

# Remove test data
if [ -d "$TEST_DATA_DIR" ]; then
    echo "🗑️  Removing test databases..."

    # List what will be deleted
    echo ""
    echo "   Files to remove:"
    find "$TEST_DATA_DIR" -type f -name "*.db*" | while read file; do
        SIZE=$(du -h "$file" | cut -f1)
        echo "   - $(basename "$file") ($SIZE)"
    done
    echo ""

    # Confirm deletion
    read -p "Delete these test files? [y/N] " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$TEST_DATA_DIR"
        echo "✅ Test data removed"
    else
        echo "⏭️  Skipped deletion"
    fi
else
    echo "✅ No test data found (already clean)"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔄 Environment Reset Instructions"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Unset test environment variables:"
echo ""
echo "  unset MCP_MEMORY_SQLITE_PATH"
echo "  unset MCP_HTTP_PORT"
echo "  unset MCP_API_KEY"
echo "  export MCP_ALLOW_ANONYMOUS_ACCESS=true"
echo ""
echo "Start production server:"
echo ""
echo "  memory server --http"
echo ""
echo "Access at: http://localhost:8000/"
echo ""
