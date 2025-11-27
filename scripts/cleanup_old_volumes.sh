#!/bin/bash
# Clean up old embedded mode volumes after migration

set -e

echo "=========================================="
echo "Cleanup Old Qdrant Volumes"
echo "=========================================="
echo

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Volumes to remove
OLD_VOLUMES=(
    "mcp-memory-service_mcp-memory-data"
    "mcp-memory-service_mcp-memory-data-mcp"
    "mcp-memory-data"
)

echo "This script will remove the following volumes:"
for vol in "${OLD_VOLUMES[@]}"; do
    echo "  - $vol"
done
echo

echo -e "${YELLOW}WARNING: This action cannot be undone!${NC}"
echo
echo "Make sure you have:"
echo "  1. Successfully migrated to Qdrant server mode"
echo "  2. Verified all memories are accessible"
echo "  3. Tested multi-client access"
echo

read -p "Are you sure you want to proceed? [y/N]: " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cleanup aborted."
    exit 0
fi

echo
echo "Removing old volumes..."
echo

for vol in "${OLD_VOLUMES[@]}"; do
    echo -n "Removing $vol... "
    if docker volume inspect "$vol" > /dev/null 2>&1; then
        if docker volume rm "$vol" 2>&1; then
            echo -e "${GREEN}✓${NC}"
        else
            echo -e "${RED}✗${NC}"
            echo "   Error: Failed to remove volume (is it in use?)"
            echo "   Try: docker compose down"
        fi
    else
        echo -e "${YELLOW}⚠${NC} (not found)"
    fi
done

echo
echo "=========================================="
echo -e "${GREEN}Cleanup Complete${NC}"
echo "=========================================="
echo
echo "Remaining volumes:"
docker volume ls | grep -E "(qdrant|mcp-memory)" || echo "  None"
echo
