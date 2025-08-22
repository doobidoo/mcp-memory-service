#!/bin/bash
# Simple conflict resolution helper

STAGING_DB="/Users/hkr/Library/Application Support/mcp-memory/sqlite_vec_staging.db"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

if [ ! -f "$STAGING_DB" ]; then
    echo -e "${RED}No staging database found${NC}"
    exit 1
fi

# Get conflicts
CONFLICTS=$(sqlite3 "$STAGING_DB" "
SELECT id, content, staged_at, conflict_status 
FROM staged_memories 
WHERE conflict_status IN ('detected', 'push_failed')
ORDER BY staged_at DESC;
")

if [ -z "$CONFLICTS" ]; then
    echo -e "${GREEN}No conflicts to resolve${NC}"
    exit 0
fi

echo -e "${YELLOW}Found conflicts to resolve:${NC}"
echo ""

echo "$CONFLICTS" | while IFS='|' read -r id content staged_at status; do
    echo -e "${RED}Conflict: $status${NC}"
    echo -e "Content: ${content:0:80}..."
    echo -e "Staged: $staged_at"
    echo -e "ID: $id"
    echo ""
    echo "Actions:"
    echo "  1. Keep and retry push"
    echo "  2. Delete (abandon change)"
    echo "  3. Skip for now"
    echo ""
    
    read -p "Choose action (1/2/3): " action
    
    case $action in
        1)
            sqlite3 "$STAGING_DB" "
            UPDATE staged_memories 
            SET conflict_status = 'none' 
            WHERE id = '$id';
            "
            echo -e "${GREEN}Marked for retry${NC}"
            ;;
        2)
            sqlite3 "$STAGING_DB" "DELETE FROM staged_memories WHERE id = '$id';"
            echo -e "${YELLOW}Deleted${NC}"
            ;;
        3)
            echo -e "${YELLOW}Skipped${NC}"
            ;;
        *)
            echo -e "${YELLOW}Invalid choice, skipped${NC}"
            ;;
    esac
    echo ""
done

echo -e "${GREEN}Conflict resolution completed${NC}"