#!/bin/bash
# Verify Qdrant server mode is working correctly

set -e

echo "=========================================="
echo "Qdrant Server Verification"
echo "=========================================="
echo

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check 1: Qdrant server is running
echo -n "1. Checking Qdrant server is running... "
if docker ps | grep -q mcp-memory-qdrant; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    echo "   Error: Qdrant server container not running"
    exit 1
fi

# Check 2: Qdrant is healthy
echo -n "2. Checking Qdrant health... "
if docker ps | grep -q "mcp-memory-qdrant.*healthy"; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${YELLOW}⚠${NC}"
    echo "   Warning: Qdrant not yet healthy (may be starting)"
fi

# Check 3: Qdrant API is accessible
echo -n "3. Checking Qdrant API... "
if curl -s -f http://localhost:6333/ > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    echo "   Error: Cannot access Qdrant API at http://localhost:6333"
    exit 1
fi

# Check 4: Collection exists
echo -n "4. Checking 'memories' collection... "
COLLECTION_INFO=$(curl -s http://localhost:6333/collections/memories 2>/dev/null || echo "")
if echo "$COLLECTION_INFO" | grep -q "\"status\":\"ok\""; then
    echo -e "${GREEN}✓${NC}"

    # Get memory count
    POINTS_COUNT=$(echo "$COLLECTION_INFO" | grep -o '"points_count":[0-9]*' | grep -o '[0-9]*')
    if [ -n "$POINTS_COUNT" ]; then
        # Subtract 1 for metadata point
        MEMORY_COUNT=$((POINTS_COUNT - 1))
        echo "   Memories: $MEMORY_COUNT (+ 1 metadata point)"
    fi
else
    echo -e "${YELLOW}⚠${NC}"
    echo "   Warning: Collection not found (will be created on first use)"
fi

# Check 5: HTTP service is connected
echo -n "5. Checking HTTP service connectivity... "
if docker ps | grep -q mcp-memory-http; then
    HTTP_LOGS=$(docker logs mcp-memory-http 2>&1 | tail -20)
    if echo "$HTTP_LOGS" | grep -q "Qdrant server mode"; then
        echo -e "${GREEN}✓${NC}"
    elif echo "$HTTP_LOGS" | grep -q "Connected to Qdrant server"; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${YELLOW}⚠${NC}"
        echo "   Warning: Cannot confirm HTTP service is in server mode"
    fi
else
    echo -e "${RED}✗${NC}"
    echo "   Error: HTTP service not running"
fi

# Check 6: MCP service is connected
echo -n "6. Checking MCP service connectivity... "
if docker ps | grep -q mcp-memory-mcp; then
    MCP_LOGS=$(docker logs mcp-memory-mcp 2>&1 | tail -20)
    if echo "$MCP_LOGS" | grep -q "Qdrant server mode"; then
        echo -e "${GREEN}✓${NC}"
    elif echo "$MCP_LOGS" | grep -q "Connected to Qdrant server"; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${YELLOW}⚠${NC}"
        echo "   Warning: Cannot confirm MCP service is in server mode"
    fi
else
    echo -e "${RED}✗${NC}"
    echo "   Error: MCP service not running"
fi

# Check 7: Network connectivity
echo -n "7. Checking Docker network... "
if docker network inspect mcp-memory-network > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC}"
else
    echo -e "${RED}✗${NC}"
    echo "   Error: Docker network 'mcp-memory-network' not found"
fi

echo
echo "=========================================="
echo -e "${GREEN}Verification Complete${NC}"
echo "=========================================="
echo
echo "Test multi-client access:"
echo "  Terminal 1: docker exec -i mcp-memory-mcp mcp-memory-server"
echo "  Terminal 2: docker exec -i mcp-memory-mcp mcp-memory-server"
echo
echo "Both should work simultaneously without lock conflicts."
echo
