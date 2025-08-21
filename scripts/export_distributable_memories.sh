#!/bin/bash

# Export distributable reference memories for sharing across local network
# Usage: ./export_distributable_memories.sh [output_file]

OUTPUT_FILE="${1:-mcp_reference_memories_$(date +%Y%m%d).json}"
MCP_ENDPOINT="https://10.0.1.30:8443/mcp"
API_KEY="test-key-123"

echo "Exporting distributable reference memories..."
echo "Output file: $OUTPUT_FILE"

curl -k -s -X POST "$MCP_ENDPOINT" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "jsonrpc": "2.0", 
    "id": 1, 
    "method": "tools/call", 
    "params": {
      "name": "search_by_tag", 
      "arguments": {
        "tags": ["distributable-reference"]
      }
    }
  }' | jq -r '.result.content[0].text' > "$OUTPUT_FILE"

if [ $? -eq 0 ]; then
    echo "✅ Export completed: $OUTPUT_FILE"
    echo "📊 Memory count: $(cat "$OUTPUT_FILE" | jq '. | length' 2>/dev/null || echo "Unknown")"
    echo ""
    echo "To import to another MCP Memory Service:"
    echo "1. Copy $OUTPUT_FILE to target machine"
    echo "2. Use store_memory calls for each entry"
    echo "3. Update CLAUDE.md with new memory hashes"
else
    echo "❌ Export failed"
    exit 1
fi