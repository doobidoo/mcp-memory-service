#!/bin/bash
# Push staged changes to remote MCP Memory Service API

STAGING_DB="/Users/hkr/Library/Application Support/mcp-memory/sqlite_vec_staging.db"
REMOTE_API="https://narrowbox.local:8443/api/memories"
# API_KEY can be set as environment variable or read from config
API_KEY="${MCP_API_KEY:-}"
HOSTNAME=$(hostname)

echo "$(date): Pushing staged changes to remote API..."

if [ ! -f "$STAGING_DB" ]; then
    echo "$(date): No staging database found - nothing to push"
    exit 0
fi

# Check if we have API key (if required)
if [ -z "$API_KEY" ]; then
    echo "$(date): WARNING: No API key configured. Set MCP_API_KEY environment variable if required"
fi

# Get count of changes ready to push
PUSH_COUNT=$(sqlite3 "$STAGING_DB" "
SELECT COUNT(*) FROM staged_memories 
WHERE conflict_status = 'none' 
  AND operation IN ('INSERT', 'UPDATE');
" 2>/dev/null || echo "0")

if [ "$PUSH_COUNT" -eq 0 ]; then
    echo "$(date): No changes ready to push"
    exit 0
fi

echo "$(date): Found $PUSH_COUNT changes ready to push to remote API"

# Test connectivity to remote API
echo "$(date): Testing connectivity to remote API..."
HTTP_STATUS=$(curl -k -s -o /dev/null -w "%{http_code}" "$REMOTE_API" --connect-timeout 10)

if [ "$HTTP_STATUS" -eq 000 ]; then
    echo "$(date): ERROR: Cannot connect to remote API at $REMOTE_API"
    echo "$(date): Changes will remain staged for next attempt"
    exit 1
elif [ "$HTTP_STATUS" -eq 404 ]; then
    echo "$(date): WARNING: API endpoint not found (404). Checking if server is running..."
elif [ "$HTTP_STATUS" -ge 200 ] && [ "$HTTP_STATUS" -lt 300 ]; then
    echo "$(date): API connectivity confirmed (HTTP $HTTP_STATUS)"
else
    echo "$(date): WARNING: Unexpected HTTP status: $HTTP_STATUS"
fi

# Process each change ready for push
PUSHED_COUNT=0
FAILED_COUNT=0

sqlite3 "$STAGING_DB" "
SELECT id, content, content_hash, tags, metadata, memory_type, 
       operation, staged_at, original_created_at, source_machine
FROM staged_memories 
WHERE conflict_status = 'none' 
  AND operation IN ('INSERT', 'UPDATE')
ORDER BY staged_at ASC;
" | while IFS='|' read -r id content content_hash tags metadata memory_type operation staged_at created_at source_machine; do

    echo "$(date): Pushing: ${content:0:50}..."
    
    # Prepare JSON payload
    # Note: This assumes the API accepts the memory service format
    JSON_PAYLOAD=$(cat << EOF
{
    "content": $(echo "$content" | jq -R .),
    "tags": $tags,
    "metadata": $metadata,
    "memory_type": "$memory_type",
    "client_hostname": "$HOSTNAME"
}
EOF
)
    
    # Prepare curl command with optional API key
    CURL_CMD="curl -k -s -X POST"
    CURL_CMD="$CURL_CMD -H 'Content-Type: application/json'"
    CURL_CMD="$CURL_CMD -H 'X-Client-Hostname: $HOSTNAME'"
    
    if [ -n "$API_KEY" ]; then
        CURL_CMD="$CURL_CMD -H 'Authorization: Bearer $API_KEY'"
    fi
    
    CURL_CMD="$CURL_CMD -d '$JSON_PAYLOAD'"
    CURL_CMD="$CURL_CMD '$REMOTE_API'"
    
    # Execute push to remote API
    RESPONSE=$(eval "$CURL_CMD" 2>&1)
    CURL_EXIT_CODE=$?
    
    if [ $CURL_EXIT_CODE -eq 0 ]; then
        # Check if response indicates success
        if echo "$RESPONSE" | grep -q '"success":\s*true\|"status":\s*"success"\|content_hash'; then
            echo "$(date): Successfully pushed: ${content:0:30}..."
            
            # Remove from staging on successful push
            sqlite3 "$STAGING_DB" "DELETE FROM staged_memories WHERE id = '$id';"
            PUSHED_COUNT=$((PUSHED_COUNT + 1))
            
        elif echo "$RESPONSE" | grep -q '"error"\|"message"\|HTTP.*[45][0-9][0-9]'; then
            echo "$(date): API error for: ${content:0:30}..."
            echo "$(date): Response: $RESPONSE"
            FAILED_COUNT=$((FAILED_COUNT + 1))
            
            # Mark as failed but don't delete (for retry)
            sqlite3 "$STAGING_DB" "
            UPDATE staged_memories 
            SET conflict_status = 'push_failed' 
            WHERE id = '$id';
            "
        else
            echo "$(date): Unexpected response: $RESPONSE"
            FAILED_COUNT=$((FAILED_COUNT + 1))
        fi
    else
        echo "$(date): Network error pushing: ${content:0:30}..."
        echo "$(date): Error: $RESPONSE"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        
        # Don't mark as failed if it's a network issue - keep for retry
    fi
    
    # Small delay to avoid overwhelming the API
    sleep 0.5
done

# Get final counts
REMAINING_COUNT=$(sqlite3 "$STAGING_DB" "SELECT COUNT(*) FROM staged_memories WHERE conflict_status = 'none';" 2>/dev/null || echo "0")
FAILED_FINAL=$(sqlite3 "$STAGING_DB" "SELECT COUNT(*) FROM staged_memories WHERE conflict_status = 'push_failed';" 2>/dev/null || echo "0")

echo "$(date): Push operation completed"
echo "$(date): Successfully pushed: $PUSHED_COUNT changes"
echo "$(date): Failed to push: $FAILED_FINAL changes"
echo "$(date): Remaining staged: $REMAINING_COUNT changes"

# Update sync status
sqlite3 "$STAGING_DB" "
INSERT OR REPLACE INTO sync_status (key, value) VALUES 
('last_push_attempt', datetime('now'));
"

if [ "$FAILED_FINAL" -gt 0 ]; then
    echo "$(date): WARNING: $FAILED_FINAL changes failed to push"
    echo "$(date): These changes will be retried on next push attempt"
fi

if [ "$REMAINING_COUNT" -gt 0 ]; then
    echo "$(date): NOTE: $REMAINING_COUNT changes still staged"
fi

echo "$(date): Push to remote API completed"