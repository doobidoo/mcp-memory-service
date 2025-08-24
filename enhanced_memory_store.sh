#!/bin/bash
# Enhanced memory store with remote-first + local staging fallback

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_API="https://narrowbox.local:8443/api/memories"
STAGING_DB="/Users/hkr/Library/Application Support/mcp-memory/sqlite_vec_staging.db"
API_KEY="${MCP_API_KEY:-}"
HOSTNAME=$(hostname)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

store_memory() {
    local content="$1"
    local tags="$2"
    local memory_type="${3:-note}"
    local project_name="$4"
    
    if [ -z "$content" ]; then
        echo -e "${RED}Error: No content provided${NC}"
        return 1
    fi
    
    # Generate content hash
    local content_hash=$(echo -n "$content" | shasum -a 256 | cut -d' ' -f1)
    
    # Auto-detect project context
    if [ -z "$project_name" ]; then
        project_name=$(basename "$(pwd)")
    fi
    
    # Auto-generate tags
    local auto_tags="source:$HOSTNAME,project:$project_name"
    
    # Add git context if available
    if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        local git_branch=$(git branch --show-current 2>/dev/null || echo "unknown")
        auto_tags="$auto_tags,git:$git_branch"
    fi
    
    # Combine with user tags
    if [ -n "$tags" ]; then
        auto_tags="$auto_tags,$tags"
    fi
    
    # Convert comma-separated tags to JSON array
    local json_tags="[\"$(echo "$auto_tags" | sed 's/,/","/g')\"]"
    
    # Prepare JSON payload
    local json_payload=$(cat << EOF
{
    "content": $(echo "$content" | jq -R .),
    "tags": $json_tags,
    "metadata": {
        "project": "$project_name",
        "hostname": "$HOSTNAME",
        "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
        "pwd": "$(pwd)"
    },
    "memory_type": "$memory_type",
    "client_hostname": "$HOSTNAME"
}
EOF
)
    
    echo "Storing memory: ${content:0:60}..."
    
    # Try remote API first
    echo "Attempting remote storage..."
    local curl_cmd="curl -k -s -X POST --connect-timeout 10"
    curl_cmd="$curl_cmd -H 'Content-Type: application/json'"
    curl_cmd="$curl_cmd -H 'X-Client-Hostname: $HOSTNAME'"
    
    if [ -n "$API_KEY" ]; then
        curl_cmd="$curl_cmd -H 'Authorization: Bearer $API_KEY'"
    fi
    
    local response=$(eval "$curl_cmd -d '$json_payload' '$REMOTE_API'" 2>&1)
    local curl_exit_code=$?
    
    if [ $curl_exit_code -eq 0 ]; then
        # Check if response indicates success
        if echo "$response" | grep -q '"success":\s*true\|"status":\s*"success"\|content_hash\|stored'; then
            echo -e "${GREEN}✓ Successfully stored to remote server${NC}"
            echo -e "${GREEN}  Content hash: ${content_hash:0:16}...${NC}"
            echo -e "${GREEN}  Tags applied: $auto_tags${NC}"
            return 0
        else
            echo -e "${YELLOW}⚠ Remote API returned unexpected response${NC}"
            echo "Response: $response"
        fi
    else
        echo -e "${YELLOW}⚠ Remote API not reachable (exit code: $curl_exit_code)${NC}"
    fi
    
    # Fallback to local staging
    echo "Falling back to local staging..."
    
    # Initialize staging DB if needed
    if [ ! -f "$STAGING_DB" ]; then
        echo "Initializing staging database..."
        "$SCRIPT_DIR/init_staging_db.sh"
    fi
    
    # Store in staging database
    local id=$(echo -n "$content$HOSTNAME$(date)" | shasum -a 256 | cut -d' ' -f1 | head -c 16)
    
    # Escape single quotes for SQL
    local content_escaped=$(echo "$content" | sed "s/'/''/g")
    local metadata_escaped=$(echo "{\"project\":\"$project_name\",\"hostname\":\"$HOSTNAME\"}" | sed "s/'/''/g")
    
    sqlite3 "$STAGING_DB" "
    INSERT OR REPLACE INTO staged_memories (
        id, content, content_hash, tags, metadata, memory_type,
        operation, staged_at, source_machine
    ) VALUES (
        '$id',
        '$content_escaped',
        '$content_hash',
        '$json_tags',
        '$metadata_escaped',
        '$memory_type',
        'INSERT',
        datetime('now'),
        '$HOSTNAME'
    );
    " 2>/dev/null
    
    if [ $? -eq 0 ]; then
        echo -e "${YELLOW}✓ Stored locally (staged for sync)${NC}"
        echo -e "${YELLOW}  Content hash: ${content_hash:0:16}...${NC}"
        echo -e "${YELLOW}  Tags applied: $auto_tags${NC}"
        echo -e "${YELLOW}  Run './sync/memory_sync.sh sync' to push to remote${NC}"
        
        # Show current staging status
        local staged_count=$(sqlite3 "$STAGING_DB" "SELECT COUNT(*) FROM staged_memories WHERE conflict_status = 'none';" 2>/dev/null || echo "0")
        echo -e "${YELLOW}  Total staged changes: $staged_count${NC}"
        
        return 0
    else
        echo -e "${RED}✗ Failed to store locally${NC}"
        return 1
    fi
}

show_help() {
    echo "Enhanced Memory Store - Remote-first with local staging fallback"
    echo ""
    echo "Usage: $0 [options] \"content\""
    echo ""
    echo "Options:"
    echo "  --tags \"tag1,tag2\"      Additional tags to apply"
    echo "  --type \"note|task|...\"   Memory type (default: note)"
    echo "  --project \"name\"        Override project name detection"
    echo "  --help, -h              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 \"Fixed the sync issue with conflict resolution\""
    echo "  $0 --tags \"bug,fix\" \"Resolved database deadlock in apply script\""
    echo "  $0 --type \"decision\" \"Chose remote-first approach for reliability\""
    echo ""
    echo "Environment Variables:"
    echo "  MCP_API_KEY             API key for remote server authentication"
    echo ""
    echo "Storage Strategy:"
    echo "  1. Try remote API first (https://narrowbox.local:8443/api/memories)"
    echo "  2. Fallback to local staging if remote fails"
    echo "  3. Use './sync/memory_sync.sh sync' to sync staged changes"
}

# Parse arguments
CONTENT=""
TAGS=""
MEMORY_TYPE="note"
PROJECT_NAME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --tags)
            TAGS="$2"
            shift 2
            ;;
        --type)
            MEMORY_TYPE="$2"
            shift 2
            ;;
        --project)
            PROJECT_NAME="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
        *)
            if [ -z "$CONTENT" ]; then
                CONTENT="$1"
            else
                CONTENT="$CONTENT $1"
            fi
            shift
            ;;
    esac
done

if [ -z "$CONTENT" ]; then
    echo "Error: No content provided"
    show_help
    exit 1
fi

store_memory "$CONTENT" "$TAGS" "$MEMORY_TYPE" "$PROJECT_NAME"