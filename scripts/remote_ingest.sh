#!/bin/bash
# Remote Document Ingestion Script for MCP Memory Service
# This script uploads and ingests documents on the remote server

set -e

# Configuration
REMOTE_HOST="${REMOTE_HOST:-10.0.1.30}"
REMOTE_USER="${REMOTE_USER:-hkr}"
# Auto-detect the mcp-memory-service repository location
REMOTE_PATH=$(ssh ${REMOTE_USER}@${REMOTE_HOST} "find /home/${REMOTE_USER} -iname 'mcp-memory-service' -type d -exec test -f {}/pyproject.toml \; -print 2>/dev/null | head -n1")
REMOTE_PATH="${REMOTE_PATH:-/home/${REMOTE_USER}/repositories/mcp-memory-service}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Function to show usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] <file_or_directory>

Remote document ingestion for MCP Memory Service

Options:
    -t, --tags TAGS         Comma-separated tags to apply (default: "documentation")
    -c, --chunk-size SIZE   Chunk size in characters (default: 800)
    -r, --recursive         Process directories recursively
    -e, --extensions EXTS   File extensions to process (default: all supported)
    -h, --host HOST         Remote host (default: 10.0.1.30)
    -u, --user USER         Remote user (default: hkr)
    --help                  Show this help message

Examples:
    # Ingest a single file
    $0 README.md

    # Ingest with custom tags
    $0 -t "documentation,important" CLAUDE.md

    # Ingest entire directory
    $0 -r docs/

    # Ingest specific file types
    $0 -r -e "md,txt" docs/

EOF
    exit 0
}

# Parse command line arguments
TAGS="documentation"
CHUNK_SIZE="800"
RECURSIVE=""
EXTENSIONS=""
FILES=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--tags)
            TAGS="$2"
            shift 2
            ;;
        -c|--chunk-size)
            CHUNK_SIZE="$2"
            shift 2
            ;;
        -r|--recursive)
            RECURSIVE="--recursive"
            shift
            ;;
        -e|--extensions)
            EXTENSIONS="--extensions $2"
            shift 2
            ;;
        -h|--host)
            REMOTE_HOST="$2"
            shift 2
            ;;
        -u|--user)
            REMOTE_USER="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            FILES+=("$1")
            shift
            ;;
    esac
done

# Check if files were provided
if [ ${#FILES[@]} -eq 0 ]; then
    print_error "No files or directories specified"
    usage
fi

# Process each file/directory
for item in "${FILES[@]}"; do
    if [ ! -e "$item" ]; then
        print_error "File or directory not found: $item"
        continue
    fi
    
    # Get absolute path
    ITEM_PATH=$(realpath "$item")
    ITEM_NAME=$(basename "$item")
    
    print_info "Processing: $ITEM_NAME"
    
    if [ -f "$item" ]; then
        # Single file ingestion
        print_info "Uploading file to remote server..."
        
        # Create temp directory on remote
        REMOTE_TEMP=$(ssh ${REMOTE_USER}@${REMOTE_HOST} "mktemp -d")
        
        # Upload file
        scp -q "$ITEM_PATH" ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_TEMP}/
        
        # Run ingestion on remote
        print_info "Running remote ingestion..."
        ssh ${REMOTE_USER}@${REMOTE_HOST} "cd \"${REMOTE_PATH}\" && \
            .venv/bin/python -m mcp_memory_service.cli.main ingest-document \
            ${REMOTE_TEMP}/${ITEM_NAME} \
            --tags '${TAGS}' \
            --chunk-size ${CHUNK_SIZE} \
            --verbose 2>&1 | grep -E '✅|📄|💾|⚡|⏱️'"
        
        # Cleanup
        ssh ${REMOTE_USER}@${REMOTE_HOST} "rm -rf ${REMOTE_TEMP}"
        
        print_success "Completed ingestion of $ITEM_NAME"
        
    elif [ -d "$item" ]; then
        # Directory ingestion
        print_info "Uploading directory to remote server..."
        
        # Create temp directory on remote
        REMOTE_TEMP=$(ssh ${REMOTE_USER}@${REMOTE_HOST} "mktemp -d")
        
        # Upload directory
        scp -rq "$ITEM_PATH" ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_TEMP}/
        
        # Run ingestion on remote
        print_info "Running remote directory ingestion..."
        ssh ${REMOTE_USER}@${REMOTE_HOST} "cd \"${REMOTE_PATH}\" && \
            .venv/bin/python -m mcp_memory_service.cli.main ingest-directory \
            ${REMOTE_TEMP}/${ITEM_NAME} \
            --tags '${TAGS}' \
            --chunk-size ${CHUNK_SIZE} \
            ${RECURSIVE} \
            ${EXTENSIONS} \
            --verbose 2>&1 | grep -E '✅|📁|📄|💾|⚡|⏱️|❌'"
        
        # Cleanup
        ssh ${REMOTE_USER}@${REMOTE_HOST} "rm -rf ${REMOTE_TEMP}"
        
        print_success "Completed ingestion of directory $ITEM_NAME"
    fi
done

print_success "Remote ingestion complete!"
print_info "View memories at: https://${REMOTE_HOST}:8443/"