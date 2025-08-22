#!/bin/bash
# Main memory synchronization orchestrator
# Implements Git-like workflow: stash → pull → apply → push

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGING_DB="/Users/hkr/Library/Application Support/mcp-memory/sqlite_vec_staging.db"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

show_status() {
    print_header "Memory Sync Status"
    
    if [ ! -f "$STAGING_DB" ]; then
        echo "Staging database not initialized"
        return
    fi
    
    STAGED_COUNT=$(sqlite3 "$STAGING_DB" "SELECT COUNT(*) FROM staged_memories WHERE conflict_status = 'none';" 2>/dev/null || echo "0")
    CONFLICT_COUNT=$(sqlite3 "$STAGING_DB" "SELECT COUNT(*) FROM staged_memories WHERE conflict_status = 'detected';" 2>/dev/null || echo "0")
    FAILED_COUNT=$(sqlite3 "$STAGING_DB" "SELECT COUNT(*) FROM staged_memories WHERE conflict_status = 'push_failed';" 2>/dev/null || echo "0")
    
    LAST_REMOTE_SYNC=$(sqlite3 "$STAGING_DB" "SELECT value FROM sync_status WHERE key = 'last_remote_sync';" 2>/dev/null || echo "Never")
    LAST_PUSH=$(sqlite3 "$STAGING_DB" "SELECT value FROM sync_status WHERE key = 'last_push_attempt';" 2>/dev/null || echo "Never")
    
    echo "Staged changes ready: $STAGED_COUNT"
    echo "Conflicts detected: $CONFLICT_COUNT"
    echo "Failed pushes: $FAILED_COUNT"
    echo "Last remote sync: $LAST_REMOTE_SYNC"
    echo "Last push attempt: $LAST_PUSH"
}

full_sync() {
    print_header "Starting Full Memory Sync"
    
    # Step 1: Stash local changes
    print_header "Step 1: Stashing Local Changes"
    if ! "$SCRIPT_DIR/stash_local_changes.sh"; then
        print_error "Failed to stash local changes"
        return 1
    fi
    print_success "Local changes stashed"
    
    # Step 2: Pull remote changes  
    print_header "Step 2: Pulling Remote Changes"
    if ! "$SCRIPT_DIR/pull_remote_changes.sh"; then
        print_error "Failed to pull remote changes"
        return 1
    fi
    print_success "Remote changes pulled"
    
    # Step 3: Apply staged changes
    print_header "Step 3: Applying Staged Changes"
    if ! "$SCRIPT_DIR/apply_local_changes.sh"; then
        print_warning "Some issues applying staged changes (check output above)"
    else
        print_success "Staged changes applied"
    fi
    
    # Step 4: Push remaining changes to remote
    print_header "Step 4: Pushing to Remote API"
    if ! "$SCRIPT_DIR/push_to_remote.sh"; then
        print_warning "Some issues pushing to remote (check output above)"
    else
        print_success "Changes pushed to remote"
    fi
    
    print_header "Full Sync Completed"
    show_status
}

quick_push() {
    print_header "Quick Push to Remote"
    
    if ! "$SCRIPT_DIR/push_to_remote.sh"; then
        print_error "Push failed"
        return 1
    fi
    
    print_success "Quick push completed"
    show_status
}

quick_pull() {
    print_header "Quick Pull from Remote"
    
    if ! "$SCRIPT_DIR/pull_remote_changes.sh"; then
        print_error "Pull failed"
        return 1
    fi
    
    print_success "Quick pull completed" 
    show_status
}

show_help() {
    echo "Memory Sync Tool - Git-like workflow for MCP Memory Service"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  sync, full     - Full synchronization (stash → pull → apply → push)"
    echo "  status, st     - Show current sync status"
    echo "  push           - Push staged changes to remote API"
    echo "  pull           - Pull latest changes from remote"
    echo "  stash          - Stash local changes to staging area"
    echo "  apply          - Apply staged changes to local database"
    echo "  init           - Initialize staging database"
    echo "  help, -h       - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 sync        # Full synchronization workflow"
    echo "  $0 status      # Check sync status"
    echo "  $0 push        # Push staged changes only"
    echo ""
    echo "Environment Variables:"
    echo "  MCP_API_KEY    - API key for remote server authentication"
}

# Main command handling
case "${1:-status}" in
    "sync"|"full")
        full_sync
        ;;
    "status"|"st")
        show_status
        ;;
    "push")
        quick_push
        ;;
    "pull")
        quick_pull
        ;;
    "stash")
        print_header "Stashing Changes"
        "$SCRIPT_DIR/stash_local_changes.sh"
        ;;
    "apply")
        print_header "Applying Changes"
        "$SCRIPT_DIR/apply_local_changes.sh"
        ;;
    "init")
        print_header "Initializing Staging Database"
        "$SCRIPT_DIR/init_staging_db.sh"
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        echo "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac