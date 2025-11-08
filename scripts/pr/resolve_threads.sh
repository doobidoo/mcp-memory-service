#!/bin/bash
# scripts/pr/resolve_threads.sh - Smart PR review thread resolution
#
# Automatically resolves review threads when the commented code has been modified.
# Uses GitHub GraphQL API to resolve threads (REST API cannot do this).
#
# Usage: bash scripts/pr/resolve_threads.sh <PR_NUMBER> [COMMIT_SHA] [--auto]
# Example: bash scripts/pr/resolve_threads.sh 212 HEAD --auto
#
# Modes:
#   --auto: Automatically resolve threads without confirmation
#   (default): Prompt for confirmation before resolving each thread

set -e

# Get script directory for sourcing helpers
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source GraphQL helpers
if [ -f "$SCRIPT_DIR/lib/graphql_helpers.sh" ]; then
    source "$SCRIPT_DIR/lib/graphql_helpers.sh"
else
    echo "Error: GraphQL helpers not found at $SCRIPT_DIR/lib/graphql_helpers.sh"
    exit 1
fi

# Parse arguments
PR_NUMBER=$1
COMMIT_SHA=${2:-HEAD}
AUTO_MODE=false

if [ -z "$PR_NUMBER" ]; then
    echo "Usage: $0 <PR_NUMBER> [COMMIT_SHA] [--auto]"
    echo "Example: $0 212 HEAD --auto"
    exit 1
fi

# Check for --auto flag
if [ "$2" = "--auto" ] || [ "$3" = "--auto" ]; then
    AUTO_MODE=true
fi

# Verify gh CLI supports GraphQL
if ! check_graphql_support; then
    exit 1
fi

echo "========================================"
echo "  Smart PR Review Thread Resolution"
echo "========================================"
echo "PR Number: #$PR_NUMBER"
echo "Commit: $COMMIT_SHA"
echo "Mode: $([ "$AUTO_MODE" = true ] && echo "Automatic" || echo "Interactive")"
echo ""

# Get all review threads
echo "Fetching review threads..."
threads_json=$(get_review_threads "$PR_NUMBER")

# Check if there are any threads
total_threads=$(echo "$threads_json" | jq '.data.repository.pullRequest.reviewThreads.nodes | length')

if [ "$total_threads" -eq 0 ]; then
    echo "✅ No review threads found for PR #$PR_NUMBER"
    exit 0
fi

# Count unresolved threads
unresolved_count=$(echo "$threads_json" | jq '[.data.repository.pullRequest.reviewThreads.nodes[] | select(.isResolved == false)] | length')

echo "Total threads: $total_threads"
echo "Unresolved threads: $unresolved_count"
echo ""

if [ "$unresolved_count" -eq 0 ]; then
    echo "✅ All review threads are already resolved!"
    exit 0
fi

# Get files modified in the commit
echo "Analyzing commit $COMMIT_SHA..."
modified_files=$(get_modified_files "$COMMIT_SHA")

if [ -z "$modified_files" ]; then
    echo "⚠️  No files modified in commit $COMMIT_SHA"
    echo "Cannot determine which threads to resolve."
    exit 1
fi

echo "Modified files:"
echo "$modified_files" | sed 's/^/  - /'
echo ""

# Process each unresolved thread
resolved_count=0
skipped_count=0
failed_count=0

echo "Processing unresolved threads..."
echo "========================================"

echo "$threads_json" | jq -r '.data.repository.pullRequest.reviewThreads.nodes[] | select(.isResolved == false) | @json' | while IFS= read -r thread_json; do
    thread_id=$(echo "$thread_json" | jq -r '.id')
    path=$(echo "$thread_json" | jq -r '.path // "unknown"')
    line=$(echo "$thread_json" | jq -r '.line // 0')
    is_outdated=$(echo "$thread_json" | jq -r '.isOutdated')
    comment_body=$(echo "$thread_json" | jq -r '.comments.nodes[0].body // "No comment"' | head -c 100)

    echo ""
    echo "Thread: $thread_id"
    echo "  File: $path:$line"
    echo "  Outdated: $is_outdated"
    echo "  Comment: ${comment_body}..."

    # Determine if we should resolve this thread
    should_resolve=false
    resolution_reason=""

    # Check if file was modified in the commit
    if echo "$modified_files" | grep -q "^${path}$"; then
        # File was modified - check if the specific line was changed
        if was_line_modified "$path" "$line" "$COMMIT_SHA"; then
            should_resolve=true
            resolution_reason="Line $line in $path was modified in commit $(git rev-parse --short "$COMMIT_SHA")"
        else
            resolution_reason="File modified but line $line unchanged"
        fi
    elif [ "$is_outdated" = "true" ]; then
        # Thread is marked as outdated by GitHub
        should_resolve=true
        resolution_reason="Thread marked as outdated by GitHub (code changed in subsequent commits)"
    else
        resolution_reason="File not modified in this commit"
    fi

    echo "  Decision: $resolution_reason"

    if [ "$should_resolve" = true ]; then
        # Resolve the thread
        if [ "$AUTO_MODE" = true ]; then
            echo "  Action: Auto-resolving..."

            # Add explanatory comment and resolve
            comment_text="✅ Resolved: $resolution_reason

Verified by automated thread resolution script."

            if resolve_review_thread "$thread_id" "$comment_text" 2>/dev/null; then
                echo "  ✅ Resolved successfully"
                resolved_count=$((resolved_count + 1))
            else
                echo "  ❌ Failed to resolve"
                failed_count=$((failed_count + 1))
            fi
        else
            # Interactive mode - ask for confirmation
            read -p "  Resolve this thread? (y/N): " -n 1 -r
            echo ""

            if [[ $REPLY =~ ^[Yy]$ ]]; then
                # Optionally ask for custom comment
                read -p "  Add custom comment? (leave empty for auto): " custom_comment

                if [ -n "$custom_comment" ]; then
                    comment_text="✅ $custom_comment"
                else
                    comment_text="✅ Resolved: $resolution_reason"
                fi

                if resolve_review_thread "$thread_id" "$comment_text" 2>/dev/null; then
                    echo "  ✅ Resolved successfully"
                    resolved_count=$((resolved_count + 1))
                else
                    echo "  ❌ Failed to resolve"
                    failed_count=$((failed_count + 1))
                fi
            else
                echo "  ⏭️  Skipped"
                skipped_count=$((skipped_count + 1))
            fi
        fi
    else
        echo "  ⏭️  Skipped (no changes detected)"
        skipped_count=$((skipped_count + 1))
    fi
done

echo ""
echo "========================================"
echo "  Resolution Summary"
echo "========================================"
echo "Resolved: $resolved_count"
echo "Skipped: $skipped_count"
echo "Failed: $failed_count"
echo ""

# Get updated thread stats
echo "Fetching updated thread status..."
updated_stats=$(get_thread_stats "$PR_NUMBER")

echo "Final Thread Status:"
echo "$updated_stats" | jq -r 'to_entries | .[] | "  \(.key | ascii_upcase): \(.value)"'
echo ""

# Exit with success if we resolved any threads or if there were none to resolve
if [ "$resolved_count" -gt 0 ] || [ "$unresolved_count" -eq 0 ]; then
    echo "✅ Thread resolution complete!"
    exit 0
else
    echo "⚠️  No threads were resolved"
    exit 0
fi
