#!/bin/bash
# scripts/pr/thread_status.sh - Display PR review thread status
#
# Shows comprehensive status of all review threads on a PR with filtering options.
# Uses GitHub GraphQL API to access review thread data.
#
# Usage: bash scripts/pr/thread_status.sh <PR_NUMBER> [--unresolved|--resolved|--outdated]
# Example: bash scripts/pr/thread_status.sh 212 --unresolved
#
# Flags:
#   --unresolved: Show only unresolved threads
#   --resolved: Show only resolved threads
#   --outdated: Show only outdated threads
#   (no flag): Show all threads with summary

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
FILTER=${2:-all}

if [ -z "$PR_NUMBER" ]; then
    echo "Usage: $0 <PR_NUMBER> [--unresolved|--resolved|--outdated]"
    echo "Example: $0 212 --unresolved"
    exit 1
fi

# Verify gh CLI supports GraphQL
if ! check_graphql_support; then
    exit 1
fi

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
GRAY='\033[0;90m'
NC='\033[0m' # No Color

echo "========================================"
echo "  PR Review Thread Status"
echo "========================================"
echo "PR Number: #$PR_NUMBER"
echo "Filter: ${FILTER/--/}"
echo ""

# Get all review threads
echo "Fetching review threads..."
threads_json=$(get_review_threads "$PR_NUMBER")

# Get thread statistics
stats=$(get_thread_stats "$PR_NUMBER")

total=$(echo "$stats" | jq -r '.total')
resolved=$(echo "$stats" | jq -r '.resolved')
unresolved=$(echo "$stats" | jq -r '.unresolved')
outdated=$(echo "$stats" | jq -r '.outdated')

# Display summary
echo "========================================"
echo "  Summary"
echo "========================================"
echo -e "Total Threads:      $total"
echo -e "${GREEN}Resolved:${NC}           $resolved"
echo -e "${RED}Unresolved:${NC}         $unresolved"
echo -e "${YELLOW}Outdated:${NC}           $outdated"
echo ""

if [ "$total" -eq 0 ]; then
    echo "✅ No review threads found for PR #$PR_NUMBER"
    exit 0
fi

# Display detailed thread list
echo "========================================"
echo "  Thread Details"
echo "========================================"

# Determine jq filter based on flag
case "$FILTER" in
    --unresolved)
        jq_filter='select(.isResolved == false)'
        ;;
    --resolved)
        jq_filter='select(.isResolved == true)'
        ;;
    --outdated)
        jq_filter='select(.isOutdated == true)'
        ;;
    *)
        jq_filter='.'
        ;;
esac

# Process and display threads
thread_count=0

echo "$threads_json" | jq -r ".data.repository.pullRequest.reviewThreads.nodes[] | $jq_filter | @json" | while IFS= read -r thread_json; do
    thread_count=$((thread_count + 1))

    thread_id=$(echo "$thread_json" | jq -r '.id')
    path=$(echo "$thread_json" | jq -r '.path // "unknown"')
    line=$(echo "$thread_json" | jq -r '.line // 0')
    original_line=$(echo "$thread_json" | jq -r '.originalLine // 0')
    diff_side=$(echo "$thread_json" | jq -r '.diffSide // "unknown"')
    is_resolved=$(echo "$thread_json" | jq -r '.isResolved')
    is_outdated=$(echo "$thread_json" | jq -r '.isOutdated')

    # Get first comment details
    author=$(echo "$thread_json" | jq -r '.comments.nodes[0].author.login // "unknown"')
    comment_body=$(echo "$thread_json" | jq -r '.comments.nodes[0].body // "No comment"')
    created_at=$(echo "$thread_json" | jq -r '.comments.nodes[0].createdAt // "unknown"')
    comment_count=$(echo "$thread_json" | jq -r '.comments.nodes | length')

    # Truncate comment to 150 chars for display
    comment_preview=$(echo "$comment_body" | head -c 150 | tr '\n' ' ')
    if [ ${#comment_body} -gt 150 ]; then
        comment_preview="${comment_preview}..."
    fi

    # Format status indicators
    if [ "$is_resolved" = "true" ]; then
        status_icon="${GREEN}✓${NC}"
        status_text="${GREEN}RESOLVED${NC}"
    else
        status_icon="${RED}○${NC}"
        status_text="${RED}UNRESOLVED${NC}"
    fi

    if [ "$is_outdated" = "true" ]; then
        outdated_icon="${YELLOW}⚠${NC}"
        outdated_text="${YELLOW}OUTDATED${NC}"
    else
        outdated_icon=" "
        outdated_text="${GRAY}current${NC}"
    fi

    # Display thread
    echo ""
    echo -e "$status_icon Thread #$thread_count"
    echo -e "  Status: $status_text | $outdated_text"
    echo -e "  File: ${BLUE}$path${NC}:$line (original: $original_line)"
    echo -e "  Side: $diff_side"
    echo -e "  Author: $author"
    echo -e "  Created: $created_at"
    echo -e "  Comments: $comment_count"
    echo -e "  ${GRAY}\"${comment_preview}\"${NC}"

    # Show thread ID for reference (can be used with resolve_threads.sh)
    echo -e "  ${GRAY}Thread ID: ${thread_id:0:20}...${NC}"
done

echo ""
echo "========================================"

# Provide actionable next steps
if [ "$unresolved" -gt 0 ]; then
    echo ""
    echo "📝 Next Steps:"
    echo ""
    echo "  1. Review unresolved threads:"
    echo "     gh pr view $PR_NUMBER --web"
    echo ""
    echo "  2. After fixing issues and pushing commits, resolve threads:"
    echo "     bash scripts/pr/resolve_threads.sh $PR_NUMBER HEAD --auto"
    echo ""
    echo "  3. Manually resolve specific threads via GitHub web interface"
    echo ""
    echo "  4. Trigger new Gemini review after fixes:"
    echo "     gh pr comment $PR_NUMBER --body '/gemini review'"
    echo ""
fi

# Exit with status indicating unresolved threads
if [ "$unresolved" -gt 0 ]; then
    exit 1
else
    echo "✅ All review threads resolved!"
    exit 0
fi
