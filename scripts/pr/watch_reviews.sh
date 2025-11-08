#!/bin/bash
# scripts/pr/watch_reviews.sh - Watch for Gemini reviews and auto-respond
#
# Usage: bash scripts/pr/watch_reviews.sh <PR_NUMBER> [CHECK_INTERVAL_SECONDS]
# Example: bash scripts/pr/watch_reviews.sh 212 180
#
# Press Ctrl+C to stop watching

set -e

# Get script directory for sourcing helpers
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source GraphQL helpers for thread resolution
if [ -f "$SCRIPT_DIR/lib/graphql_helpers.sh" ]; then
    source "$SCRIPT_DIR/lib/graphql_helpers.sh"
    GRAPHQL_AVAILABLE=true
else
    echo "Warning: GraphQL helpers not available, thread status disabled"
    GRAPHQL_AVAILABLE=false
fi

PR_NUMBER=$1
CHECK_INTERVAL=${2:-180}  # Default: 3 minutes

if [ -z "$PR_NUMBER" ]; then
    echo "Usage: $0 <PR_NUMBER> [CHECK_INTERVAL_SECONDS]"
    echo "Example: $0 212 180"
    exit 1
fi

echo "========================================"
echo "  Gemini PR Review Watch Mode"
echo "========================================"
echo "PR Number: #$PR_NUMBER"
echo "Check Interval: ${CHECK_INTERVAL}s"
echo "GraphQL Thread Tracking: $([ "$GRAPHQL_AVAILABLE" = true ] && echo "Enabled" || echo "Disabled")"
echo "Press Ctrl+C to stop"
echo ""

# Get repository from git remote (portable across forks)
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null || echo "doobidoo/mcp-memory-service")

# Track last review timestamp to detect new reviews
last_review_time=""

while true; do
    echo "[$(date '+%H:%M:%S')] Checking for new reviews..."

    # Get latest Gemini review timestamp
    current_review_time=$(gh api "repos/$REPO/pulls/$PR_NUMBER/reviews" 2>/dev/null | \
        jq -r '[.[] | select(.user.login == "gemini-code-assist[bot]")] | last | .submitted_at' 2>/dev/null || echo "")

    # Get review state
    review_state=$(gh pr view $PR_NUMBER --json reviews --jq '[.reviews[] | select(.author.login == "gemini-code-assist[bot]")] | last | .state' 2>/dev/null || echo "")

    # Get inline comments count (from latest review)
    comments_count=$(gh api "repos/$REPO/pulls/$PR_NUMBER/comments" 2>/dev/null | \
        jq '[.[] | select(.user.login == "gemini-code-assist[bot]")] | length' 2>/dev/null || echo "0")

    echo "  Review State: ${review_state:-none}"
    echo "  Inline Comments: $comments_count"
    echo "  Last Review: ${current_review_time:-never}"

    # Display thread status if GraphQL available
    if [ "$GRAPHQL_AVAILABLE" = true ]; then
        thread_stats=$(get_thread_stats "$PR_NUMBER" 2>/dev/null || echo '{"total":0,"resolved":0,"unresolved":0}')
        total_threads=$(echo "$thread_stats" | jq -r '.total // 0')
        resolved_threads=$(echo "$thread_stats" | jq -r '.resolved // 0')
        unresolved_threads=$(echo "$thread_stats" | jq -r '.unresolved // 0')
        echo "  Review Threads: $total_threads total, $resolved_threads resolved, $unresolved_threads unresolved"
    fi

    # Check if there's a new review
    if [ -n "$current_review_time" ] && [ "$current_review_time" != "$last_review_time" ]; then
        echo ""
        echo "🔔 NEW REVIEW DETECTED!"
        echo "  Timestamp: $current_review_time"
        echo "  State: $review_state"
        echo ""

        last_review_time="$current_review_time"

        # Check if approved
        if [ "$review_state" = "APPROVED" ]; then
            echo "✅ PR APPROVED by Gemini!"
            echo "  No further action needed"
            echo ""
            echo "You can now merge the PR:"
            echo "  gh pr merge $PR_NUMBER --squash"
            echo ""
            echo "Watch mode will continue monitoring..."

        elif [ "$review_state" = "CHANGES_REQUESTED" ] || [ "$comments_count" -gt 0 ]; then
            echo "📝 Review feedback received ($comments_count inline comments)"
            echo ""

            # Display detailed thread status if GraphQL available
            if [ "$GRAPHQL_AVAILABLE" = true ] && [ "$unresolved_threads" -gt 0 ]; then
                echo "Thread Status:"
                bash "$SCRIPT_DIR/thread_status.sh" "$PR_NUMBER" --unresolved 2>/dev/null || true
                echo ""
            fi

            echo "Options:"
            echo "  1. View detailed thread status:"
            echo "     bash scripts/pr/thread_status.sh $PR_NUMBER"
            echo ""
            echo "  2. View inline comments on GitHub:"
            echo "     gh pr view $PR_NUMBER --web"
            echo ""
            echo "  3. Run auto-review to fix issues automatically:"
            echo "     bash scripts/pr/auto_review.sh $PR_NUMBER 5 true"
            echo ""
            echo "  4. Fix manually, push, and resolve threads:"
            echo "     # After pushing fixes:"
            echo "     bash scripts/pr/resolve_threads.sh $PR_NUMBER HEAD --auto"
            echo "     gh pr comment $PR_NUMBER --body '/gemini review'"
            echo ""

            # Optionally auto-trigger review cycle
            read -t 30 -p "Auto-run review cycle? (y/N): " response || response="n"
            echo ""

            if [[ "$response" =~ ^[Yy]$ ]]; then
                echo "🤖 Starting automated review cycle..."
                bash scripts/pr/auto_review.sh $PR_NUMBER 3 true
                echo ""
                echo "✅ Auto-review cycle completed"
                echo "   Watch mode resuming..."
            else
                echo "⏭️  Skipped auto-review"
                echo "   Manual fixes expected"
            fi

        elif [ "$review_state" = "COMMENTED" ]; then
            echo "💬 General comments received (no changes requested)"
            echo "  Review: $review_state"

        else
            echo "ℹ️  Review state: ${review_state:-unknown}"
        fi

        echo ""
        echo "----------------------------------------"
    fi

    echo "  Next check in ${CHECK_INTERVAL}s..."
    echo ""
    sleep $CHECK_INTERVAL
done
