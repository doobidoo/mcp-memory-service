#!/bin/bash
# scripts/pr/auto_review.sh - Automated PR review loop using Gemini CLI
#
# Usage: bash scripts/pr/auto_review.sh <PR_NUMBER> [MAX_ITERATIONS] [SAFE_FIX_MODE]
# Example: bash scripts/pr/auto_review.sh 123 5 true

set -e

# Get script directory for sourcing helpers
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source GraphQL helpers for thread resolution
if [ -f "$SCRIPT_DIR/lib/graphql_helpers.sh" ]; then
    source "$SCRIPT_DIR/lib/graphql_helpers.sh"
    GRAPHQL_AVAILABLE=true
else
    echo "Warning: GraphQL helpers not available, thread auto-resolution disabled"
    GRAPHQL_AVAILABLE=false
fi

PR_NUMBER=$1
MAX_ITERATIONS=${2:-5}
SAFE_FIX_MODE=${3:-true}

if [ -z "$PR_NUMBER" ]; then
    echo "Usage: $0 <PR_NUMBER> [MAX_ITERATIONS] [SAFE_FIX_MODE]"
    echo "Example: $0 123 5 true"
    exit 1
fi

# Check dependencies
if ! command -v gh &> /dev/null; then
    echo "Error: GitHub CLI (gh) is not installed"
    echo "Install: https://cli.github.com/"
    exit 1
fi

if ! command -v gemini &> /dev/null; then
    echo "Error: Gemini CLI is not installed"
    exit 1
fi

echo "=== Automated PR Review Loop ==="
echo "PR Number: #$PR_NUMBER"
echo "Max Iterations: $MAX_ITERATIONS"
echo "Safe Fix Mode: $SAFE_FIX_MODE"
echo "GraphQL Thread Resolution: $([ "$GRAPHQL_AVAILABLE" = true ] && echo "Enabled" || echo "Disabled")"
echo ""

# Get repository from git remote (portable across forks)
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null || echo "doobidoo/mcp-memory-service")

iteration=1
approved=false

while [ $iteration -le $MAX_ITERATIONS ] && [ "$approved" = false ]; do
    echo "=== Iteration $iteration/$MAX_ITERATIONS ==="

    # Trigger Gemini review (use /gemini review for inline comments)
    echo "Requesting Gemini code review (inline comments)..."
    gh pr comment $PR_NUMBER --body "/gemini review"

    # Wait for Gemini to process
    echo "Waiting for Gemini review (90 seconds)..."
    sleep 90

    # Fetch latest review status and comments
    echo "Fetching review feedback..."

    # Get review state (APPROVED, CHANGES_REQUESTED, COMMENTED)
    review_state=$(gh pr view $PR_NUMBER --json reviews --jq '[.reviews[] | select(.author.login == "gemini-code-assist[bot]")] | last | .state')

    # Fetch actual comment bodies for categorization first
    review_comments=$(gh api "repos/$REPO/pulls/$PR_NUMBER/comments" | \
        jq -r '[.[] | select(.user.login == "gemini-code-assist[bot]")] | .[] | "- \(.path):\(.line) - \(.body[0:200])"' | \
        head -50)

    # Get inline review comments count
    review_comments_count=$(gh api "repos/$REPO/pulls/$PR_NUMBER/comments" | jq '[.[] | select(.user.login == "gemini-code-assist[bot]")] | length')

    echo "Review State: $review_state"
    echo "Inline Comments: $review_comments_count"

    # Display thread status if GraphQL available
    if [ "$GRAPHQL_AVAILABLE" = true ]; then
        thread_stats=$(get_thread_stats "$PR_NUMBER" 2>/dev/null || echo '{"total":0,"resolved":0,"unresolved":0}')
        total_threads=$(echo "$thread_stats" | jq -r '.total // 0')
        resolved_threads=$(echo "$thread_stats" | jq -r '.resolved // 0')
        unresolved_threads=$(echo "$thread_stats" | jq -r '.unresolved // 0')
        echo "Review Threads: $total_threads total, $resolved_threads resolved, $unresolved_threads unresolved"
    fi

    echo ""

    # Check if approved
    if [ "$review_state" = "APPROVED" ]; then
        echo "✅ PR approved by Gemini!"
        approved=true
        break
    fi

    # If no inline comments, we're done
    if [ "$review_comments_count" -eq 0 ] && [ "$review_state" != "CHANGES_REQUESTED" ]; then
        echo "✅ No issues found in review"
        approved=true
        break
    fi

    # Extract actionable issues
    if [ "$SAFE_FIX_MODE" = true ]; then
        echo "Analyzing feedback for safe auto-fixes..."

        # Get PR diff
        pr_diff=$(gh pr diff $PR_NUMBER)

        # Use Gemini to categorize issues (request JSON format)
        categorization=$(gemini "Categorize these code review comments into a JSON object.

Review feedback:
$review_comments

Categories:
- safe: Simple fixes (formatting, imports, type hints, docstrings, variable renaming)
- unsafe: Logic changes, API modifications, security-critical code
- non_code: Documentation, discussion, questions

IMPORTANT: Output ONLY valid JSON in this exact format:
{
  \"safe\": [\"issue 1\", \"issue 2\"],
  \"unsafe\": [\"issue 1\"],
  \"non_code\": [\"comment 1\"]
}")

        echo "$categorization"

        # Extract safe issues using jq
        safe_issues=$(echo "$categorization" | jq -r '.safe[]' 2>/dev/null || echo "")

        if [ -z "$safe_issues" ]; then
            echo "No safe auto-fixable issues found. Manual intervention required."
            break
        fi

        echo ""
        echo "Safe issues to auto-fix:"
        echo "$safe_issues"
        echo ""

        # Generate fixes for safe issues
        echo "Generating code fixes..."
        fixes=$(gemini "Generate git diff patches for these safe fixes:

Issues to fix:
$safe_issues

Current code (PR diff):
$pr_diff

Output only the git diff patch that can be applied with 'git apply'. Include file paths and line numbers.")

        # Use mktemp for patch file
        patch_file=$(mktemp -t pr_fixes_${PR_NUMBER}_${iteration}.XXXXXX)
        echo "$fixes" > "$patch_file"

        # Attempt to apply fixes
        echo "Attempting to apply fixes..."
        if git apply --check "$patch_file" 2>/dev/null; then
            git apply "$patch_file"
            git add -A

            # Create commit message
            commit_msg="fix: apply Gemini review feedback (iteration $iteration)

Addressed:
$safe_issues

Co-Authored-By: Gemini Code Assist <gemini@google.com>"

            git commit -m "$commit_msg"
            git push

            echo "✅ Fixes applied and pushed"

            # Auto-resolve review threads for files we just fixed
            if [ "$GRAPHQL_AVAILABLE" = true ]; then
                echo ""
                echo "Resolving review threads for fixed code..."

                # Get the commit SHA we just created
                latest_commit=$(git rev-parse HEAD)

                # Run thread resolution in auto mode
                if bash "$SCRIPT_DIR/resolve_threads.sh" "$PR_NUMBER" "$latest_commit" --auto 2>&1 | grep -q "Resolved:"; then
                    echo "✅ Review threads auto-resolved"
                else
                    echo "ℹ️  No threads needed resolution"
                fi
            fi

            # Clean up temp file
            rm -f "$patch_file"
        else
            echo "⚠️  Could not auto-apply fixes. Patch saved to $patch_file"
            echo "Manual application required."
            break
        fi
    else
        echo "Manual fix mode enabled. Review feedback above and apply manually."
        break
    fi

    iteration=$((iteration + 1))
    echo ""
    echo "Waiting 10 seconds before next iteration..."
    sleep 10
done

echo ""
echo "=== Review Loop Complete ==="

if [ "$approved" = true ]; then
    echo "🎉 PR #$PR_NUMBER is approved and ready to merge!"
    gh pr comment $PR_NUMBER --body "✅ **Automated Review Complete**

All review iterations completed successfully. PR is approved and ready for merge.

Iterations: $((iteration - 1))/$MAX_ITERATIONS"
    exit 0
else
    echo "⚠️  Max iterations reached or manual intervention needed"
    gh pr comment $PR_NUMBER --body "⚠️ **Automated Review Incomplete**

Review loop completed $((iteration - 1)) iterations but approval not received.
Manual review and intervention may be required.

Please review the latest feedback and apply necessary changes."
    exit 1
fi
