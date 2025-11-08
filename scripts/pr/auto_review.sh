#!/bin/bash
# scripts/pr/auto_review.sh - Automated PR review loop using Gemini CLI
#
# Usage: bash scripts/pr/auto_review.sh <PR_NUMBER> [MAX_ITERATIONS] [SAFE_FIX_MODE]
# Example: bash scripts/pr/auto_review.sh 123 5 true

set -e

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
echo ""

iteration=1
approved=false

while [ $iteration -le $MAX_ITERATIONS ] && [ "$approved" = false ]; do
    echo "=== Iteration $iteration/$MAX_ITERATIONS ==="

    # Trigger Gemini review (comment on PR)
    echo "Requesting Gemini review..."
    gh pr comment $PR_NUMBER --body "Please review this PR for code quality, security, and best practices. Focus on:
- Code complexity and readability
- Security vulnerabilities
- Performance implications
- Breaking changes
- Test coverage

Provide specific, actionable feedback."

    # Wait for Gemini to process
    echo "Waiting for Gemini review (90 seconds)..."
    sleep 90

    # Fetch latest review comments
    echo "Fetching review feedback..."
    review_comments=$(gh pr view $PR_NUMBER --json comments --jq '[.comments[] | select(.author.login == "gemini-code-assist")] | last | .body')

    echo ""
    echo "--- Review Feedback ---"
    echo "$review_comments"
    echo "--- End Feedback ---"
    echo ""

    # Check if approved
    if echo "$review_comments" | grep -qi "looks good\|approved\|lgtm\|ready to merge"; then
        echo "✅ PR approved by Gemini!"
        approved=true
        break
    fi

    # Extract actionable issues
    if [ "$SAFE_FIX_MODE" = true ]; then
        echo "Analyzing feedback for safe auto-fixes..."

        # Get PR diff
        pr_diff=$(gh pr diff $PR_NUMBER)

        # Use Gemini to categorize issues
        categorization=$(gemini "Categorize these code review comments:

Review feedback:
$review_comments

Categories:
- SAFE: Simple fixes (formatting, imports, type hints, docstrings, variable renaming)
- UNSAFE: Logic changes, API modifications, security-critical code
- NON_CODE: Documentation, discussion, questions

Output format:
SAFE:
- [specific issue 1]
- [specific issue 2]

UNSAFE:
- [specific issue 1]

NON_CODE:
- [specific comment 1]")

        echo "$categorization"

        # Extract safe issues
        safe_issues=$(echo "$categorization" | sed -n '/SAFE:/,/UNSAFE:/p' | grep '^-' | sed 's/^- //')

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

        echo "$fixes" > /tmp/pr_fixes_${PR_NUMBER}_${iteration}.patch

        # Attempt to apply fixes
        echo "Attempting to apply fixes..."
        if git apply --check /tmp/pr_fixes_${PR_NUMBER}_${iteration}.patch 2>/dev/null; then
            git apply /tmp/pr_fixes_${PR_NUMBER}_${iteration}.patch
            git add -A

            # Create commit message
            commit_msg="fix: apply Gemini review feedback (iteration $iteration)

Addressed:
$safe_issues

Co-Authored-By: Gemini Code Assist <gemini@google.com>"

            git commit -m "$commit_msg"
            git push

            echo "✅ Fixes applied and pushed"
        else
            echo "⚠️  Could not auto-apply fixes. Patch saved to /tmp/pr_fixes_${PR_NUMBER}_${iteration}.patch"
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
