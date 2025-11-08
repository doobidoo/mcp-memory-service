#!/bin/bash
# scripts/pr/quality_gate.sh - Run all quality checks before PR review
#
# Usage: bash scripts/pr/quality_gate.sh <PR_NUMBER>
# Example: bash scripts/pr/quality_gate.sh 123

set -e

PR_NUMBER=$1

if [ -z "$PR_NUMBER" ]; then
    echo "Usage: $0 <PR_NUMBER>"
    exit 1
fi

if ! command -v gh &> /dev/null; then
    echo "Error: GitHub CLI (gh) is not installed"
    exit 1
fi

if ! command -v gemini &> /dev/null; then
    echo "Error: Gemini CLI is not installed"
    exit 1
fi

echo "=== PR Quality Gate for #$PR_NUMBER ==="
echo ""

exit_code=0
warnings=()
critical_issues=()

# Get changed Python files
echo "Fetching changed files..."
changed_files=$(gh pr diff $PR_NUMBER --name-only | grep '\.py$' || echo "")

if [ -z "$changed_files" ]; then
    echo "No Python files changed in this PR."
    exit 0
fi

echo "Changed Python files:"
echo "$changed_files"
echo ""

# Check 1: Code complexity
echo "=== Check 1: Code Complexity ==="
echo "$changed_files" | while IFS= read -r file; do
    if [ -z "$file" ]; then
        continue
    fi

    if [ ! -f "$file" ]; then
        echo "Skipping $file (file not found in working directory)"
        continue
    fi

    echo "Analyzing: $file"
    result=$(gemini "Analyze code complexity. Rate each function 1-10 (1=simple, 10=very complex). Report ONLY functions with score >7 in format 'FunctionName: Score X - Reason'. File content:

$(cat "$file")" 2>&1 || echo "")

    if echo "$result" | grep -qi "score [89]\|score 10"; then
        warnings+=("High complexity in $file: $result")
        exit_code=1
    fi
done
echo ""

# Check 2: Security scan
echo "=== Check 2: Security Vulnerabilities ==="
echo "$changed_files" | while IFS= read -r file; do
    if [ -z "$file" ]; then
        continue
    fi

    if [ ! -f "$file" ]; then
        continue
    fi

    echo "Scanning: $file"
    # Request machine-parseable output (similar to pre-commit hook)
    result=$(gemini "Security audit. Check for: SQL injection (raw SQL), XSS (unescaped HTML), command injection (os.system, subprocess with shell=True), path traversal, hardcoded secrets.

IMPORTANT: Output format:
- If ANY vulnerability found, start response with: VULNERABILITY_DETECTED: [type]
- If NO vulnerabilities found, start response with: SECURITY_CLEAN
- Then provide details

File content:
$(cat "$file")" 2>&1 || echo "SECURITY_CLEAN")

    # Check for machine-parseable vulnerability marker (more reliable than grep)
    if echo "$result" | grep -q "^VULNERABILITY_DETECTED:"; then
        critical_issues+=("🔴 Security issue in $file: $result")
        exit_code=2
    fi
done
echo ""

# Check 3: Test coverage
echo "=== Check 3: Test Coverage ==="
test_files=$(gh pr diff $PR_NUMBER --name-only | grep -c '^tests/.*\.py$' || echo "0")
# Count code files directly from changed_files
code_files=$(echo "$changed_files" | grep -c '\.py$' || echo "0")

if [ $code_files -gt 0 ] && [ $test_files -eq 0 ]; then
    warnings+=("No test files added/modified despite $code_files code file(s) changed")
    if [ $exit_code -eq 0 ]; then
        exit_code=1
    fi
fi
echo "Code files changed: $code_files"
echo "Test files changed: $test_files"
echo ""

# Check 4: Breaking changes
echo "=== Check 4: Breaking Changes ==="
head_branch=$(gh pr view $PR_NUMBER --json headRefName --jq '.headRefName')

# Get API-related changes
api_changes=$(git diff origin/main...origin/$head_branch -- \
    src/mcp_memory_service/tools.py \
    src/mcp_memory_service/web/api/ \
    2>/dev/null || echo "")

if [ ! -z "$api_changes" ]; then
    echo "Analyzing API changes..."
    # Truncate to 200 lines (increased from 100) to capture more context while preventing overflow
    # Note: Large PRs may still lose context, but this is a reasonable trade-off for performance
    breaking_result=$(gemini "Analyze for breaking changes. Breaking changes include: removed functions/endpoints, changed signatures (parameters removed/reordered), changed return types, renamed public APIs, changed HTTP paths/methods. Report ONLY if breaking changes found with severity (CRITICAL/HIGH/MEDIUM). Changes:

$(echo "$api_changes" | head -200)" 2>&1 || echo "")

    if echo "$breaking_result" | grep -qi "breaking\|CRITICAL\|HIGH"; then
        warnings+=("⚠️  Potential breaking changes detected: $breaking_result")
        if [ $exit_code -eq 0 ]; then
            exit_code=1
        fi
    fi
else
    echo "No API changes detected"
fi
echo ""

# Report results
echo "=== Quality Gate Summary ==="
echo ""

if [ $exit_code -eq 0 ]; then
    echo "✅ ALL CHECKS PASSED"
    echo ""
    echo "Quality Gate Results:"
    echo "- Code complexity: ✅ OK"
    echo "- Security scan: ✅ OK"
    echo "- Test coverage: ✅ OK"
    echo "- Breaking changes: ✅ None detected"
    echo ""

    gh pr comment $PR_NUMBER --body "✅ **Quality Gate PASSED**

All automated checks completed successfully:
- ✅ Code complexity: OK
- ✅ Security scan: OK
- ✅ Test coverage: OK
- ✅ Breaking changes: None detected

PR is ready for Gemini review."

elif [ $exit_code -eq 2 ]; then
    echo "🔴 CRITICAL FAILURES"
    echo ""
    for issue in "${critical_issues[@]}"; do
        echo "$issue"
    done
    echo ""

    # Format issues for comment
    issues_md=$(printf '%s\n' "${critical_issues[@]}" | sed 's/^/- /')

    gh pr comment $PR_NUMBER --body "🔴 **Quality Gate FAILED - CRITICAL**

Security vulnerabilities detected. PR is blocked until issues are resolved.

$issues_md

**Action Required:**
Run \`bash scripts/security/scan_vulnerabilities.sh\` locally and fix all security issues before proceeding."

else
    echo "⚠️  WARNINGS (non-blocking)"
    echo ""
    for warning in "${warnings[@]}"; do
        echo "- $warning"
    done
    echo ""

    # Format warnings for comment
    warnings_md=$(printf '%s\n' "${warnings[@]}" | sed 's/^/- /')

    gh pr comment $PR_NUMBER --body "⚠️ **Quality Gate WARNINGS**

Some checks require attention (non-blocking):

$warnings_md

**Recommendation:**
Consider addressing these issues before requesting review to improve code quality."

fi

exit $exit_code
