#!/bin/bash
# scripts/pr/quality_gate.sh - Run all quality checks before PR review
#
# Two modes:
#   bash scripts/pr/quality_gate.sh --staged [--with-pyscn]      # staged changes (pre-PR)
#   bash scripts/pr/quality_gate.sh <PR_NUMBER> [--with-pyscn]   # a GitHub-era PR
#
# The staged mode is what pre_pr_check.sh calls; it needs no forge access.
#
# The complexity and security checks need a text model. By default they talk to a
# local OpenAI-compatible endpoint via scripts/pr/lib/llm_prompt.py, which keeps the
# gate off paid APIs. Set MCP_QUALITY_LLM=gemini to use the Gemini CLI instead.
# Endpoint configuration lives in that helper's docstring.
#
# Exit codes: 0 passed, 1 warnings, 2 critical (security), 3 skipped (no backend).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLM_HELPER="$SCRIPT_DIR/lib/llm_prompt.py"
EXIT_SKIPPED=3

MODE=""
PR_NUMBER=""
RUN_PYSCN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --staged)
            MODE="staged"
            shift
            ;;
        --with-pyscn)
            RUN_PYSCN=true
            shift
            ;;
        [0-9]*)
            MODE="pr"
            PR_NUMBER=$1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 (--staged | <PR_NUMBER>) [--with-pyscn]"
            exit 1
            ;;
    esac
done

if [ -z "$MODE" ]; then
    echo "Usage: $0 (--staged | <PR_NUMBER>) [--with-pyscn]"
    exit 1
fi

if [ "$MODE" = "pr" ] && ! command -v gh &> /dev/null; then
    echo "Error: PR mode needs the GitHub CLI (gh). Use --staged for local checks."
    exit 1
fi

# Resolve the analysis backend before running anything, so "no backend" is
# reported as skipped rather than as a silent pass on every file.
LLM_BACKEND="${MCP_QUALITY_LLM:-local}"

if [ "$LLM_BACKEND" = "gemini" ]; then
    if ! command -v gemini &> /dev/null; then
        echo "WARNING: MCP_QUALITY_LLM=gemini but the Gemini CLI is not installed."
        echo "   Skipped, NOT passed: complexity and security were not evaluated."
        exit $EXIT_SKIPPED
    fi
elif ! echo "reply with READY" | python3 "$LLM_HELPER" > /dev/null 2>&1; then
    echo "WARNING: no local analysis model reachable - skipping AI-based quality checks."
    echo "   Tried ${MCP_QUALITY_LLM_URL:-http://127.0.0.1:11437/v1} via $LLM_HELPER."
    echo "   Skipped, NOT passed: complexity and security were not evaluated."
    echo "   Start the local endpoint, point MCP_QUALITY_LLM_URL at another one,"
    echo "   or set MCP_QUALITY_LLM=gemini to use the Gemini CLI."
    # Exit 3 = skipped, distinct from 0 (passed) and 1 (failed). Callers only see
    # the status code, and pre_pr_check.sh used to report this as a green check.
    exit $EXIT_SKIPPED
fi

# analyze <prompt> - one model call, empty output on failure so callers stay simple.
analyze() {
    if [ "$LLM_BACKEND" = "gemini" ]; then
        gemini "$1" 2>&1 || echo ""
    else
        printf '%s' "$1" | python3 "$LLM_HELPER" 2>/dev/null || echo ""
    fi
}

if [ "$MODE" = "staged" ]; then
    echo "=== Quality Gate for staged changes ==="
else
    echo "=== PR Quality Gate for #$PR_NUMBER ==="
fi
echo "Analysis backend: $LLM_BACKEND"
echo ""

exit_code=0
warnings=()
critical_issues=()

# Get changed files
echo "Fetching changed files..."
pr_head_branch=""
if [ "$MODE" = "staged" ]; then
    all_changed=$(git diff --cached --name-only --diff-filter=ACMR)
else
    all_changed=$(gh pr diff $PR_NUMBER --name-only)
    pr_head_branch=$(gh pr view $PR_NUMBER --json headRefName --jq '.headRefName')
fi
changed_files=$(echo "$all_changed" | grep '\.py$' || echo "")

if [ -z "$changed_files" ]; then
    echo "No Python files changed."
    exit 0
fi

echo "Changed Python files:"
echo "$changed_files"
echo ""

# Check 1: Code complexity
# The loops below read from process substitution, not a pipe: a piped `while`
# runs in a subshell, so exit_code and the warnings array never made it out and
# these two checks could not fail the gate.
echo "=== Check 1: Code Complexity ==="
while IFS= read -r file; do
    if [ -z "$file" ]; then
        continue
    fi

    if [ ! -f "$file" ]; then
        echo "Skipping $file (file not found in working directory)"
        continue
    fi

    # Which functions does this change actually touch? The model scores the whole
    # file, so without this every change to a file that already holds a complex
    # function failed the gate -- proximity, not the change (#1118).
    if [ "$MODE" = "staged" ]; then
        touched=$(python3 "$SCRIPT_DIR/lib/touched_functions.py" --staged "$file")
    else
        touched=$(python3 "$SCRIPT_DIR/lib/touched_functions.py" --range "origin/main...origin/$pr_head_branch" "$file")
    fi

    if [ -z "$touched" ]; then
        echo "Skipping $file (no function bodies touched)"
        continue
    fi

    echo "Analyzing: $file"
    result=$(analyze "Analyze code complexity. Rate each function 1-10 (1=simple, 10=very complex). Report ONLY functions with score >7 in format 'FunctionName: Score X - Reason'. File content:

$(cat "$file")")

    # The documented budget is grade A-B, complexity <= 8, so 8 is acceptable and
    # only 9 and 10 are findings. Findings on functions this diff did not touch
    # are dropped: they are the file's pre-existing state, and the author of an
    # unrelated change is not the person to refactor them.
    scoped=$(printf '%s' "$result" | python3 "$SCRIPT_DIR/lib/scope_findings.py" $touched) && {
        warnings+=("High complexity in $file: $scoped")
        exit_code=1
    }
done < <(echo "$changed_files")
echo ""

# Check 2: Security scan
echo "=== Check 2: Security Vulnerabilities ==="
while IFS= read -r file; do
    if [ -z "$file" ]; then
        continue
    fi

    if [ ! -f "$file" ]; then
        continue
    fi

    echo "Scanning: $file"
    # Request machine-parseable output (similar to pre-commit hook)
    result=$(analyze "Security audit. Check for: SQL injection (raw SQL), XSS (unescaped HTML), command injection (os.system, subprocess with shell=True), path traversal, hardcoded secrets.

IMPORTANT: Output format:
- If ANY vulnerability found, start response with: VULNERABILITY_DETECTED: [type]
- If NO vulnerabilities found, start response with: SECURITY_CLEAN
- Then provide details

File content:
$(cat "$file")")

    # Check for machine-parseable vulnerability marker (more reliable than grep)
    if echo "$result" | grep -q "^VULNERABILITY_DETECTED:"; then
        critical_issues+=("Security issue in $file: $result")
        exit_code=2
    fi
done < <(echo "$changed_files")
echo ""

# Check 3: Test coverage
echo "=== Check 3: Test Coverage ==="
# `grep -c` prints 0 and exits 1 on no match, so an `|| echo 0` fallback yielded
# "0\n0" and the arithmetic test below died on it. `|| true` keeps grep's own count.
# .sh too: the gate scripts themselves are covered by tests/ci/*.sh, and counting
# only .py made every shell-tested change look untested.
test_files=$(echo "$all_changed" | grep -Ec '^tests/.*\.(py|sh)$' || true)
# Count code files directly from changed_files
code_files=$(echo "$changed_files" | grep -c '\.py$' || true)

# Dead-code removal has no new behavior to cover, so demanding a test file for it
# made the gate unpassable for exactly the cleanup it keeps asking for. Anything
# that adds real behavior still needs a test -- see lib/is_cleanup_only.py.
if [ "$MODE" = "staged" ]; then
    py_diff=$(git diff --cached -- '*.py')
else
    py_diff=$(gh pr diff $PR_NUMBER)
fi
cleanup_only=false
if [ -n "$py_diff" ] && printf '%s' "$py_diff" | python3 "$SCRIPT_DIR/lib/is_cleanup_only.py"; then
    cleanup_only=true
fi

if [ $code_files -gt 0 ] && [ $test_files -eq 0 ] && [ "$cleanup_only" = false ]; then
    warnings+=("No test files added/modified despite $code_files code file(s) changed")
    if [ $exit_code -eq 0 ]; then
        exit_code=1
    fi
fi
echo "Code files changed: $code_files"
echo "Test files changed: $test_files"
if [ $code_files -gt 0 ] && [ "$cleanup_only" = true ]; then
    echo "Cleanup-only change (no added behavior) - test requirement not applied"
fi
echo ""

# Check 4: Breaking changes
echo "=== Check 4: Breaking Changes ==="
api_paths=(src/mcp_memory_service/tools src/mcp_memory_service/web/api)

if [ "$MODE" = "staged" ]; then
    api_changes=$(git diff --cached -- "${api_paths[@]}" 2>/dev/null || echo "")
else
    api_changes=$(git diff "origin/main...origin/$pr_head_branch" -- "${api_paths[@]}" 2>/dev/null || echo "")
fi

if [ ! -z "$api_changes" ]; then
    echo "Analyzing API changes..."
    # Truncate to 200 lines to bound the prompt. Large diffs still lose context,
    # which is an accepted trade-off here.
    # The marker is required for the same reason the security check has one: the
    # old pattern grepped for the word "breaking", which matches the model's own
    # "No breaking changes found." Every API change reported a finding whose text
    # said there was none.
    breaking_result=$(analyze "Analyze for breaking changes. Breaking changes include: removed functions/endpoints, changed signatures (parameters removed/reordered), changed return types, renamed public APIs, changed HTTP paths/methods.

IMPORTANT: Output format:
- If ANY breaking change is found, start the response with: BREAKING_CHANGE_DETECTED: [severity CRITICAL/HIGH/MEDIUM]
- If there are none, start the response with: NO_BREAKING_CHANGES
- Then provide details

Changes:

$(echo "$api_changes" | head -200)")

    if echo "$breaking_result" | grep -q "^BREAKING_CHANGE_DETECTED:"; then
        warnings+=("Potential breaking changes detected: $breaking_result")
        if [ $exit_code -eq 0 ]; then
            exit_code=1
        fi
    fi
else
    echo "No API changes detected"
fi
echo ""

# Check 5: pyscn comprehensive analysis (optional)
pyscn_ran=false
if [ "$RUN_PYSCN" = true ]; then
    echo "=== Check 5: pyscn Comprehensive Analysis ==="

    if command -v pyscn &> /dev/null; then
        pyscn_ran=true
        echo "Running pyscn static analysis..."

        pyscn_args=(--threshold 50)
        if [ "$MODE" = "pr" ]; then
            pyscn_args+=(--pr "$PR_NUMBER")
        fi

        if bash "$SCRIPT_DIR/run_pyscn_analysis.sh" "${pyscn_args[@]}"; then
            echo "pyscn analysis passed"
        else
            echo "pyscn analysis found quality issues (health score <50)"
            # Block on pyscn failures when health score below threshold
            if [ $exit_code -eq 0 ]; then
                exit_code=1
            fi
        fi
    else
        echo "pyscn not installed, skipping comprehensive analysis"
        echo "Install with: pip install pyscn"
    fi
    echo ""
fi

# Report results
echo "=== Quality Gate Summary ==="
echo ""

if [ $exit_code -eq 0 ]; then
    echo "ALL CHECKS PASSED"
    echo ""
    echo "Quality Gate Results:"
    echo "- Code complexity: OK"
    echo "- Security scan: OK"
    echo "- Test coverage: OK"
    echo "- Breaking changes: none detected"
    # Only claim pyscn passed when it actually ran; --with-pyscn without the tool
    # installed used to print OK for an analysis that never happened.
    if [ "$pyscn_ran" = true ]; then
        echo "- pyscn analysis: OK"
    elif [ "$RUN_PYSCN" = true ]; then
        echo "- pyscn analysis: SKIPPED (pyscn not installed)"
    fi
    echo ""

    if [ "$MODE" = "pr" ]; then
        pyscn_note=""
        if [ "$pyscn_ran" = true ]; then
            pyscn_note="
- pyscn comprehensive analysis: see detailed report comment"
        fi

        gh pr comment $PR_NUMBER --body "**Quality Gate PASSED**

All automated checks completed successfully:
- Code complexity: OK
- Security scan: OK
- Test coverage: OK
- Breaking changes: none detected${pyscn_note}"
    fi

elif [ $exit_code -eq 2 ]; then
    echo "CRITICAL FAILURES"
    echo ""
    for issue in "${critical_issues[@]}"; do
        echo "$issue"
    done
    echo ""

    if [ "$MODE" = "pr" ]; then
        issues_md=$(printf '%s\n' "${critical_issues[@]}" | sed 's/^/- /')

        gh pr comment $PR_NUMBER --body "**Quality Gate FAILED - CRITICAL**

Security vulnerabilities detected. PR is blocked until issues are resolved.

$issues_md

**Action required:**
Run \`bash scripts/security/scan_vulnerabilities.sh\` locally and fix all security issues before proceeding."
    fi

else
    # These block: exit 1 is what pre_pr_check.sh reads to fail check 2. The
    # heading used to say "non-blocking", so the gate contradicted itself in the
    # same breath (#1118).
    echo "FINDINGS (blocking)"
    echo ""
    for warning in "${warnings[@]}"; do
        echo "- $warning"
    done
    echo ""

    if [ "$MODE" = "pr" ]; then
        warnings_md=$(printf '%s\n' "${warnings[@]}" | sed 's/^/- /')

        gh pr comment $PR_NUMBER --body "**Quality Gate FAILED**

These findings block the gate:

$warnings_md

Each names a function this change touched. Findings on untouched functions in the same file are not reported."
    fi

fi

exit $exit_code
