#!/bin/bash
#
# Pre-PR Quality Gate Script
#
# Runs comprehensive quality checks BEFORE creating a pull request.
# This prevents the multi-iteration review nightmare by catching issues early.
#
# Usage:
#   bash scripts/pr/pre_pr_check.sh [--fix]
#
# Options:
#   --fix    Attempt to auto-fix issues (black formatting, isort, etc.)
#
# Exit codes:
#   0 - All checks passed, safe to create PR
#   1 - Quality checks failed, DO NOT create PR yet
#   2 - Script error or missing dependencies
#

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

FIX_MODE=false
if [ "$1" = "--fix" ]; then
    FIX_MODE=true
fi

# Resolve Python: prefer project .venv (where the editable install lives),
# then $VIRTUAL_ENV, then bare `python3`/`python`. Without this, system Python
# misses the editable mcp_memory_service package and checks fail with
# "Package not installed" / "ModuleNotFoundError" errors.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
    PIP_BIN="$REPO_ROOT/.venv/bin/pip"
    PYTEST_BIN="$REPO_ROOT/.venv/bin/pytest"
elif [ -n "$VIRTUAL_ENV" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python"
    PIP_BIN="$VIRTUAL_ENV/bin/pip"
    PYTEST_BIN="$VIRTUAL_ENV/bin/pytest"
else
    PYTHON_BIN="$(command -v python3 || command -v python)"
    PIP_BIN="$(command -v pip3 || command -v pip)"
    PYTEST_BIN="$(command -v pytest)"
fi
[ -x "$PYTEST_BIN" ] || PYTEST_BIN="$PYTHON_BIN -m pytest"

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         Pre-PR Quality Gate - MCP Memory Service             ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

FAILED_CHECKS=0
SKIPPED_CHECKS=0
TOTAL_CHECKS=0

# Helper function for check status
# status: 0 = pass, 3 = skipped (tool missing — neither pass nor fail), else fail
check_status() {
    local name="$1"
    local status=$2
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))

    if [ $status -eq 0 ]; then
        echo -e "${GREEN}✅ PASS${NC} - $name"
    elif [ $status -eq 3 ]; then
        echo -e "${YELLOW}SKIP${NC} - $name (not enforced — see note below)"
        SKIPPED_CHECKS=$((SKIPPED_CHECKS + 1))
    else
        echo -e "${RED}❌ FAIL${NC} - $name"
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
    fi
}

# Check 1: Staged files exist
echo -e "\n${YELLOW}[1/9]${NC} Checking for staged files..."
STAGED_FILES=$(git diff --cached --name-only)
if [ -z "$STAGED_FILES" ]; then
    echo -e "${RED}❌ No staged files found. Stage your changes first: git add .${NC}"
    exit 2
fi
echo -e "${GREEN}✅${NC} Found $(echo "$STAGED_FILES" | wc -l) staged files"

# Check 2: Run full quality gate
echo -e "\n${YELLOW}[2/9]${NC} Running quality_gate.sh (complexity, security, PEP 8)..."
set +e
bash scripts/pr/quality_gate.sh --staged --with-pyscn
QUALITY_GATE_EXIT=$?
set -e
if [ $QUALITY_GATE_EXIT -eq 0 ]; then
    check_status "Quality gate (complexity, security, test coverage, breaking changes)" 0
elif [ $QUALITY_GATE_EXIT -eq 3 ]; then
    # Gemini CLI missing — the gate ran nothing. Reporting this as a pass made
    # the whole check meaningless on machines without the CLI.
    check_status "Quality gate (complexity, security, test coverage, breaking changes)" 3
    echo -e "${YELLOW}   Complexity and security were not evaluated locally — CI still checks them${NC}"
else
    check_status "Quality gate (complexity, security, test coverage, breaking changes)" 1
    echo -e "${RED}   See the FINDINGS list above - it names the check that failed${NC}"
fi

# Check 3: Run test suite with coverage
echo -e "\n${YELLOW}[3/9]${NC} Running test suite with coverage..."

# Check if pytest-cov is installed
if ! "$PYTHON_BIN" -c "import pytest_cov" 2>/dev/null; then
    echo -e "${YELLOW}   Installing pytest-cov...${NC}"
    "$PIP_BIN" install pytest-cov > /dev/null 2>&1
fi

# Run tests with coverage.
# `set +e` around the assignment is load-bearing: under `set -e` a failing
# pytest inside $(...) aborts this script immediately, so the TEST_EXIT_CODE
# handling below never ran and a failing suite looked like the gate itself
# crashing with no message.
# The selection mirrors .github/workflows/ci.yml so that a green gate here
# means the same thing CI will say. Benchmarks, consolidation and integration
# are excluded there (heavy, network, or services the runner lacks); running
# them here made the gate fail locally on tests CI never executes.
set +e
COVERAGE_OUTPUT=$($PYTEST_BIN tests/ -q --tb=short \
    --ignore=tests/consolidation \
    --ignore=tests/benchmarks \
    --ignore=tests/integration \
    -m "not benchmark" \
    --timeout=120 \
    --cov=src/mcp_memory_service \
    --cov-report=term-missing 2>&1)
TEST_EXIT_CODE=$?
set -e
COVERAGE_PERCENT=$(echo "$COVERAGE_OUTPUT" | grep "TOTAL" | awk '{print $4}' | sed 's/%//')

if [ $TEST_EXIT_CODE -eq 0 ]; then
    check_status "Test suite" 0
else
    check_status "Test suite" 1
    echo -e "${RED}   Fix failing tests before creating PR${NC}"
    # Show what actually failed — the output was captured, so without this the
    # user is told to fix failing tests without being told which ones.
    echo "$COVERAGE_OUTPUT" | grep -E "^(FAILED|ERROR)" | head -20 | sed 's/^/     /'
fi

# Coverage threshold check.
#
# Advisory, not blocking — same stance as .github/workflows/ci.yml, which runs
# coverage without --cov-fail-under and says so: "coverage is report-only for
# now, re-introduce a gate once the deterministic-subset baseline is known and
# stable". The deterministic subset currently sits near 60%, so a hard 80% here
# meant this gate could not be passed by anyone, on any branch.
COVERAGE_TARGET=80
if [ -n "$COVERAGE_PERCENT" ] && [ "$COVERAGE_PERCENT" -ge "$COVERAGE_TARGET" ]; then
    check_status "Test coverage (target ${COVERAGE_TARGET}%)" 0
    echo -e "${GREEN}   Current coverage: ${COVERAGE_PERCENT}%${NC}"
else
    check_status "Test coverage (target ${COVERAGE_TARGET}%)" 3
    echo -e "${YELLOW}   Current coverage: ${COVERAGE_PERCENT}% (target: ${COVERAGE_TARGET}%, advisory)${NC}"
    echo -e "${YELLOW}   Add tests for the code this PR touches; the target is not enforced yet${NC}"
fi

# Check 3.5: Handler coverage check
echo -e "\n${YELLOW}[3.5/9]${NC} Checking handler test coverage..."
if "$PYTHON_BIN" scripts/validation/check_handler_coverage.py; then
    check_status "Handler coverage (all 17 handlers tested)" 0
else
    check_status "Handler coverage (all 17 handlers tested)" 1
    echo -e "${RED}   Add integration tests for untested handlers${NC}"
    echo -e "${YELLOW}   See: tests/integration/test_all_memory_handlers.py for examples${NC}"
fi

# Check 4: Import validation
echo -e "\n${YELLOW}[4/9]${NC} Validating imports (regression check for Issue #299)..."
if bash scripts/ci/validate_imports.sh > /dev/null 2>&1; then
    check_status "Import validation (no ModuleNotFoundError)" 0
else
    check_status "Import validation (no ModuleNotFoundError)" 1
    echo -e "${RED}   Fix import errors:${NC}"
    bash scripts/ci/validate_imports.sh
fi

# Check 5: PEP 8 compliance (imports)
echo -e "\n${YELLOW}[5/9]${NC} Checking import ordering (PEP 8)..."
IMPORT_ISSUES=0
for file in $(echo "$STAGED_FILES" | grep '\.py$' | grep -v '^claude-hooks/' || true); do
    if [ -f "$file" ]; then
        # Check for inline imports (not at top of file). The trailing space is
        # load-bearing: without it the pattern also matches any indented line
        # starting with `importlib`, e.g. `    importlib.util.find_spec(...)`,
        # and reports it as a misplaced import.
        if grep -n "^    import \|^        import " "$file" | grep -v "# inline import" > /dev/null; then
            echo -e "${RED}   Found inline imports in $file (should be at top)${NC}"
            IMPORT_ISSUES=$((IMPORT_ISSUES + 1))
        fi
    fi
done

if [ $IMPORT_ISSUES -eq 0 ]; then
    check_status "Import ordering (PEP 8)" 0
else
    check_status "Import ordering (PEP 8)" 1
    if [ "$FIX_MODE" = true ]; then
        echo -e "${BLUE}   Running isort to fix imports...${NC}"
        isort $(echo "$STAGED_FILES" | grep '\.py$') 2>/dev/null || true
    fi
fi

# Check 6: No debug code
echo -e "\n${YELLOW}[6/9]${NC} Checking for debug code..."
DEBUG_ISSUES=0
for file in $(echo "$STAGED_FILES" | grep '\.py$' | grep -v '^claude-hooks/' || true); do
    if [ -f "$file" ]; then
        # A debugger left behind is always wrong, wherever it sits.
        DEBUG_PATTERN="import pdb\|breakpoint()"

        # print() only counts as debug code in the library. CLI scripts and
        # tests print as their normal output, so flagging it there made the
        # gate unpassable for any change under scripts/ (#188).
        case "$file" in
            src/*) DEBUG_PATTERN="$DEBUG_PATTERN\|print(" ;;
        esac

        if grep -n "$DEBUG_PATTERN" "$file" | grep -v "logger.debug\|# debug\|\"print" > /dev/null 2>&1; then
            echo -e "${YELLOW}   Found potential debug code in $file${NC}"
            DEBUG_ISSUES=$((DEBUG_ISSUES + 1))
        fi
    fi
done

if [ $DEBUG_ISSUES -eq 0 ]; then
    check_status "No debug code" 0
else
    check_status "No debug code" 1
    echo -e "${YELLOW}   Review and remove debug statements (or add '# debug' comment if intentional)${NC}"
fi

# Check 6.5: Log injection guard
echo -e "\n${YELLOW}[6.5/9]${NC} Checking for unsanitized log calls (log-injection guard)..."
LOG_INJECT_ISSUES=0
for file in $(echo "$STAGED_FILES" | grep '\.py$' | grep -v '^claude-hooks/' || true); do
    if [ -f "$file" ]; then
        # Flag f-string logger calls without _sanitize_log_value — potential log injection
        HITS=$(grep -En 'logger\.(info|warning|error|debug|critical)\(f"' "$file" \
               | grep -v '_sanitize_log_value' || true)
        if [ -n "$HITS" ]; then
            echo -e "${YELLOW}   Potential log injection in $file:${NC}"
            echo "$HITS" | head -5 | sed 's/^/     /'
            LOG_INJECT_ISSUES=$((LOG_INJECT_ISSUES + 1))
        fi
    fi
done

if [ $LOG_INJECT_ISSUES -eq 0 ]; then
    check_status "Log injection guard (f-string logger calls sanitized)" 0
else
    check_status "Log injection guard (f-string logger calls sanitized)" 1
    echo -e "${RED}   Wrap user-provided values with _sanitize_log_value() from compat.py${NC}"
    echo -e "${YELLOW}   Example: logger.info(f\"Stored: {_sanitize_log_value(content)}\")${NC}"
    echo -e "${YELLOW}   See: src/mcp_memory_service/compat.py for the utility function${NC}"
fi

# Check 6.6: Bundled hooks config carries no credentials
echo -e "\n${YELLOW}[6.6/9]${NC} Checking bundled hooks config for credentials..."
if bash scripts/ci/check_hooks_config_secrets.sh > /dev/null 2>&1; then
    check_status "Bundled hooks config free of credentials/local paths" 0
else
    check_status "Bundled hooks config free of credentials/local paths" 1
    echo -e "${RED}   claude-hooks/config.json ships to users - replace personal values:${NC}"
    bash scripts/ci/check_hooks_config_secrets.sh || true
fi

# Check 6.7: Bundled hooks config and its template have not drifted apart
echo -e "\n${YELLOW}[6.7/9]${NC} Checking bundled hooks config against its template..."
if bash scripts/ci/check_hooks_config_drift.sh > /dev/null 2>&1; then
    check_status "Hooks config and template share one key structure" 0
else
    check_status "Hooks config and template share one key structure" 1
    echo -e "${RED}   Add the same keys to whichever file is missing them:${NC}"
    bash scripts/ci/check_hooks_config_drift.sh || true
fi

# Check 6.8: Dead references in active docs
# scripts/ci/check_dead_refs.sh existed since #702 but was invoked by nothing —
# no CI workflow, no hook, not this gate — while CLAUDE.md told contributors
# it ran in CI on docs changes. Running it here is the cheap half of the fix
# (0.5s, whole-tree scan of docs/ and README.md); the CI trigger is issue #312.
echo -e "\n${YELLOW}[6.8/9]${NC} Checking active docs for dead references..."
if bash scripts/ci/check_dead_refs.sh > /dev/null 2>&1; then
    check_status "No dead references in active docs" 0
else
    check_status "No dead references in active docs" 1
    echo -e "${RED}   Removed features/ports/commands are still referenced in docs:${NC}"
    bash scripts/ci/check_dead_refs.sh || true
fi

# Check 7: Docstring coverage
echo -e "\n${YELLOW}[7/9]${NC} Checking docstring coverage..."
MISSING_DOCSTRINGS=0
for file in $(echo "$STAGED_FILES" | grep '\.py$' | grep -v '^claude-hooks/' || true); do
    if [ -f "$file" ]; then
        # Simple heuristic: Check for functions without docstrings
        FUNC_COUNT=$(grep -c "^def \|^    def " "$file" 2>/dev/null || echo "0")
        DOCSTRING_COUNT=$(grep -c "\"\"\"" "$file" 2>/dev/null || echo "0")

        if [ "$FUNC_COUNT" -gt 0 ] && [ "$DOCSTRING_COUNT" -eq 0 ]; then
            echo -e "${YELLOW}   $file has functions but no docstrings${NC}"
            MISSING_DOCSTRINGS=$((MISSING_DOCSTRINGS + 1))
        fi
    fi
done

if [ $MISSING_DOCSTRINGS -eq 0 ]; then
    check_status "Docstring coverage" 0
else
    check_status "Docstring coverage" 1
    echo -e "${YELLOW}   Add docstrings to new functions (Args, Returns, Raises)${NC}"
fi

# Check 8: Final validation summary
echo -e "\n${YELLOW}[8/9]${NC} Final validation summary..."
check_status "All automated checks completed" 0

# Check 9: Code-quality-guard agent recommendation
echo -e "\n${YELLOW}[9/9]${NC} Code-quality-guard agent check..."
echo -e "${BLUE}   RECOMMENDATION: Run code-quality-guard agent for deep analysis${NC}"
echo -e "${BLUE}   Command: @agent code-quality-guard \"Analyze staged files\"${NC}"
echo -e "${YELLOW}   ⚠️  This PR template requires agent usage - mark checkbox when done${NC}"

# Summary
echo -e "\n${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                      Quality Gate Summary                      ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

if [ $FAILED_CHECKS -eq 0 ]; then
    PASSED_CHECKS=$((TOTAL_CHECKS - SKIPPED_CHECKS))
    echo -e "${GREEN}✅ ALL CHECKS PASSED${NC} ($PASSED_CHECKS/$TOTAL_CHECKS)"
    if [ $SKIPPED_CHECKS -gt 0 ]; then
        echo -e "${YELLOW}   $SKIPPED_CHECKS check(s) skipped — see SKIP above; those were not evaluated${NC}"
    fi
    echo ""
    echo -e "${GREEN}Safe to create PR!${NC}"
    echo ""
    echo -e "Next steps:"
    echo -e "  1. Run code-quality-guard agent for final review"
    echo -e "  2. Create PR: ${BLUE}gh pr create --title '<title>' --body '<body>'${NC}"
    echo ""
    exit 0
else
    echo -e "${RED}❌ FAILED${NC} ($FAILED_CHECKS/$TOTAL_CHECKS checks failed)"
    echo ""
    echo -e "${RED}DO NOT create PR yet!${NC}"
    echo ""
    echo -e "Fix the issues above, then re-run:"
    echo -e "  ${BLUE}bash scripts/pr/pre_pr_check.sh${NC}"
    echo ""

    if [ "$FIX_MODE" = false ]; then
        echo -e "Or try auto-fix mode:"
        echo -e "  ${BLUE}bash scripts/pr/pre_pr_check.sh --fix${NC}"
        echo ""
    fi

    exit 1
fi
