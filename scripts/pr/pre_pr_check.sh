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

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         Pre-PR Quality Gate - MCP Memory Service             ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

FAILED_CHECKS=0
TOTAL_CHECKS=0

# Helper function for check status
check_status() {
    local name="$1"
    local status=$2
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))

    if [ $status -eq 0 ]; then
        echo -e "${GREEN}✅ PASS${NC} - $name"
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
if bash scripts/pr/quality_gate.sh --staged --with-pyscn; then
    check_status "Quality gate (complexity ≤8, no security issues)" 0
else
    check_status "Quality gate (complexity ≤8, no security issues)" 1
    echo -e "${RED}   Fix high-complexity functions or security issues before creating PR${NC}"
fi

# Check 3: Run test suite with coverage
echo -e "\n${YELLOW}[3/9]${NC} Running test suite with coverage..."

# Check if pytest-cov is installed
if ! python -c "import pytest_cov" 2>/dev/null; then
    echo -e "${YELLOW}   Installing pytest-cov...${NC}"
    pip install pytest-cov > /dev/null 2>&1
fi

# Run tests with coverage
COVERAGE_OUTPUT=$(pytest tests/ -q --tb=short \
    --cov=src/mcp_memory_service \
    --cov-report=term-missing 2>&1)

TEST_EXIT_CODE=$?
COVERAGE_PERCENT=$(echo "$COVERAGE_OUTPUT" | grep "TOTAL" | awk '{print $4}' | sed 's/%//')

if [ $TEST_EXIT_CODE -eq 0 ]; then
    check_status "Test suite" 0
else
    check_status "Test suite" 1
    echo -e "${RED}   Fix failing tests before creating PR${NC}"
fi

# Coverage threshold check
if [ -n "$COVERAGE_PERCENT" ] && [ "$COVERAGE_PERCENT" -ge 80 ]; then
    check_status "Test coverage (≥80%)" 0
    echo -e "${GREEN}   Current coverage: ${COVERAGE_PERCENT}%${NC}"
else
    check_status "Test coverage (≥80%)" 1
    echo -e "${RED}   Current coverage: ${COVERAGE_PERCENT}% (minimum: 80%)${NC}"
    echo -e "${YELLOW}   Add tests for untested code${NC}"
fi

# Check 3.5: Handler coverage check
echo -e "\n${YELLOW}[3.5/9]${NC} Checking handler test coverage..."
if python scripts/validation/check_handler_coverage.py; then
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
for file in $(echo "$STAGED_FILES" | grep '\.py$' || true); do
    if [ -f "$file" ]; then
        # Check for inline imports (not at top of file)
        if grep -n "^    import\|^        import" "$file" | grep -v "# inline import" > /dev/null; then
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
for file in $(echo "$STAGED_FILES" | grep '\.py$' || true); do
    if [ -f "$file" ]; then
        # Check for common debug patterns
        if grep -n "import pdb\|breakpoint()\|print(" "$file" | grep -v "logger.debug\|# debug\|\"print" > /dev/null 2>&1; then
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

# Check 7: Docstring coverage
echo -e "\n${YELLOW}[7/9]${NC} Checking docstring coverage..."
MISSING_DOCSTRINGS=0
for file in $(echo "$STAGED_FILES" | grep '\.py$' || true); do
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
    echo -e "${GREEN}✅ ALL CHECKS PASSED${NC} ($TOTAL_CHECKS/$TOTAL_CHECKS)"
    echo ""
    echo -e "${GREEN}Safe to create PR!${NC}"
    echo ""
    echo -e "Next steps:"
    echo -e "  1. Run code-quality-guard agent for final review"
    echo -e "  2. Create PR: ${BLUE}gh pr create --fill${NC}"
    echo -e "  3. Request Gemini review: ${BLUE}gh pr comment <PR#> --body '/gemini review'${NC}"
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
