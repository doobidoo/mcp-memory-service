#!/bin/bash
# scripts/pr/run_pyscn_analysis.sh - Run pyscn comprehensive code quality analysis
#
# Usage:
#   bash scripts/pr/run_pyscn_analysis.sh [--pr PR_NUMBER] [--threshold SCORE]
#
# Options:
#   --pr PR_NUMBER    Post results as PR comment (requires gh CLI)
#   --threshold SCORE Minimum health score (default: 50, blocks below this)
#
# Examples:
#   bash scripts/pr/run_pyscn_analysis.sh                    # Local analysis
#   bash scripts/pr/run_pyscn_analysis.sh --pr 123           # Analyze and comment on PR #123
#   bash scripts/pr/run_pyscn_analysis.sh --threshold 70     # Require health score ≥70

set -e
set -o pipefail

# Colors for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
PR_NUMBER=""
THRESHOLD=50  # Default: block if health score <50

while [[ $# -gt 0 ]]; do
    case $1 in
        --pr)
            PR_NUMBER="$2"
            shift 2
            ;;
        --threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--pr PR_NUMBER] [--threshold SCORE]"
            exit 1
            ;;
    esac
done

# Check for pyscn
if ! command -v pyscn &> /dev/null; then
    echo -e "${RED}❌ pyscn not found${NC}"
    echo ""
    echo "Install pyscn with:"
    echo "  pip install pyscn"
    echo ""
    echo "Repository: https://github.com/ludo-technologies/pyscn"
    exit 1
fi

echo -e "${BLUE}=== pyscn Code Quality Analysis ===${NC}"
echo ""

# Create reports directory if needed
mkdir -p .pyscn/reports

# Run pyscn analysis
echo "Running pyscn analysis (this may take 30-60 seconds)..."
echo ""

# Mark the start so stale reports from previous runs cannot be selected.
RUN_MARKER=$(mktemp)

# Run analysis using pyscn's stable machine-readable format.
if pyscn analyze --json --no-open . 2>&1 | tee /tmp/pyscn_output.log; then
    echo -e "${GREEN}✓${NC} Analysis complete"
else
    rm -f "$RUN_MARKER"
    echo -e "${RED}❌ Analysis failed${NC}"
    cat /tmp/pyscn_output.log
    exit 1
fi

JSON_FILE=$(find .pyscn/reports -type f -name 'analyze_*.json' -newer "$RUN_MARKER" -print | sort | tail -1)
rm -f "$RUN_MARKER"
if [ -z "$JSON_FILE" ]; then
    echo -e "${RED}❌ Analysis produced no new JSON report${NC}"
    exit 1
fi

METRICS=$(python3 "$(dirname "$0")/../quality/read_pyscn_summary.py" "$JSON_FILE") || exit 1
IFS=$'\t' read -r HEALTH_SCORE COMPLEXITY_SCORE DEAD_CODE_SCORE DUPLICATION_SCORE \
    _COUPLING_SCORE _DEPENDENCIES_SCORE ARCHITECTURE_SCORE AVG_COMPLEXITY \
    MAX_COMPLEXITY DUPLICATION_NUM DEAD_CODE_ISSUES <<< "$METRICS"
DUPLICATION_PCT="${DUPLICATION_NUM}%"

echo ""
echo -e "${BLUE}=== Analysis Results ===${NC}"
echo ""
echo "📊 Overall Health Score: $HEALTH_SCORE/100"
echo ""
echo "Quality Metrics:"
echo "  - Complexity: $COMPLEXITY_SCORE/100 (Avg: $AVG_COMPLEXITY, Max: $MAX_COMPLEXITY)"
echo "  - Dead Code: $DEAD_CODE_SCORE/100 ($DEAD_CODE_ISSUES issues)"
echo "  - Duplication: $DUPLICATION_SCORE/100 ($DUPLICATION_PCT duplication)"
echo ""
echo "📄 Report: $JSON_FILE"
echo ""

# Determine status
EXIT_CODE=0
STATUS="✅ PASSED"
EMOJI="✅"
COLOR=$GREEN

if [ "$HEALTH_SCORE" -lt "$THRESHOLD" ]; then
    STATUS="🔴 BLOCKED"
    EMOJI="🔴"
    COLOR=$RED
    EXIT_CODE=1
elif [ "$HEALTH_SCORE" -lt 70 ]; then
    STATUS="⚠️  WARNING"
    EMOJI="⚠️"
    COLOR=$YELLOW
fi

echo -e "${COLOR}${STATUS}${NC} - Health score: $HEALTH_SCORE (threshold: $THRESHOLD)"
echo ""

# Generate recommendations
RECOMMENDATIONS=""

if [ "$HEALTH_SCORE" -lt 50 ]; then
    RECOMMENDATIONS="**🚨 Critical Action Required:**
- Health score below 50 is a release blocker
- Focus on top 5 highest complexity functions
- Remove dead code before proceeding
"
elif [ "$HEALTH_SCORE" -lt 70 ]; then
    RECOMMENDATIONS="**⚠️  Improvement Recommended:**
- Plan refactoring sprint within 2 weeks
- Track high-complexity functions on project board
- Review duplication patterns for consolidation opportunities
"
fi

# Check for critical issues
CRITICAL_COMPLEXITY=""
if [ "$MAX_COMPLEXITY" -gt 10 ]; then
    CRITICAL_COMPLEXITY="- ⚠️  Functions with complexity >10 detected (Max: $MAX_COMPLEXITY)
"
fi

CRITICAL_DUPLICATION=""
if (( $(echo "$DUPLICATION_NUM > 5.0" | bc -l) )); then
    CRITICAL_DUPLICATION="- ⚠️  Code duplication above 5% threshold ($DUPLICATION_PCT)
"
fi

# Post to PR if requested
if [ -n "$PR_NUMBER" ]; then
    if ! command -v gh &> /dev/null; then
        echo -e "${YELLOW}⚠️  gh CLI not found, skipping PR comment${NC}"
    else
        echo "Posting results to PR #$PR_NUMBER..."

        COMMENT_BODY="## ${EMOJI} pyscn Code Quality Analysis

**Health Score:** $HEALTH_SCORE/100

### Quality Metrics

| Metric | Score | Details |
|--------|-------|---------|
| 🔢 Complexity | $COMPLEXITY_SCORE/100 | Avg: $AVG_COMPLEXITY, Max: $MAX_COMPLEXITY |
| 💀 Dead Code | $DEAD_CODE_SCORE/100 | $DEAD_CODE_ISSUES issues |
| 📋 Duplication | $DUPLICATION_SCORE/100 | $DUPLICATION_PCT code duplication |
| 🏗️  Architecture | $ARCHITECTURE_SCORE/100 | pyscn architecture score |

### Status

$STATUS (Threshold: $THRESHOLD)

${CRITICAL_COMPLEXITY}${CRITICAL_DUPLICATION}${RECOMMENDATIONS}

### Full Report

View detailed analysis: [JSON Report]($JSON_FILE)

---

<details>
<summary>📖 About pyscn</summary>

pyscn (Python Static Code Navigator) provides comprehensive static analysis including:
- Cyclomatic complexity analysis
- Dead code detection
- Code duplication (clone detection)
- Coupling metrics (CBO)
- Dependency graph analysis
- Architecture violation detection

Repository: https://github.com/ludo-technologies/pyscn
</details>"

        echo "$COMMENT_BODY" | gh pr comment "$PR_NUMBER" --body-file -
        echo -e "${GREEN}✓${NC} Posted comment to PR #$PR_NUMBER"
    fi
fi

# Summary
echo ""
echo -e "${BLUE}=== Summary ===${NC}"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ Quality checks passed${NC}"
    echo ""
    echo "Health score ($HEALTH_SCORE) meets threshold ($THRESHOLD)"
else
    echo -e "${RED}❌ Quality checks failed${NC}"
    echo ""
    echo "Health score ($HEALTH_SCORE) below threshold ($THRESHOLD)"
    echo ""
    echo "Action required before merging:"
    echo "  1. Review full report: $JSON_FILE"
    echo "  2. Address high-complexity functions (complexity >10)"
    echo "  3. Remove dead code ($DEAD_CODE_ISSUES issues)"
    echo "  4. Reduce duplication where feasible"
    echo ""
fi

exit $EXIT_CODE
