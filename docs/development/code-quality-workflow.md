# Code Quality Workflow Documentation

> **Version**: 1.0.0
> **Last Updated**: November 2025
> **Status**: Active

## Overview

This document describes the comprehensive code quality workflow for the MCP Memory Service project, integrating LLM-based analysis (Groq/Gemini) with static analysis (pyscn) for multi-layer quality assurance.

## Table of Contents

- [Quality Strategy](#quality-strategy)
- [Layer 1: Pre-commit Checks](#layer-1-pre-commit-checks)
- [Layer 2: PR Quality Gates](#layer-2-pr-quality-gates)
- [Layer 3: Periodic Reviews](#layer-3-periodic-reviews)
- [pyscn Integration](#pyscn-integration)
- [Health Score Thresholds](#health-score-thresholds)
- [Troubleshooting](#troubleshooting)
- [Appendix](#appendix)

## Quality Strategy

### Three-Layer Approach

The workflow uses three complementary layers to ensure code quality:

```
Layer 1: Pre-commit     → Fast (<5s)      → Every commit
Layer 2: PR Gate        → Moderate (30s)  → PR creation
Layer 3: Periodic       → Deep (60s)      → Weekly review
```

### Tool Selection

| Tool | Purpose | Speed | Blocking | When |
|------|---------|-------|----------|------|
| **Groq API** | LLM complexity checks | <5s | Yes (>8) | Pre-commit |
| **Gemini CLI** | LLM fallback | ~3s | Yes (>8) | Pre-commit |
| **pyscn** | Static analysis | 30-60s | Yes (<50) | PR + weekly |
| **code-quality-guard** | Manual review | Variable | No | On-demand |

## Layer 1: Pre-commit Checks

### Purpose

Catch quality issues before they enter the codebase.

### Checks Performed

1. **Development Environment Validation**
   - Verify editable install (`pip install -e .`)
   - Check version consistency (source vs installed)
   - Prevent stale package issues

2. **Complexity Analysis** (Groq/Gemini)
   - Rate functions 1-10
   - Block if any function >8
   - Warn if any function =7

3. **Security Scanning**
   - SQL injection (raw SQL queries)
   - XSS (unescaped HTML)
   - Command injection (shell=True)
   - Hardcoded secrets

### Usage

**Installation:**
```bash
ln -s ../../scripts/hooks/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

**Configuration:**
```bash
# Primary LLM: Groq (fast, simple auth)
export GROQ_API_KEY="your-groq-api-key"

# Fallback: Gemini CLI
npm install -g @google/generative-ai-cli
```

**Example Output:**
```
Running pre-commit quality checks...

✓ Using Groq API (fast mode)

Verifying development environment...
✓ Development environment OK

=== Checking: src/mcp_memory_service/storage/sqlite_vec.py ===
Checking complexity...
⚠️  High complexity detected (score 7)
initialize: Score 7 - Multiple nested conditions and error handling paths

Checking for security issues...
✓ No security issues

=== Pre-commit Check Summary ===

⚠️  HIGH COMPLEXITY WARNING

Some functions have high complexity (score 7).
Consider refactoring to improve maintainability.

Continue with commit anyway? (y/n)
```

### Thresholds

- **Block**: Complexity >8, any security issues
- **Warn**: Complexity =7
- **Pass**: Complexity <7, no security issues

## Layer 2: PR Quality Gates

### Purpose

Comprehensive checks before code review and merge.

### Standard Checks

Run automatically on PR creation:

```bash
bash scripts/pr/quality_gate.sh <PR_NUMBER>
```

**Checks:**
1. Code complexity (Gemini CLI)
2. Security vulnerabilities
3. Test coverage (code files vs test files)
4. Breaking changes detection

**Duration:** ~10-30 seconds

### Comprehensive Checks (with pyscn)

Optional deep analysis:

```bash
bash scripts/pr/quality_gate.sh <PR_NUMBER> --with-pyscn
```

**Additional Checks:**
- Cyclomatic complexity scoring
- Dead code detection
- Code duplication analysis
- Coupling metrics (CBO)
- Architecture score

**Duration:** ~30-60 seconds

### Example Output

**Standard Checks:**
```
=== PR Quality Gate for #123 ===

Fetching changed files...
Changed Python files:
src/mcp_memory_service/storage/hybrid.py
tests/test_hybrid_storage.py

=== Check 1: Code Complexity ===
Analyzing: src/mcp_memory_service/storage/hybrid.py
✓ Complexity OK

=== Check 2: Security Vulnerabilities ===
Scanning: src/mcp_memory_service/storage/hybrid.py
✓ No security issues

=== Check 3: Test Coverage ===
Code files changed: 1
Test files changed: 1
✓ Test coverage OK

=== Check 4: Breaking Changes ===
No API changes detected
✓ No breaking changes

=== Quality Gate Summary ===

✅ ALL CHECKS PASSED

Quality Gate Results:
- Code complexity: ✅ OK
- Security scan: ✅ OK
- Test coverage: ✅ OK
- Breaking changes: ✅ None detected
```

**Comprehensive Checks (with pyscn):**
```
=== Check 5: pyscn Comprehensive Analysis ===
Running pyscn static analysis...

📊 Overall Health Score: 68/100

Quality Metrics:
  - Complexity: 45/100 (Avg: 8.2, Max: 15)
  - Dead Code: 75/100 (12 issues)
  - Duplication: 40/100 (4.2% duplication)

⚠️  WARNING - Health score: 68 (threshold: 50)

✓ pyscn analysis completed
```

### Thresholds

- **Block PR**: Security issues, health score <50
- **Warn**: Complexity >7, health score 50-69
- **Pass**: No security issues, health score ≥70

## Layer 3: Periodic Reviews

### Purpose

Track quality trends, detect regressions, plan refactoring.

### Metrics Tracking

**Run manually or via cron:**
```bash
bash scripts/quality/track_pyscn_metrics.sh
```

**Frequency:** Weekly or after major changes

**Stored Data:**
- Health score over time
- Complexity metrics (avg, max)
- Duplication percentage
- Dead code issues
- Architecture violations

**Output:**
- CSV file: `.pyscn/history/metrics.csv`
- JSON report: `.pyscn/reports/analyze_*.json`

**Example Output:**
```
=== pyscn Metrics Tracking ===

Running pyscn analysis (this may take 30-60 seconds)...
✓ Analysis complete

=== Metrics Extracted ===
Health Score: 68/100
Complexity: 45/100 (Avg: 8.2, Max: 15)
Dead Code: 75/100 (12 issues)
Duplication: 40/100 (4.2%)
Coupling: 100/100
Dependencies: 90/100
Architecture: 80/100

✓ Metrics saved to .pyscn/history/metrics.csv

=== Comparison to Previous Run ===
Previous: 70/100 (2025-11-16)
Current:  68/100 (2025-11-23)
Change:   -2 points

⚠️  Regression: -2 points

=== Trend Summary ===
Total measurements: 5
Average health score: 69/100
Highest: 72/100
Lowest: 65/100
```

### Weekly Review

**Run manually or via cron:**
```bash
bash scripts/quality/weekly_quality_review.sh [--create-issue]
```

**Features:**
- Compare current vs last week's metrics
- Generate markdown trend report
- Identify regressions (>5% health score drop)
- Optionally create GitHub issue for significant regressions

**Output:** `docs/development/quality-review-YYYYMMDD.md`

**Example Report:**
```markdown
# Weekly Quality Review - November 23, 2025

## Summary

**Overall Trend:** ➡️  Stable

| Metric | Previous | Current | Change |
|--------|----------|---------|--------|
| Health Score | 70/100 | 68/100 | -2 |
| Complexity | 48/100 | 45/100 | -3 |
| Duplication | 42/100 | 40/100 | -2 |

## Status

### ✅ Acceptable

Health score ≥70 indicates good code quality:
- Continue current development practices
- Monitor trends for regressions
- Address new issues proactively

## Observations

- ⚠️  Complexity score decreased - New complex code introduced
- ⚠️  Code duplication increased - Review for consolidation opportunities
```

## pyscn Integration

### Installation

```bash
pip install pyscn
```

**Repository:** https://github.com/ludo-technologies/pyscn

### Capabilities

1. **Cyclomatic Complexity**
   - Function-level scoring (1-100)
   - Average, maximum, high-risk functions
   - Detailed complexity breakdown

2. **Dead Code Detection**
   - Unreachable code after returns
   - Unused imports
   - Unused variables/functions

3. **Clone Detection**
   - Exact duplicates
   - Near-exact duplicates (>90% similarity)
   - Clone groups and fragments

4. **Coupling Metrics (CBO)**
   - Coupling Between Objects
   - High-coupling classes
   - Average coupling score

5. **Dependency Analysis**
   - Module dependencies
   - Circular dependency detection
   - Dependency depth

6. **Architecture Validation**
   - Layered architecture compliance
   - Layer violation detection
   - Cross-layer dependencies

### Usage

**Full Analysis:**
```bash
pyscn analyze .
```

**Generate and view the machine-readable report:**
```bash
pyscn analyze --json --no-open .
python3 -m json.tool "$(ls -t .pyscn/reports/analyze_*.json | head -1)" | less
```

The PR and trend scripts read metrics from the JSON `summary` contract through
`scripts/quality/read_pyscn_summary.py`. They deliberately reject missing or
non-numeric metrics instead of recording a misleading zero score.

### Report Interpretation

**Health Score Breakdown:**

| Component | Score | Grade | Interpretation |
|-----------|-------|-------|----------------|
| **Complexity** | 40/100 | Poor | 28 high-risk functions (>7), avg 9.5 |
| **Dead Code** | 70/100 | Fair | 27 issues, 2 critical |
| **Duplication** | 30/100 | Poor | 6.0% duplication, 18 clone groups |
| **Coupling** | 100/100 | Excellent | Avg CBO 1.5, 0 high-coupling |
| **Dependencies** | 85/100 | Good | 0 cycles, depth 7 |
| **Architecture** | 75/100 | Good | 58 violations, 75.5% compliance |

**Example: Complexity Report**

```
Top 5 High-Complexity Functions:
1. install.py::main() - Complexity: 62, Nesting: 6
2. config.py::__main__() - Complexity: 42, Nesting: 0
3. sqlite_vec.py::initialize() - Complexity: 38, Nesting: 10
4. oauth/authorization.py::token() - Complexity: 35, Nesting: 4
5. install.py::install_package() - Complexity: 33, Nesting: 4
```

**Action:** Refactor functions with complexity >10 using:
- Extract method refactoring
- Strategy pattern for conditionals
- Helper functions for complex operations

## Health Score Thresholds

### Release Blocker (<50)

**Status:** 🔴 **Cannot merge or release**

**Required Actions:**
1. Review full pyscn report
2. Identify top 5 complexity hotspots
3. Create refactoring tasks
4. Schedule immediate refactoring sprint
5. Track progress in issue #240

**Timeline:** Must resolve before any merges

### Action Required (50-69)

**Status:** 🟡 **Plan refactoring within 2 weeks**

**Recommended Actions:**
1. Analyze complexity trends
2. Create project board for tracking
3. Allocate 20% sprint capacity to quality
4. Review duplication for consolidation
5. Remove dead code

**Timeline:** 2-week improvement plan

### Good (70-84)

**Status:** ✅ **Monitor trends, continue development**

**Maintenance:**
- Monthly quality reviews
- Track complexity trends
- Keep health score above 70
- Address new issues proactively

### Excellent (85+)

**Status:** 🎯 **Maintain current standards**

**Best Practices:**
- Document quality patterns
- Share refactoring techniques
- Mentor team members
- Celebrate wins

## Troubleshooting

### Common Issues

**Issue:** pyscn not found
```bash
# Solution
pip install pyscn
```

**Issue:** Pre-commit hook not running
```bash
# Solution
chmod +x .git/hooks/pre-commit
ls -la .git/hooks/pre-commit  # Verify symlink
```

**Issue:** Groq API errors
```bash
# Solution 1: Check API key
echo $GROQ_API_KEY  # Should not be empty

# Solution 2: Test Groq connection
curl https://api.groq.com/openai/v1/models \
  -H "Authorization: Bearer $GROQ_API_KEY"

# Solution 3: Fall back to Gemini
unset GROQ_API_KEY  # Temporarily disable Groq
```

**Issue:** pyscn analysis too slow
```bash
# Solution: Run on specific directories
pyscn analyze src/  # Exclude tests, scripts
pyscn analyze --exclude "tests/*,scripts/*"
```

**Issue:** False positive security warnings
```bash
# Solution: Review and whitelist
# Add comment explaining why code is safe
# Example:
# SAFE: User input sanitized via parameterized query
```

### Performance Tuning

**Pre-commit Hooks:**
- Use Groq API (200-300ms vs Gemini 2-3s)
- Analyze only staged files
- Skip checks if no Python files

**PR Quality Gates:**
- Run standard checks first (fast)
- Use `--with-pyscn` for comprehensive analysis
- Cache pyscn reports for repeated checks

**Periodic Reviews:**
- Schedule during off-hours (cron)
- Use JSON output for scripting
- Archive old reports (keep last 30 days)

## Appendix

### Script Reference

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/hooks/pre-commit` | Pre-commit quality checks | Auto-runs on `git commit` |
| `scripts/pr/quality_gate.sh` | PR quality gates | `bash scripts/pr/quality_gate.sh <PR>` |
| `scripts/pr/run_pyscn_analysis.sh` | pyscn PR analysis | `bash scripts/pr/run_pyscn_analysis.sh --pr <PR>` |
| `scripts/quality/track_pyscn_metrics.sh` | Metrics tracking | `bash scripts/quality/track_pyscn_metrics.sh` |
| `scripts/quality/read_pyscn_summary.py` | Validate and extract pyscn JSON metrics | Called by both pyscn scripts |
| `scripts/quality/weekly_quality_review.sh` | Weekly review | `bash scripts/quality/weekly_quality_review.sh` |

### Configuration Files

| File | Purpose |
|------|---------|
| `.pyscn/.gitignore` | Ignore pyscn reports and history |
| `.pyscn/history/metrics.csv` | Historical quality metrics |
| `.pyscn/reports/*.json` | pyscn machine-readable reports |
| `.claude/agents/code-quality-guard.md` | Code quality agent specification |

### Related Documentation

- [CLAUDE.md](../../CLAUDE.md) - Project conventions and workflows
- [`.claude/agents/code-quality-guard.md`](../../.claude/agents/code-quality-guard.md) - Agent workflows
- [scripts/README.md](../../scripts/README.md) - Script documentation
- [Issue #240](https://github.com/doobidoo/mcp-memory-service/issues/240) - Quality improvements tracking

### External Resources

- [pyscn GitHub](https://github.com/ludo-technologies/pyscn) - pyscn documentation
- [Groq API Docs](https://console.groq.com/docs) - Groq API reference
- [Gemini CLI](https://www.npmjs.com/package/@google/generative-ai-cli) - Gemini CLI docs

---

**Document Version History:**

- v1.0.0 (2025-11-24): Initial comprehensive documentation with pyscn integration
