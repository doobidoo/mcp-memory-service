# Code Quality Workflow - Multi-Layer Strategy

## Three-Layer Quality Assurance

The QA workflow uses three complementary layers for comprehensive code quality assurance:

### Layer 1: Pre-commit (Fast - <5s)
- Groq/Gemini LLM complexity checks
- Security scanning (SQL injection, XSS, command injection)
- Dev environment validation
- **Blocking**: Complexity >8, any security issues

### Layer 2: PR Quality Gate (Moderate - 10-60s)
- Standard checks: complexity, security, test coverage, breaking changes
- Comprehensive checks (`--with-pyscn`): + duplication, dead code, architecture
- **Blocking**: Security issues, health score <50

### Layer 3: Periodic Review (Weekly)
- pyscn codebase-wide analysis
- Trend tracking and regression detection
- Refactoring sprint planning

## pyscn Integration

[pyscn](https://github.com/ludo-technologies/pyscn) provides comprehensive static analysis:

**Capabilities:**
- Cyclomatic complexity scoring
- Dead code detection
- Clone detection (duplication)
- Coupling metrics (CBO)
- Dependency graph analysis
- Architecture violation detection

**Usage:**

```bash
# PR creation (automated)
bash scripts/pr/quality_gate.sh 123 --with-pyscn

# Local pre-PR check
pyscn analyze .
open .pyscn/reports/analyze_*.html

# Track metrics over time
bash scripts/quality/track_pyscn_metrics.sh

# Weekly review
bash scripts/quality/weekly_quality_review.sh
```

## Health Score Thresholds

| Score | Status | Action Required |
|-------|--------|----------------|
| **<50** | 🔴 **Release Blocker** | Cannot merge - immediate refactoring required |
| **50-69** | 🟡 **Action Required** | Plan refactoring sprint within 2 weeks |
| **70-84** | ✅ **Good** | Monitor trends, continue development |
| **85+** | 🎯 **Excellent** | Maintain current standards |

## Quality Standards

**Release Blockers** (Health Score <50):
- ❌ Cannot merge to main
- ❌ Cannot create release
- 🔧 Required: Immediate refactoring

**Action Required** (Health Score 50-69):
- ⚠️ Plan refactoring sprint within 2 weeks
- 📊 Track on project board
- 🎯 Focus on top 5 complexity offenders

**Acceptable** (Health Score ≥70):
- ✅ Continue normal development
- 📈 Monitor trends monthly
- 🎯 Address new issues proactively

## Tool Complementarity

| Tool | Speed | Scope | Use Case | Blocking |
|------|-------|-------|----------|----------|
| **Groq/Gemini (pre-commit)** | <5s | Changed files | Every commit | Yes (complexity >8) |
| **quality_gate.sh** | 10-30s | PR files | PR creation | Yes (security) |
| **pyscn (PR)** | 30-60s | Full codebase | PR + periodic | Yes (health <50) |
| **code-quality-guard** | Manual | Targeted | Refactoring | No (advisory) |

## Integration Points

- **Pre-commit**: Fast LLM checks (Groq primary, Gemini fallback)
- **PR Quality Gate**: `--with-pyscn` flag for comprehensive analysis
- **Periodic**: Weekly pyscn analysis with trend tracking

## PR Quality Gate: Local Model Selection

The local quality gate reads `MCP_QUALITY_LLM_URL` (default
`http://127.0.0.1:11437/v1`) and probes the models advertised by that endpoint
in order. It pins the first model that completes the readiness request and names
it in the gate output. Set `MCP_QUALITY_LLM_MODEL` to require a specific model,
and `MCP_QUALITY_LLM_API_KEY` when the endpoint requires a bearer token.

If an oMLX endpoint returns HTTP 507 for one listed model, that model may be
unavailable even when disk space is healthy. Leave `MCP_QUALITY_LLM_MODEL`
unset to try the remaining advertised models, or set it to a known working ID.

## Pre-commit Hook Setup

```bash
# Recommended: Groq for fast, non-interactive checks
export GROQ_API_KEY="your-groq-api-key"  # Primary (200-300ms, no OAuth)

# Falls back to Gemini CLI if Groq unavailable
# Skips checks gracefully if neither available

# Enable pre-commit hook
ln -s ../../scripts/hooks/pre-commit .git/hooks/pre-commit
```

**LLM Priority:**
1. **Groq API** (Primary) - Fast (200-300ms), simple API key auth
2. **Gemini CLI** (Fallback) - Slower (2-3s), OAuth browser flow
3. **Skip checks** - If neither available, commit proceeds

See [`.claude/agents/code-quality-guard.md`](../.claude/agents/code-quality-guard.md) for detailed workflows.
